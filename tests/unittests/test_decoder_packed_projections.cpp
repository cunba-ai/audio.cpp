#include "engine/framework/core/backend.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/transformers/qwen_causal_decoder.h"
#include "engine/framework/modules/transformers/qwen_causal_decode_runtime.h"
#include "engine/framework/modules/transformers/qwen_decoder.h"
#include "engine/framework/modules/optimizations/fast_kv_modules.h"
#include "engine/framework/modules/optimizations/fast_projection_modules.h"

#include <cmath>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr size_t kGraphBytes = 16 * 1024 * 1024;
constexpr size_t kGraphNodes = 4096;

std::vector<float> patterned(size_t count, float phase, float scale) {
    std::vector<float> values(count);
    for (size_t i = 0; i < count; ++i) {
        const float x = static_cast<float>(i);
        values[i] = scale * (std::sin(phase + 0.19f * x) + 0.35f * std::cos(phase + 0.07f * x));
    }
    return values;
}

void require_allclose(
    const std::vector<float> & actual,
    const std::vector<float> & expected,
    float tolerance,
    const std::string & label) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error(label + " size mismatch");
    }
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) {
            throw std::runtime_error(label + " contains non-finite values");
        }
        const float diff = std::fabs(actual[i] - expected[i]);
        if (diff > tolerance) {
            std::ostringstream message;
            message << label << " mismatch at " << i << ": expected " << expected[i]
                    << ", got " << actual[i] << ", diff=" << diff;
            throw std::runtime_error(message.str());
        }
    }
}

struct LayerResult {
    std::vector<float> output;
    std::vector<float> key;
    std::vector<float> value;
};

LayerResult run_layer(bool packed,
    engine::core::BackendType backend_type = engine::core::BackendType::Cpu,
    bool batched_decode = false) {
    const int64_t batch = batched_decode ? 2 : 1;
    const int64_t steps = batched_decode ? 1 : 3;
    constexpr int64_t cache_steps = 8;
    constexpr int64_t hidden = 8;
    constexpr int64_t heads = 2;
    constexpr int64_t kv_heads = 1;
    constexpr int64_t head_dim = 4;
    constexpr int64_t intermediate = 12;
    constexpr int64_t q_out = heads * head_dim;
    constexpr int64_t kv_out = kv_heads * head_dim;

    engine::core::BackendConfig backend_config{backend_type, 0, 8};
    ggml_backend_t backend = engine::core::init_backend(backend_config);
    if (backend == nullptr) {
        throw std::runtime_error("failed to initialize test backend");
    }

    ggml_init_params params{kGraphBytes, nullptr, true};
    ggml_context * ggml = ggml_init(params);
    if (ggml == nullptr) {
        ggml_backend_free(backend);
        throw std::runtime_error("failed to initialize GGML context");
    }

    ggml_backend_buffer_t buffer = nullptr;
    try {
        engine::core::ModuleBuildContext ctx{ggml, "qwen_packed_projection_test", backend_type};
        auto make_f32 = [&](std::initializer_list<int64_t> dims) {
            return engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims(dims));
        };

        auto input = make_f32({batch, steps, hidden});
        auto positions = engine::core::make_tensor(
            ctx,
            GGML_TYPE_I32,
            engine::core::TensorShape::from_dims({batched_decode ? batch : steps}));

        engine::modules::QwenDecoderLayerWeights weights;
        weights.input_norm = {make_f32({hidden}), std::nullopt};
        weights.post_norm = {make_f32({hidden}), std::nullopt};
        weights.self_attention.out_weight = make_f32({hidden, hidden});
        weights.mlp.down_proj = {make_f32({hidden, intermediate}), std::nullopt};

        const auto q_values = patterned(static_cast<size_t>(q_out * hidden), 0.1f, 0.12f);
        const auto k_values = patterned(static_cast<size_t>(kv_out * hidden), 0.5f, 0.10f);
        const auto v_values = patterned(static_cast<size_t>(kv_out * hidden), 0.9f, 0.08f);
        const auto gate_values = patterned(static_cast<size_t>(intermediate * hidden), 1.3f, 0.11f);
        const auto up_values = patterned(static_cast<size_t>(intermediate * hidden), 1.7f, 0.09f);

        if (packed) {
            weights.self_attention.qkv_weight = make_f32({q_out + 2 * kv_out, hidden});
            weights.mlp.gate_up_proj = engine::modules::LinearWeights{
                make_f32({intermediate * 2, hidden}),
                std::nullopt,
            };
        } else {
            weights.self_attention.q_weight = make_f32({q_out, hidden});
            weights.self_attention.k_weight = make_f32({kv_out, hidden});
            weights.self_attention.v_weight = make_f32({kv_out, hidden});
            weights.mlp.gate_proj = {make_f32({intermediate, hidden}), std::nullopt};
            weights.mlp.up_proj = {make_f32({intermediate, hidden}), std::nullopt};
        }

        engine::modules::QwenDecoderLayerConfig config;
        config.hidden_size = hidden;
        config.num_attention_heads = heads;
        config.num_key_value_heads = kv_heads;
        config.head_dim = head_dim;
        config.intermediate_size = intermediate;
        config.rms_norm_eps = 1e-5f;
        config.qkv_layout = packed
            ? engine::modules::QwenDecoderQKVLayout::PackedQKV
            : engine::modules::QwenDecoderQKVLayout::Separate;
        config.runtime.mlp.mode = packed
            ? engine::modules::QwenDecoderMLPMode::PackedGateUp
            : engine::modules::QwenDecoderMLPMode::Exact;
        config.use_qk_norm = false;
        config.runtime.attention.prefill_mode = engine::modules::QwenDecoderAttentionMode::ManualRepeat;

        ggml_cgraph * graph = ggml_new_graph_custom(ggml, kGraphNodes, false);
        engine::modules::QwenDecoderLayerOutputs outputs;
        engine::core::TensorValue cache_key, cache_value, cache_slot, mask;
        if (batched_decode) {
            cache_key = make_f32({batch, cache_steps, kv_heads, head_dim});
            cache_value = make_f32({batch, cache_steps, kv_heads, head_dim});
            cache_slot = engine::core::make_tensor(ctx, GGML_TYPE_I32,
                engine::core::TensorShape::from_dims({batch}));
            mask = make_f32({batch, 1, 1, cache_steps});
            config.runtime.static_cache.update_mode = engine::modules::QwenDecoderStaticCacheUpdateMode::DirectSetRows;
            config.runtime.attention.static_mode = engine::modules::QwenDecoderAttentionMode::ManualRepeat;
            outputs = engine::modules::QwenDecoderLayerModule(config).build_with_static_cache_tail_batched(
                ctx, graph, input, positions, weights, cache_key, cache_value, cache_slot, mask);
        } else {
            outputs = engine::modules::QwenDecoderLayerModule(config).build(ctx, input, positions, weights);
        }
        ggml_build_forward_expand(graph, outputs.output.tensor);
        if (batched_decode) {
            int rope_nodes = 0;
            int copied_positions = 0;
            for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
                const auto * node = ggml_graph_node(graph, i);
                if (node->op != GGML_OP_ROPE) {
                    continue;
                }
                ++rope_nodes;
                if (node->src[1]->op == GGML_OP_CONT) {
                    ++copied_positions;
                }
                if (backend_type == engine::core::BackendType::Vulkan && node->src[1]->view_offs != 0) {
                    throw std::runtime_error("Vulkan batched RoPE retains an offset position view");
                }
            }
            const int expected_copies = backend_type == engine::core::BackendType::Vulkan ? 2 : 0;
            if (rope_nodes != 4 || copied_positions != expected_copies) {
                throw std::runtime_error("Unexpected batched RoPE position layout");
            }
        }
        buffer = ggml_backend_alloc_ctx_tensors(ggml, backend);
        if (buffer == nullptr) {
            throw std::runtime_error("failed to allocate test tensors");
        }

        engine::core::write_tensor_f32(input, patterned(static_cast<size_t>(batch * steps * hidden), 2.1f, 0.20f));
        engine::core::write_tensor_i32(positions, batched_decode ? std::vector<int32_t>{2, 5} : std::vector<int32_t>{0, 1, 2});
        if (batched_decode) {
            engine::core::write_tensor_f32(cache_key, patterned(batch * cache_steps * kv_out, 0.4f, 0.1f));
            engine::core::write_tensor_f32(cache_value, patterned(batch * cache_steps * kv_out, 0.8f, 0.1f));
            engine::core::write_tensor_i32(cache_slot, {2, cache_steps + 5});
            std::vector<float> mask_values(batch * cache_steps, -std::numeric_limits<float>::infinity());
            for (int64_t b = 0; b < batch; ++b) {
                for (int64_t t = 0; t <= (b == 0 ? 2 : 5); ++t) {
                    mask_values[b * cache_steps + t] = 0.0f;
                }
            }
            engine::core::write_tensor_f32(mask, mask_values);
        }
        engine::core::write_tensor_f32(*weights.input_norm.weight, patterned(hidden, 0.3f, 0.7f));
        engine::core::write_tensor_f32(*weights.post_norm.weight, patterned(hidden, 0.7f, 0.8f));
        engine::core::write_tensor_f32(
            weights.self_attention.out_weight,
            patterned(static_cast<size_t>(hidden * hidden), 1.1f, 0.10f));
        engine::core::write_tensor_f32(
            weights.mlp.down_proj.weight,
            patterned(static_cast<size_t>(hidden * intermediate), 1.9f, 0.10f));

        if (packed) {
            std::vector<float> qkv_values;
            qkv_values.reserve(q_values.size() + k_values.size() + v_values.size());
            qkv_values.insert(qkv_values.end(), q_values.begin(), q_values.end());
            qkv_values.insert(qkv_values.end(), k_values.begin(), k_values.end());
            qkv_values.insert(qkv_values.end(), v_values.begin(), v_values.end());
            engine::core::write_tensor_f32(*weights.self_attention.qkv_weight, qkv_values);

            std::vector<float> gate_up_values;
            gate_up_values.reserve(gate_values.size() + up_values.size());
            gate_up_values.insert(gate_up_values.end(), gate_values.begin(), gate_values.end());
            gate_up_values.insert(gate_up_values.end(), up_values.begin(), up_values.end());
            engine::core::write_tensor_f32(weights.mlp.gate_up_proj->weight, gate_up_values);
        } else {
            engine::core::write_tensor_f32(weights.self_attention.q_weight, q_values);
            engine::core::write_tensor_f32(weights.self_attention.k_weight, k_values);
            engine::core::write_tensor_f32(weights.self_attention.v_weight, v_values);
            engine::core::write_tensor_f32(weights.mlp.gate_proj.weight, gate_values);
            engine::core::write_tensor_f32(weights.mlp.up_proj.weight, up_values);
        }

        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("test graph execution failed");
        }
        LayerResult result;
        engine::core::read_tensor_f32_into(outputs.output.tensor, result.output);
        engine::core::read_tensor_f32_into(outputs.key.tensor, result.key);
        engine::core::read_tensor_f32_into(outputs.value.tensor, result.value);

        ggml_backend_buffer_free(buffer);
        buffer = nullptr;
        ggml_free(ggml);
        ggml_backend_free(backend);
        return result;
    } catch (...) {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
        ggml_free(ggml);
        ggml_backend_free(backend);
        throw;
    }
}

void test_packed_qkv_and_gate_up_match_separate_projections() {
    const auto separate = run_layer(false);
    const auto packed = run_layer(true);
    require_allclose(packed.output, separate.output, 2.0e-5f, "decoder output");
    require_allclose(packed.key, separate.key, 2.0e-5f, "decoder key");
    require_allclose(packed.value, separate.value, 2.0e-5f, "decoder value");
}

void test_suffix_causal_mask() {
    const auto values = engine::modules::qwen_causal_suffix_mask_values(2, 3, 2);
    if (values.size() != 30) {
        throw std::runtime_error("suffix causal mask size mismatch");
    }
    const std::vector<bool> expected{
        true, true, true, false, false,
        true, true, true, true, false,
        true, true, true, true, true,
    };
    for (int batch = 0; batch < 2; ++batch) {
        for (size_t i = 0; i < expected.size(); ++i) {
            const float actual = ggml_fp16_to_fp32(values[static_cast<size_t>(batch) * expected.size() + i]);
            if ((expected[i] && actual != 0.0F) || (!expected[i] && !std::isinf(actual))) {
                throw std::runtime_error("suffix causal mask visibility mismatch");
            }
        }
    }
}

void test_f16_kv_set_rows() {
    engine::core::BackendConfig backend_config{engine::core::BackendType::Cpu, 0, 4};
    ggml_backend_t backend = engine::core::init_backend(backend_config);
    if (backend == nullptr) {
        throw std::runtime_error("failed to initialize CPU backend");
    }

    ggml_init_params params{kGraphBytes, nullptr, true};
    ggml_context * ggml = ggml_init(params);
    if (ggml == nullptr) {
        ggml_backend_free(backend);
        throw std::runtime_error("failed to initialize GGML context");
    }

    ggml_backend_buffer_t buffer = nullptr;
    try {
        engine::core::ModuleBuildContext ctx{ggml, "f16_kv_set_rows_test", engine::core::BackendType::Cpu};
        const auto cache = engine::core::make_tensor(
            ctx,
            GGML_TYPE_F16,
            engine::core::TensorShape::from_dims({1, 3, 1, 2}));
        const auto row = engine::core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({1, 1, 1, 2}));
        const auto row_index = engine::core::make_tensor(
            ctx,
            GGML_TYPE_I64,
            engine::core::TensorShape::from_dims({1}));
        const auto output = engine::modules::FastKVSetRowsModule({
            engine::modules::FastKVSetRowsMode::BackendViewOptimized,
        }).build(ctx, cache, row, row_index);

        ggml_cgraph * graph = ggml_new_graph_custom(ggml, kGraphNodes, false);
        ggml_build_forward_expand(graph, output.tensor);
        buffer = ggml_backend_alloc_ctx_tensors(ggml, backend);
        if (buffer == nullptr) {
            throw std::runtime_error("failed to allocate f16 KV test tensors");
        }

        engine::core::write_tensor_f16(cache, std::vector<float>(6, 0.0F));
        engine::core::write_tensor_f32(row, {1.25F, -2.5F});
        const int64_t index = 1;
        ggml_backend_tensor_set(row_index.tensor, &index, 0, sizeof(index));
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("f16 KV set-rows graph compute failed");
        }

        const auto values = engine::core::read_tensor_f16(output.tensor);
        require_allclose(values, {0.0F, 0.0F, 1.25F, -2.5F, 0.0F, 0.0F}, 1.0e-3F, "f16 KV cache");

        ggml_backend_buffer_free(buffer);
        buffer = nullptr;
        ggml_free(ggml);
        ggml_backend_free(backend);
    } catch (...) {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
        ggml_free(ggml);
        ggml_backend_free(backend);
        throw;
    }
}

void test_f16_kv_set_rows_batched() {
    engine::core::BackendConfig backend_config{engine::core::BackendType::Cpu, 0, 4};
    ggml_backend_t backend = engine::core::init_backend(backend_config);
    if (backend == nullptr) {
        throw std::runtime_error("failed to initialize CPU backend");
    }

    ggml_init_params params{kGraphBytes, nullptr, true};
    ggml_context * ggml = ggml_init(params);
    if (ggml == nullptr) {
        ggml_backend_free(backend);
        throw std::runtime_error("failed to initialize GGML context");
    }

    ggml_backend_buffer_t buffer = nullptr;
    try {
        engine::core::ModuleBuildContext ctx{ggml, "f16_kv_set_rows_batched_test", engine::core::BackendType::Cpu};
        const auto cache = engine::core::make_tensor(
            ctx,
            GGML_TYPE_F16,
            engine::core::TensorShape::from_dims({2, 3, 1, 2}));
        const auto rows = engine::core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({2, 1, 1, 2}));
        const auto row_indices = engine::core::make_tensor(
            ctx,
            GGML_TYPE_I64,
            engine::core::TensorShape::from_dims({2}));
        const auto output = engine::modules::FastKVSetRowsModule({
            engine::modules::FastKVSetRowsMode::BackendViewOptimized,
        }).build(ctx, cache, rows, row_indices);

        ggml_cgraph * graph = ggml_new_graph_custom(ggml, kGraphNodes, false);
        ggml_build_forward_expand(graph, output.tensor);
        buffer = ggml_backend_alloc_ctx_tensors(ggml, backend);
        if (buffer == nullptr) {
            throw std::runtime_error("failed to allocate batched F16 KV test tensors");
        }

        engine::core::write_tensor_f16(cache, std::vector<float>(12, 0.0F));
        engine::core::write_tensor_f32(rows, {1.25F, -2.5F, 3.5F, -4.5F});
        const std::vector<int64_t> indices{1, 4};
        ggml_backend_tensor_set(row_indices.tensor, indices.data(), 0, indices.size() * sizeof(int64_t));
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("batched F16 KV set-rows graph compute failed");
        }

        const auto values = engine::core::read_tensor_f16(output.tensor);
        require_allclose(
            values,
            {0.0F, 0.0F, 1.25F, -2.5F, 0.0F, 0.0F,
             0.0F, 0.0F, 3.5F, -4.5F, 0.0F, 0.0F},
            1.0e-3F,
            "batched f16 KV cache");

        ggml_backend_buffer_free(buffer);
        buffer = nullptr;
        ggml_free(ggml);
        ggml_backend_free(backend);
    } catch (...) {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
        ggml_free(ggml);
        ggml_backend_free(backend);
        throw;
    }
}

int count_graph_op(ggml_cgraph * graph, ggml_op op) {
    int count = 0;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        const ggml_tensor * node = ggml_graph_node(graph, i);
        count += node != nullptr && node->op == op ? 1 : 0;
    }
    return count;
}

bool graph_contains_sequence(ggml_cgraph * graph, std::initializer_list<ggml_op> ops) {
    if (ops.size() == 0 || static_cast<int>(ops.size()) > ggml_graph_n_nodes(graph)) {
        return false;
    }
    for (int start = 0; start + static_cast<int>(ops.size()) <= ggml_graph_n_nodes(graph); ++start) {
        bool matches = true;
        int offset = 0;
        for (const ggml_op op : ops) {
            const ggml_tensor * node = ggml_graph_node(graph, start + offset++);
            matches = matches && node != nullptr && node->op == op;
        }
        if (matches) {
            return true;
        }
    }
    return false;
}

std::string graph_ops(ggml_cgraph * graph) {
    std::ostringstream out;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        const ggml_tensor * node = ggml_graph_node(graph, i);
        if (i != 0) {
            out << ',';
        }
        out << (node != nullptr ? ggml_op_name(node->op) : "null");
    }
    return out.str();
}

void test_fast_projection_accepts_cuda_or_hip_backends() {
    constexpr int64_t in_features = 8;
    constexpr int64_t out_features = 12;
    for (const auto backend_type : {
             engine::core::BackendType::Cuda,
             engine::core::BackendType::Hip,
         }) {
        ggml_init_params params{kGraphBytes, nullptr, true};
        ggml_context * ggml = ggml_init(params);
        if (ggml == nullptr) {
            throw std::runtime_error("failed to initialize fast projection graph test context");
        }
        try {
            engine::core::ModuleBuildContext ctx{ggml, "fast_projection_cuda_or_hip_test", backend_type};
            const auto input = engine::core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                engine::core::TensorShape::from_dims({1, in_features}));
            const auto weight = engine::core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                engine::core::TensorShape::from_dims({out_features, in_features}));
            const auto output = engine::modules::FastPackedProjection4Module({in_features, out_features})
                                    .build(ctx, input, {weight, std::nullopt});
            if (output.shape.rank != 2 || output.shape.dims[0] != 1 || output.shape.dims[1] != out_features) {
                throw std::runtime_error("fast projection CUDA/HIP output shape mismatch");
            }
            ggml_free(ggml);
        } catch (...) {
            ggml_free(ggml);
            throw;
        }
    }

    for (const auto rejected_backend : {engine::core::BackendType::Cpu, engine::core::BackendType::Metal}) {
        ggml_init_params params{kGraphBytes, nullptr, true};
        ggml_context * ggml = ggml_init(params);
        if (ggml == nullptr) {
            throw std::runtime_error("failed to initialize fast projection rejection test context");
        }
        bool rejected = false;
        try {
            engine::core::ModuleBuildContext ctx{ggml, "fast_projection_rejection_test", rejected_backend};
            const auto input = engine::core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                engine::core::TensorShape::from_dims({1, in_features}));
            const auto weight = engine::core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                engine::core::TensorShape::from_dims({out_features, in_features}));
            try {
                (void) engine::modules::FastPackedProjection4Module({in_features, out_features})
                    .build(ctx, input, {weight, std::nullopt});
            } catch (const std::runtime_error &) {
                rejected = true;
            }
            ggml_free(ggml);
        } catch (...) {
            ggml_free(ggml);
            throw;
        }
        if (!rejected) {
            throw std::runtime_error("fast projection must reject non-CUDA/HIP backends");
        }
    }
}

// Single-token (or batched single-token) static-cache decode tail with
// optional fused q/k norms and optional packed q|k|v biases: the graph shapes
// the QwenCausalDecodeRuntime families build on their hot decode path. When
// `fused_qk_norm` is set the packed run uses qk_norm_packed + PackedQKV and
// must hit the fused 1-projection + 1-norm + 1-rope fast path (verified via
// the graph op counts); when `packed_bias` is set the run compares the packed
// qkv weight|bias against three separate biased projections.
struct DecodeLayerResult {
    LayerResult layer;
    int rope_nodes = 0;
    int rms_norm_nodes = 0;
    int mul_mat_nodes = 0;
};

enum class DecodeVariant {
    Separate,
    PackedFusedQKNorm,
    SeparateBias,
    PackedBias,
};

DecodeLayerResult run_decode_layer(DecodeVariant variant, bool batched_decode) {
    const int64_t batch = batched_decode ? 2 : 1;
    constexpr int64_t cache_steps = 8;
    constexpr int64_t hidden = 8;
    constexpr int64_t heads = 2;
    constexpr int64_t kv_heads = 1;
    constexpr int64_t head_dim = 4;
    constexpr int64_t intermediate = 12;
    constexpr int64_t q_out = heads * head_dim;
    constexpr int64_t kv_out = kv_heads * head_dim;
    const bool packed = variant == DecodeVariant::PackedFusedQKNorm || variant == DecodeVariant::PackedBias;
    const bool biased = variant == DecodeVariant::SeparateBias || variant == DecodeVariant::PackedBias;
    const bool fused_qk_norm = variant == DecodeVariant::PackedFusedQKNorm;

    engine::core::BackendConfig backend_config{engine::core::BackendType::Cpu, 0, 8};
    ggml_backend_t backend = engine::core::init_backend(backend_config);
    if (backend == nullptr) {
        throw std::runtime_error("failed to initialize test backend");
    }

    ggml_init_params params{kGraphBytes, nullptr, true};
    ggml_context * ggml = ggml_init(params);
    if (ggml == nullptr) {
        ggml_backend_free(backend);
        throw std::runtime_error("failed to initialize GGML context");
    }

    ggml_backend_buffer_t buffer = nullptr;
    try {
        engine::core::ModuleBuildContext ctx{ggml, "qwen_packed_decode_tail_test", engine::core::BackendType::Cpu};
        auto make_f32 = [&](std::initializer_list<int64_t> dims) {
            return engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims(dims));
        };

        const auto input = make_f32({batch, 1, hidden});
        const auto positions = engine::core::make_tensor(
            ctx,
            GGML_TYPE_I32,
            engine::core::TensorShape::from_dims({batched_decode ? batch : 1}));
        const auto cache_key = make_f32({batch, cache_steps, kv_heads, head_dim});
        const auto cache_value = make_f32({batch, cache_steps, kv_heads, head_dim});
        const auto cache_slot = engine::core::make_tensor(
            ctx,
            GGML_TYPE_I32,
            engine::core::TensorShape::from_dims({batched_decode ? batch : 1}));
        // FlashGroupedViewKV static attention consumes an F16 additive mask.
        const auto mask = engine::core::make_tensor(
            ctx,
            GGML_TYPE_F16,
            engine::core::TensorShape::from_dims({batch, 1, 1, cache_steps}));

        engine::modules::QwenDecoderLayerWeights weights;
        weights.input_norm = {make_f32({hidden}), std::nullopt};
        weights.post_norm = {make_f32({hidden}), std::nullopt};
        weights.self_attention.out_weight = make_f32({hidden, hidden});
        weights.mlp.down_proj = {make_f32({hidden, intermediate}), std::nullopt};

        const auto q_values = patterned(static_cast<size_t>(q_out * hidden), 0.11f, 0.12f);
        const auto k_values = patterned(static_cast<size_t>(kv_out * hidden), 0.51f, 0.10f);
        const auto v_values = patterned(static_cast<size_t>(kv_out * hidden), 0.91f, 0.08f);
        const auto q_bias_values = patterned(static_cast<size_t>(q_out), 0.13f, 0.30f);
        const auto k_bias_values = patterned(static_cast<size_t>(kv_out), 0.53f, 0.20f);
        const auto v_bias_values = patterned(static_cast<size_t>(kv_out), 0.93f, 0.10f);
        std::vector<float> qkv_bias_values;
        const auto gate_values = patterned(static_cast<size_t>(intermediate * hidden), 1.31f, 0.11f);
        const auto up_values = patterned(static_cast<size_t>(intermediate * hidden), 1.71f, 0.09f);
        const auto q_norm_values = patterned(static_cast<size_t>(head_dim), 0.21f, 0.4f);
        const auto k_norm_values = patterned(static_cast<size_t>(head_dim), 0.61f, 0.3f);

        if (packed) {
            weights.self_attention.qkv_weight = make_f32({q_out + 2 * kv_out, hidden});
            if (biased) {
                qkv_bias_values.reserve(q_bias_values.size() + k_bias_values.size() + v_bias_values.size());
                qkv_bias_values.insert(qkv_bias_values.end(), q_bias_values.begin(), q_bias_values.end());
                qkv_bias_values.insert(qkv_bias_values.end(), k_bias_values.begin(), k_bias_values.end());
                qkv_bias_values.insert(qkv_bias_values.end(), v_bias_values.begin(), v_bias_values.end());
                weights.self_attention.qkv_bias = make_f32({q_out + 2 * kv_out});
            }
            weights.mlp.gate_up_proj = engine::modules::LinearWeights{
                make_f32({intermediate * 2, hidden}),
                std::nullopt,
            };
        } else {
            weights.self_attention.q_weight = make_f32({q_out, hidden});
            weights.self_attention.k_weight = make_f32({kv_out, hidden});
            weights.self_attention.v_weight = make_f32({kv_out, hidden});
            if (biased) {
                weights.self_attention.q_bias = make_f32({q_out});
                weights.self_attention.k_bias = make_f32({kv_out});
                weights.self_attention.v_bias = make_f32({kv_out});
            }
            weights.mlp.gate_proj = {make_f32({intermediate, hidden}), std::nullopt};
            weights.mlp.up_proj = {make_f32({intermediate, hidden}), std::nullopt};
        }
        std::vector<float> qk_norm_tiled;
        if (!biased) {
            weights.q_norm = {make_f32({head_dim}), std::nullopt};
            weights.k_norm = {make_f32({head_dim}), std::nullopt};
        }
        if (fused_qk_norm) {
            qk_norm_tiled.reserve(static_cast<size_t>((heads + kv_heads) * head_dim));
            for (int64_t head = 0; head < heads; ++head) {
                qk_norm_tiled.insert(qk_norm_tiled.end(), q_norm_values.begin(), q_norm_values.end());
            }
            for (int64_t head = 0; head < kv_heads; ++head) {
                qk_norm_tiled.insert(qk_norm_tiled.end(), k_norm_values.begin(), k_norm_values.end());
            }
            weights.qk_norm_packed = make_f32({heads + kv_heads, head_dim});
        }

        engine::modules::QwenDecoderLayerConfig config;
        config.hidden_size = hidden;
        config.num_attention_heads = heads;
        config.num_key_value_heads = kv_heads;
        config.head_dim = head_dim;
        config.intermediate_size = intermediate;
        config.rms_norm_eps = 1e-5f;
        config.qkv_layout = packed
            ? engine::modules::QwenDecoderQKVLayout::PackedQKV
            : engine::modules::QwenDecoderQKVLayout::Separate;
        config.runtime.mlp.mode = packed
            ? engine::modules::QwenDecoderMLPMode::PackedGateUp
            : engine::modules::QwenDecoderMLPMode::Exact;
        config.use_qk_norm = !biased;
        config.runtime.attention.static_mode = engine::modules::QwenDecoderAttentionMode::FlashGroupedViewKV;
        config.runtime.static_cache.update_mode =
            engine::modules::QwenDecoderStaticCacheUpdateMode::DirectSetRows;
        config.runtime.static_cache.set_rows_mode =
            engine::modules::QwenDecoderStaticCacheSetRowsMode::BackendViewOptimized;

        ggml_cgraph * graph = ggml_new_graph_custom(ggml, kGraphNodes, false);
        engine::modules::QwenDecoderLayerOutputs outputs;
        if (batched_decode) {
            outputs = engine::modules::QwenDecoderLayerModule(config).build_with_static_cache_tail_batched(
                ctx, graph, input, positions, weights, cache_key, cache_value, cache_slot, mask);
        } else {
            outputs = engine::modules::QwenDecoderLayerModule(config).build_with_static_cache_tail(
                ctx, graph, input, positions, weights, cache_key, cache_value, cache_slot, mask);
        }
        ggml_build_forward_expand(graph, outputs.output.tensor);

        DecodeLayerResult result;
        result.rope_nodes = count_graph_op(graph, GGML_OP_ROPE);
        result.rms_norm_nodes = count_graph_op(graph, GGML_OP_RMS_NORM);
        result.mul_mat_nodes = count_graph_op(graph, GGML_OP_MUL_MAT);

        buffer = ggml_backend_alloc_ctx_tensors(ggml, backend);
        if (buffer == nullptr) {
            throw std::runtime_error("failed to allocate test tensors");
        }

        engine::core::write_tensor_f32(input, patterned(static_cast<size_t>(batch * hidden), 2.1f, 0.20f));
        engine::core::write_tensor_i32(positions, batched_decode
            ? std::vector<int32_t>{2, 5}
            : std::vector<int32_t>{2});
        engine::core::write_tensor_f32(cache_key, patterned(batch * cache_steps * kv_out, 0.4f, 0.1f));
        engine::core::write_tensor_f32(cache_value, patterned(batch * cache_steps * kv_out, 0.8f, 0.1f));
        if (batched_decode) {
            engine::core::write_tensor_i32(cache_slot, {2, cache_steps + 5});
        } else {
            engine::core::write_tensor_i32(cache_slot, {2});
        }
        std::vector<float> mask_values(batch * cache_steps, -std::numeric_limits<float>::infinity());
        for (int64_t b = 0; b < batch; ++b) {
            for (int t = 0; t <= 2; ++t) {
                mask_values[b * cache_steps + t] = 0.0f;
            }
        }
        engine::core::write_tensor_f16(mask, mask_values);
        engine::core::write_tensor_f32(*weights.input_norm.weight, patterned(hidden, 0.3f, 0.7f));
        engine::core::write_tensor_f32(*weights.post_norm.weight, patterned(hidden, 0.7f, 0.8f));
        if (!biased) {
            engine::core::write_tensor_f32(*weights.q_norm.weight, q_norm_values);
            engine::core::write_tensor_f32(*weights.k_norm.weight, k_norm_values);
        }
        if (fused_qk_norm) {
            engine::core::write_tensor_f32(*weights.qk_norm_packed, qk_norm_tiled);
        }
        engine::core::write_tensor_f32(
            weights.self_attention.out_weight,
            patterned(static_cast<size_t>(hidden * hidden), 1.1f, 0.10f));
        engine::core::write_tensor_f32(
            weights.mlp.down_proj.weight,
            patterned(static_cast<size_t>(hidden * intermediate), 1.9f, 0.10f));

        if (packed) {
            std::vector<float> qkv_values;
            qkv_values.reserve(q_values.size() + k_values.size() + v_values.size());
            qkv_values.insert(qkv_values.end(), q_values.begin(), q_values.end());
            qkv_values.insert(qkv_values.end(), k_values.begin(), k_values.end());
            qkv_values.insert(qkv_values.end(), v_values.begin(), v_values.end());
            engine::core::write_tensor_f32(*weights.self_attention.qkv_weight, qkv_values);
            if (biased) {
                engine::core::write_tensor_f32(*weights.self_attention.qkv_bias, qkv_bias_values);
            }

            std::vector<float> gate_up_values;
            gate_up_values.reserve(gate_values.size() + up_values.size());
            gate_up_values.insert(gate_up_values.end(), gate_values.begin(), gate_values.end());
            gate_up_values.insert(gate_up_values.end(), up_values.begin(), up_values.end());
            engine::core::write_tensor_f32(weights.mlp.gate_up_proj->weight, gate_up_values);
        } else {
            engine::core::write_tensor_f32(weights.self_attention.q_weight, q_values);
            engine::core::write_tensor_f32(weights.self_attention.k_weight, k_values);
            engine::core::write_tensor_f32(weights.self_attention.v_weight, v_values);
            if (biased) {
                engine::core::write_tensor_f32(*weights.self_attention.q_bias, q_bias_values);
                engine::core::write_tensor_f32(*weights.self_attention.k_bias, k_bias_values);
                engine::core::write_tensor_f32(*weights.self_attention.v_bias, v_bias_values);
            }
            engine::core::write_tensor_f32(weights.mlp.gate_proj.weight, gate_values);
            engine::core::write_tensor_f32(weights.mlp.up_proj.weight, up_values);
        }

        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("decode tail graph execution failed");
        }
        engine::core::read_tensor_f32_into(outputs.output.tensor, result.layer.output);
        engine::core::read_tensor_f32_into(outputs.key.tensor, result.layer.key);
        engine::core::read_tensor_f32_into(outputs.value.tensor, result.layer.value);

        ggml_backend_buffer_free(buffer);
        buffer = nullptr;
        ggml_free(ggml);
        ggml_backend_free(backend);
        return result;
    } catch (...) {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
        ggml_free(ggml);
        ggml_backend_free(backend);
        throw;
    }
}

void test_fused_qkv_qk_norm_decode_tail_matches_separate() {
    for (bool batched : {false, true}) {
        const auto separate = run_decode_layer(DecodeVariant::Separate, batched);
        // Packed-vs-separate GEMMs accumulate in different orders; both compute
        // the same math, so allow fp reassociation slack beyond the layer test.
        const auto packed = run_decode_layer(DecodeVariant::PackedFusedQKNorm, batched);
        require_allclose(packed.layer.output, separate.layer.output, 1.0e-4f, "fused decode output");
        require_allclose(packed.layer.key, separate.layer.key, 1.0e-4f, "fused decode key");
        require_allclose(packed.layer.value, separate.layer.value, 1.0e-4f, "fused decode value");
        // The fused fast path must actually be engaged: exactly one rope over
        // the packed q|k rows (the per-row-position separate path slices and
        // ropes each row's q and k individually), one fused qk RMSNorm instead
        // of separate q/k norms (input + post norms are counted on both
        // sides), and strictly fewer projection GEMMs.
        if (packed.rope_nodes != 1 ||
            packed.rope_nodes >= separate.rope_nodes ||
            packed.rms_norm_nodes != separate.rms_norm_nodes - 1 ||
            packed.mul_mat_nodes >= separate.mul_mat_nodes) {
            std::ostringstream message;
            message << "fused decode fast path not engaged (batched=" << batched
                    << "): rope " << separate.rope_nodes << "->" << packed.rope_nodes
                    << ", rms_norm " << separate.rms_norm_nodes << "->" << packed.rms_norm_nodes
                    << ", mul_mat " << separate.mul_mat_nodes << "->" << packed.mul_mat_nodes;
            throw std::runtime_error(message.str());
        }
    }
}

void test_packed_qkv_bias_decode_tail_matches_separate() {
    for (bool batched : {false, true}) {
        const auto separate = run_decode_layer(DecodeVariant::SeparateBias, batched);
        const auto packed = run_decode_layer(DecodeVariant::PackedBias, batched);
        require_allclose(packed.layer.output, separate.layer.output, 1.0e-4f, "packed-bias decode output");
        require_allclose(packed.layer.key, separate.layer.key, 1.0e-4f, "packed-bias decode key");
        require_allclose(packed.layer.value, separate.layer.value, 1.0e-4f, "packed-bias decode value");
        if (packed.mul_mat_nodes >= separate.mul_mat_nodes) {
            throw std::runtime_error("packed-bias decode did not reduce projection GEMMs");
        }
    }
}

// Runtime-level equivalence: a full QwenCausalDecodeRuntime stack with the
// packed loading (fused qk-norm variant and packed-bias variant) must produce
// the same logits as the separately loaded stack across prefill, decode and
// the batched decode path - the exact graphs the integrated model families
// build.
void test_runtime_packed_matches_separate(bool biased) {
    constexpr int64_t hidden = 64;
    constexpr int64_t heads = 4;
    constexpr int64_t kv_heads = 2;
    constexpr int64_t head_dim = 16;
    constexpr int64_t intermediate = 128;
    constexpr int64_t layers = 2;
    constexpr int64_t vocab = 97;
    constexpr int64_t q_out = heads * head_dim;
    constexpr int64_t kv_out = kv_heads * head_dim;

    engine::core::BackendConfig backend_config{engine::core::BackendType::Cpu, 0, 8};
    engine::core::ExecutionContext execution(backend_config);
    auto * context = ggml_init({8 * 1024 * 1024, nullptr, true});
    if (context == nullptr) {
        throw std::runtime_error("weight context allocation failed");
    }
    struct ContextGuard { ggml_context * p; ~ContextGuard() { ggml_free(p); } } context_guard{context};
    engine::core::ModuleBuildContext ctx{context, "qwen_packed_runtime_test", engine::core::BackendType::Cpu};
    std::vector<engine::core::TensorValue> tensors;
    auto tensor = [&](std::initializer_list<int64_t> shape) {
        auto value = engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims(shape));
        tensors.push_back(value);
        return value;
    };

    const auto q_norm_values = patterned(static_cast<size_t>(head_dim), 0.21f, 0.4f);
    const auto k_norm_values = patterned(static_cast<size_t>(head_dim), 0.61f, 0.3f);
    std::vector<float> qk_norm_tiled;
    for (int64_t head = 0; head < heads; ++head) {
        qk_norm_tiled.insert(qk_norm_tiled.end(), q_norm_values.begin(), q_norm_values.end());
    }
    for (int64_t head = 0; head < kv_heads; ++head) {
        qk_norm_tiled.insert(qk_norm_tiled.end(), k_norm_values.begin(), k_norm_values.end());
    }

    auto make_weights = [&](bool packed) {
        engine::modules::QwenCausalDecodeRuntimeWeights weights;
        weights.token_embedding = tensor({vocab, hidden});
        weights.final_norm = {tensor({hidden}), std::nullopt};
        weights.lm_head = engine::modules::LinearWeights{weights.token_embedding, std::nullopt};
        for (int layer = 0; layer < layers; ++layer) {
            engine::modules::QwenDecoderLayerWeights w;
            w.input_norm = {tensor({hidden}), std::nullopt};
            w.post_norm = {tensor({hidden}), std::nullopt};
            if (!biased) {
                w.q_norm = {tensor({head_dim}), std::nullopt};
                w.k_norm = {tensor({head_dim}), std::nullopt};
            }
            if (packed) {
                w.self_attention.qkv_weight = tensor({q_out + 2 * kv_out, hidden});
                if (biased) {
                    w.self_attention.qkv_bias = tensor({q_out + 2 * kv_out});
                }
                if (!biased) {
                    w.qk_norm_packed = tensor({heads + kv_heads, head_dim});
                }
                w.mlp.gate_up_proj = engine::modules::LinearWeights{tensor({intermediate * 2, hidden}), std::nullopt};
            } else {
                w.self_attention.q_weight = tensor({q_out, hidden});
                w.self_attention.k_weight = tensor({kv_out, hidden});
                w.self_attention.v_weight = tensor({kv_out, hidden});
                if (biased) {
                    w.self_attention.q_bias = tensor({q_out});
                    w.self_attention.k_bias = tensor({kv_out});
                    w.self_attention.v_bias = tensor({kv_out});
                }
                w.mlp.gate_proj = {tensor({intermediate, hidden}), std::nullopt};
                w.mlp.up_proj = {tensor({intermediate, hidden}), std::nullopt};
            }
            w.self_attention.out_weight = tensor({hidden, q_out});
            w.mlp.down_proj = {tensor({hidden, intermediate}), std::nullopt};
            weights.stack.layers.push_back(w);
        }
        return weights;
    };

    auto separate_weights = make_weights(false);
    auto packed_weights = make_weights(true);
    auto * buffer = ggml_backend_alloc_ctx_tensors(context, execution.backend());
    if (buffer == nullptr) {
        throw std::runtime_error("weight buffer allocation failed");
    }
    struct BufferGuard { ggml_backend_buffer_t p; ~BufferGuard() { ggml_backend_buffer_free(p); } } buffer_guard{buffer};
    for (const auto & value : tensors) {
        auto values = patterned(static_cast<size_t>(ggml_nelements(value.tensor)), 0.05f, 0.3f);
        engine::core::write_tensor_f32(value, values);
    }
    // Rebuild the packed tensors as exact concatenations of the separate ones
    // (and tile the q/k norm weights) so both runtimes see identical values.
    const auto read_f32 = [](const engine::core::TensorValue & value) {
        std::vector<float> out;
        engine::core::read_tensor_f32_into(value.tensor, out);
        return out;
    };
    {
        const auto q_ref = read_f32(separate_weights.stack.layers[0].self_attention.q_weight);
        const auto k_ref = read_f32(separate_weights.stack.layers[0].self_attention.k_weight);
        const auto v_ref = read_f32(separate_weights.stack.layers[0].self_attention.v_weight);
        const auto gate_ref = read_f32(separate_weights.stack.layers[0].mlp.gate_proj.weight);
        const auto up_ref = read_f32(separate_weights.stack.layers[0].mlp.up_proj.weight);
        std::vector<float> qkv_ref;
        qkv_ref.reserve(q_ref.size() + k_ref.size() + v_ref.size());
        qkv_ref.insert(qkv_ref.end(), q_ref.begin(), q_ref.end());
        qkv_ref.insert(qkv_ref.end(), k_ref.begin(), k_ref.end());
        qkv_ref.insert(qkv_ref.end(), v_ref.begin(), v_ref.end());
        std::vector<float> gate_up_ref;
        gate_up_ref.reserve(gate_ref.size() + up_ref.size());
        gate_up_ref.insert(gate_up_ref.end(), gate_ref.begin(), gate_ref.end());
        gate_up_ref.insert(gate_up_ref.end(), up_ref.begin(), up_ref.end());
        for (int layer = 0; layer < layers; ++layer) {
            auto & w = packed_weights.stack.layers[layer];
            engine::core::write_tensor_f32(*w.self_attention.qkv_weight, qkv_ref);
            if (biased) {
                std::vector<float> qkv_bias_ref;
                const auto q_bias = read_f32(*separate_weights.stack.layers[0].self_attention.q_bias);
                const auto k_bias = read_f32(*separate_weights.stack.layers[0].self_attention.k_bias);
                const auto v_bias = read_f32(*separate_weights.stack.layers[0].self_attention.v_bias);
                qkv_bias_ref.insert(qkv_bias_ref.end(), q_bias.begin(), q_bias.end());
                qkv_bias_ref.insert(qkv_bias_ref.end(), k_bias.begin(), k_bias.end());
                qkv_bias_ref.insert(qkv_bias_ref.end(), v_bias.begin(), v_bias.end());
                engine::core::write_tensor_f32(*w.self_attention.qkv_bias, qkv_bias_ref);
            } else {
                engine::core::write_tensor_f32(*w.qk_norm_packed, qk_norm_tiled);
            }
            engine::core::write_tensor_f32(w.mlp.gate_up_proj->weight, gate_up_ref);
        }
    }

    auto make_config = [&](bool packed) {
        engine::modules::QwenCausalDecodeRuntimeConfig config;
        config.trace_name = biased ? "qwen_packed_bias_runtime_test" : "qwen_packed_fused_runtime_test";
        auto & stack = config.decoder.stack;
        stack.hidden_size = hidden;
        stack.num_attention_heads = heads;
        stack.num_key_value_heads = kv_heads;
        stack.head_dim = head_dim;
        stack.intermediate_size = intermediate;
        stack.layers = layers;
        stack.rms_norm_eps = 1e-6f;
        stack.use_qk_norm = !biased;
        stack.qkv_layout = packed
            ? engine::modules::QwenDecoderQKVLayout::PackedQKV
            : engine::modules::QwenDecoderQKVLayout::Separate;
        stack.runtime.mlp.mode = packed
            ? engine::modules::QwenDecoderMLPMode::PackedGateUp
            : engine::modules::QwenDecoderMLPMode::Exact;
        stack.runtime.attention.prefill_mode = engine::modules::QwenDecoderAttentionMode::FlashGroupedViewKV;
        stack.runtime.attention.static_mode = engine::modules::QwenDecoderAttentionMode::FlashGroupedViewKV;
        stack.runtime.static_cache.update_mode =
            engine::modules::QwenDecoderStaticCacheUpdateMode::DirectSetRows;
        stack.runtime.static_cache.set_rows_mode =
            engine::modules::QwenDecoderStaticCacheSetRowsMode::BackendViewOptimized;
        config.decoder.logits_size = vocab;
        config.decoder.static_cache_type = GGML_TYPE_F16;
        config.prefill_graph_arena_bytes = 8 * 1024 * 1024;
        config.decode_graph_arena_bytes = 8 * 1024 * 1024;
        return config;
    };

    engine::modules::QwenCausalDecodeRuntime separate(execution, make_config(false), separate_weights);
    engine::modules::QwenCausalDecodeRuntime packed(execution, make_config(true), packed_weights);

    const std::vector<int32_t> prompt{3, 17, 91, 42, 8, 64, 1, 29, 55, 12};
    auto expected = separate.prefill_tokens(prompt);
    auto actual = packed.prefill_tokens(prompt);
    require_allclose(actual.logits, expected.logits, 3.0e-3f, "packed prefill logits");

    separate.start_decode_tokens(expected.state, 32);
    packed.start_decode_tokens(expected.state, 32);
    // Decode drift bound: the packed-bias variant (packed GEMMs + swiglu, no
    // fused norms) is bit-exact against separate decode, and the fused
    // qk-norm variant is bit-exact at the single-layer level; the remaining
    // bounded drift here (~1e-2 on logits of order 1, non-diverging over
    // steps) is the fp reassociation floor of the fused rms_norm/rope shapes
    // amplified through the stack. Wrong-row/layout bugs produce O(1)
    // diverging differences, far above this bound.
    for (int token = 0; token < 5; ++token) {
        const auto reference_logits = separate.decode_token(token + 7).logits;
        const auto actual_logits = packed.decode_token(token + 7).logits;
        require_allclose(actual_logits, reference_logits, 0.15F, "packed decode logits");
    }

    // Batched decode path (music3-style CFG hot loop).
    constexpr int64_t batch = 2;
    std::vector<int32_t> batched_prompt;
    batched_prompt.reserve(prompt.size() * batch);
    batched_prompt.insert(batched_prompt.end(), prompt.begin(), prompt.end());
    batched_prompt.insert(batched_prompt.end(), prompt.begin(), prompt.end());
    auto batched_expected = separate.prefill_tokens_batched(batched_prompt, batch, static_cast<int64_t>(prompt.size()));
    auto batched_actual = packed.prefill_tokens_batched(batched_prompt, batch, static_cast<int64_t>(prompt.size()));
    require_allclose(batched_actual.logits, batched_expected.logits, 3.0e-3f, "packed batched prefill logits");
    separate.start_decode_tokens_batched(batched_expected.state, 32);
    packed.start_decode_tokens_batched(batched_expected.state, 32);
    for (int step = 0; step < 3; ++step) {
        const auto reference_logits = separate.decode_tokens_batched({step + 3, step + 5}).logits;
        require_allclose(
            packed.decode_tokens_batched({step + 3, step + 5}).logits,
            reference_logits,
            0.15F,
            "packed batched decode logits");
    }
}

void test_higgs_decode_graph_exposes_cuda_fast_paths() {
    constexpr int64_t hidden = 8;
    constexpr int64_t heads = 2;
    constexpr int64_t kv_heads = 1;
    constexpr int64_t head_dim = 4;
    constexpr int64_t intermediate = 12;
    constexpr int64_t cache_steps = 8;
    constexpr int64_t qkv_out = heads * head_dim + 2 * kv_heads * head_dim;

    ggml_init_params params{kGraphBytes, nullptr, true};
    ggml_context * ggml = ggml_init(params);
    if (ggml == nullptr) {
        throw std::runtime_error("failed to initialize Higgs decode graph test context");
    }

    try {
        engine::core::ModuleBuildContext ctx{ggml, "higgs_decode_fast_path_test", engine::core::BackendType::Cuda};
        auto make_tensor = [&](ggml_type type, std::initializer_list<int64_t> dims) {
            return engine::core::make_tensor(ctx, type, engine::core::TensorShape::from_dims(dims));
        };

        const auto input = make_tensor(GGML_TYPE_F32, {1, 1, hidden});
        const auto positions = make_tensor(GGML_TYPE_I32, {1});
        const auto cache_key = make_tensor(GGML_TYPE_F16, {1, cache_steps, kv_heads, head_dim});
        const auto cache_value = make_tensor(GGML_TYPE_F16, {1, cache_steps, kv_heads, head_dim});
        const auto cache_slot = make_tensor(GGML_TYPE_I64, {1});
        const auto attention_mask = make_tensor(GGML_TYPE_F16, {1, 1, 1, cache_steps});

        engine::modules::QwenDecoderLayerWeights weights;
        weights.input_norm = {make_tensor(GGML_TYPE_F32, {hidden}), std::nullopt};
        weights.q_norm = {make_tensor(GGML_TYPE_F32, {head_dim}), std::nullopt};
        weights.k_norm = {make_tensor(GGML_TYPE_F32, {head_dim}), std::nullopt};
        weights.post_norm = {make_tensor(GGML_TYPE_F32, {hidden}), std::nullopt};
        weights.self_attention.qkv_weight = make_tensor(GGML_TYPE_F32, {qkv_out, hidden});
        weights.self_attention.out_weight = make_tensor(GGML_TYPE_F32, {hidden, hidden});
        weights.mlp.gate_up_proj = engine::modules::LinearWeights{
            make_tensor(GGML_TYPE_F32, {intermediate * 2, hidden}),
            std::nullopt,
        };
        weights.mlp.down_proj = {
            make_tensor(GGML_TYPE_F32, {hidden, intermediate}),
            std::nullopt,
        };

        engine::modules::QwenDecoderLayerConfig config;
        config.hidden_size = hidden;
        config.num_attention_heads = heads;
        config.num_key_value_heads = kv_heads;
        config.head_dim = head_dim;
        config.intermediate_size = intermediate;
        config.qkv_layout = engine::modules::QwenDecoderQKVLayout::PackedQKV;
        config.use_qk_norm = true;
        config.runtime.attention.static_mode =
            engine::modules::QwenDecoderAttentionMode::FlashGroupedViewKV;
        config.runtime.static_cache.update_mode =
            engine::modules::QwenDecoderStaticCacheUpdateMode::DirectSetRows;
        config.runtime.static_cache.set_rows_mode =
            engine::modules::QwenDecoderStaticCacheSetRowsMode::BackendViewOptimized;
        config.runtime.mlp.mode = engine::modules::QwenDecoderMLPMode::PackedGateUp;

        ggml_cgraph * graph = ggml_new_graph_custom(ggml, kGraphNodes, false);
        const auto outputs = engine::modules::QwenDecoderLayerModule(config).build_with_static_cache_tail(
            ctx,
            graph,
            input,
            positions,
            weights,
            cache_key,
            cache_value,
            cache_slot,
            attention_mask);
        ggml_build_forward_expand(graph, outputs.output.tensor);

        if (count_graph_op(graph, GGML_OP_FLASH_ATTN_EXT) != 1) {
            throw std::runtime_error("Higgs decode graph must contain one grouped FlashAttention op");
        }
        if (count_graph_op(graph, GGML_OP_SET_ROWS) != 2) {
            throw std::runtime_error("Higgs decode graph must update both F16 KV caches with set-rows");
        }
        if (count_graph_op(graph, GGML_OP_GLU) != 1) {
            throw std::runtime_error("Higgs decode graph must contain one packed SwiGLU op");
        }
        if (count_graph_op(graph, GGML_OP_REPEAT) != 0) {
            throw std::runtime_error("grouped FlashAttention must not materialize repeated KV heads");
        }
        if (!graph_contains_sequence(graph, {GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS})) {
            throw std::runtime_error(
                "Higgs key-cache update must expose CUDA RoPE/view/set-rows fusion; graph=" +
                graph_ops(graph));
        }

        ggml_free(ggml);
    } catch (...) {
        ggml_free(ggml);
        throw;
    }
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        bool vulkan = false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--vulkan") {
                vulkan = true;
            } else if (arg == "--log") {
                engine::debug::configure_logging({true, ""});
            } else {
                throw std::runtime_error("Unknown argument: " + arg);
            }
        }
        test_packed_qkv_and_gate_up_match_separate_projections();
        test_fused_qkv_qk_norm_decode_tail_matches_separate();
        test_packed_qkv_bias_decode_tail_matches_separate();
        test_runtime_packed_matches_separate(/*biased=*/false);
        test_runtime_packed_matches_separate(/*biased=*/true);
        for (bool packed : {false, true}) {
            const auto reference = run_layer(packed, engine::core::BackendType::Cpu, true);
            if (vulkan) {
                const auto actual = run_layer(packed, engine::core::BackendType::Vulkan, true);
                require_allclose(actual.output, reference.output, 2.0e-5f, "Vulkan batched decoder output");
                require_allclose(actual.key, reference.key, 2.0e-5f, "Vulkan batched decoder key");
                require_allclose(actual.value, reference.value, 2.0e-5f, "Vulkan batched decoder value");
            }
        }
        test_suffix_causal_mask();
        test_f16_kv_set_rows();
        test_f16_kv_set_rows_batched();
        test_fast_projection_accepts_cuda_or_hip_backends();
        test_higgs_decode_graph_exposes_cuda_fast_paths();
        std::cout << "qwen_decoder_packed_projection_test: ok\n";
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "qwen_decoder_packed_projection_test: failed: " << ex.what() << "\n";
        return 1;
    }
}
