// Fork-regression test: irodori codec decoder padding=0 + slice crop list
// (audio.cpp fork, src/models/irodori_tts/codec.cpp).
//
// ggml's ggml_conv_transpose_1d only allows p0 == 0, so the irodori codec
// decoder_block() runs the upsample conv-transpose with padding=0 on the
// backends without native padding support and crops the excess with a
// SliceModule. The fork's backend list is:
//
//   crop_backend_output = Cpu || Vulkan || Metal || Sycl   (CUDA absent)
//
// i.e. SYCL is in the list and CUDA is not (CUDA takes the col2im lowering
// with native padding). This test builds the public
// build_irodori_codec_decode() graph under different ModuleBuildContext
// backend types on a CPU machine (graph building never touches a device) and
// pins:
//
//   1. Sycl-typed and Cuda-typed builds share the identical col2im lowering;
//      the ONLY structural difference is the crop: one extra VIEW node per
//      decoder block, and each Sycl col2im produces exactly
//      2*((stride+1)/2) more frames than the Cuda one (padding 0 vs native).
//   2. Vulkan-typed build matches the Sycl structure (also in the list).
//   3. A Cpu-typed build (generic conv-transpose path) also uses padding=0
//      (same upsample frame counts as Sycl) and keeps the crops.
//   4. All builds converge to the identical final output shape - the crop
//      end-points reproduce the native-padding length exactly.
//   5. The Cpu-typed graph computes on the CPU backend with real weights and
//      produces finite, non-degenerate audio (happy-path smoke).

#include "test_assert.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/module.h"
#include "engine/models/irodori_tts/codec.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr size_t kGraphBytes = 64 * 1024 * 1024;
constexpr size_t kGraphNodes = 8192;

// Small codec geometry (see IrodoriCodecConfig defaults for the real one).
constexpr int64_t kCodebookDim = 8;
constexpr int64_t kLatentDim = 16;
constexpr int64_t kDecoderDim = 32;
constexpr int64_t kLatentSteps = 3;
constexpr int64_t kBlocks = 4;
const int64_t kRates[kBlocks] = {12, 10, 8, 2};

int64_t crop_padding(int64_t stride) {
    return (stride + 1) / 2;
}

struct BuiltGraph {
    ggml_context * ctx = nullptr;
    ggml_tensor * latent = nullptr;
    ggml_tensor * output = nullptr;
    std::vector<ggml_tensor *> weight_tensors;

    // Facts collected from the built cgraph.
    int64_t output_frames = 0;
    std::vector<int64_t> upsample_frames;  // per decoder block, in graph order
    int64_t view_nodes = 0;
    int64_t col2im_nodes = 0;
    int64_t conv_transpose_nodes = 0;
};

// Builds the decode graph for a backend type over fabricated zero weights.
BuiltGraph build_decode(engine::core::BackendType backend_type) {
    using engine::models::irodori_tts::IrodoriCodecWeights;
    using engine::models::irodori_tts::IrodoriCodecDecoderBlockWeights;
    using engine::models::irodori_tts::IrodoriCodecResidualUnitWeights;
    namespace modules = engine::modules;
    namespace core = engine::core;

    ggml_init_params init{};
    init.mem_size = kGraphBytes;
    init.no_alloc = true;
    BuiltGraph built;
    built.ctx = ggml_init(init);
    if (built.ctx == nullptr) {
        throw std::runtime_error("ggml_init failed");
    }
    ggml_context * ctx = built.ctx;

    // Engine logical shapes are row-major and reversed into ggml ne, so
    // create weights through core::make_tensor instead of raw ggml calls.
    core::ModuleBuildContext helper_ctx;
    helper_ctx.ggml = ctx;
    helper_ctx.module_instance_name = "fork_irodori_crop_test.weights";
    helper_ctx.backend_type = backend_type;
    auto track = [&](std::initializer_list<int64_t> dims) {
        auto value = core::make_tensor(helper_ctx, GGML_TYPE_F32,
                                       core::TensorShape::from_dims(dims));
        built.weight_tensors.push_back(value.tensor);
        return value;
    };
    auto conv1d = [&](int64_t out_c, int64_t in_c, int64_t k) {
        modules::Conv1dWeights w;
        w.weight = track({out_c, in_c, k});
        w.bias = track({out_c});
        return w;
    };
    auto conv_t1d = [&](int64_t in_c, int64_t out_c, int64_t k) {
        modules::ConvTranspose1dWeights w;
        w.weight = track({in_c, out_c, k});
        w.bias = track({out_c});
        return w;
    };
    auto snake = [&](int64_t channels) {
        modules::Snake1dWeights w;
        w.alpha = track({channels});
        return w;
    };
    auto residual = [&](int64_t channels) {
        IrodoriCodecResidualUnitWeights w;
        w.snake0 = snake(channels);
        w.conv0 = conv1d(channels, channels, 7);
        w.snake1 = snake(channels);
        w.conv1 = conv1d(channels, channels, 1);
        return w;
    };

    IrodoriCodecWeights weights;
    weights.quantizer_out_proj = conv1d(kLatentDim, kCodebookDim, 1);
    weights.decoder_input = conv1d(kDecoderDim, kLatentDim, 7);
    int64_t in_channels = kDecoderDim;
    for (int64_t block = 0; block < kBlocks; ++block) {
        const int64_t out_channels = in_channels / 2;
        IrodoriCodecDecoderBlockWeights dec;
        dec.up_snake = snake(in_channels);
        dec.up_conv = conv_t1d(in_channels, out_channels, 2 * kRates[block]);
        dec.residual_0 = residual(out_channels);
        dec.residual_1 = residual(out_channels);
        dec.residual_2 = residual(out_channels);
        weights.decoder_blocks.push_back(dec);
        in_channels = out_channels;
    }
    weights.watermark_passthrough.snake = snake(in_channels);
    weights.watermark_passthrough.conv = conv1d(1, in_channels, 7);

    engine::core::ModuleBuildContext build_ctx;
    build_ctx.ggml = ctx;
    build_ctx.module_instance_name = "fork_irodori_crop_test";
    build_ctx.backend_type = backend_type;

    auto latent = core::make_tensor(
        build_ctx, GGML_TYPE_F32,
        core::TensorShape::from_dims({1, kLatentSteps, kCodebookDim}));
    built.latent = latent.tensor;

    engine::models::irodori_tts::IrodoriCodecConfig config;
    config.codebook_dim = kCodebookDim;
    config.latent_dim = kLatentDim;
    config.decoder_dim = kDecoderDim;

    auto output = engine::models::irodori_tts::build_irodori_codec_decode(
        build_ctx, latent, weights, config);
    built.output = output.tensor;
    built.output_frames = output.shape.dims[2];

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, kGraphNodes, false);
    ggml_build_forward_expand(graph, output.tensor);
    const int node_count = ggml_graph_n_nodes(graph);
    for (int i = 0; i < node_count; i++) {
        const ggml_tensor * node = ggml_graph_node(graph, i);
        switch (node->op) {
            case GGML_OP_COL2IM_1D:
                built.col2im_nodes++;
                built.upsample_frames.push_back(node->ne[0]);
                break;
            case GGML_OP_CONV_TRANSPOSE_1D:
                built.conv_transpose_nodes++;
                built.upsample_frames.push_back(node->ne[0]);
                break;
            case GGML_OP_VIEW:
                built.view_nodes++;
                break;
            default:
                break;
        }
    }
    return built;
}

}  // namespace

int main() try {
    // ---- Structural comparison across backend types.
    BuiltGraph cuda_graph = build_decode(engine::core::BackendType::Cuda);
    BuiltGraph sycl_graph = build_decode(engine::core::BackendType::Sycl);
    BuiltGraph vulkan_graph = build_decode(engine::core::BackendType::Vulkan);
    BuiltGraph cpu_graph = build_decode(engine::core::BackendType::Cpu);

    // 1. Sycl and Cuda share the col2im lowering; Cpu takes the generic path.
    engine::test::require_eq(sycl_graph.col2im_nodes, int64_t{kBlocks},
                             "SYCL build uses the col2im upsample lowering");
    engine::test::require_eq(cuda_graph.col2im_nodes, int64_t{kBlocks},
                             "CUDA build uses the col2im upsample lowering");
    engine::test::require_eq(cpu_graph.conv_transpose_nodes, int64_t{kBlocks},
                             "CPU build uses the generic conv-transpose path");

    // 2. The crop list: exactly one extra VIEW node per decoder block on the
    //    Sycl build vs the Cuda build (CUDA is not in the crop list).
    engine::test::require_eq(sycl_graph.view_nodes - cuda_graph.view_nodes, int64_t{kBlocks},
                             "SYCL build must insert one crop view per decoder block (CUDA none)");

    // 3. Vulkan is in the crop list too: identical structure to Sycl.
    engine::test::require_eq(vulkan_graph.view_nodes - cuda_graph.view_nodes, int64_t{kBlocks},
                             "Vulkan build must crop like the SYCL build");
    engine::test::require_eq(vulkan_graph.view_nodes, sycl_graph.view_nodes,
                             "Vulkan and SYCL crop structures must match");

    // 4. Upsample frame counts: padding=0 vs native padding differ by exactly
    //    2*((stride+1)/2) per block; CPU (padding=0 generic path) matches SYCL.
    engine::test::require_eq(sycl_graph.upsample_frames.size(), cuda_graph.upsample_frames.size(),
                             "upsample node counts");
    for (size_t i = 0; i < sycl_graph.upsample_frames.size(); i++) {
        const int64_t expected_diff = 2 * crop_padding(kRates[i]);
        engine::test::require_eq(sycl_graph.upsample_frames[i] - cuda_graph.upsample_frames[i],
                                 expected_diff,
                                 "SYCL upsample must run padding=0 (native-padding diff, block " +
                                     std::to_string(i) + ")");
        engine::test::require_eq(cpu_graph.upsample_frames[i], sycl_graph.upsample_frames[i],
                                 "CPU upsample must run padding=0 like SYCL (block " +
                                     std::to_string(i) + ")");
    }

    // 5. All builds converge to the same final output shape (crop end-points
    //    reproduce the native-padding length).
    engine::test::require(cpu_graph.output_frames == cuda_graph.output_frames &&
                              sycl_graph.output_frames == cuda_graph.output_frames &&
                              vulkan_graph.output_frames == cuda_graph.output_frames,
                          "all backend builds must produce the same decode output length");
    {
        int64_t frames = kLatentSteps;
        for (int64_t i = 0; i < kBlocks; i++) {
            // (in-1)*s + 2s (padding=0) minus the 2*crop_padding crop.
            frames = (frames - 1) * kRates[i] + 2 * kRates[i] - 2 * crop_padding(kRates[i]);
        }
        engine::test::require_eq(cuda_graph.output_frames, frames,
                                 "decode output length closed-form check");
    }

    // ---- CPU compute smoke over the crop path (Cpu-typed graph).
    {
        engine::core::BackendConfig backend_config{engine::core::BackendType::Cpu, 0, 4};
        ggml_backend_t backend = engine::core::init_backend(backend_config);
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(cpu_graph.ctx, backend);
        engine::test::require(buffer != nullptr, "CPU tensor allocation");

        std::mt19937 rng(31);
        for (ggml_tensor * t : cpu_graph.weight_tensors) {
            const int64_t n = ggml_nelements(t);
            std::vector<float> values(static_cast<size_t>(n));
            for (auto & v : values) {
                v = (static_cast<float>(rng() % 2000) / 1000.0f - 1.0f) * 0.25f;
            }
            // Keep Snake alphas away from zero (snake divides by alpha).
            if (ggml_n_dims(t) == 1 && n <= kDecoderDim) {
                for (auto & v : values) {
                    if (std::fabs(v) < 0.05f) {
                        v = v < 0 ? -0.05f : 0.05f;
                    }
                }
            }
            std::memcpy(t->data, values.data(), values.size() * sizeof(float));
        }
        {
            std::vector<float> values(ggml_nelements(cpu_graph.latent));
            for (size_t i = 0; i < values.size(); i++) {
                values[i] = std::sin(0.7f * static_cast<float>(i)) * 0.5f;
            }
            std::memcpy(cpu_graph.latent->data, values.data(), values.size() * sizeof(float));
        }

        ggml_cgraph * graph = ggml_new_graph_custom(cpu_graph.ctx, kGraphNodes, false);
        ggml_build_forward_expand(graph, cpu_graph.output);
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("CPU graph compute failed");
        }
        std::vector<float> values;
        engine::core::read_tensor_f32_into(cpu_graph.output, values);
        engine::test::require_eq(values.size(), static_cast<size_t>(cpu_graph.output_frames),
                                 "computed output size");
        float magnitude = 0.0f;
        for (const float v : values) {
            engine::test::require(std::isfinite(v), "decode output must be finite");
            magnitude = std::max(magnitude, std::fabs(v));
        }
        engine::test::require(magnitude > 0.0f, "decode output must be non-degenerate");
        ggml_backend_buffer_free(buffer);
        ggml_backend_free(backend);
    }

    ggml_free(cuda_graph.ctx);
    ggml_free(sycl_graph.ctx);
    ggml_free(vulkan_graph.ctx);
    ggml_free(cpu_graph.ctx);

    std::printf("test_fork_irodori_codec_crop: all cases passed\n");
    return 0;
} catch (const std::exception & error) {
    std::fprintf(stderr, "test_fork_irodori_codec_crop FAILED: %s\n", error.what());
    return 1;
}
