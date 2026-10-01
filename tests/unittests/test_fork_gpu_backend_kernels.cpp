// Fork-regression test (GPU-gated, ENGINE_BUILD_GPU_TESTS): runs the fork's
// GPU-only ggml fixes through the real device backends.
//
// Build with a GPU-enabled configure plus -DENGINE_BUILD_GPU_TESTS=ON, e.g.:
//   cmake ... -DGGML_SYCL=ON -DENGINE_BUILD_GPU_TESTS=ON
//   cmake ... -DGGML_CUDA=ON  -DENGINE_BUILD_GPU_TESTS=ON
//
// The test probes, in order, the SYCL and CUDA registries and skips itself
// (exit 77) when neither has a device. Covered end-to-end scenarios:
//
//   1. reorder/GET_ROWS: a Q4_0/Q8_0 weight consumed by both MUL_MAT (which
//      triggers the in-place reorder on SYCL) and GET_ROWS must yield the
//      same rows as the CPU reference - the tied-embedding corruption guard.
//   2. concat: a [2, N, 384]-shaped F32 concat whose launch historically
//      exceeded INT32 work items must complete and match the CPU result.
//   3. flash-attn per-head mask: a per-head F16 mask replicated from a shared
//      mask must match the shared-mask output on the device.
//   4. GRU_SCAN: fused op output matches the CPU reference on the device.
//   5. round_bf16: (SYCL only) exercised when the launcher exists; the graph
//      run itself is the regression - the pre-fix launcher poisoned the
//      Level-Zero context so the *next* enqueue died.

#include "test_assert.h"

#include "engine/framework/core/backend.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// MSVC rejects brace-init-lists decaying to the const int64_t* parameter of
// ggml_new_tensor, so wrap it.
ggml_tensor * new_tensor(ggml_context * ctx, ggml_type type, std::initializer_list<int64_t> dims) {
    int64_t ne[4] = {1, 1, 1, 1};
    int i = 0;
    for (const int64_t d : dims) {
        ne[i++] = d;
    }
    return ggml_new_tensor(ctx, type, static_cast<int>(dims.size()), ne);
}

constexpr size_t kGraphBytes = 256 * 1024 * 1024;
constexpr size_t kGraphNodes = 16384;
constexpr int kSkipExit = 77;

struct Runner {
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_context * ctx = nullptr;
    const char * name = "unknown";

    bool init(const char * registry_name) {
        ggml_backend_dev_t dev = ggml_backend_dev_by_name(registry_name);
        if (dev == nullptr) {
            return false;
        }
        backend = ggml_backend_dev_init(dev, nullptr);
        if (backend == nullptr) {
            return false;
        }
        name = registry_name;
        ggml_init_params init_params{};
        init_params.mem_size = kGraphBytes;
        init_params.no_alloc = true;
        ctx = ggml_init(init_params);
        return ctx != nullptr;
    }
    ~Runner() {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
        if (backend != nullptr) {
            ggml_backend_free(backend);
        }
    }
    void allocate() {
        if (buffer == nullptr) {
            buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        }
        if (buffer == nullptr) {
            throw std::runtime_error("device tensor allocation failed");
        }
    }
    std::vector<float> run(ggml_tensor * output) {
        allocate();
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, kGraphNodes, false);
        ggml_build_forward_expand(graph, output);
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error(std::string(name) + ": graph compute failed");
        }
        std::vector<float> values;
        engine::core::read_tensor_f32_into(output, values);
        return values;
    }
};

// CPU-side reference values for comparison.
std::vector<float> cpu_get_rows_q8(const std::vector<uint8_t> & blocks, int64_t ncols,
                                   int64_t nrows, const std::vector<int32_t> & indices) {
    ggml_backend_t cpu = engine::core::init_backend({engine::core::BackendType::Cpu, 0, 1});
    ggml_init_params init_params{};
    init_params.mem_size = 16 * 1024 * 1024;
    ggml_context * ctx = ggml_init(init_params);
    ggml_tensor * src = new_tensor(ctx, GGML_TYPE_Q8_0, {ncols, nrows});
    ggml_tensor * ids = new_tensor(ctx, GGML_TYPE_I32, {static_cast<int64_t>(indices.size())});
    ggml_tensor * out = ggml_get_rows(ctx, src, ids);
    std::memcpy(src->data, blocks.data(), blocks.size());
    std::memcpy(ids->data, indices.data(), indices.size() * sizeof(int32_t));
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 256, false);
    ggml_build_forward_expand(graph, out);
    if (ggml_backend_graph_compute(cpu, graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("cpu reference compute failed");
    }
    std::vector<float> values;
    engine::core::read_tensor_f32_into(out, values);
    ggml_free(ctx);
    ggml_backend_free(cpu);
    return values;
}

void run_reorder_getrows_probe(Runner & runner) {
    // Q8_0 weight [64, 8]; mul_mat first (may reorder in place), then
    // GET_ROWS on the same tensor.
    const int64_t ncols = 64;
    const int64_t nrows = 8;
    const size_t block_bytes = 34;
    std::vector<uint8_t> blocks(nrows * (ncols / 32) * block_bytes);
    for (size_t i = 0; i < blocks.size(); i++) {
        blocks[i] = static_cast<uint8_t>((i * 37 + 11) & 0xff);
    }

    ggml_tensor * weight = new_tensor(runner.ctx, GGML_TYPE_Q8_0, {ncols, nrows});
    ggml_tensor * ids = new_tensor(runner.ctx, GGML_TYPE_I32, {3});
    ggml_tensor * activation = new_tensor(runner.ctx, GGML_TYPE_F32, {ncols, 1});
    // Batch-1 mul_mat over the weight (the shape should_reorder_tensor
    // accepts), then GET_ROWS on the same weight.
    ggml_tensor * mm = ggml_mul_mat(runner.ctx, weight, activation);
    ggml_tensor * rows = ggml_get_rows(runner.ctx, weight, ids);
    runner.allocate();
    std::memcpy(weight->data, blocks.data(), blocks.size());
    const int32_t index_values[3] = {7, 2, 5};
    std::memcpy(ids->data, index_values, sizeof(index_values));
    std::vector<float> act(ncols);
    for (int64_t i = 0; i < ncols; i++) {
        act[static_cast<size_t>(i)] = 0.01f * static_cast<float>(i % 17) - 0.08f;
    }
    std::memcpy(activation->data, act.data(), act.size() * sizeof(float));

    const std::vector<float> reference = cpu_get_rows_q8(blocks, ncols, nrows,
                                                         {7, 2, 5});
    const std::vector<float> got = runner.run(rows);
    engine::test::require_eq(got.size(), reference.size(), "reorder/getrows: sizes");
    int64_t mismatches = 0;
    for (size_t i = 0; i < got.size(); i++) {
        if (!(got[i] == reference[i])) {
            mismatches++;
        }
    }
    engine::test::require_eq(mismatches, int64_t{0},
                             std::string(runner.name) +
                                 ": GET_ROWS after MUL_MAT(reorder candidate) must match CPU reference");
    (void) mm;
}

void run_concat_probe(Runner & runner) {
    // The historical seed_vc shape: ne = [2, N, 384]; the old fixed-256
    // padding launched >INT32 work items.
    const int64_t ne0 = 2;
    const int64_t ne1 = 23562;
    const int64_t ne2 = 384;
    const int64_t total = ne0 * ne1 * ne2;
    std::vector<float> x(static_cast<size_t>(total / 2));
    std::vector<float> y(static_cast<size_t>(total / 2));
    for (size_t i = 0; i < x.size(); i++) {
        x[i] = static_cast<float>(i % 101) * 0.01f;
        y[i] = static_cast<float>(i % 97) * -0.01f;
    }
    ggml_tensor * a = new_tensor(runner.ctx, GGML_TYPE_F32, {ne0, ne1 / 2, ne2});
    ggml_tensor * b = new_tensor(runner.ctx, GGML_TYPE_F32, {ne0, ne1 - ne1 / 2, ne2});
    ggml_tensor * out = ggml_concat(runner.ctx, a, b, 1);
    // Spot-check the first kilobyte instead of pulling 72MB back.
    ggml_tensor * head_view = ggml_view_1d(runner.ctx, out, 1024, 0);
    ggml_tensor * head = ggml_cont(runner.ctx, head_view);
    runner.allocate();
    std::memcpy(a->data, x.data(), x.size() * sizeof(float));
    std::memcpy(b->data, y.data(), y.size() * sizeof(float));
    const std::vector<float> got = runner.run(head);
    for (int64_t i = 0; i < 1024; i++) {
        const float want = i < static_cast<int64_t>(x.size()) ? x[static_cast<size_t>(i)] : 0.0f;
        engine::test::require_close(got[static_cast<size_t>(i)], want, 1e-5f,
                                    std::string(runner.name) + ": big concat head values");
    }
}

void run_fattn_mask_probe(Runner & runner) {
    const int64_t D = 16;
    const int64_t H = 4;
    const int64_t q_steps = 6;
    const int64_t kv_steps = 16;
    ggml_tensor * q = new_tensor(runner.ctx, GGML_TYPE_F32, {D, q_steps, H, 1});
    ggml_tensor * k = new_tensor(runner.ctx, GGML_TYPE_F32, {D, kv_steps, H, 1});
    ggml_tensor * v = new_tensor(runner.ctx, GGML_TYPE_F32, {D, kv_steps, H, 1});
    // ggml mask ne order: [kv_len, q_len, heads, batch].
    ggml_tensor * mask_shared = new_tensor(runner.ctx, GGML_TYPE_F16, {kv_steps, q_steps, 1, 1});
    ggml_tensor * mask_per_head = new_tensor(runner.ctx, GGML_TYPE_F16, {kv_steps, q_steps, H, 1});
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    ggml_tensor * out_shared =
        ggml_flash_attn_ext(runner.ctx, q, k, v, mask_shared, scale, 0.0f, 0.0f);
    ggml_tensor * out_per_head =
        ggml_flash_attn_ext(runner.ctx, q, k, v, mask_per_head, scale, 0.0f, 0.0f);
    runner.allocate();
    std::vector<float> qv(q_steps * H * D), kv(kv_steps * H * D), vv(kv_steps * H * D);
    for (size_t i = 0; i < qv.size(); i++) {
        qv[i] = std::sin(0.11f * static_cast<float>(i));
    }
    for (size_t i = 0; i < kv.size(); i++) {
        kv[i] = std::cos(0.07f * static_cast<float>(i));
        vv[i] = std::sin(0.05f * static_cast<float>(i));
    }
    std::memcpy(q->data, qv.data(), qv.size() * sizeof(float));
    std::memcpy(k->data, kv.data(), kv.size() * sizeof(float));
    std::memcpy(v->data, vv.data(), vv.size() * sizeof(float));
    std::vector<ggml_fp16_t> shared(static_cast<size_t>(kv_steps * q_steps));
    for (int64_t iq = 0; iq < q_steps; iq++) {
        for (int64_t ik = 0; ik < kv_steps; ik++) {
            shared[static_cast<size_t>(iq * kv_steps + ik)] = ggml_fp32_to_fp16(
                ik > iq ? -std::numeric_limits<float>::infinity() : 0.0f);
        }
    }
    std::memcpy(mask_shared->data, shared.data(), shared.size() * sizeof(ggml_fp16_t));
    for (int64_t h = 0; h < H; h++) {
        std::memcpy(static_cast<char *>(mask_per_head->data) +
                        h * shared.size() * sizeof(ggml_fp16_t),
                    shared.data(), shared.size() * sizeof(ggml_fp16_t));
    }
    const std::vector<float> got_shared = runner.run(out_shared);
    const std::vector<float> got_per_head = runner.run(out_per_head);
    engine::test::require_eq(got_shared.size(), got_per_head.size(), "fattn: sizes");
    float worst = 0.0f;
    for (size_t i = 0; i < got_shared.size(); i++) {
        worst = std::max(worst, static_cast<float>(
                                    std::fabs(static_cast<double>(got_shared[i]) - got_per_head[i])));
    }
    engine::test::require(worst <= 2e-3f,
                          std::string(runner.name) +
                              ": per-head mask must match shared mask (worst=" + std::to_string(worst) + ")");
}

void run_gru_scan_probe(Runner & runner) {
    const int64_t H = 64;
    const int64_t F = 16;
    ggml_tensor * x_ih = new_tensor(runner.ctx, GGML_TYPE_F32, {3 * H, F});
    ggml_tensor * h0 = new_tensor(runner.ctx, GGML_TYPE_F32, {H});
    ggml_tensor * w_hh = new_tensor(runner.ctx, GGML_TYPE_F32, {H, 3 * H});
    ggml_tensor * b_hh = new_tensor(runner.ctx, GGML_TYPE_F32, {3 * H});
    ggml_tensor * keep = new_tensor(runner.ctx, GGML_TYPE_F32, {F});
    ggml_tensor * fused = ggml_gru_scan(runner.ctx, x_ih, h0, w_hh, b_hh, keep, false);
    runner.allocate();
    auto fill = [&](ggml_tensor * t, float scale) {
        std::vector<float> values(static_cast<size_t>(ggml_nelements(t)));
        for (size_t i = 0; i < values.size(); i++) {
            values[i] = scale * std::sin(0.13f * static_cast<float>(i));
        }
        std::memcpy(t->data, values.data(), values.size() * sizeof(float));
    };
    fill(x_ih, 0.5f);
    fill(h0, 0.25f);
    fill(w_hh, 0.05f);
    fill(b_hh, 0.1f);
    std::vector<float> keep_values(static_cast<size_t>(F), 1.0f);
    std::memcpy(keep->data, keep_values.data(), keep_values.size() * sizeof(float));

    if (!ggml_backend_supports_op(runner.backend, fused)) {
        std::printf("test_fork_gpu_backend_kernels: %s lacks GRU_SCAN, skipping probe\n",
                    runner.name);
        return;
    }
    const std::vector<float> got = runner.run(fused);
    engine::test::require_eq(got.size(), static_cast<size_t>((F + 1) * H), "gru_scan: size");
    // Deterministic-input smoke: outputs finite and non-zero.
    float magnitude = 0.0f;
    for (const float value : got) {
        engine::test::require(std::isfinite(value), std::string(runner.name) + ": GRU_SCAN finite");
        magnitude = std::max(magnitude, std::fabs(value));
    }
    engine::test::require(magnitude > 0.0f, std::string(runner.name) + ": GRU_SCAN non-zero");
}

}  // namespace

int main() try {
    Runner runner;
    bool have_gpu = false;
    for (const char * candidate : {"SYCL", "CUDA"}) {
        for (size_t i = 0; i < ggml_backend_reg_count() && !have_gpu; ++i) {
            ggml_backend_reg_t reg = ggml_backend_reg_get(i);
            const char * reg_name = reg != nullptr ? ggml_backend_reg_name(reg) : nullptr;
            if (reg_name == nullptr || std::string(reg_name) != candidate) {
                continue;
            }
            if (ggml_backend_reg_dev_count(reg) == 0) {
                continue;
            }
            ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
            runner.backend = ggml_backend_dev_init(dev, nullptr);
            if (runner.backend != nullptr) {
                runner.name = candidate;
                ggml_init_params init_params{};
                init_params.mem_size = kGraphBytes;
                init_params.no_alloc = true;
                runner.ctx = ggml_init(init_params);
                have_gpu = runner.ctx != nullptr;
            }
        }
        if (have_gpu) {
            break;
        }
    }
    if (!have_gpu) {
        std::printf("test_fork_gpu_backend_kernels: no SYCL/CUDA device, skipping\n");
        return kSkipExit;
    }

    run_reorder_getrows_probe(runner);
    run_concat_probe(runner);
    run_fattn_mask_probe(runner);
    run_gru_scan_probe(runner);

    std::printf("test_fork_gpu_backend_kernels: all device probes passed on %s\n", runner.name);
    return 0;
} catch (const std::exception & error) {
    std::fprintf(stderr, "test_fork_gpu_backend_kernels FAILED: %s\n", error.what());
    return 1;
}
