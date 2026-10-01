// Fork-regression test: per-head flash-attention masks (audio.cpp fork,
// restored by dc6d1b59 after PR #39 dropped it).
//
// chatterbox passes an F16 mask with mask->ne[2] != 1 (one mask slice per
// query head). The SYCL tile/vec kernels (fattn-tile.hpp / fattn-vec.hpp)
// learned to offset the mask pointer by nb32 * (head % ne32), and the kernel
// selection in fattn.cpp stopped rejecting per-head masks outright (they now
// only disable the GQA fused-block optimization and must divide Q heads).
//
// The SYCL kernel changes need a GPU (see test_fork_gpu_backend_kernels.cpp);
// this CPU test pins the two halves of the contract that runs everywhere:
//
//   1. Numerics: on the CPU backend, ggml_flash_attn_ext with a per-head
//      mask (every head slice a copy of the shared mask) must produce the
//      exact same output as the shared single-slice mask. The SYCL kernels
//      exist to restore exactly this equivalence, so any upstream change
//      that breaks per-head mask semantics on CPU (or the mask layout
//      contract) fails here too.
//   2. Dispatch gates (host-side mirror of ggml_sycl_get_best_fattn_kernel):
//      a per-head mask disables gqa_opt and a non-dividing head count is
//      rejected, while shared masks keep the historical behavior.

#include "test_assert.h"

#include "engine/framework/core/backend.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <random>
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

constexpr size_t kGraphBytes = 64 * 1024 * 1024;
constexpr size_t kGraphNodes = 4096;

// FATTN_KQ_STRIDE from ggml-sycl/fattn-common.hpp.
constexpr int64_t kFattnKqStride = 256;

struct CpuRunner {
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_context * ctx = nullptr;

    CpuRunner() {
        backend = engine::core::init_backend({engine::core::BackendType::Cpu, 0, 4});
        ggml_init_params init{};
        init.mem_size = kGraphBytes;
        init.no_alloc = true;
        ctx = ggml_init(init);
        if (ctx == nullptr) {
            throw std::runtime_error("ggml_init failed");
        }
    }
    ~CpuRunner() {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
        ggml_free(ctx);
        ggml_backend_free(backend);
    }
    void allocate() {
        if (buffer == nullptr) {
            buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        }
        if (buffer == nullptr) {
            throw std::runtime_error("tensor allocation failed");
        }
    }
    std::vector<float> run(ggml_tensor * output) {
        allocate();
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, kGraphNodes, false);
        ggml_build_forward_expand(graph, output);
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("graph compute failed");
        }
        std::vector<float> values;
        engine::core::read_tensor_f32_into(output, values);
        return values;
    }
};

struct FattnCase {
    const char * name;
    int64_t heads;      // Q heads
    int64_t kv_heads;   // K/V heads (GQA ratio = heads / kv_heads)
    int64_t q_steps;
    int64_t kv_steps;
    int64_t head_dim;
    unsigned seed;
};

// Builds Q [D, q, H], K/V [D, kv, Hkv], shared mask [kv, q, 1, 1] F16 and the
// replicated per-head mask [kv, q, H, 1] F16 (ggml mask ne order:
// [kv_len, q_len, heads, batch]); returns flash-attn outputs.
void compare_shared_vs_per_head(const FattnCase & c) {
    CpuRunner runner;
    const int64_t D = c.head_dim;
    const int64_t H = c.heads;
    const int64_t Hkv = c.kv_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    const float max_bias = 0.0f;
    const float logit_softcap = 0.0f;

    ggml_tensor * q = new_tensor(runner.ctx, GGML_TYPE_F32, {D, c.q_steps, H, 1});
    ggml_tensor * k = new_tensor(runner.ctx, GGML_TYPE_F32, {D, c.kv_steps, Hkv, 1});
    ggml_tensor * v = new_tensor(runner.ctx, GGML_TYPE_F32, {D, c.kv_steps, Hkv, 1});
    ggml_tensor * mask_shared = new_tensor(runner.ctx, GGML_TYPE_F16, {c.kv_steps, c.q_steps, 1, 1});
    ggml_tensor * mask_per_head = new_tensor(runner.ctx, GGML_TYPE_F16, {c.kv_steps, c.q_steps, H, 1});
    ggml_tensor * out_shared = ggml_flash_attn_ext(runner.ctx, q, k, v, mask_shared,
                                                   scale, max_bias, logit_softcap);
    ggml_tensor * out_per_head = ggml_flash_attn_ext(runner.ctx, q, k, v, mask_per_head,
                                                     scale, max_bias, logit_softcap);
    runner.allocate();

    std::mt19937 rng(c.seed);
    auto sample = [&rng]() {
        return -1.0f + 2.0f * static_cast<float>(rng() % 10000u) / 9999.0f;
    };
    {
        std::vector<float> buf;
        auto fill = [&](ggml_tensor * t) {
            buf.resize(static_cast<size_t>(ggml_nelements(t)));
            for (auto & value : buf) {
                value = sample();
            }
            if (t->type == GGML_TYPE_F32) {
                std::memcpy(t->data, buf.data(), buf.size() * sizeof(float));
            } else {
                std::vector<ggml_fp16_t> hbuf(buf.size());
                for (size_t i = 0; i < buf.size(); i++) {
                    hbuf[i] = ggml_fp32_to_fp16(buf[i]);
                }
                std::memcpy(t->data, hbuf.data(), hbuf.size() * sizeof(ggml_fp16_t));
            }
        };
        fill(q);
        fill(k);
        fill(v);
    }
    // Shared mask: causal-ish pattern with a mix of 0 and -inf.
    const size_t shared_elems = static_cast<size_t>(c.kv_steps * c.q_steps);
    std::vector<ggml_fp16_t> shared(shared_elems);
    for (int64_t iq = 0; iq < c.q_steps; iq++) {
        for (int64_t ik = 0; ik < c.kv_steps; ik++) {
            const bool masked = ik > iq + c.kv_steps - c.q_steps;
            shared[static_cast<size_t>(iq * c.kv_steps + ik)] =
                ggml_fp32_to_fp16(masked ? -std::numeric_limits<float>::infinity() : 0.0f);
        }
    }
    std::memcpy(mask_shared->data, shared.data(), shared.size() * sizeof(ggml_fp16_t));
    // Per-head mask: every head slice is a copy of the shared slice.
    for (int64_t h = 0; h < H; h++) {
        std::memcpy(static_cast<char *>(mask_per_head->data) + h * c.kv_steps * c.q_steps * sizeof(ggml_fp16_t),
                    shared.data(), shared.size() * sizeof(ggml_fp16_t));
    }

    const std::vector<float> got_shared = runner.run(out_shared);
    const std::vector<float> got_per_head = runner.run(out_per_head);

    engine::test::require_eq(got_shared.size(), got_per_head.size(),
                             std::string(c.name) + ": output sizes");

    // Both graphs read the same mask values per (head, query, key) - the
    // outputs must be bit-identical.
    int64_t mismatches = 0;
    float worst = 0.0f;
    for (size_t i = 0; i < got_shared.size(); i++) {
        if (!(got_shared[i] == got_per_head[i])) {
            mismatches++;
            worst = std::max(worst, static_cast<float>(std::fabs(static_cast<double>(got_shared[i]) - got_per_head[i])));
        }
    }
    engine::test::require_eq(mismatches, int64_t{0},
                             std::string(c.name) +
                                 ": per-head replicated mask must match the shared mask bit-for-bit (worst diff=" +
                                 std::to_string(worst) + ")");

    // Sanity: outputs are finite and non-degenerate.
    float magnitude = 0.0f;
    for (const float value : got_shared) {
        engine::test::require(std::isfinite(value), std::string(c.name) + ": non-finite attention output");
        magnitude = std::max(magnitude, std::fabs(value));
    }
    engine::test::require(magnitude > 0.0f, std::string(c.name) + ": attention output must be non-zero");
}

// ---- Mirror of the dispatch gates in ggml_sycl_get_best_fattn_kernel ------
struct FattnGateQuery {
    bool has_mask = true;
    int64_t mask_heads = 1;       // mask->ne[2]
    int64_t q_heads = 8;          // Q->ne[2]
    int64_t gqa_ratio = 1;
    float max_bias = 0.0f;
    int64_t kv_len = 512;
};

enum class BestKernel { Tile, Vec, GqaFused, None };

// Mirrors the three touched conditions in fattn.cpp:
//   per_head_mask := mask && mask->ne[2] != 1
//   gqa_opt_applies := gqa_ratio >= 2 && mask && !per_head_mask &&
//                      max_bias == 0 && K->ne[1] % FATTN_KQ_STRIDE == 0
//   rejection: mask && mask->ne[2] != 1 && Q->ne[2] % mask->ne[2] != 0 -> NONE
BestKernel mirror_select_kernel(const FattnGateQuery & q) {
    const bool per_head_mask = q.has_mask && q.mask_heads != 1;
    if (q.has_mask && q.mask_heads != 1 && q.q_heads % q.mask_heads != 0) {
        return BestKernel::None;
    }
    const bool gqa_opt = q.gqa_ratio >= 2 && q.has_mask && !per_head_mask &&
                        q.max_bias == 0.0f && q.kv_len % kFattnKqStride == 0;
    return gqa_opt ? BestKernel::GqaFused : BestKernel::Tile;
}

}  // namespace

int main() try {
    // ---- Numerics: replicated per-head mask == shared mask (CPU backend).
    compare_shared_vs_per_head({"MHA shared-vs-perhead", 4, 4, 6, 16, 32, 21});
    compare_shared_vs_per_head({"GQA2", 8, 4, 5, 20, 16, 22});
    compare_shared_vs_per_head({"GQA4 tall", 8, 2, 8, 24, 8, 23});
    compare_shared_vs_per_head({"single head", 1, 1, 4, 12, 8, 24});

    // ---- Dispatch-gate mirror.
    {
        // Shared mask + GQA2 + aligned KV: the fused GQA block stays eligible.
        FattnGateQuery gqa;
        gqa.gqa_ratio = 2;
        gqa.kv_len = 512;
        engine::test::require(mirror_select_kernel(gqa) == BestKernel::GqaFused,
                              "shared mask + GQA2 must keep the fused GQA optimization");

        // Per-head mask disables the GQA fused block but stays dispatchable.
        FattnGateQuery per_head = gqa;
        per_head.mask_heads = 8;
        per_head.q_heads = 8;
        engine::test::require(mirror_select_kernel(per_head) == BestKernel::Tile,
                              "per-head mask must disable gqa_opt but remain dispatchable (tile/vec path)");

        // Head count that does not divide the Q heads is rejected outright.
        FattnGateQuery indivisible = gqa;
        indivisible.mask_heads = 3;
        indivisible.q_heads = 8;
        engine::test::require(mirror_select_kernel(indivisible) == BestKernel::None,
                              "mask heads not dividing Q heads must be rejected (kernel NONE)");

        // Divisible per-head mask with a larger factor stays on the tile path.
        FattnGateQuery divisible = gqa;
        divisible.mask_heads = 4;
        divisible.q_heads = 8;
        engine::test::require(mirror_select_kernel(divisible) == BestKernel::Tile,
                              "divisible per-head mask dispatches to the tile/vec path");

        // No mask: the gqa_opt condition in fattn.cpp requires a mask
        // (&& mask), so the mask-less dispatch takes the plain tile/vec path.
        FattnGateQuery no_mask = gqa;
        no_mask.has_mask = false;
        engine::test::require(mirror_select_kernel(no_mask) == BestKernel::Tile,
                              "mask-less GQA dispatch takes the tile/vec path (gqa_opt requires a mask)");

        // max_bias != 0 also disables the fused path (unchanged behavior).
        FattnGateQuery biased = gqa;
        biased.max_bias = 0.5f;
        engine::test::require(mirror_select_kernel(biased) == BestKernel::Tile,
                              "alibi bias keeps the tile/vec path");
    }

    std::printf("test_fork_sycl_fattn_per_head_mask: all cases passed\n");
    return 0;
} catch (const std::exception & error) {
    std::fprintf(stderr, "test_fork_sycl_fattn_per_head_mask FAILED: %s\n", error.what());
    return 1;
}
