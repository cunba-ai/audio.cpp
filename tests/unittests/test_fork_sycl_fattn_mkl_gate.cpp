// Fork-regression test: SYCL MKL flash-attention gate requires F16 K/V
// (audio.cpp fork, fe01cab4; re-ported onto the current baseline).
//
// The MKL GEMM prompt-processing path (fattn-mkl.cpp) was ~200x slower than
// the TILE kernel for F32 K/V caches on B50 (dramabox gemma3 encoder: 4.2s vs
// 21.7ms per call - the path does ~50 host-blocking section syncs per call
// and each sync costs tens of milliseconds in the engine process). The fork
// gate therefore only ever routes F16 K/V to MKL; F32 K/V (and anything else
// outside the validated envelope) falls through to the TILE/VEC kernels.
//
// The gate lives in ggml_sycl_get_best_fattn_kernel (fattn.cpp, static,
// SYCL-only TU), so this CPU test mirrors the decision table and pins:
//
//   1. F32 K/V -> TILE even when every other envelope condition holds
//      (the dramabox regression this fork change was written for).
//   2. F16 K/V inside the envelope -> MKL.
//   3. F16 K/V outside the envelope (head dim, KV length, GQA ratio, bias,
//      softcap, sinks, batch mismatch, non-multiple row stride) -> TILE.
//   4. GGML_SYCL_ENABLE_MKL_FA=0 disables the MKL path entirely.
// The gate wiring itself (source anchors) is covered by
// test_fork_source_anchors.cpp.

#include "test_assert.h"

#include <ggml.h>

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace {

enum class FattnKernel { Tile, Vec, Mkl };

// Mirror of the MKL branch in ggml_sycl_get_best_fattn_kernel (fattn.cpp).
struct MklGateQuery {
    bool env_enabled = true;
    bool has_mask = true;
    bool has_sinks = false;
    int64_t gqa_ratio = 2;
    ggml_type k_type = GGML_TYPE_F16;
    ggml_type v_type = GGML_TYPE_F16;
    int64_t head_dim = 128;       // Q->ne[0] == V->ne[0]
    int64_t q_len = 512;          // Q->ne[1]
    int64_t kv_len = 1024;        // K->ne[1]
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    bool batch_mismatch = false;  // Q->ne[3] != K->ne[3] && K->ne[3] != 1
    int64_t kv_row_stride = -1;   // K/V nb[1]; -1 means natural ne[0]*2
};

FattnKernel mirror_mkl_gate(const MklGateQuery & q) {
    const int64_t kv_stride = q.kv_row_stride >= 0 ? q.kv_row_stride : q.head_dim * 2;
    if (q.env_enabled && q.has_mask && !q.has_sinks && q.gqa_ratio >= 2 &&
        q.k_type == GGML_TYPE_F16 && q.v_type == GGML_TYPE_F16 &&
        q.head_dim >= 64 && q.head_dim <= 512 && q.head_dim % 64 == 0 &&
        q.q_len >= 32 && q.kv_len >= 1024 &&
        q.max_bias == 0.0f && q.logit_softcap == 0.0f &&
        !q.batch_mismatch &&
        kv_stride % (q.head_dim * 2) == 0) {
        return FattnKernel::Mkl;
    }
    return FattnKernel::Tile;
}

}  // namespace

int main() try {
    // ---- Happy path: F16 K/V inside the envelope takes MKL.
    {
        MklGateQuery q;
        engine::test::require(mirror_mkl_gate(q) == FattnKernel::Mkl,
                              "F16 K/V inside the envelope must take the MKL path");
    }

    // ---- The dramabox regression: F32 K/V must fall back to TILE.
    {
        MklGateQuery q;
        q.k_type = GGML_TYPE_F32;
        q.v_type = GGML_TYPE_F32;
        engine::test::require(mirror_mkl_gate(q) == FattnKernel::Tile,
                              "F32 K/V must NOT take the MKL path (falls back to TILE)");
    }
    // Mixed F32/F16 also falls through.
    {
        MklGateQuery q;
        q.k_type = GGML_TYPE_F32;
        engine::test::require(mirror_mkl_gate(q) == FattnKernel::Tile,
                              "mixed F32 K with F16 V must fall back to TILE");
    }
    // Quantized K/V also falls through.
    {
        MklGateQuery q;
        q.k_type = GGML_TYPE_Q8_0;
        q.v_type = GGML_TYPE_Q8_0;
        engine::test::require(mirror_mkl_gate(q) == FattnKernel::Tile,
                              "quantized K/V must fall back to TILE");
    }

    // ---- Envelope edges (each violates exactly one condition -> TILE).
    {
        const struct {
            const char * name;
            MklGateQuery query;
        } cases[] = {
            {"head_dim below 64", [] { MklGateQuery q; q.head_dim = 32; return q; }()},
            {"head_dim above 512", [] { MklGateQuery q; q.head_dim = 640; return q; }()},
            {"head_dim not multiple of 64", [] { MklGateQuery q; q.head_dim = 96; return q; }()},
            {"q_len below 32", [] { MklGateQuery q; q.q_len = 16; return q; }()},
            {"kv_len below 1024", [] { MklGateQuery q; q.kv_len = 512; return q; }()},
            {"gqa ratio 1", [] { MklGateQuery q; q.gqa_ratio = 1; return q; }()},
            {"alibi bias", [] { MklGateQuery q; q.max_bias = 0.5f; return q; }()},
            {"logit softcap", [] { MklGateQuery q; q.logit_softcap = 30.0f; return q; }()},
            {"attention sinks", [] { MklGateQuery q; q.has_sinks = true; return q; }()},
            {"no mask", [] { MklGateQuery q; q.has_mask = false; return q; }()},
            {"batch mismatch", [] { MklGateQuery q; q.batch_mismatch = true; return q; }()},
        };
        for (const auto & test_case : cases) {
            engine::test::require(mirror_mkl_gate(test_case.query) == FattnKernel::Tile,
                                  std::string("envelope violation must fall back to TILE: ") +
                                      test_case.name);
        }
        // Interleaved-but-aligned KV strides stay on MKL (nb1 % ne0*2 == 0
        // holds for both dense and interleaved layouts).
        {
            MklGateQuery q;
            q.kv_row_stride = 8 * 128 * 2;
            engine::test::require(mirror_mkl_gate(q) == FattnKernel::Mkl,
                                  "interleaved but aligned KV strides stay on MKL");
        }
    }
    // Pathological stride (nb1 not a multiple of the row size) falls through.
    {
        MklGateQuery q;
        q.head_dim = 128;
        q.kv_row_stride = 75;
        engine::test::require(mirror_mkl_gate(q) == FattnKernel::Tile,
                              "non-multiple KV row stride must fall back to TILE");
    }

    // ---- Kill switch: GGML_SYCL_ENABLE_MKL_FA=0.
    {
        MklGateQuery q;
        q.env_enabled = false;
        engine::test::require(mirror_mkl_gate(q) == FattnKernel::Tile,
                              "GGML_SYCL_ENABLE_MKL_FA=0 must disable the MKL path");
    }

    std::printf("test_fork_sycl_fattn_mkl_gate: all cases passed\n");
    return 0;
} catch (const std::exception & error) {
    std::fprintf(stderr, "test_fork_sycl_fattn_mkl_gate FAILED: %s\n", error.what());
    return 1;
}
