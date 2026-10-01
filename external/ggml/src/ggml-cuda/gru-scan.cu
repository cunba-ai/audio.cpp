//
// audio.cpp fork op: fused single-layer GRU scan (RMVPE pitch extractor).
// CUDA twin of ggml-sycl/gru-scan.cpp — one block runs the whole sequential
// scan; phase A computes hi = W_hh . h + b_hh one row per warp with coalesced
// reads + warp reduction, phase B applies the gates one lane per element.
// Tensor contract documented in ggml-sycl/gru-scan.cpp.
//

#include "ggml-cuda/common.cuh"
#include "gru-scan.cuh"

// HIP 6.4 has no __shfl_down_sync (only __shfl_down); the CUDA twin is
// stubbed out for HIP/MUSA builds following the convrot-linear.cu precedent.
// The engine queries ggml_backend_supports_op() before using the fused op and
// falls back to the unrolled per-timestep graph, so HIP keeps working.
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)

void ggml_cuda_op_gru_scan(ggml_backend_cuda_context &, ggml_tensor *) {
    GGML_ABORT("gru_scan is only implemented for CUDA");
}

#else

static __global__ void gru_scan_f32_cuda(
        const float * __restrict__ x_ih,   // [3H, F]
        const float * __restrict__ h0,     // [H]
        const float * __restrict__ w_hh,   // [H, 3H] (row-major out-major)
        const float * __restrict__ b_hh,   // [3H]
        const float * __restrict__ keep,   // [F]
        float       * __restrict__ out,    // [H, F+1]
        const int64_t H,
        const int64_t F,
        
        const int     reverse) {
    constexpr int nth = CUDA_GRU_SCAN_THREADS;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    constexpr int n_warps = nth / 32;

    extern __shared__ float smem[];
    float * h  = smem;          // H
    float * hi = smem + H;      // 3H

    for (int64_t i = tid; i < H; i += nth) {
        h[i] = h0[i];
    }
    __syncthreads();

    for (int64_t s = 0; s < F; ++s) {
        const int64_t t = reverse ? (F - 1 - s) : s;

        // phase A: hi = W_hh . h + b_hh
        for (int64_t row = warp; row < 3 * H; row += n_warps) {
            const float * wrow = w_hh + row * H;
            float acc = 0.0f;
            for (int64_t i = lane; i < H; i += 32) {
                acc += h[i] * wrow[i];
            }
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                acc += __shfl_down_sync(0xffffffffu, acc, off);
            }
            if (lane == 0) {
                hi[row] = acc + b_hh[row];
            }
        }
        __syncthreads();

        // phase B: gates + keep update
        const float * xt = x_ih + t * (3 * H);
        for (int64_t j = tid; j < H; j += nth) {
            const float r = 1.0f / (1.0f + __expf(-(xt[j]       + hi[j])));
            const float z = 1.0f / (1.0f + __expf(-(xt[H + j]   + hi[H + j])));
            const float n = tanhf(xt[2 * H + j] + r * hi[2 * H + j]);
            const float updated = n + z * (h[j] - n);
            const float kv = keep[t];
            const float h_new = h[j] + kv * (updated - h[j]);
            out[t * H + j] = h_new;
            h[j] = h_new;
        }
        __syncthreads();
    }

    for (int64_t j = tid; j < H; j += nth) {
        out[F * H + j] = h[j];
    }
}

void ggml_cuda_op_gru_scan(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x_ih_t = dst->src[0];
    const ggml_tensor * h0_t   = dst->src[1];
    const ggml_tensor * w_hh_t = dst->src[2];
    const ggml_tensor * b_hh_t = dst->src[3];
    const ggml_tensor * keep_t = dst->src[4];

    const int64_t H = x_ih_t->ne[0] / 3;
    const int64_t F = x_ih_t->ne[1];
    const int reverse = ggml_get_op_params_i32(dst, 0) != 0;

    const float * x_ih = (const float *) x_ih_t->data;
    const float * h0   = (const float *) h0_t->data;
    const float * w_hh = (const float *) w_hh_t->data;
    const float * b_hh = (const float *) b_hh_t->data;
    const float * keep = (const float *) keep_t->data;
    float * out = (float *) dst->data;

    cudaStream_t stream = ctx.stream();

    const size_t smem = (H + 3 * H) * sizeof(float);
    static_assert(CUDA_GRU_SCAN_THREADS % 32 == 0, "");
    gru_scan_f32_cuda<<<1, CUDA_GRU_SCAN_THREADS, smem, stream>>>(
        x_ih, h0, w_hh, b_hh, keep, out, H, F, reverse);
}

#endif  // !GGML_USE_HIP && !GGML_USE_MUSA
