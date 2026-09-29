//
// audio.cpp fork op: fused single-layer GRU scan (RMVPE pitch extractor).
//
// Replaces the engine's per-timestep unrolled GRU graph (~24 tiny dispatches
// per step; a 512-frame chunk = ~12k dispatches) with one op per scan.
// One work-group executes the whole sequential scan. Within a step:
//   phase A: hi = W_hh . h + b_hh — one row per sub-group, lanes stride over
//            H with coalesced reads + sub-group reduction (W streams from L2,
//            ~786 KB/step for H=256);
//   phase B: gates + keep update — one lane per hidden element, coalesced.
// Gate math is identical to the CPU reference (see ggml-cpu.c); the dot
// product reduction order differs (sub-group tree vs sequential), i.e.
// last-ulp level differences only.
//
// src0 x_ih : [3H, F] f32, input projection (r|z|n rows, bias_ih included)
// src1 h0   : [H] f32 initial hidden
// src2 w_hh : [H, 3H] f32 (ne0 = H in, ne1 = 3H out)
// src3 b_hh : [3H] f32
// src4 keep : [F] f32 (contiguous; [1,F] and [F,1] both accepted)
// dst       : [H, F+1] f32 (cols 0..F-1 per-step hidden in time order,
//             col F = final hidden)
// op_params : { i32 reverse }
//

#include <sycl/sycl.hpp>
#include "dpct/helper.hpp"
#include "common.hpp"

#include <cmath>

namespace {
constexpr int kGruThreads = 256;
}  // namespace

void ggml_sycl_gru_scan(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x_ih_t = dst->src[0];
    const ggml_tensor * h0_t   = dst->src[1];
    const ggml_tensor * w_hh_t = dst->src[2];
    const ggml_tensor * b_hh_t = dst->src[3];
    const ggml_tensor * keep_t = dst->src[4];

    const int64_t H = x_ih_t->ne[0] / 3;
    const int64_t F = x_ih_t->ne[1];
    const bool reverse = ggml_get_op_params_i32(dst, 0) != 0;
    // keep: [F] or [1, F]; contiguous layouts put frame t at element t

    const float * x_ih = (const float *) x_ih_t->data;
    const float * h0   = (const float *) h0_t->data;
    const float * w_hh = (const float *) w_hh_t->data;
    const float * b_hh = (const float *) b_hh_t->data;
    const float * keep = (const float *) keep_t->data;
    float * out = (float *) dst->data;

    ggml_sycl_set_device(ctx.device);
    dpct::queue_ptr stream = ctx.stream();

    const sycl::range<1> local_size(kGruThreads);
    const sycl::range<1> global_size(kGruThreads);  // exactly one work-group

    stream->submit([&](sycl::handler & cgh) {
        // hidden state (H) + hidden projection (3H) live in SLM across steps
        sycl::local_accessor<float, 1> h_acc(sycl::range<1>(H), cgh);
        sycl::local_accessor<float, 1> hi_acc(sycl::range<1>(3 * H), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(global_size, local_size),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                const int tid = (int) item.get_local_id(0);
                const int nth = (int) item.get_local_range(0);
                const sycl::sub_group sg = item.get_sub_group();
                const int lane   = (int) sg.get_local_id();
                const int sg_sz  = (int) sg.get_local_range().size();
                const int sg_id  = (int) sg.get_group_id();
                const int n_sg   = (int) item.get_local_range().size() / sg_sz;

                // load initial hidden into SLM
                for (int64_t i = tid; i < H; i += nth) {
                    h_acc[i] = h0[i];
                }
                item.barrier(sycl::access::fence_space::local_space);

                for (int64_t s = 0; s < F; ++s) {
                    const int64_t t = reverse ? (F - 1 - s) : s;

                    // phase A: hi = W_hh . h + b_hh (rows round-robin over
                    // sub-groups; lanes stride over H, coalesced reads)
                    for (int64_t row = sg_id; row < 3 * H; row += n_sg) {
                        const float * wrow = w_hh + row * H;
                        float acc = 0.0f;
                        for (int64_t i = lane; i < H; i += sg_sz) {
                            acc += h_acc[i] * wrow[i];
                        }
                        acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
                        if (lane == 0) {
                            hi_acc[row] = acc + b_hh[row];
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    // phase B: gates + keep update, one lane per element
                    const float * xt = x_ih + t * (3 * H);
                    for (int64_t j = tid; j < H; j += nth) {
                        const float r = 1.0f / (1.0f + sycl::exp(-(xt[j]           + hi_acc[j])));
                        const float z = 1.0f / (1.0f + sycl::exp(-(xt[H + j]       + hi_acc[H + j])));
                        const float n = sycl::tanh(xt[2 * H + j] + r * hi_acc[2 * H + j]);
                        const float updated = n + z * (h_acc[j] - n);
                        const float kv = keep[t];
                        const float h_new = h_acc[j] + kv * (updated - h_acc[j]);
                        out[t * H + j] = h_new;
                        h_acc[j] = h_new;
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                // final hidden -> last column
                for (int64_t j = tid; j < H; j += nth) {
                    out[F * H + j] = h_acc[j];
                }
            });
    });
}

bool ggml_sycl_gru_scan_supported(const ggml_tensor * op) {
    if (op->type != GGML_TYPE_F32) {
        return false;
    }
    for (int i = 0; i < 5; ++i) {
        const ggml_tensor * t = op->src[i];
        if (t == nullptr || t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) {
            return false;
        }
    }
    const ggml_tensor * x_ih = op->src[0];
    const ggml_tensor * h0   = op->src[1];
    const ggml_tensor * w_hh = op->src[2];
    const ggml_tensor * b_hh = op->src[3];
    const ggml_tensor * keep = op->src[4];
    const int64_t H = h0->ne[0];
    return H > 0 && H % 8 == 0 &&
        x_ih->ne[0] == 3 * H && x_ih->ne[2] == 1 && x_ih->ne[3] == 1 &&
        w_hh->ne[0] == H && w_hh->ne[1] == 3 * H &&
        b_hh->ne[0] == 3 * H &&
        ggml_nelements(keep) == x_ih->ne[1];
}
