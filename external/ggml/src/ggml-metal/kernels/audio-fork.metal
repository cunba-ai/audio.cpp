// audio.cpp fork kernels ported from the vendored ggml 0.12.0 single-file
// ggml-metal.metal. Each kernel is self-contained and reads its parameters via
// the ggml_metal_kargs_* structs declared in ggml-metal-impl.h (inlined by the
// metallib build).
#include "common.h"

// ---------------------------------------------------------------------------
// GGML_OP_SNAKE_1D: x + sin(alpha*x)^2 / alpha  (music models, dual input)
// ---------------------------------------------------------------------------
kernel void kernel_snake_1d_f32(
        constant ggml_metal_kargs_snake_1d & args,
        device  const char * src0,
        device  const char * src1,
        device        char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int n = args.ne00*args.ne01;

    const int ith = tgpig.x*ntg.x + tpitg.x;

    if (ith >= n) {
        return;
    }

    const int i0 = ith % args.ne00;
    const int i1 = ith / args.ne00;

    device const float * x = (device const float *)(src0 + i0*args.nb00 + i1*args.nb01);
    device const float * a = (device const float *)(src1 + i0*4); // alpha [C,1] contiguous f32
    device       float * y = (device       float *)(dst  + i0*args.nb0  + i1*args.nb1);

    const float xv = x[0];
    const float av = a[0];
    const float ax = xv * av;
    const float s  = sin(ax);
    const float s2 = s * s;

    y[0] = xv + s2/av;
}


// ---------------------------------------------------------------------------
// broadcast binary op (GGML_OP_ADD/MUL/... with non-contiguous broadcasting)
// ---------------------------------------------------------------------------

template<typename T>
kernel void kernel_bin_bcast_impl(
        constant ggml_metal_kargs_bin & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int i0 = tgpig.x*ntg.x + tpitg.x;
    const int i1 = tgpig.y;
    const int i2 = tgpig.z % args.ne2;
    const int i3 = tgpig.z / args.ne2;

    if (i0 >= args.ne0) {
        return;
    }

    const int i00 = i0 % args.ne00;
    const int i01 = i1 % args.ne01;
    const int i02 = i2 % args.ne02;
    const int i03 = i3 % args.ne03;

    const int i10 = i0 % args.ne10;
    const int i11 = i1 % args.ne11;
    const int i12 = i2 % args.ne12;
    const int i13 = i3 % args.ne13;

    device const T * src0_ptr = (device const T *) (src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01 + i00*args.nb00);
    device const T * src1_ptr = (device const T *) (src1 + i13*args.nb13 + i12*args.nb12 + i11*args.nb11 + i10*args.nb10);
    device       T * dst_ptr  = (device       T *) (dst  + i3 *args.nb3  + i2 *args.nb2  + i1 *args.nb1  + i0 *args.nb0);

    if (FC_bin_op == 0) {
        *dst_ptr = *src0_ptr + *src1_ptr;
    }

    if (FC_bin_op == 1) {
        *dst_ptr = *src0_ptr - *src1_ptr;
    }
}

typedef decltype(kernel_bin_bcast_impl<float>) kernel_bin_bcast_f32_t;
typedef decltype(kernel_bin_bcast_impl<half>)  kernel_bin_bcast_f16_t;

template [[host_name("kernel_bin_bcast_f32")]] kernel kernel_bin_bcast_f32_t kernel_bin_bcast_impl<float>;
template [[host_name("kernel_bin_bcast_f16")]] kernel kernel_bin_bcast_f16_t kernel_bin_bcast_impl<half>;

// ---------------------------------------------------------------------------
// GGML_OP_CONV_TRANSPOSE_2D batched linear variant (N > 1 fast path)
// ---------------------------------------------------------------------------

template <typename T>
kernel void kernel_conv_transpose_2d_linear(
        constant ggml_metal_kargs_conv_transpose_2d_linear & args,
        device const T * src0,
        device const float * src1,
        device       float * dst,
        uint         tgpig [[threadgroup_position_in_grid]],
        uint         tpitg [[thread_position_in_threadgroup]],
        uint         ntg   [[threads_per_threadgroup]]) {

    const int global_idx = tgpig * ntg + tpitg;
    if (global_idx >= args.total) {
        return;
    }

    const int out_x = global_idx % args.OW;
    const int out_y = (global_idx / args.OW) % args.OH;
    const int out_c = (global_idx / (args.OW * args.OH)) % args.OC;
    const int out_n = global_idx / (args.OW * args.OH * args.OC);

    float acc = 0.0f;

    if (args.IH == 1 && args.OH == 1 && args.KH == 1) {
        for (int in_c = 0; in_c < args.IC; ++in_c) {
            const int input_base = (args.IW * args.IC) * out_n + args.IW * in_c;
            const int kernel_base = (args.KW * args.OC) * in_c + args.KW * out_c;
            for (int kw = 0; kw < args.KW; ++kw) {
                int in_x = out_x - kw;
                if (in_x < 0 || in_x % args.s0) {
                    continue;
                }
                in_x /= args.s0;
                if (in_x >= args.IW) {
                    continue;
                }

                acc += src1[input_base + in_x] * float(src0[kernel_base + kw]);
            }
        }

        dst[global_idx] = acc;
        return;
    }

    for (int in_c = 0; in_c < args.IC; ++in_c) {
        for (int kh = 0; kh < args.KH; ++kh) {
            int in_y = out_y - kh;
            if (in_y < 0 || in_y % args.s0) {
                continue;
            }
            in_y /= args.s0;
            if (in_y >= args.IH) {
                continue;
            }

            for (int kw = 0; kw < args.KW; ++kw) {
                int in_x = out_x - kw;
                if (in_x < 0 || in_x % args.s0) {
                    continue;
                }
                in_x /= args.s0;
                if (in_x >= args.IW) {
                    continue;
                }

                const int input_idx =
                    (args.IW * args.IH * args.IC) * out_n + (args.IW * args.IH) * in_c + (args.IW) * in_y + in_x;
                const int kernel_idx =
                    (args.KH * args.KW * args.OC) * in_c + (args.KH * args.KW) * out_c + (args.KW) * kh + kw;

                acc += src1[input_idx] * float(src0[kernel_idx]);
            }
        }
    }

    dst[global_idx] = acc;
}

template [[host_name("kernel_conv_transpose_2d_linear_f32_f32")]]
kernel void kernel_conv_transpose_2d_linear<float>(
    constant ggml_metal_kargs_conv_transpose_2d_linear & args,
    device const float * src0,
    device const float * src1,
    device       float * dst,
    uint         tgpig [[threadgroup_position_in_grid]],
    uint         tpitg [[thread_position_in_threadgroup]],
    uint         ntg   [[threads_per_threadgroup]]);

template [[host_name("kernel_conv_transpose_2d_linear_f16_f32")]]
kernel void kernel_conv_transpose_2d_linear<half>(

// ---------------------------------------------------------------------------
// GGML_OP_DIAG_MASK_INF (retired upstream in favour of the TRI ops)
// ---------------------------------------------------------------------------

kernel void kernel_diag_mask_inf_f32(
        constant ggml_metal_kargs_diag_mask_inf & args,
        device const char * src0,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort3 tptg[[threads_per_threadgroup]]) {
    const int32_t i0 = tgpig.x*tptg.x + tiitg;
    const int32_t i1 = tgpig.y;
    const int32_t i2 = tgpig.z % args.ne2;
    const int32_t i3 = tgpig.z / args.ne2;

    if (i0 >= args.ne0) {
        return;
    }

    device const float * src0_ptr = (device const float *)(src0 + i3*args.nb03 + i2*args.nb02 + i1*args.nb01 + i0*args.nb00);
    device       float * dst_ptr  = (device       float *)(dst  + i3*args.nb3  + i2*args.nb2  + i1*args.nb1  + i0*args.nb0);

    *dst_ptr = i0 > args.n_past + i1 ? -INFINITY : *src0_ptr;
}

