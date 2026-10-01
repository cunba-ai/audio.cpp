// Fork-regression test: ggml_round_bf16() unary op (audio.cpp fork).
//
// ROUND_BF16 rounds f32 values onto the bf16 grid (round-half-to-even, NaN
// forced quiet) and is used by the fork's bf16 weight-packing paths (e.g.
// yue2 NAR). The op spans ggml.c (builder), the UNARY op table and the CPU
// kernel (unary-ops.cpp op_round_bf16). This test pins:
//
//   1. The builder: result is always F32, op params carry
//      GGML_UNARY_OP_ROUND_BF16, source may be F32/F16/BF16.
//   2. CPU numerics vs the exact bf16 bit formula
//      ((u.i + 0x7fff + lsb) >> 16): representable values are unchanged,
//      midpoints round to even, NaN stays NaN (quieted), +/-inf survives,
//      zero signs are preserved.
//   3. F16 input widens through the same rounding; BF16 input is a fixpoint.
//
// Note on the original SYCL fix (d6c77c75 "round_bf16 launcher dereferenced a
// host pointer inside the device lambda"): the current tree (upstream 0.12.0
// baseline after PR #39) has no SYCL round_bf16 launcher at all, so there is
// no host-pointer-extraction code to guard here; that part is N/A and tracked
// by test_fork_source_anchors.cpp asserting the absence so a merge that
// reintroduces the old broken launcher shape gets flagged for review.

#include "test_assert.h"

#include "engine/framework/core/backend.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <cmath>
#include <cstdint>
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

constexpr size_t kGraphBytes = 8 * 1024 * 1024;
constexpr size_t kGraphNodes = 1024;

// Reference: the exact bit formula from ggml-impl.h
// ggml_compute_fp32_to_bf16() followed by widening back to f32.
float reference_round_bf16(float x) {
    union {
        float f;
        uint32_t i;
    } u;
    u.f = x;
    uint16_t h;
    if ((u.i & 0x7fffffff) > 0x7f800000) {  // NaN
        h = static_cast<uint16_t>((u.i >> 16) | 64);
    } else {
        h = static_cast<uint16_t>((u.i + (0x7fff + ((u.i >> 16) & 1))) >> 16);
    }
    union {
        uint16_t h;
        float f;
    } out;  // bf16 -> f32 is a 16-bit left shift into the top half
    uint32_t widened = static_cast<uint32_t>(h) << 16;
    std::memcpy(&out.f, &widened, sizeof(out.f));
    return out.f;
}

struct CpuRunner {
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_context * ctx = nullptr;

    CpuRunner() {
        backend = engine::core::init_backend({engine::core::BackendType::Cpu, 0, 2});
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

uint32_t f32_bits(float x) {
    uint32_t bits = 0;
    std::memcpy(&bits, &x, sizeof(bits));
    return bits;
}

}  // namespace

int main() try {
    // ---- Builder contract.
    {
        CpuRunner runner;
        ggml_tensor * src = new_tensor(runner.ctx, GGML_TYPE_F32, {16});
        ggml_tensor * out = ggml_round_bf16(runner.ctx, src);
        engine::test::require_eq(out->type, GGML_TYPE_F32, "round_bf16 result type must be F32");
        engine::test::require_eq(out->op, GGML_OP_UNARY, "round_bf16 lowers to GGML_OP_UNARY");
        engine::test::require_eq(ggml_get_unary_op(out), GGML_UNARY_OP_ROUND_BF16,
                                 "round_bf16 unary op selector");
        engine::test::require(out->src[0] == src, "round_bf16 source wiring");
        // F16 / BF16 sources are accepted by the builder (widened to F32 out).
        ggml_tensor * src_f16 = new_tensor(runner.ctx, GGML_TYPE_F16, {4});
        ggml_tensor * src_bf16 = new_tensor(runner.ctx, GGML_TYPE_BF16, {4});
        ggml_round_bf16(runner.ctx, src_f16);
        ggml_round_bf16(runner.ctx, src_bf16);
    }

    // ---- F32 numerics vs the bit formula.
    {
        std::vector<float> inputs = {
            0.0f,
            -0.0f,
            1.0f,
            -1.0f,
            0.5f,
            2.0f,
            3.14159265f,
            -2.718281828f,
            1e-38f,                 // denormal-scale tiny
            65504.0f,
            1e30f,
            std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity(),
            // bf16 grid midpoints: between 1.0 (0x3F80) and the next bf16 step
            // 1.0078125 the midpoint 1.00390625 must round to EVEN (1.0).
            1.00390625f,
            // and one ulp above the midpoint must round up.
            1.00390628f,
            // midpoint between 2.0 (even mantissa) and 2.015625 -> rounds to 2.0.
            2.0078125f,
            // midpoint between 3.0 (odd mantissa) and 3.03125 -> rounds to 3.03125.
            3.015625f,
        };
        // Sweep a dense range around 1..2 (all bf16 midpoints there).
        for (int i = 0; i < 256; i++) {
            const float base = 1.0f + static_cast<float>(i) * (1.0f / 256.0f);
            inputs.push_back(base);
            inputs.push_back(-base);
        }

        CpuRunner runner;
        ggml_tensor * src = new_tensor(runner.ctx, GGML_TYPE_F32, {static_cast<int64_t>(inputs.size())});
        ggml_tensor * out = ggml_round_bf16(runner.ctx, src);
        runner.allocate();
        std::memcpy(src->data, inputs.data(), inputs.size() * sizeof(float));
        const std::vector<float> got = runner.run(out);

        for (size_t i = 0; i < inputs.size(); i++) {
            const float want = reference_round_bf16(inputs[i]);
            if (std::isnan(want)) {
                engine::test::require(std::isnan(got[i]),
                                      "round_bf16(NaN) must stay NaN at index " + std::to_string(i));
                continue;
            }
            engine::test::require(
                f32_bits(got[i]) == f32_bits(want),
                "round_bf16 bit mismatch at index " + std::to_string(i) + ": input=" +
                    std::to_string(inputs[i]) + " got=" + std::to_string(got[i]) +
                    " want=" + std::to_string(want));
        }
        // Signed zeros survive the round trip.
        engine::test::require(f32_bits(got[0]) == 0x00000000u, "+0.0 preserved");
        engine::test::require(f32_bits(got[1]) == 0x80000000u, "-0.0 preserved");
    }

    // ---- NaN input quieted but still NaN.
    {
        CpuRunner runner;
        ggml_tensor * src = new_tensor(runner.ctx, GGML_TYPE_F32, {2});
        ggml_tensor * out = ggml_round_bf16(runner.ctx, src);
        runner.allocate();
        const float nan_value = std::numeric_limits<float>::quiet_NaN();
        std::memcpy(src->data, &nan_value, sizeof(float));
        const float infs = std::numeric_limits<float>::infinity();
        std::memcpy(static_cast<char *>(src->data) + sizeof(float), &infs, sizeof(float));
        const std::vector<float> got = runner.run(out);
        engine::test::require(std::isnan(got[0]), "NaN input stays NaN");
        engine::test::require(std::isinf(got[1]) && got[1] > 0, "+inf survives rounding");
    }

    // ---- F16 input: widen then round; BF16 input: fixpoint.
    {
        CpuRunner runner;
        const std::vector<float> values = {0.25f, 1.5f, 2.75f, -3.25f, 100.125f, 0.0078125f};
        ggml_tensor * src_f16 = new_tensor(runner.ctx, GGML_TYPE_F16, {static_cast<int64_t>(values.size())});
        ggml_tensor * src_bf16 = new_tensor(runner.ctx, GGML_TYPE_BF16, {static_cast<int64_t>(values.size())});
        ggml_tensor * out_f16 = ggml_round_bf16(runner.ctx, src_f16);
        ggml_tensor * out_bf16 = ggml_round_bf16(runner.ctx, src_bf16);
        runner.allocate();
        for (size_t i = 0; i < values.size(); i++) {
            const ggml_fp16_t h = ggml_fp32_to_fp16(values[i]);
            std::memcpy(static_cast<char *>(src_f16->data) + i * sizeof(ggml_fp16_t), &h, sizeof(h));
            const uint16_t b = static_cast<uint16_t>(f32_bits(values[i]) >> 16);  // truncate only for input construction
            std::memcpy(static_cast<char *>(src_bf16->data) + i * sizeof(uint16_t), &b, sizeof(b));
        }
        const std::vector<float> got_f16 = runner.run(out_f16);
        for (size_t i = 0; i < values.size(); i++) {
            // f16 values are exactly representable in bf16? Not all (f16 has 10
            // mantissa bits, bf16 has 7), so compare against the formula on the
            // widened f16 value.
            const float widened = ggml_fp16_to_fp32(ggml_fp32_to_fp16(values[i]));
            engine::test::require(f32_bits(got_f16[i]) == f32_bits(reference_round_bf16(widened)),
                                  "round_bf16(F16) must equal the formula applied to the widened value");
        }

        const std::vector<float> got_bf16 = runner.run(out_bf16);
        const std::vector<float> bf16_inputs = engine::core::read_tensor_bf16(src_bf16);
        for (size_t i = 0; i < values.size(); i++) {
            engine::test::require(f32_bits(got_bf16[i]) == f32_bits(bf16_inputs[i]),
                                  "round_bf16(BF16 input) must be a fixpoint");
        }
    }

    std::printf("test_fork_round_bf16: all cases passed\n");
    return 0;
} catch (const std::exception & error) {
    std::fprintf(stderr, "test_fork_round_bf16 FAILED: %s\n", error.what());
    return 1;
}
