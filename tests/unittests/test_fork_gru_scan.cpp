// Fork-regression test: GGML_OP_GRU_SCAN fused op (audio.cpp fork).
//
// The fork replaces the RMVPE per-frame unrolled GRU graph (~12k dispatches)
// with one ggml_gru_scan() op (fe01cab4, restored by dc6d1b59 after the
// PR #39 merge dropped it). This test guards the full registration chain and
// the CPU reference kernel:
//
//   1. Registration: GGML_OP_GRU_SCAN exists in the op table, carries the
//      "GRU_SCAN" name / "gru_scan(x_ih, h0, w_hh, b_hh, keep)" symbol, and
//      the CPU backend reports supports_op() == true (the engine's fused-path
//      guard in rmvpe_pitch_extractor.cpp relies on exactly that query).
//   2. Numerics: fused output == scalar reference (same op order as
//      ggml-cpu.c's ggml_compute_forward_gru_scan) bit-for-bit, and fused
//      output == ggml unrolled graph (the engine's pre-fusion lowering)
//      within float GEMM tolerance.
//   3. Edge cases: keep=0 freezes the state exactly, keep=1 applies the full
//      update, reverse=true scans frames back-to-front, F=1 single step, and
//      H=1024 (GGML_GRU_SCAN_MAX_H) still works.
//
// An upstream merge that drops the op breaks the ggml_gru_scan() link symbol
// (compile error) or the CPU kernel (assertion below) - never silently.

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

constexpr size_t kGraphBytes = 128 * 1024 * 1024;
constexpr size_t kGraphNodes = 8192;

struct GruParams {
    int64_t hidden = 8;   // H, must be % 8 == 0 (ggml_gru_scan contract)
    int64_t frames = 5;   // F
    bool reverse = false;
    unsigned seed = 1;
};

struct GruTensors {
    ggml_tensor * x_ih = nullptr;  // [3H, F]  x @ W_ih + b_ih (bias_ih included)
    ggml_tensor * h0 = nullptr;    // [H]
    ggml_tensor * w_hh = nullptr;  // [H, 3H] row-major [o][i]
    ggml_tensor * b_hh = nullptr;  // [3H]
    ggml_tensor * keep = nullptr;  // [F]
};

// Host RNG helpers.
float sample(std::mt19937 & rng) {
    return -1.0f + 2.0f * (static_cast<float>(rng()) - rng.min()) /
                       static_cast<float>(rng.max() - rng.min());
}

void fill_uniform(const GruTensors & t, std::mt19937 & rng) {
    std::vector<float> buf;
    auto fill = [&](ggml_tensor * tensor) {
        const int64_t count = ggml_nelements(tensor);
        buf.resize(static_cast<size_t>(count));
        for (auto & v : buf) {
            v = sample(rng);
        }
        std::memcpy(tensor->data, buf.data(), buf.size() * sizeof(float));
    };
    fill(t.x_ih);
    fill(t.h0);
    fill(t.w_hh);
    fill(t.b_hh);
    fill(t.keep);
}

// Scalar reference with the exact operation order of the CPU kernel in
// external/ggml/src/ggml-cpu/ggml-cpu.c (ggml_compute_forward_gru_scan):
// per-output sequential dot, 1/(1+expf(-x)) sigmoid, tanhf candidate.
struct GruReference {
    std::vector<float> sequence;  // [F][H]
    std::vector<float> final_hidden;
};

GruReference run_reference(const GruParams & p, const GruTensors & t, const std::vector<float> & keep_override) {
    const int64_t H = p.hidden;
    const int64_t F = t.x_ih->ne[1];
    const auto * xw = (const float *) t.x_ih->data;
    const auto * hw = (const float *) t.w_hh->data;
    const auto * bw = (const float *) t.b_hh->data;
    const auto * kw = (const float *) t.keep->data;
    const auto * h_init = (const float *) t.h0->data;

    GruReference out;
    out.sequence.assign(static_cast<size_t>(F * H), 0.0f);
    out.final_hidden.assign(static_cast<size_t>(H), 0.0f);

    std::vector<float> h_cur(static_cast<size_t>(H));
    for (int64_t i = 0; i < H; i++) {
        h_cur[static_cast<size_t>(i)] = h_init[i];
    }

    for (int64_t s = 0; s < F; s++) {
        const int64_t step = p.reverse ? (F - 1 - s) : s;
        std::vector<float> hi(static_cast<size_t>(3 * H));
        for (int64_t o = 0; o < 3 * H; o++) {
            float acc = bw[o];
            const float * wrow = hw + o * H;
            for (int64_t i = 0; i < H; i++) {
                acc += h_cur[static_cast<size_t>(i)] * wrow[i];
            }
            hi[static_cast<size_t>(o)] = acc;
        }
        const float * xt = xw + step * (3 * H);
        const float kv = keep_override.empty() ? kw[step] : keep_override[static_cast<size_t>(step)];
        for (int64_t j = 0; j < H; j++) {
            const float r = 1.0f / (1.0f + expf(-(xt[j] + hi[static_cast<size_t>(j)])));
            const float z = 1.0f / (1.0f + expf(-(xt[H + j] + hi[static_cast<size_t>(H + j)])));
            const float n = tanhf(xt[2 * H + j] + r * hi[static_cast<size_t>(2 * H + j)]);
            const float updated = n + z * (h_cur[static_cast<size_t>(j)] - n);
            const float h_new = h_cur[static_cast<size_t>(j)] + kv * (updated - h_cur[static_cast<size_t>(j)]);
            out.sequence[static_cast<size_t>(step * H + j)] = h_new;
            h_cur[static_cast<size_t>(j)] = h_new;
        }
    }
    out.final_hidden.assign(h_cur.begin(), h_cur.end());
    return out;
}

// The engine's pre-fusion lowering (rmvpe_pitch_extractor.cpp
// build_gru_chunk_graph) expressed directly over ggml ops: per-frame
// GEMM + gate math + concat. Used for a second, graph-level comparison.
struct CpuRunner {
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_context * ctx = nullptr;

    explicit CpuRunner(size_t graph_bytes) {
        backend = engine::core::init_backend({engine::core::BackendType::Cpu, 0, 4});
        ggml_init_params init{};
        init.mem_size = graph_bytes;
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
    void compute(ggml_tensor * output) {
        allocate();
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, kGraphNodes, false);
        ggml_build_forward_expand(graph, output);
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("graph compute failed");
        }
    }
};

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

GruTensors make_gru_tensors(ggml_context * ctx, int64_t H, int64_t F) {
    GruTensors t;
    t.x_ih = new_tensor(ctx, GGML_TYPE_F32, {3 * H, F});
    t.h0 = new_tensor(ctx, GGML_TYPE_F32, {H});
    t.w_hh = new_tensor(ctx, GGML_TYPE_F32, {H, 3 * H});
    t.b_hh = new_tensor(ctx, GGML_TYPE_F32, {3 * H});
    t.keep = new_tensor(ctx, GGML_TYPE_F32, {F});
    return t;
}

// Builds ggml_gru_scan() over pre-filled input tensors and returns the raw
// [H, F+1] output tensor (sequence rows + final hidden row).
ggml_tensor * build_fused(CpuRunner & runner, const GruTensors & t, bool reverse) {
    return ggml_gru_scan(runner.ctx, t.x_ih, t.h0, t.w_hh, t.b_hh, t.keep, reverse);
}

ggml_tensor * build_unrolled(CpuRunner & runner, const GruTensors & t, bool reverse) {
    ggml_context * ctx = runner.ctx;
    const int64_t H = t.h0->ne[0];
    const int64_t F = t.x_ih->ne[1];
    ggml_tensor * hidden = t.h0;
    std::vector<ggml_tensor *> steps(static_cast<size_t>(F));
    for (int64_t index = 0; index < F; index++) {
        const int64_t frame = reverse ? (F - 1 - index) : index;
        // xi = x_ih[:, frame]
        auto * xi = ggml_view_1d(ctx, t.x_ih, 3 * H, static_cast<size_t>(frame) * 3 * H * sizeof(float));
        // hi = W_hh^T . h + b_hh  (row vector times matrix)
        auto * h2d = ggml_reshape_2d(ctx, hidden, H, 1);  // [H] -> [H, 1] row vector
        auto * hi = ggml_add(ctx,
                             ggml_mul_mat(ctx, t.w_hh, h2d),
                             ggml_reshape_2d(ctx, t.b_hh, 3 * H, 1));
        auto * i_r = ggml_view_1d(ctx, xi, H, 0);
        auto * i_z = ggml_view_1d(ctx, xi, H, static_cast<size_t>(H) * sizeof(float));
        auto * i_n = ggml_view_1d(ctx, xi, H, static_cast<size_t>(2 * H) * sizeof(float));
        auto * h_r = ggml_view_1d(ctx, hi, H, 0);
        auto * h_z = ggml_view_1d(ctx, hi, H, static_cast<size_t>(H) * sizeof(float));
        auto * h_n = ggml_view_1d(ctx, hi, H, static_cast<size_t>(2 * H) * sizeof(float));
        auto * reset = ggml_sigmoid(ctx, ggml_add(ctx, i_r, h_r));
        auto * update = ggml_sigmoid(ctx, ggml_add(ctx, i_z, h_z));
        auto * candidate = ggml_tanh(ctx, ggml_add(ctx, i_n, ggml_mul(ctx, reset, h_n)));
        auto * updated = ggml_add(ctx, candidate, ggml_mul(ctx, update, ggml_sub(ctx, hidden, candidate)));
        auto * keep_t = ggml_view_1d(ctx, t.keep, 1, static_cast<size_t>(frame) * sizeof(float));
        auto * keep_rep = ggml_repeat(ctx, keep_t, hidden);
        hidden = ggml_add(ctx, hidden, ggml_mul(ctx, keep_rep, ggml_sub(ctx, updated, hidden)));
        steps[static_cast<size_t>(frame)] = hidden;
    }
    auto * sequence = steps[0];
    for (int64_t i = 1; i < F; i++) {
        sequence = ggml_concat(ctx, sequence, steps[static_cast<size_t>(i)], 0);
    }
    return ggml_concat(ctx, sequence, hidden, 0);  // [H, F+1] like the fused op
}

std::vector<float> read_out(const ggml_tensor * t) {
    std::vector<float> values;
    engine::core::read_tensor_f32_into(t, values);
    return values;
}

void compare_fused_vs_reference(const char * label, const GruParams & p, const std::vector<float> & keep_override = {}) {
    CpuRunner runner(kGraphBytes);
    GruTensors t = make_gru_tensors(runner.ctx, p.hidden, p.frames);
    ggml_tensor * fused = build_fused(runner, t, p.reverse);
    runner.allocate();
    std::mt19937 rng(p.seed);
    fill_uniform(t, rng);
    if (!keep_override.empty()) {
        std::memcpy(t.keep->data, keep_override.data(), keep_override.size() * sizeof(float));
    }
    runner.compute(fused);

    const GruReference ref = run_reference(p, t, keep_override);
    const std::vector<float> got = read_out(fused);

    const int64_t H = p.hidden;
    const int64_t F = p.frames;
    engine::test::require_eq(ggml_nelements(fused), (F + 1) * H, std::string(label) + ": fused element count");

    int64_t mismatches = 0;
    float worst = 0.0f;
    for (int64_t i = 0; i < F * H; i++) {
        const float a = got[static_cast<size_t>(i)];
        const float b = ref.sequence[static_cast<size_t>(i)];
        if (!(a == b)) {
            mismatches++;
            worst = std::max(worst, static_cast<float>(fabs(static_cast<double>(a) - b)));
        }
    }
    for (int64_t j = 0; j < H; j++) {
        const float a = got[static_cast<size_t>(F * H + j)];
        const float b = ref.final_hidden[static_cast<size_t>(j)];
        if (!(a == b)) {
            mismatches++;
            worst = std::max(worst, static_cast<float>(fabs(static_cast<double>(a) - b)));
        }
    }
    engine::test::require_eq(mismatches, int64_t{0},
        std::string(label) + ": fused vs scalar reference must be bit-exact (CPU kernel and reference share op order); worst diff=" +
            std::to_string(worst));
}

void compare_fused_vs_unrolled(const char * label, const GruParams & p) {
    // Fused graph.
    CpuRunner fused_runner(kGraphBytes);
    GruTensors ft = make_gru_tensors(fused_runner.ctx, p.hidden, p.frames);
    ggml_tensor * fused = build_fused(fused_runner, ft, p.reverse);
    fused_runner.allocate();
    std::mt19937 rng(p.seed);
    fill_uniform(ft, rng);
    fused_runner.compute(fused);
    const std::vector<float> fused_values = read_out(fused);

    // Unrolled graph over the same input values.
    CpuRunner unrolled_runner(kGraphBytes);
    GruTensors ut = make_gru_tensors(unrolled_runner.ctx, p.hidden, p.frames);
    ggml_tensor * unrolled_pre = build_unrolled(unrolled_runner, ut, p.reverse);
    unrolled_runner.allocate();
    std::memcpy(ut.x_ih->data, ft.x_ih->data, ggml_nbytes(ut.x_ih));
    std::memcpy(ut.h0->data, ft.h0->data, ggml_nbytes(ut.h0));
    std::memcpy(ut.w_hh->data, ft.w_hh->data, ggml_nbytes(ut.w_hh));
    std::memcpy(ut.b_hh->data, ft.b_hh->data, ggml_nbytes(ut.b_hh));
    std::memcpy(ut.keep->data, ft.keep->data, ggml_nbytes(ut.keep));
    unrolled_runner.compute(unrolled_pre);
    const std::vector<float> unrolled_values = read_out(unrolled_pre);

    const size_t count = fused_values.size();
    engine::test::require_eq(count, unrolled_values.size(), std::string(label) + ": unrolled element count");
    float worst = 0.0f;
    for (size_t i = 0; i < count; i++) {
        const float diff = static_cast<float>(fabs(static_cast<double>(fused_values[i]) - unrolled_values[i]));
        worst = std::max(worst, diff);
    }
    // GEMM accumulation order differs from the scalar kernel; the fork's own
    // on-device verification measured 4.7e-7 for the CPU reference. Allow a
    // small epsilon scaled by H (dot-product length).
    const float tolerance = 4e-5f * static_cast<float>(p.hidden);
    engine::test::require(worst <= tolerance,
        std::string(label) + ": fused vs unrolled graph drift too large: " + std::to_string(worst));
}

}  // namespace

int main() try {
    // ---- Registration chain (a dropped op breaks the link or these names).
    engine::test::require_eq(std::string(ggml_op_name(GGML_OP_GRU_SCAN)), std::string("GRU_SCAN"),
                             "GGML_OP_GRU_SCAN must be registered with its op name");
    engine::test::require_eq(std::string(ggml_op_symbol(GGML_OP_GRU_SCAN)),
                             std::string("gru_scan(x_ih, h0, w_hh, b_hh, keep)"),
                             "GGML_OP_GRU_SCAN symbol table entry");

    {
        // CPU backend must accept the op (rmvpe fused-path guard queries this).
        CpuRunner runner(kGraphBytes);
        GruTensors t = make_gru_tensors(runner.ctx, 8, 4);
        ggml_tensor * node = build_fused(runner, t, false);
        engine::test::require(ggml_backend_supports_op(runner.backend, node),
                              "CPU backend must support GGML_OP_GRU_SCAN (engine fused-path guard)");
    }

    // ---- Happy paths: fused vs scalar reference, bit-exact.
    compare_fused_vs_reference("forward H=8 F=5", {8, 5, false, 1});
    compare_fused_vs_reference("forward H=16 F=7 seed2", {16, 7, false, 2});
    compare_fused_vs_reference("reverse H=8 F=5", {8, 5, true, 3});
    compare_fused_vs_reference("reverse H=32 F=9 seed4", {32, 9, true, 4});

    // ---- Fused vs unrolled engine lowering (float tolerance).
    compare_fused_vs_unrolled("unrolled H=8 F=5", {8, 5, false, 5});
    compare_fused_vs_unrolled("unrolled reverse H=8 F=5", {8, 5, true, 5});
    compare_fused_vs_unrolled("unrolled H=64 F=16", {64, 16, false, 6});

    // ---- Edge cases.
    // keep=0: state frozen, every sequence row equals h0 exactly.
    {
        GruParams p{8, 6, false, 7};
        std::vector<float> keep_zero(6, 0.0f);
        compare_fused_vs_reference("keep=0", p, keep_zero);
        CpuRunner runner(kGraphBytes);
        GruTensors t = make_gru_tensors(runner.ctx, p.hidden, p.frames);
        ggml_tensor * fused = build_fused(runner, t, false);
        runner.allocate();
        std::mt19937 rng(p.seed);
        fill_uniform(t, rng);
        std::memcpy(t.keep->data, keep_zero.data(), keep_zero.size() * sizeof(float));
        runner.compute(fused);
        const std::vector<float> got = read_out(fused);
        const auto * h0 = (const float *) t.h0->data;
        for (int64_t f = 0; f < p.frames; f++) {
            for (int64_t j = 0; j < p.hidden; j++) {
                engine::test::require(
                    got[static_cast<size_t>(f * p.hidden + j)] == h0[j],
                    "keep=0 must freeze the hidden state bit-exactly at every step");
            }
        }
    }
    // keep=1: full update path (also covered bit-exact against the reference).
    compare_fused_vs_reference("keep=1", {8, 4, false, 8}, std::vector<float>(4, 1.0f));
    // Mixed keep pattern, reverse direction.
    {
        std::vector<float> mixed{1.0f, 0.0f, 0.25f, 1.0f, 0.75f};
        compare_fused_vs_reference("keep=mixed reverse", {8, 5, true, 9}, mixed);
    }
    // F=1: single step.
    compare_fused_vs_reference("F=1", {8, 1, false, 10});
    compare_fused_vs_reference("F=1 reverse", {8, 1, true, 11});
    // H = 1024: GGML_GRU_SCAN_MAX_H boundary (RMVPE uses 256; 1024 must pass).
    compare_fused_vs_reference("H=1024 boundary", {1024, 3, false, 12});
    // RMVPE-shaped smoke: H=256, F=32.
    compare_fused_vs_reference("H=256 F=32", {256, 32, false, 13});
    compare_fused_vs_reference("H=256 F=32 reverse", {256, 32, true, 14});

    std::printf("test_fork_gru_scan: all cases passed\n");
    return 0;
} catch (const std::exception & error) {
    std::fprintf(stderr, "test_fork_gru_scan FAILED: %s\n", error.what());
    return 1;
}
