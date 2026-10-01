// Fork-regression test: SYCL weight-reorder / GET_ROWS interplay (audio.cpp
// fork, restored by dc6d1b59 after PR #39 dropped it).
//
// The SYCL backend rewrites Q4_0/Q8_0 weight tensors in place into an SoA
// layout (all quants first, then all fp16 scales - see reorder_qw_q4_0()/
// reorder_qw_q8_0() in ggml-sycl.cpp). Two fork fixes guard that layout:
//
//   a. should_reorder_tensor() never reorders a tensor that any executed
//      graph has read through GET_ROWS (tracked via
//      ctx.get_rows_sources) - without it, tied-embedding models
//      (whisper/qwen3/ace_step) read garbage from the second decode step.
//   b. getrows.cpp gained SoA read kernels (k_get_rows_quant_soa) for Q4_0/
//      Q8_0 so that even an already-reordered tensor decodes correctly.
//
// The SYCL kernels themselves need a GPU (see the GPU-gated
// test_fork_gpu_backend_kernels.cpp); this CPU test pins the two contracts
// the kernels rely on:
//
//   1. Layout contract: a host mirror of the reorder pass + a host mirror of
//      the SoA decode kernel reproduce, bit-for-bit, the reference GET_ROWS
//      output that ggml's CPU backend produces on the standard layout.
//   2. Predicate contract: a mirror of should_reorder_tensor's decision
//      table - GET_ROWS sources stay in AoS layout, non-GET_ROWS mul_mat
//      sources with batch-1 activations still reorder, other ops don't.

#include "test_assert.h"

#include "engine/framework/core/backend.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <set>
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

constexpr size_t kGraphBytes = 4 * 1024 * 1024;
constexpr size_t kGraphNodes = 1024;

constexpr int kQk = 32;  // QK4_0 == QK8_0 == 32

struct BlockSpec {
    float scale;             // fp16-representable scale of the block
    std::vector<uint8_t> payload;  // 16 bytes (Q4_0) or 32 bytes (Q8_0)
};

// Builds a standard-layout quantized tensor [ncols, nrows] from explicit
// block specs (ncols must be a multiple of 32; nrows * ncols/32 blocks).
std::vector<uint8_t> pack_blocks(const std::vector<BlockSpec> & blocks, bool q8) {
    const size_t payload = q8 ? 32 : 16;
    std::vector<uint8_t> data;
    data.reserve(blocks.size() * (2 + payload));
    for (const auto & block : blocks) {
        const ggml_fp16_t d = ggml_fp32_to_fp16(block.scale);
        const uint8_t d_bytes[2] = {static_cast<uint8_t>(d & 0xff), static_cast<uint8_t>(d >> 8)};
        data.push_back(d_bytes[0]);
        data.push_back(d_bytes[1]);
        engine::test::require(block.payload.size() == payload, "test block payload size");
        data.insert(data.end(), block.payload.begin(), block.payload.end());
    }
    return data;
}

// Host mirror of reorder_qw_q4_0()/reorder_qw_q8_0() (ggml-sycl.cpp):
// quants region first (one byte per element for Q8_0, one nibble per element
// for Q4_0), then the fp16 scales indexed by the GLOBAL block number.
std::vector<uint8_t> mirror_reorder_to_soa(const std::vector<uint8_t> & blocks, bool q8,
                                           int64_t ncols, int64_t nrows) {
    const size_t block_bytes = q8 ? 34 : 18;
    const size_t payload = q8 ? 32 : 16;
    const int64_t blocks_total = ncols * nrows / kQk;
    engine::test::require_eq(blocks.size() / block_bytes, static_cast<size_t>(blocks_total),
                             "block buffer size");
    const int64_t qs_region = q8 ? ncols * nrows : ncols * nrows / 2;
    std::vector<uint8_t> soa(static_cast<size_t>(qs_region + 2 * blocks_total), 0);
    for (int64_t ib = 0; ib < blocks_total; ib++) {
        const uint8_t * src = blocks.data() + ib * block_bytes;
        // quants
        if (q8) {
            std::memcpy(soa.data() + ib * 32, src + 2, 32);
        } else {
            std::memcpy(soa.data() + ib * 16, src + 2, 16);
        }
        // scale (global block index)
        uint16_t d = static_cast<uint16_t>(src[0] | (src[1] << 8));
        std::memcpy(soa.data() + qs_region + ib * 2, &d, sizeof(d));
    }
    return soa;
}

// Host mirror of k_get_rows_quant_soa<qk, qr> (getrows.cpp): decodes element
// (i00, i01) of a SoA-reordered tensor.
float mirror_soa_get_row(const std::vector<uint8_t> & soa, bool q8,
                         int64_t ncols, int64_t nrows, int64_t i00, int64_t i01) {
    const int qr = q8 ? 1 : 2;
    const int blocks_per_row = static_cast<int>(ncols / kQk);
    const int64_t qs_region_bytes = ncols * nrows / qr;
    const int64_t ib = i01 * blocks_per_row + i00 / kQk;
    const int iqs = static_cast<int>(i00 % kQk);

    uint16_t d_bits = 0;
    std::memcpy(&d_bits, soa.data() + qs_region_bytes + ib * 2, sizeof(d_bits));
    const float d = ggml_fp16_to_fp32(d_bits);

    if (q8) {
        const int8_t q = static_cast<int8_t>(soa[static_cast<size_t>(ib * kQk + iqs)]);
        return d * static_cast<float>(q);
    }
    // Q4_0 split-nibble layout: the 16 low nibbles of a block hold elements
    // 0..15, the 16 high nibbles hold 16..31 (dequantize_row_q4_0).
    const int half = kQk / 2;
    const uint8_t byte = soa[static_cast<size_t>(ib * half + (iqs % half))];
    const int v = (byte >> ((iqs / half) * 4)) & 0xf;
    return d * static_cast<float>(v - 8);
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

// Runs GET_ROWS over a standard-layout quantized tensor on the CPU backend
// and returns [ncols x indices] rows (reference output).
std::vector<float> reference_get_rows(ggml_type type, const std::vector<uint8_t> & block_data,
                                      int64_t ncols, int64_t nrows, const std::vector<int32_t> & indices) {
    CpuRunner runner;
    ggml_tensor * src = new_tensor(runner.ctx, type, {ncols, nrows});
    ggml_tensor * ids = new_tensor(runner.ctx, GGML_TYPE_I32, {static_cast<int64_t>(indices.size())});
    ggml_tensor * out = ggml_get_rows(runner.ctx, src, ids);
    runner.allocate();
    std::memcpy(src->data, block_data.data(), block_data.size());
    std::memcpy(ids->data, indices.data(), indices.size() * sizeof(int32_t));
    return runner.run(out);
}

std::vector<BlockSpec> make_blocks(int64_t block_count, bool q8, unsigned seed) {
    std::vector<BlockSpec> blocks;
    const size_t payload = q8 ? 32 : 16;
    const float scales[] = {0.5f, 1.0f, 1.25f, 0.75f, 2.0f, 0.625f, 1.5f, 0.875f};
    unsigned state = seed;
    for (int64_t b = 0; b < block_count; b++) {
        BlockSpec spec;
        spec.scale = scales[b % 8];
        spec.payload.resize(payload);
        for (size_t i = 0; i < payload; i++) {
            state = state * 1664525u + 1013904223u;
            spec.payload[i] = static_cast<uint8_t>((state >> 16) & 0xff);
        }
        blocks.push_back(spec);
    }
    return blocks;
}

void check_type(bool q8, int64_t ncols, int64_t nrows, const std::vector<int32_t> & indices, unsigned seed) {
    const ggml_type type = q8 ? GGML_TYPE_Q8_0 : GGML_TYPE_Q4_0;
    const auto blocks = make_blocks(ncols * nrows / kQk, q8, seed);
    const auto block_data = pack_blocks(blocks, q8);

    // Reference: standard layout through the real CPU GET_ROWS kernel.
    const std::vector<float> reference = reference_get_rows(type, block_data, ncols, nrows, indices);

    // Fork layout: reorder mirror + SoA decode mirror.
    const auto soa = mirror_reorder_to_soa(block_data, q8, ncols, nrows);
    const int64_t out_rows = static_cast<int64_t>(indices.size());
    engine::test::require_eq(reference.size(), static_cast<size_t>(out_rows * ncols),
                             "reference GET_ROWS size");

    int64_t mismatches = 0;
    for (int64_t r = 0; r < out_rows; r++) {
        const int64_t row = indices[static_cast<size_t>(r)];
        for (int64_t c = 0; c < ncols; c++) {
            const float want = reference[static_cast<size_t>(r * ncols + c)];
            const float got = mirror_soa_get_row(soa, q8, ncols, nrows, c, row);
            if (!(got == want)) {
                if (++mismatches <= 5) {
                    std::fprintf(stderr,
                                 "  SoA mismatch type=%s ncols=%lld nrows=%lld row=%lld col=%lld"
                                 " got=%f want=%f\n",
                                 q8 ? "Q8_0" : "Q4_0", static_cast<long long>(ncols),
                                 static_cast<long long>(nrows), static_cast<long long>(row),
                                 static_cast<long long>(c), got, want);
                }
            }
        }
    }
    engine::test::require_eq(mismatches, int64_t{0},
                             std::string(q8 ? "Q8_0" : "Q4_0") +
                                 ": SoA reorder+decode mirror must reproduce the reference GET_ROWS");
}

// ---- Mirror of should_reorder_tensor (ggml-sycl.cpp). ----------------------
struct ReorderQuery {
    bool optimize_enabled = true;
    bool device_reorder_ok = true;
    ggml_op op = GGML_OP_MUL_MAT;
    bool is_get_rows_source = false;
    int64_t src1_rows = 1;
    int64_t src1_planes = 1;
    int64_t src1_batches = 1;
};

bool mirror_should_reorder(const ReorderQuery & q) {
    return q.optimize_enabled &&
           q.device_reorder_ok &&
           q.op == GGML_OP_MUL_MAT &&
           !q.is_get_rows_source &&
           q.src1_rows == 1 && q.src1_planes == 1 && q.src1_batches == 1;
}

}  // namespace

int main() try {
    // ---- Layout contract, happy paths.
    // Q4_0: 4 rows x 64 cols (2 blocks/row), out-of-order + duplicated indices.
    check_type(/*q8=*/false, 64, 4, {3, 1, 0, 3, 2}, 11);
    // Q8_0 same shape.
    check_type(/*q8=*/true, 64, 4, {3, 1, 0, 3, 2}, 12);
    // Single row, single index.
    check_type(false, 32, 1, {0}, 13);
    check_type(true, 32, 1, {0}, 14);
    // Wider than one block span and not a multiple of the 256-thread span
    // (exercises per-block indexing across many blocks).
    check_type(false, 96, 3, {2, 0}, 15);
    check_type(true, 96, 3, {2, 0}, 16);
    // Tall table: 16 rows.
    check_type(true, 32, 16, {15, 7, 0, 8}, 17);

    // ---- Predicate contract (should_reorder_tensor decision table).
    {
        // Happy: batch-1 mul_mat over a never-fetched weight reorders.
        ReorderQuery q;
        engine::test::require(mirror_should_reorder(q),
                              "plain batch-1 MUL_MAT weight must be reorderable");

        // A tensor that an executed graph read through GET_ROWS must NOT be
        // reordered in place - the core fork fix.
        ReorderQuery fetched = q;
        fetched.is_get_rows_source = true;
        engine::test::require(!mirror_should_reorder(fetched),
                              "GET_ROWS source tensors must be excluded from reorder");

        // Batched activation disables reorder.
        ReorderQuery batched = q;
        batched.src1_rows = 2;
        engine::test::require(!mirror_should_reorder(batched),
                              "batch>1 MUL_MAT must not reorder");

        // Non-mul_mat ops never reorder.
        ReorderQuery other_op = q;
        other_op.op = GGML_OP_ADD;
        engine::test::require(!mirror_should_reorder(other_op),
                              "non-MUL_MAT ops must not reorder");

        // Global opt-out and device capability gates.
        ReorderQuery disabled = q;
        disabled.optimize_enabled = false;
        engine::test::require(!mirror_should_reorder(disabled), "GGML_SYCL_DISABLE_OPT must disable reorder");
        ReorderQuery bad_device = q;
        bad_device.device_reorder_ok = false;
        engine::test::require(!mirror_should_reorder(bad_device), "device without reorder perf must skip");

        // Multi-plane/batch activation shapes also disable it.
        ReorderQuery planes = q;
        planes.src1_planes = 2;
        engine::test::require(!mirror_should_reorder(planes), "2D+ activations must not reorder");
    }

    std::printf("test_fork_sycl_reorder_getrows: all cases passed\n");
    return 0;
} catch (const std::exception & error) {
    std::fprintf(stderr, "test_fork_sycl_reorder_getrows FAILED: %s\n", error.what());
    return 1;
}
