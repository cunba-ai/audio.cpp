// Fork-regression test: Vulkan per-row dispatch clamping (audio.cpp fork,
// restored by dc6d1b59 after PR #39 dropped it).
//
// rms_norm dispatches one workgroup per row along x. Devices with a low
// maxComputeWorkGroupCount[0] (Intel ICDs report 65536 vs 2^31-1 on NVIDIA)
// hit ggml_vk_dispatch_pipeline's GGML_ASSERT when a tensor has more rows
// than that (vibevoice conv features with ~264k rows). The fork's
// ggml_vk_dispatch_rms_norm_row_chunked() chunks the dispatch along x and
// passes each chunk's base row through the push-constant row-offset field
// (decoded in rms_norm.comp with floatBitsToUint, which derives the dst row
// stride from p.ne01 instead of gl_NumWorkGroups.x).
//
// Pure C++ logic - no Vulkan needed. This test mirrors the chunking loop and
// verifies:
//   1. rows <= max: single dispatch, row base 0, full extent.
//   2. rows > max: every chunk fits inside max, chunks tile [0, rows)
//      exactly, and the encoded base row decodes back losslessly.
//   3. The vibevoice shape (264k rows on a 65536-limit ICD) partitions into
//      ceil(264k/65536) = 5 chunks.
//   4. Edge boundaries: rows == max (single), rows == max + 1 (two chunks),
//      rows == 2*max exactly (two full chunks).
//
// The source-side wiring (helper exists, both rms_norm call sites use it,
// rms_norm.comp takes the row base) is pinned by test_fork_source_anchors.cpp.

#include "test_assert.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// Push-constant encoding used by the helper and rms_norm.comp: the base row
// travels as the bit pattern of a float and is decoded with floatBitsToUint.
float encode_row_base(uint32_t base) {
    float bits = 0.0f;
    std::memcpy(&bits, &base, sizeof(bits));
    return bits;
}

uint32_t decode_row_base(float field) {
    uint32_t base = 0;
    std::memcpy(&base, &field, sizeof(base));
    return base;
}

struct ChunkDispatch {
    std::vector<std::pair<uint32_t, uint32_t>> chunks;  // (base_row, rows)
    bool single = false;
};

// Mirror of ggml_vk_dispatch_rms_norm_row_chunked's partitioning.
ChunkDispatch mirror_row_chunked(uint32_t rows, uint32_t max_x) {
    ChunkDispatch out;
    if (rows <= max_x) {
        out.single = true;
        out.chunks.push_back({0, rows});
        return out;
    }
    for (uint32_t base = 0; base < rows; base += max_x) {
        const uint32_t chunk_rows = std::min(rows - base, max_x);
        out.chunks.push_back({base, chunk_rows});
    }
    return out;
}

void check_partition(const char * label, uint32_t rows, uint32_t max_x, size_t expect_chunks) {
    const ChunkDispatch dispatch = mirror_row_chunked(rows, max_x);

    engine::test::require(dispatch.single == (rows <= max_x),
                          std::string(label) + ": single-dispatch classification");
    engine::test::require_eq(dispatch.chunks.size(), expect_chunks,
                             std::string(label) + ": chunk count");

    // Chunks tile [0, rows) exactly, in order, each within the device limit.
    uint32_t covered = 0;
    for (const auto & [base, chunk_rows] : dispatch.chunks) {
        engine::test::require(chunk_rows >= 1 && chunk_rows <= max_x,
                              std::string(label) + ": chunk rows within device limit");
        engine::test::require_eq(base, covered, std::string(label) + ": chunk base continuity");
        // The base row survives the float-bits push-constant round trip.
        engine::test::require_eq(decode_row_base(encode_row_base(base)), base,
                                 std::string(label) + ": row-base floatBitsToUint round trip");
        covered += chunk_rows;
    }
    engine::test::require_eq(covered, rows, std::string(label) + ": rows fully covered");

    // Total dispatched work equals the full extent (no lost/duplicated rows).
    uint64_t total = 0;
    for (const auto & [base, chunk_rows] : dispatch.chunks) {
        total += chunk_rows;
    }
    engine::test::require_eq(total, static_cast<uint64_t>(rows),
                             std::string(label) + ": dispatched row total");
}

}  // namespace

int main() try {
    // ---- Happy path: within-limit rows take a single dispatch.
    check_partition("rows below limit", 1000, 65536, 1);
    check_partition("exactly the limit", 65536, 65536, 1);

    // ---- Edge boundaries just above the limit.
    check_partition("limit plus one", 65537, 65536, 2);
    check_partition("twice the limit", 131072, 65536, 2);
    check_partition("twice plus one", 131073, 65536, 3);

    // ---- The vibevoice failure shape: ~264k conv-feature rows on an Intel
    // ICD with maxComputeWorkGroupCount[0] = 65536.
    check_partition("vibevoice 264k rows", 264192, 65536, 5);

    // ---- A hypothetical stricter ICD limit (2^16 is common; some report 256
    // for individual dimensions - not x, but the math must hold regardless).
    check_partition("tiny limit", 1000, 256, 4);
    check_partition("single row", 1, 65536, 1);

    // ---- The float-bits encoding never aliases small-norm floats: any base
    // row up to 2^32-1 must round trip (rms_norm.comp decodes with
    // floatBitsToUint, so arbitrary bit patterns must be preserved).
    {
        const uint32_t probes[] = {0u, 1u, 255u, 65535u, 65536u, 0x7fffffffu, 0xffffffffu};
        for (const uint32_t probe : probes) {
            engine::test::require_eq(decode_row_base(encode_row_base(probe)), probe,
                                     "floatBitsToUint round trip for base row " + std::to_string(probe));
        }
    }

    // ---- Regression guard on the underlying arithmetic the old code got
    // wrong: the per-chunk dispatch extent is uint32_t, so the final chunk
    // (rows % max) must be computed from the remaining extent, not from a
    // signed difference.
    {
        const ChunkDispatch dispatch = mirror_row_chunked(200000, 65536);
        engine::test::require_eq(dispatch.chunks.size(), static_cast<size_t>(4),
                                 "200000/65536 chunk count");
        // 200000 = 3*65536 + 3392.
        engine::test::require_eq(dispatch.chunks.back().second, 3392u, "final partial chunk size");
        engine::test::require_eq(dispatch.chunks.back().first, 196608u, "final chunk base");
    }

    std::printf("test_fork_vulkan_dispatch_clamp: all cases passed\n");
    return 0;
} catch (const std::exception & error) {
    std::fprintf(stderr, "test_fork_vulkan_dispatch_clamp FAILED: %s\n", error.what());
    return 1;
}
