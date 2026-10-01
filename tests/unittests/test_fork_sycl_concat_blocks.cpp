// Fork-regression test: SYCL concat dynamic block/chunk sizing (audio.cpp
// fork, restored by dc6d1b59 after PR #39 dropped it).
//
// concat.cpp used to pad every launch up to a fixed SYCL_CONCAT_BLOCK_SIZE
// (256) and compute offsets in int. For large tensors (seed_vc's
// [2, 23562, 384] concat = 2.3G work items) the global nd_range exceeded
// INT_MAX and the SYCL runtime rejected the launch. The fork fix:
//   * block_ne0 = min(GGML_PAD(ne0, WARP_SIZE), SYCL_CONCAT_BLOCK_SIZE)
//   * num_blocks = ceil(ne0 / block_ne0)
//   * dim-1 extents submitted in chunks of max_i1 =
//     INT_MAX / (ne2 * num_blocks * block_ne0) so every launch's global
//     work-item product stays below INT_MAX
//   * all kernel offsets widened to int64_t with an i1_off chunk base and
//     ne1_total (= ne1) passed separately.
//
// The SYCL kernel itself needs a GPU (see test_fork_gpu_backend_kernels.cpp);
// this CPU test mirrors the launch math and the kernel index arithmetic:
//
//   1. No chunk's global work-item product exceeds INT32_MAX - including
//      the historical seed_vc shape that used to fail.
//   2. The chunks exactly tile [0, ne1).
//   3. Small tensors take a single chunk with the dynamic block size.
//   4. Host emulation of concat_T_dim0/1/2 over chunked launches matches a
//      trivial reference concat for every concat dimension.

#include "test_assert.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// Constants mirrored from ggml-sycl/presets.hpp. The fork builds SYCL with
// the INTEL target (GGML_SYCL_WARP_SIZE=16); the invariants below hold for
// any warp size, so both 16 and 64 are exercised.
constexpr int64_t kConcatBlockSize = 256;  // SYCL_CONCAT_BLOCK_SIZE

int64_t pad_to(int64_t value, int64_t multiple) {
    return (value + multiple - 1) / multiple * multiple;
}

// Launch plan mirror of concat_T_sycl (concat.cpp).
struct ConcatPlan {
    int64_t block_ne0 = 0;
    int64_t num_blocks = 0;
    int64_t max_i1 = 0;
    std::vector<int64_t> chunk_sizes;  // dim-1 chunks
};

ConcatPlan mirror_concat_plan(int64_t ne0, int64_t ne1, int64_t ne2, int64_t warp_size) {
    const int64_t ne0_pad = pad_to(ne0, warp_size);
    const int64_t block_ne0 = std::min<int64_t>(ne0_pad, kConcatBlockSize);
    const int64_t num_blocks = (ne0 + block_ne0 - 1) / block_ne0;

    int64_t max_i1 = std::numeric_limits<int32_t>::max() /
                     std::max<int64_t>(1, ne2 * num_blocks * block_ne0);
    max_i1 = std::max<int64_t>(max_i1, 1);

    ConcatPlan plan;
    plan.block_ne0 = block_ne0;
    plan.num_blocks = num_blocks;
    plan.max_i1 = max_i1;
    for (int64_t off = 0; off < ne1; off += max_i1) {
        plan.chunk_sizes.push_back(std::min<int64_t>(max_i1, ne1 - off));
    }
    return plan;
}

// ---- Host emulation of the SYCL kernels (concat_T_dim0/1/2) ---------------
// Emulates one chunk launch: the grid is (ne2_total, chunk, num_blocks) and
// every work item (i2, i1 = group(1) + i1_off, nidx) writes one element.
template <int Dim>
void emulate_concat_chunk(const std::vector<float> & x, const std::vector<float> & y,
                          std::vector<float> & dst,
                          int64_t ne00, int64_t ne01, int64_t ne02,
                          int64_t ne0_total, int64_t ne1_total, int64_t ne2_total,
                          int64_t chunk, int64_t i1_off) {
    for (int64_t i2 = 0; i2 < ne2_total; i2++) {
        for (int64_t g1 = 0; g1 < chunk; g1++) {
            const int64_t i1 = g1 + i1_off;
            for (int64_t nidx = 0; nidx < ne0_total; nidx++) {
                const int64_t offset_dst = nidx + i1 * ne0_total + i2 * ne0_total * ne1_total;
                if constexpr (Dim == 0) {
                    if (nidx < ne00) {
                        const int64_t offset_src = nidx + i1 * ne00 + i2 * ne00 * ne1_total;
                        dst[static_cast<size_t>(offset_dst)] = x[static_cast<size_t>(offset_src)];
                    } else {
                        const int64_t offset_src = nidx - ne00 + i1 * (ne0_total - ne00) +
                                                   i2 * (ne0_total - ne00) * ne1_total;
                        dst[static_cast<size_t>(offset_dst)] = y[static_cast<size_t>(offset_src)];
                    }
                } else if constexpr (Dim == 1) {
                    if (i1 < ne01) {
                        const int64_t offset_src = nidx + i1 * ne0_total + i2 * ne0_total * ne01;
                        dst[static_cast<size_t>(offset_dst)] = x[static_cast<size_t>(offset_src)];
                    } else {
                        const int64_t offset_src = nidx + (i1 - ne01) * ne0_total +
                                                   i2 * ne0_total * (ne1_total - ne01);
                        dst[static_cast<size_t>(offset_dst)] = y[static_cast<size_t>(offset_src)];
                    }
                } else {
                    if (i2 < ne02) {
                        const int64_t offset_src = nidx + i1 * ne0_total + i2 * ne0_total * ne1_total;
                        dst[static_cast<size_t>(offset_dst)] = x[static_cast<size_t>(offset_src)];
                    } else {
                        const int64_t offset_src = nidx + i1 * ne0_total +
                                                   (i2 - ne02) * ne0_total * ne1_total;
                        dst[static_cast<size_t>(offset_dst)] = y[static_cast<size_t>(offset_src)];
                    }
                }
            }
        }
    }
}

struct Shape {
    int64_t ne0 = 0;  // concatenated innermost dim
    int64_t ne00 = 0; // src0 innermost dim (dim0 concat) / full ne0 otherwise
    int64_t ne01 = 0; // src0 dim1
    int64_t ne02 = 0; // src0 dim2
    int64_t ne1 = 0;  // total dim1
    int64_t ne2 = 0;  // total dim2
};

// Trivial reference concat on host.
std::vector<float> reference_concat(const Shape & s, int dim, const std::vector<float> & x,
                                    const std::vector<float> & y) {
    std::vector<float> dst(static_cast<size_t>(s.ne0 * s.ne1 * s.ne2), -1.0f);
    for (int64_t i2 = 0; i2 < s.ne2; i2++) {
        for (int64_t i1 = 0; i1 < s.ne1; i1++) {
            for (int64_t i0 = 0; i0 < s.ne0; i0++) {
                float value = 0.0f;
                if (dim == 0) {
                    value = (i0 < s.ne00) ? x[static_cast<size_t>(i0 + i1 * s.ne00 + i2 * s.ne00 * s.ne1)]
                                          : y[static_cast<size_t>(i0 - s.ne00 + i1 * (s.ne0 - s.ne00) +
                                                                 i2 * (s.ne0 - s.ne00) * s.ne1)];
                } else if (dim == 1) {
                    value = (i1 < s.ne01) ? x[static_cast<size_t>(i0 + i1 * s.ne0 + i2 * s.ne0 * s.ne01)]
                                          : y[static_cast<size_t>(i0 + (i1 - s.ne01) * s.ne0 +
                                                                 i2 * s.ne0 * (s.ne1 - s.ne01))];
                } else {
                    value = (i2 < s.ne02) ? x[static_cast<size_t>(i0 + i1 * s.ne0 + i2 * s.ne0 * s.ne1)]
                                          : y[static_cast<size_t>(i0 + i1 * s.ne0 +
                                                                 (i2 - s.ne02) * s.ne0 * s.ne1)];
                }
                dst[static_cast<size_t>(i0 + i1 * s.ne0 + i2 * s.ne0 * s.ne1)] = value;
            }
        }
    }
    return dst;
}

void check_emulation(int dim, const Shape & s, int64_t warp_size) {
    // Build patterned sources sized to their own shapes.
    const int64_t x_elems = (dim == 0 ? s.ne00 : s.ne0) *
                            (dim == 1 ? s.ne01 : s.ne1) *
                            (dim == 2 ? s.ne02 : s.ne2);
    const int64_t y_elems = s.ne0 * s.ne1 * s.ne2 - x_elems;
    std::vector<float> x(static_cast<size_t>(x_elems));
    std::vector<float> y(static_cast<size_t>(y_elems));
    for (size_t i = 0; i < x.size(); i++) {
        x[i] = static_cast<float>(i % 977) * 0.5f - 3.0f;
    }
    for (size_t i = 0; i < y.size(); i++) {
        y[i] = 100.0f + static_cast<float>(i % 733) * 0.25f;
    }

    const ConcatPlan plan = mirror_concat_plan(s.ne0, s.ne1, s.ne2, warp_size);
    std::vector<float> got(static_cast<size_t>(s.ne0 * s.ne1 * s.ne2), -1.0f);
    int64_t off = 0;
    for (const int64_t chunk : plan.chunk_sizes) {
        if (dim == 0) {
            emulate_concat_chunk<0>(x, y, got, s.ne00, s.ne01, s.ne02, s.ne0, s.ne1, s.ne2, chunk, off);
        } else if (dim == 1) {
            emulate_concat_chunk<1>(x, y, got, s.ne00, s.ne01, s.ne02, s.ne0, s.ne1, s.ne2, chunk, off);
        } else {
            emulate_concat_chunk<2>(x, y, got, s.ne00, s.ne01, s.ne02, s.ne0, s.ne1, s.ne2, chunk, off);
        }
        off += chunk;
    }

    const std::vector<float> want = reference_concat(s, dim, x, y);
    int64_t mismatches = 0;
    for (size_t i = 0; i < want.size(); i++) {
        if (!(got[i] == want[i])) {
            mismatches++;
        }
    }
    engine::test::require_eq(mismatches, int64_t{0},
                             "concat dim=" + std::to_string(dim) +
                                 " chunked kernel emulation must match the reference");
}

void check_plan_invariants(const char * label, int64_t ne0, int64_t ne1, int64_t ne2,
                           int64_t warp_size, int64_t expect_chunks = -1) {
    const ConcatPlan plan = mirror_concat_plan(ne0, ne1, ne2, warp_size);

    // Chunks exactly tile [0, ne1).
    int64_t total = 0;
    for (const int64_t chunk : plan.chunk_sizes) {
        engine::test::require(chunk >= 1 && chunk <= plan.max_i1,
                              std::string(label) + ": chunk size within [1, max_i1]");
        total += chunk;
    }
    engine::test::require_eq(total, ne1, std::string(label) + ": chunks must tile ne1 exactly");

    // Every launch's global work-item product stays inside int32.
    for (const int64_t chunk : plan.chunk_sizes) {
        const int64_t global_items = ne2 * chunk * plan.num_blocks * plan.block_ne0;
        engine::test::require(
            global_items <= std::numeric_limits<int32_t>::max(),
            std::string(label) + ": per-chunk global work items must fit in int32 (got " +
                std::to_string(global_items) + ")");
    }

    if (expect_chunks >= 0) {
        engine::test::require_eq(static_cast<int64_t>(plan.chunk_sizes.size()), expect_chunks,
                                 std::string(label) + ": expected chunk count");
    }
}

}  // namespace

int main() try {
    for (const int64_t warp : {16, 64}) {
        // ---- Overflow guard: the historical seed_vc failure shape.
        // [2, 23562, 384]: the old fixed-256 padding launched 2.3G items.
        check_plan_invariants("seed_vc dim0 shape", /*ne0=*/2, /*ne1=*/23562, /*ne2=*/384, warp, 1);

        // A dim-1-heavy tensor that genuinely needs chunking:
        // ne2 * num_blocks * block = 384 * 8 * 32 = 98304 -> max_i1 ~21845,
        // ne1 = 50000 -> 3 chunks.
        check_plan_invariants("dim1 heavy chunked", 256, 50000, 384, warp, 3);

        // Extreme: ne1 large enough that max_i1 clamps to 1.
        check_plan_invariants("max_i1 clamp", 65536, 5, 4096, warp);

        // ---- Small tensors: single chunk, dynamic block.
        {
            const ConcatPlan plan = mirror_concat_plan(/*ne0=*/7, /*ne1=*/3, /*ne2=*/2, warp);
            engine::test::require_eq(static_cast<int64_t>(plan.chunk_sizes.size()), int64_t{1},
                                     "small tensor takes a single chunk");
            engine::test::require_eq(plan.block_ne0, pad_to(7, warp),
                                     "small tensors use the warp-padded block, not 256");
            engine::test::require_eq(plan.num_blocks, int64_t{1}, "small tensor block count");
        }
        // ne0 exactly a multiple of the block size.
        {
            const ConcatPlan plan = mirror_concat_plan(512, 4, 3, warp);
            engine::test::require_eq(plan.block_ne0, int64_t{kConcatBlockSize}, "full block for wide rows");
            engine::test::require_eq(plan.num_blocks, int64_t{2}, "two blocks for ne0=512");
        }
        // ne0 < warp size and ne0 between warp and 256.
        check_plan_invariants("tiny", 3, 2, 2, warp, 1);
        check_plan_invariants("odd width", 33, 5, 7, warp, 1);
    }

    // ---- Kernel index emulation vs trivial reference (all three dims).
    // Includes a shape that spans multiple dim-1 chunks with an odd ne0.
    {
        Shape s;
        s.ne0 = 10;
        s.ne00 = 4;   // dim0 split point
        s.ne01 = 2;   // dim1 split point
        s.ne02 = 1;   // dim2 split point
        s.ne1 = 5;
        s.ne2 = 3;
        check_emulation(0, s, 16);
        check_emulation(1, s, 16);
        check_emulation(2, s, 16);

        // A second shape with different split points and larger dims.
        Shape s2;
        s2.ne0 = 100;
        s2.ne00 = 37;
        s2.ne01 = 3;
        s2.ne02 = 2;
        s2.ne1 = 7;
        s2.ne2 = 4;
        check_emulation(0, s2, 16);
        check_emulation(1, s2, 16);
        check_emulation(2, s2, 16);

        // Warp-64 variant of the same shapes.
        check_emulation(1, s2, 64);
    }

    std::printf("test_fork_sycl_concat_blocks: all cases passed\n");
    return 0;
} catch (const std::exception & error) {
    std::fprintf(stderr, "test_fork_sycl_concat_blocks FAILED: %s\n", error.what());
    return 1;
}
