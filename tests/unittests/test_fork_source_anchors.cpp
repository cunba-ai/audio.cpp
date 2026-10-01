// Fork-regression test: source anchors for the fork's GPU-only ggml patches.
//
// The fork carries several fixes inside external/ggml SYCL/Vulkan sources
// that a CPU-only build never compiles. The in-repo policy after an upstream
// merge ("zero-GPU policy" in commit dc6d1b59) is to verify those patches by
// symbol anchors; this test automates it so a merge that silently drops or
// rewrites a patch fails `ctest -L fork_regression` even on CPU-only
// machines.
//
// Each anchor below names the file and a distinctive fragment of the fork's
// patch. The fragments are chosen to be load-bearing (the exact identifier
// the fix introduced), not incidental formatting.

#include "test_assert.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

const std::filesystem::path kRoot = std::filesystem::path(AUDIOCPP_FORK_SOURCE_DIR);

std::string read_file(const char * rel) {
    const auto path = kRoot / rel;
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("fork anchor: cannot open " + path.string());
    }
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return buffer.str();
}

void require_contains(const char * file, const std::string & needle, const std::string & why) {
    const std::string content = read_file(file);
    engine::test::require(content.find(needle) != std::string::npos,
                          std::string("fork anchor missing in ") + file + " (" + why +
                               "): expected fragment:\n" + needle);
}

void require_absent(const char * file, const std::string & needle, const std::string & why) {
    const std::string content = read_file(file);
    engine::test::require(content.find(needle) == std::string::npos,
                          std::string("forbidden pattern present in ") + file + " (" + why +
                               "): " + needle);
}

}  // namespace

int main() try {
    // ---- 1. SYCL reorder/GET_ROWS (dc6d1b59 fix #1).
    require_contains("external/ggml/src/ggml-sycl/getrows.cpp",
                     "k_get_rows_quant_soa",
                     "SoA read kernel for reordered GET_ROWS sources");
    require_contains("external/ggml/src/ggml-sycl/getrows.cpp",
                     "ctx.get_rows_sources.insert(dst->src[0]->data)",
                     "GET_ROWS source registration");
    require_contains("external/ggml/src/ggml-sycl/ggml-sycl.cpp",
                     "ctx.get_rows_sources.find(dst->src[0]->data) == ctx.get_rows_sources.end()",
                     "should_reorder_tensor excludes GET_ROWS sources");

    // ---- 2. SYCL fattn per-head mask (dc6d1b59 fix #2).
    require_contains("external/ggml/src/ggml-sycl/fattn-tile.hpp",
                     "nb32 * (head0 % ne32)",
                     "tile kernel per-head mask offset");
    require_contains("external/ggml/src/ggml-sycl/fattn-vec.hpp",
                     "nb32 * (head % ne32)",
                     "vec kernel per-head mask offset");
    require_contains("external/ggml/src/ggml-sycl/fattn.cpp",
                     "per_head_mask",
                     "kernel-selection per-head mask gate");
    require_contains("external/ggml/src/ggml-sycl/fattn.cpp",
                     "Q->ne[2] % mask->ne[2] != 0",
                     "head-divisibility rejection");

    // ---- 3. SYCL concat dynamic blocks (dc6d1b59 fix #3).
    require_contains("external/ggml/src/ggml-sycl/concat.cpp",
                     "max_i1",
                     "dim-1 chunking limit");
    require_contains("external/ggml/src/ggml-sycl/concat.cpp",
                     "i1_off",
                     "chunk base offset plumbing");
    require_contains("external/ggml/src/ggml-sycl/concat.cpp",
                     "const int64_t ne0_pad   = GGML_PAD(ne0, WARP_SIZE);",
                     "dynamic block sizing");

    // ---- 4. Vulkan dispatch clamping (dc6d1b59 fix #4).
    require_contains("external/ggml/src/ggml-vulkan/ggml-vulkan.cpp",
                     "ggml_vk_dispatch_rms_norm_row_chunked",
                     "chunked rms_norm dispatch helper");
    // Both rms_norm call sites route through the chunked helper.
    {
        const std::string content =
            read_file("external/ggml/src/ggml-vulkan/ggml-vulkan.cpp");
        const size_t calls = content.find("ggml_vk_dispatch_rms_norm_row_chunked(ctx") != std::string::npos;
        engine::test::require(calls, "vulkan: rms_norm dispatch must call the chunked helper");
    }
    require_contains("external/ggml/src/ggml-vulkan/vulkan-shaders/rms_norm.comp",
                     "param2",
                     "rms_norm.comp row-base parameter");

    // ---- 5. GRU_SCAN registration chain (dc6d1b59 fix #5).
    require_contains("external/ggml/include/ggml.h",
                     "GGML_OP_GRU_SCAN",
                     "op enum entry");
    require_contains("external/ggml/src/ggml.c",
                     "gru_scan(x_ih, h0, w_hh, b_hh, keep)",
                     "op symbol table entry");
    require_contains("external/ggml/src/ggml-cpu/ggml-cpu.c",
                     "ggml_compute_forward_gru_scan",
                     "CPU reference kernel");
    require_contains("external/ggml/src/ggml-sycl/ggml-sycl.cpp",
                     "ggml_sycl_gru_scan_supported",
                     "SYCL supports_op gate");
    // CUDA kernel files exist.
    for (const char * rel : {"external/ggml/src/ggml-cuda/gru-scan.cu",
                             "external/ggml/src/ggml-cuda/gru-scan.cuh",
                             "external/ggml/src/ggml-sycl/gru-scan.cpp",
                             "external/ggml/src/ggml-sycl/gru-scan.hpp"}) {
        engine::test::require(std::filesystem::exists(kRoot / rel),
                              std::string("GRU_SCAN kernel file missing: ") + rel);
    }

    // ---- 6. round_bf16: the CPU op stays; the SYCL launcher is absent in
    // this tree (the d6c77c75 host-pointer fix targeted a launcher that the
    // PR #39 baseline no longer ships). If a merge reintroduces a SYCL
    // round_bf16 launcher, it MUST extract the data pointer host-side (no
    // src0->data dereference inside the device lambda).
    require_contains("external/ggml/src/ggml.c",
                     "ggml_round_bf16",
                     "round_bf16 builder");
    require_contains("external/ggml/src/ggml-cpu/unary-ops.cpp",
                     "ggml_compute_forward_round_bf16",
                     "CPU round_bf16 kernel");
    {
        const std::string elementwise = read_file("external/ggml/src/ggml-sycl/element_wise.cpp");
        if (elementwise.find("round_bf16") != std::string::npos) {
            // A SYCL launcher came back with an upstream merge: it must not
            // capture the ggml_tensor* in the lambda. The broken pre-fix
            // pattern read src0->data inside the parallel_for body.
            const size_t lambda_capture = elementwise.find("(const float *) src0->data");
            const size_t lambda_capture16 = elementwise.find("(const sycl::half *) src0->data");
            engine::test::require(lambda_capture == std::string::npos &&
                                      lambda_capture16 == std::string::npos,
                                  "round_bf16 SYCL launcher must extract src pointers host-side, "
                                  "not dereference src0->data inside the device lambda (d6c77c75)");
        }
    }

    // ---- 9. MKL flash-attention gate requires F16 K/V (fe01cab4 lineage,
    // re-ported onto the current baseline): F32 K/V must never route to the
    // MKL GEMM path (the dramabox ~200x regression).
    require_contains("external/ggml/src/ggml-sycl/fattn.cpp",
                     "BEST_FATTN_KERNEL_MKL",
                     "MKL kernel enum entry");
    require_contains("external/ggml/src/ggml-sycl/fattn.cpp",
                     "K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16",
                     "MKL gate requires F16 K/V (F32 falls back to TILE)");
    require_contains("external/ggml/src/ggml-sycl/fattn.cpp",
                     "g_ggml_sycl_enable_mkl_fa",
                     "MKL gate env kill switch");
    engine::test::require(std::filesystem::exists(kRoot / "external/ggml/src/ggml-sycl/fattn-mkl.cpp"),
                          "fattn-mkl.cpp kernel file missing");
    require_contains("external/ggml/src/ggml-sycl/fattn-mkl.cpp",
                     "ggml_sycl_flash_attn_ext_mkl",
                     "MKL kernel entry point");
    require_contains("external/ggml/src/ggml-sycl/fattn.hpp",
                     "ggml_sycl_flash_attn_ext_mkl",
                     "MKL kernel declaration");

    // ---- 7. Backend family resolver: CUDA is checked before HIP (the two
    // are mutually exclusive builds, and the precedence keeps explicit-CUDA
    // builds on BackendType::Cuda).
    {
        const std::string backend = read_file("src/framework/core/backend.cpp");
        const size_t cuda_check = backend.find("find_reg_by_backend_type(BackendType::Cuda)");
        const size_t hip_check = backend.find("find_reg_by_backend_type(BackendType::Hip)");
        engine::test::require(cuda_check != std::string::npos && hip_check != std::string::npos,
                              "backend.cpp must keep both CUDA and HIP registry checks");
        engine::test::require(cuda_check < hip_check,
                              "backend.cpp must check the CUDA registry before HIP");
        require_contains("src/framework/core/backend.cpp",
                         "return BackendType::Cuda;",
                         "resolver default branch");
    }

    // ---- 10. irodori codec crop list: Sycl present, Cuda absent.
    require_contains("src/models/irodori_tts/codec.cpp",
                     "ctx.backend_type == core::BackendType::Sycl",
                     "SYCL must be in the conv-transpose crop list");
    {
        const std::string codec = read_file("src/models/irodori_tts/codec.cpp");
        // Extract the crop_backend_output definition and check membership.
        const std::string marker = "const bool crop_backend_output =";
        const size_t start = codec.find(marker);
        engine::test::require(start != std::string::npos, "codec.cpp crop list present");
        const size_t end = codec.find(';', start);
        engine::test::require(end != std::string::npos, "codec.cpp crop list terminated");
        const std::string condition = codec.substr(start, end - start);
        engine::test::require(condition.find("BackendType::Sycl") != std::string::npos,
                              "crop list must contain Sycl");
        engine::test::require(condition.find("BackendType::Cpu") != std::string::npos,
                              "crop list must contain Cpu");
        engine::test::require(condition.find("BackendType::Vulkan") != std::string::npos,
                              "crop list must contain Vulkan");
        engine::test::require(condition.find("BackendType::Metal") != std::string::npos,
                              "crop list must contain Metal");
        engine::test::require(condition.find("BackendType::Cuda") == std::string::npos,
                              "crop list must NOT contain Cuda (native padding path)");
        require_contains("src/models/irodori_tts/codec.cpp",
                         "SliceModule({2, padding,",
                         "crop slice after padding=0 conv transpose");
    }

    std::printf("test_fork_source_anchors: all fork patch anchors present\n");
    return 0;
} catch (const std::exception & error) {
    std::fprintf(stderr, "test_fork_source_anchors FAILED: %s\n", error.what());
    return 1;
}
