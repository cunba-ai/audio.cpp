// Fork-regression test: resolve_cuda_family_backend_type() (audio.cpp fork,
// backend.cpp).
//
// CUDA and HIP are mutually exclusive builds of ggml's ggml-cuda sources,
// and the resolver maps the loaded registry to the engine's BackendType so
// HIP-specific model paths (e.g. the Strix Halo graph-cache gates keyed on
// BackendType::Hip) engage on HIP-only builds without an explicit
// --backend hip request. Decision table:
//
//   CUDA/MUSA registry present  -> BackendType::Cuda
//   else ROCm registry present  -> BackendType::Hip
//   else (CPU-only build)       -> BackendType::Cuda  (default)
//
// The registry set depends on how the binary was built, so the test derives
// the expected branch from the actually loaded registries (mirroring
// find_reg_by_backend_type's name table) and asserts the resolver agrees.
// On a CPU-only build this exercises the default branch; on CUDA or HIP
// builds (CI GPU lanes) it exercises the corresponding branch. A regression
// that returns Sycl/Vulkan/Metal/Cpu or inverts the CUDA/HIP precedence
// fails in every configuration.

#include "test_assert.h"

#include "engine/framework/core/backend.h"

#include <ggml-backend.h>

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// Mirror of the k_backend_reg_names table in backend.cpp (anonymous
// namespace there, so mirrored here).
bool registry_is_cuda_family(const char * name) {
    return name != nullptr && (std::string(name) == "CUDA" || std::string(name) == "MUSA");
}

bool registry_is_hip(const char * name) {
    return name != nullptr && std::string(name) == "ROCm";
}

}  // namespace

int main() try {
    engine::core::ensure_backends_loaded();

    // Scan the loaded registries the same way find_reg_by_backend_type does.
    bool has_cuda = false;
    bool has_hip = false;
    std::vector<std::string> registry_names;
    for (size_t i = 0; i < ggml_backend_reg_count(); ++i) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        if (reg == nullptr) {
            continue;
        }
        const char * name = ggml_backend_reg_name(reg);
        registry_names.emplace_back(name != nullptr ? name : "<null>");
        has_cuda = has_cuda || registry_is_cuda_family(name);
        has_hip = has_hip || registry_is_hip(name);
    }

    engine::core::BackendType resolved = engine::core::resolve_cuda_family_backend_type();

    // Derive the expected branch from the registry state.
    engine::core::BackendType expected = engine::core::BackendType::Cuda;  // default
    if (has_cuda) {
        expected = engine::core::BackendType::Cuda;
    } else if (has_hip) {
        expected = engine::core::BackendType::Hip;
    }

    engine::test::require(resolved == expected,
                          "resolve_cuda_family_backend_type must follow the CUDA > HIP > default table; "
                          "resolved=" +
                              std::to_string(static_cast<int>(resolved)) + " expected=" +
                              std::to_string(static_cast<int>(expected)));

    // The resolver must only ever answer inside the CUDA family.
    engine::test::require(resolved == engine::core::BackendType::Cuda ||
                              resolved == engine::core::BackendType::Hip,
                          "resolve_cuda_family_backend_type must only return Cuda or Hip");

    // On a HIP-only build the resolver must NOT claim Cuda (that would
    // disable the HIP-specific engine paths this fix was written for).
    if (!has_cuda && has_hip) {
        engine::test::require(resolved == engine::core::BackendType::Hip,
                              "HIP-only build must resolve to BackendType::Hip");
    }
    // On a CPU-only build neither registry exists; the default must be Cuda
    // (safe: init_backend(Cuda) reports the missing backend explicitly).
    if (!has_cuda && !has_hip) {
        engine::test::require(resolved == engine::core::BackendType::Cuda,
                              "no CUDA/HIP registry present must default to BackendType::Cuda");
    }

    std::string registries;
    for (const auto & name : registry_names) {
        if (!registries.empty()) {
            registries += ",";
        }
        registries += name;
    }
    std::printf("test_fork_backend_family_resolver: resolved=%d (registries: %s)\n",
                static_cast<int>(resolved), registries.c_str());
    return 0;
} catch (const std::exception & error) {
    std::fprintf(stderr, "test_fork_backend_family_resolver FAILED: %s\n", error.what());
    return 1;
}
