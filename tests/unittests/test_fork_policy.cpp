// Fork-policy guard: the upstream C ABI must stay out of this fork.
//
// Upstream PR #530 added a second C ABI (include/audiocpp.h, src/capi/,
// tests/capi/, docs/c_api.md) that overlaps this fork's capi/ surface — same
// CMake target name ("audiocpp") and colliding function names
// (audiocpp_stream_start/push/finish). Fork policy keeps ONLY the fork's C
// ABI: capi/include/audiocpp.h built by AUDIOCPP_BUILD_CAPI.
//
// After every upstream merge this test re-asserts the policy: if a merge
// resolves the modify/delete conflicts the wrong way (re-adding upstream's
// files) or the fork's own CAPI files go missing, it fails loudly with the
// path that broke the invariant.

#include "test_assert.h"

#include <cstdio>
#include <filesystem>
#include <string>

namespace {

const std::filesystem::path kRoot = std::filesystem::path(AUDIOCPP_FORK_SOURCE_DIR);

// Upstream C-ABI artifacts that must NOT exist in this fork.
const char * kForbiddenPaths[] = {
    "include/audiocpp.h",
    "src/capi",
    "tests/capi",
    "docs/c_api.md",
};

// The fork's C ABI that must exist.
const char * kRequiredPaths[] = {
    "capi/include/audiocpp.h",
    "capi/src/audiocpp_capi.cpp",
};

}  // namespace

int main() {
    for (const char * rel : kForbiddenPaths) {
        const auto path = kRoot / rel;
        engine::test::require(
            !std::filesystem::exists(path),
            "fork policy violated: upstream C-ABI artifact is present: " + path.string() +
                " (fork keeps only capi/; resolve upstream modify/delete conflicts by"
                " keeping the deletion — see the FORK POLICY note in CMakeLists.txt)");
    }
    for (const char * rel : kRequiredPaths) {
        const auto path = kRoot / rel;
        engine::test::require(
            std::filesystem::exists(path),
            "fork C-ABI file missing: " + path.string());
    }
    std::printf("fork_policy_test: upstream C ABI absent, fork CAPI present\n");
    return 0;
}
