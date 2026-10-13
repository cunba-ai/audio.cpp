// Fork-regression test: unvalidated_backend_override() (audio.cpp fork,
// backend.cpp) — the AUDIOCPP_ENABLE_UNVALIDATED_BACKEND escape hatch for
// family-level "unvalidated backend" gates (RE-USE on HIP, Whistle outside
// CPU/Metal).
//
// The helper reads the environment on every call (no static caching), so this
// test toggles the variable within one process:
//
//   unset / empty / "0" / "true" / "01"  -> false (conservative default: the
//                                          family gate keeps throwing)
//   "1"                                 -> true  (gate skipped, warning logged)
//
// A regression that caches the first answer, accepts any non-empty value, or
// drops the strict == "1" check fails at least one case below. The warning
// itself is routed through ggml's log sink and is asserted only structurally
// (the call returns true without throwing); capturing the callback output is
// not stable across the log-replay the vendored ggml performs at load.

#include "test_assert.h"

#include "engine/framework/core/backend.h"

#include <cstdio>
#include <cstdlib>
#include <string>

#ifdef _WIN32
#define UNVALIDATED_TEST_SET(name, value) _putenv_s((name), (value))
static void unvalidated_test_unset() { _putenv_s("AUDIOCPP_ENABLE_UNVALIDATED_BACKEND", ""); }
#else
#define UNVALIDATED_TEST_SET(name, value) setenv((name), (value), 1)
static void unvalidated_test_unset() { unsetenv("AUDIOCPP_ENABLE_UNVALIDATED_BACKEND"); }
#endif

int main() try {
    using engine::core::unvalidated_backend_override;

    // Default: unset (or empty, which is what _putenv_s("") produces on
    // Windows) keeps the conservative behavior.
    unvalidated_test_unset();
    engine::test::require(
        !unvalidated_backend_override("RE-USE", "HIP", "this backend has not been validated"),
        "unset AUDIOCPP_ENABLE_UNVALIDATED_BACKEND must not lift family backend gates");
    engine::test::require(
        !unvalidated_backend_override("Whistle", "non-CPU/Metal", "allowlist gate"),
        "unset AUDIOCPP_ENABLE_UNVALIDATED_BACKEND must not lift allowlist gates either");

    // Only the exact value "1" opts in.
    UNVALIDATED_TEST_SET("AUDIOCPP_ENABLE_UNVALIDATED_BACKEND", "1");
    engine::test::require(
        unvalidated_backend_override("RE-USE", "HIP", "this backend has not been validated"),
        "AUDIOCPP_ENABLE_UNVALIDATED_BACKEND=1 must lift the gate (warning path must not throw)");

    // Strictness: anything else keeps the default, matching the framework's
    // ENGINE_VALIDATE_BACKEND_GRAPH convention.
    for (const std::string value : {"0", "true", "yes", "01", "1 "}) {
        UNVALIDATED_TEST_SET("AUDIOCPP_ENABLE_UNVALIDATED_BACKEND", value.c_str());
        engine::test::require(
            !unvalidated_backend_override("RE-USE", "HIP", "gate"),
            "AUDIOCPP_ENABLE_UNVALIDATED_BACKEND=\"" + value + "\" must not lift the gate");
    }

    // Toggle back without restarting the process (no caching).
    UNVALIDATED_TEST_SET("AUDIOCPP_ENABLE_UNVALIDATED_BACKEND", "1");
    engine::test::require(
        unvalidated_backend_override("RE-USE", "HIP", "gate"),
        "re-enabling within one process must take effect on the next call");
    unvalidated_test_unset();
    engine::test::require(
        !unvalidated_backend_override("RE-USE", "HIP", "gate"),
        "unsetting within one process must restore the conservative default");

    std::printf("fork_unvalidated_backend_override_test: all cases passed\n");
    return 0;
} catch (const std::exception & error) {
    std::fprintf(stderr, "fork_unvalidated_backend_override_test failed: %s\n", error.what());
    return 1;
}
