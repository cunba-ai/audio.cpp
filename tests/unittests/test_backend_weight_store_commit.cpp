// Regression test for IndexTTS2 OOM-by-overcommit (see .agents/indextts2_bug_report.md).
//
// Background:
//   BackendWeightStore constructs a ggml context pool sized by `context_bytes`
//   (commonly 4096 MB per runtime). Historically it pre-COMMITTED the entire
//   `context_bytes` up front via ggml_init -> ggml_aligned_malloc, even though
//   the store always runs with no_alloc=true and only ever uses the pool for
//   tensor METADATA (a few KB of ggml_tensor structs). The tensor DATA lives
//   in a separate backend buffer allocated in upload().
//
//   With 4096 MB pre-committed per store, a multi-runtime model like IndexTTS2
//   (7 runtimes) committed 7 * 4096 MB ~= 28 GB of private memory just to hold
//   ~1 MB of metadata, blowing past the Windows system commit limit (~51 GB on
//   the affected V100 box) and causing the next 2 GB GPT-arena allocation to
//   fail with GGML_ASSERT(ctx->mem_buffer != NULL).
//
// This test asserts the regression fix: a no_alloc metadata-only weight store
// must NOT commit a region proportional to context_bytes. It measures the
// process private commit before/after constructing a store with a large
// context_bytes and requires the committed delta to stay well below
// context_bytes (i.e. no over-commit).
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"

#include "test_assert.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#endif

namespace {

#if defined(_WIN32)
long long current_private_commit_bytes() {
    PROCESS_MEMORY_COUNTERS_EX pmc;
    std::memset(&pmc, 0, sizeof(pmc));
    pmc.cb = sizeof(pmc);
    if (!GetProcessMemoryInfo(GetCurrentProcess(),
                              reinterpret_cast<PROCESS_MEMORY_COUNTERS *>(&pmc),
                              sizeof(pmc))) {
        return -1;
    }
    return static_cast<long long>(pmc.PrivateUsage);
}
#else
long long current_private_commit_bytes() {
    return -1;  // not measured on non-Windows; test compiles everywhere but only asserts on Windows
}
#endif

void run_test() {
    auto * backend = engine::core::init_backend({engine::core::BackendType::Cpu, 0, 1});
    engine::test::require(backend != nullptr, "failed to init CPU backend");

    constexpr size_t context_bytes = static_cast<size_t>(512) * 1024 * 1024;  // 512 MB nominal

    const long long commit_before = current_private_commit_bytes();

    // Construct a metadata-only weight store with a 512 MB nominal context_bytes.
    // Under the buggy behavior the constructor commits the entire 512 MB up
    // front. Under the fix it should commit only a small metadata budget.
    {
        engine::core::BackendWeightStore store(
            backend,
            engine::core::BackendType::Cpu,
            "test.weight_store_commit",
            context_bytes);
        // Touch nothing else; the pool is committed in the constructor. Keeping
        // `store` alive across the measurement isolates the constructor's commit.
        const long long commit_after = current_private_commit_bytes();
        const long long commit_delta = (commit_before < 0 || commit_after < 0)
                                           ? -1
                                           : (commit_after - commit_before);

        std::cout << "context_bytes = " << (context_bytes / (1024 * 1024)) << " MB\n";
        std::cout << "private commit delta after constructing store: ";
        if (commit_delta < 0) {
            std::cout << "(not measured on this platform)\n";
        } else {
            std::cout << (commit_delta / (1024 * 1024)) << " MB\n";
        }

        // Regression assertion: a no_alloc metadata-only pool must not commit a
        // region proportional to context_bytes. The pool only ever holds tensor
        // metadata; 64 MB is 8x smaller than the buggy 512 MB commit and still
        // absurdly generous for metadata.
        if (commit_delta >= 0) {
            const long long acceptable_upper = static_cast<long long>(context_bytes / 8);  // 64 MB
            engine::test::require(
                commit_delta < acceptable_upper,
                "BackendWeightStore over-committed: private commit delta=" +
                    std::to_string(commit_delta / (1024 * 1024)) + " MB >= " +
                    std::to_string(acceptable_upper / (1024 * 1024)) +
                    " MB of nominal context_bytes. The no_alloc metadata pool must not pre-commit the full context_bytes.");
        }
    }

    ggml_backend_free(backend);
}

}  // namespace

int main() {
    try {
        run_test();
        std::cout << "test_backend_weight_store_commit: PASS\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "test_backend_weight_store_commit: FAIL — " << e.what() << "\n";
        return 1;
    }
}
