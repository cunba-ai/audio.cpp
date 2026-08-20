// Unit tests for engine::assets::embedded::prefer_embedded_vad_model_path —
// the shared policy used by the ASR families' internal VAD sessions
// (sense_asr, qwen3_asr, vibevoice_asr).
//
// The original bug: those sessions hardcoded the on-disk default path
// (assets/framework/models/silero_vad) for their internal silero_vad session,
// so builds with AUDIOCPP_EMBED_VAD_ASSETS never used the baked-in weights
// and threw "Silero VAD model path does not exist" on machines without the
// disk assets. load_silero_vad_model only consults the embedded weights when
// model_path is EMPTY, so the fix resolves the path at session construction:
// explicit option wins, otherwise embedded (empty path), otherwise disk.
//
// Fork-regression guard: an upstream merge that reintroduces a hardcoded disk
// path (or flips this policy) breaks these tests.

#include "engine/framework/assets/embedded.h"

#include "test_assert.h"

#include <cstdio>
#include <filesystem>
#include <optional>
#include <string>

namespace {

const std::filesystem::path kDiskDefault =
    std::filesystem::path("assets") / "framework" / "models" / "silero_vad";

}  // namespace

int main() {
    using engine::assets::embedded::prefer_embedded_vad_model_path;

    // 1. An explicitly configured path always wins — even when embedded
    //    weights are available, we never silently override user config.
    {
        const auto p = prefer_embedded_vad_model_path(
            std::optional<std::string>{"/custom/silero_vad"}, kDiskDefault, true);
        engine::test::require_eq(
            p.string(), "/custom/silero_vad",
            "explicit vad_model_path option must win over embedded weights");
    }

    // 2. No option + embedded build => empty path. This is what drives
    //    load_silero_vad_model into its embedded-weights branch; returning
    //    the disk default here is exactly the original bug.
    {
        const auto p = prefer_embedded_vad_model_path(
            std::nullopt, kDiskDefault, true);
        engine::test::require(
            p.empty(),
            "embedded build with no explicit option must yield an empty "
            "path (embedded branch), got: " + p.string());
    }

    // 3. No option + no embedded assets => the on-disk default (upstream
    //    builds without AUDIOCPP_EMBED_VAD_ASSETS keep working).
    {
        const auto p = prefer_embedded_vad_model_path(
            std::nullopt, kDiskDefault, false);
        engine::test::require_eq(
            p, kDiskDefault, "non-embedded build must fall back to disk default");
    }

    // 4. Explicit option + no embedded assets => still the explicit path.
    {
        const auto p = prefer_embedded_vad_model_path(
            std::optional<std::string>{"D:/models/silero"}, kDiskDefault, false);
        engine::test::require_eq(
            p.string(), "D:/models/silero",
            "explicit option must win regardless of embedded availability");
    }

    // 5. Integration sanity for real builds: when this test binary is built
    //    against an engine_runtime compiled with AUDIOCPP_EMBED_VAD_ASSETS
    //    (the fork's dll configuration), the real availability flag must be
    //    true for silero_vad and the policy must select the embedded branch.
    //    Non-embedded builds print a skip note instead of failing.
    {
        const bool embedded = engine::assets::embedded::has_embedded_asset("silero_vad");
#if defined(AUDIOCPP_EMBED_VAD_ASSETS)
        engine::test::require(
            embedded,
            "AUDIOCPP_EMBED_VAD_ASSETS build must expose the silero_vad asset");
        const auto p = prefer_embedded_vad_model_path(std::nullopt, kDiskDefault, embedded);
        engine::test::require(
            p.empty(),
            "policy must select the embedded branch on an embed build");
#else
        std::printf("skip: built without AUDIOCPP_EMBED_VAD_ASSETS (has_embedded_asset=%d)\n",
                    embedded ? 1 : 0);
#endif
    }

    std::printf("asr_vad_model_path_test: all cases passed\n");
    return 0;
}
