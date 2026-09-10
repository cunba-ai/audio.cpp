// End-to-end test for the embedded audio-utility denoisers in the C ABI.
//
// Fork delta under guard: GTCRN ships three checkpoints in
// assets/framework/audio_utilities/gtcrn/ and the fork added them to the
// AUDIOCPP_EMBED_AUDIO_UTILITIES manifest plus the audiocpp_denoise dispatch,
// so `audiocpp_denoise(..., "gtcrn", NULL, ...)` works with zero model files
// on disk. Upstream edits to the embed manifest (GTCRN was originally left out
// of it) or to the denoise dispatch can silently drop either half; this test
// fails loudly when that happens.
//
// Skips itself when the utilities embed option is off, so upstream CI builds
// without embedding stay green.
//
// This TU compiles audiocpp_capi.cpp directly (same static-link pattern as
// capi_enum_sync_test / capi_session_options_test).

#include "audiocpp.h"

#include "engine/framework/assets/embedded.h"

#include "test_assert.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

constexpr float kPi = 3.14159265358979323846F;

std::vector<float> make_tone(int sample_rate, int n_samples) {
    std::vector<float> pcm(static_cast<size_t>(n_samples));
    for (int i = 0; i < n_samples; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(sample_rate);
        pcm[static_cast<size_t>(i)] = 0.25F * std::sin(2.0F * kPi * 440.0F * t);
    }
    return pcm;
}

#if defined(AUDIOCPP_EMBED_AUDIO_UTILITIES) && (AUDIOCPP_EMBED_AUDIO_UTILITIES != 0)

// Denoise once with an empty model_path and assert the embedded asset carried
// the call end to end. Returns the output length in samples.
int64_t denoise_with_embedded_asset(
    const std::vector<float> & pcm, int sample_rate, const char * model_name) {
    audiocpp_error_t err = {};
    audiocpp_audio_t * out =
        audiocpp_denoise(pcm.data(), static_cast<int64_t>(pcm.size()), sample_rate,
                         model_name, /*model_path=*/nullptr, /*options_json=*/nullptr, &err);
    if (out == nullptr) {
        const std::string message =
            std::string("audiocpp_denoise(") + model_name + ", NULL) failed: " +
            (err.message != nullptr ? err.message : "(no message)");
        audiocpp_free_string(err.message);
        throw std::runtime_error(message);
    }
    engine::test::require_eq(out->sample_rate, sample_rate,
                             std::string(model_name) + " output sample_rate");
    engine::test::require(out->n_samples > 0, std::string(model_name) + " output is empty");
    const int64_t n_samples = out->n_samples;
    audiocpp_free_audio(out);
    return n_samples;
}

#endif  // AUDIOCPP_EMBED_AUDIO_UTILITIES

}  // namespace

#if !defined(AUDIOCPP_EMBED_AUDIO_UTILITIES) || (AUDIOCPP_EMBED_AUDIO_UTILITIES == 0)

int main() {
    std::printf(
        "capi_denoise_embedded_test: skipped (AUDIOCPP_EMBED_AUDIO_UTILITIES=OFF)\n");
    return 0;
}

#else

int main() {
    // 1. Every asset the CAPI can materialize for a NULL model_path must be in
    //    the generated embed table.
    const char * kAssetIds[] = {
        "deepfilternet2", "rnnoise", "zipenhancer", "flashsr",
        "gtcrn_streaming", "gtcrn_dns3",     "gtcrn_vctk",
    };
    for (const char * id : kAssetIds) {
        engine::test::require(engine::assets::embedded::has_embedded_asset(id),
                              std::string("embedded audio-utility asset missing: ") + id);
    }

    // 2. End to end: "gtcrn" (the streaming alias) denoises a tone with no
    //    model path, no options, nothing on disk.
    constexpr int kSampleRate = 16000;
    const auto pcm = make_tone(kSampleRate, kSampleRate / 2);
    (void)denoise_with_embedded_asset(pcm, kSampleRate, "gtcrn");

    // 3. Each explicit variant name must materialize its own checkpoint.
    for (const char * name : {"gtcrn_streaming", "gtcrn_dns3", "gtcrn_vctk"}) {
        (void)denoise_with_embedded_asset(pcm, kSampleRate, name);
    }

    // 4. An unknown model name still fails loudly (and names the alternatives).
    {
        audiocpp_error_t err = {};
        audiocpp_audio_t * out = audiocpp_denoise(
            pcm.data(), static_cast<int64_t>(pcm.size()), kSampleRate, "no_such_model",
            nullptr, nullptr, &err);
        engine::test::require(out == nullptr, "unknown denoise model must fail");
        engine::test::require(err.message != nullptr, "unknown model error message present");
        engine::test::require(
            std::string(err.message).find("unsupported denoise model") != std::string::npos,
            std::string("unexpected error message: ") + err.message);
        audiocpp_free_string(err.message);
    }

    std::printf("capi_denoise_embedded_test: all cases passed\n");
    return 0;
}

#endif  // AUDIOCPP_EMBED_AUDIO_UTILITIES
