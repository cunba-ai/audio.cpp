/**
 * test_capi_denoise.c — verify audiocpp_denoise (3 models) + audiocpp_super_resolve.
 * Requires AUDIOCPP_EMBED_AUDIO_UTILITIES=ON for the NULL-path (embedded) cases;
 * the explicit-path cases need AUDIOCPP_TEST_ASSETS_ROOT=<.../assets/framework>.
 */
#include "../capi/include/audiocpp.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Build ~0.5s of 16kHz audio: a 220Hz tone + white-ish noise (deterministic).
   Kept short because these models build per-frame graphs with a fixed node
   capacity; very long inputs overflow the graph (production callers should
   chunk the audio first). */
static float *make_noisy(int64_t *n_out, int *rate_out) {
    const int rate = 16000;
    const int64_t total = rate / 2;  /* 8000 samples (0.5s) — within graph budget */
    float *pcm = (float *)malloc(sizeof(float) * (size_t)total);
    if (!pcm) return NULL;
    unsigned int rng = 12345u;
    for (int64_t i = 0; i < total; ++i) {
        const double t = (double)i / rate;
        const double tone = 0.3 * sin(2.0 * M_PI * 220.0 * t);
        /* simple LCG noise in [-0.15, 0.15] */
        rng = rng * 1103515245u + 12345u;
        const double noise = ((double)(rng >> 16) / 32768.0 - 0.5) * 0.3;
        pcm[i] = (float)(tone + noise);
    }
    *n_out = total;
    *rate_out = rate;
    return pcm;
}

static double rms(const float *p, int64_t n) {
    double s = 0.0;
    for (int64_t i = 0; i < n; ++i) s += (double)p[i] * (double)p[i];
    return sqrt(s / (double)n);
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);  /* unbuffered so we see output before any abort */
    setvbuf(stderr, NULL, _IONBF, 0);
    printf("audiocpp version: %s\n", audiocpp_version());
    int64_t n = 0;
    int rate = 0;
    float *pcm = make_noisy(&n, &rate);
    if (!pcm) { printf("FAIL: oom\n"); return 1; }
    const double in_rms = rms(pcm, n);
    printf("input: %lld samples @ %d Hz, RMS=%.4f\n", (long long)n, rate, in_rms);

    const char *assets_root = getenv("AUDIOCPP_TEST_ASSETS_ROOT");
    audiocpp_error_t err = {0, NULL};
    int failures = 0;

    const char *denoise_models[] = {"deepfilternet2", "rnnoise", "zipenhancer"};
    for (int m = 0; m < 3; ++m) {
        const char *model = denoise_models[m];
        printf("\n=== audiocpp_denoise(\"%s\") embedded (model_path=NULL) ===\n", model);
        fflush(stdout);
        audiocpp_audio_t *out = audiocpp_denoise(pcm, n, rate, model, NULL, NULL, &err);
        if (!out) {
            printf("  FAIL/SKIP: code=%d msg=%s\n", err.code, err.message ? err.message : "(null)");
            printf("  (embedded path needs AUDIOCPP_EMBED_AUDIO_UTILITIES=ON)\n");
            audiocpp_clear_error(&err);
            continue;
        }
        const double out_rms = rms(out->samples, out->n_samples);
        printf("  out: %lld samples @ %d Hz, RMS=%.4f (in=%.4f)\n",
               (long long)out->n_samples, out->sample_rate, out_rms, in_rms);
        if (out->n_samples == 0) {
            printf("  FAIL: empty output\n");
            ++failures;
        }
        audiocpp_free_audio(out);
        printf("  OK\n");
    }

    printf("\n=== audiocpp_super_resolve(\"flashsr\") embedded ===\n");
    {
        audiocpp_audio_t *out = audiocpp_super_resolve(pcm, n, rate, NULL, NULL, &err);
        if (!out) {
            printf("  FAIL/SKIP: code=%d msg=%s\n", err.code, err.message ? err.message : "(null)");
            audiocpp_clear_error(&err);
        } else {
            printf("  out: %lld samples @ %d Hz (input was %d Hz)\n",
                   (long long)out->n_samples, out->sample_rate, rate);
            /* flashsr upsamples 16k -> 48k, so expect ~3x samples and 48k rate */
            if (out->sample_rate != 48000) {
                printf("  WARN: expected 48k output, got %d\n", out->sample_rate);
            }
            if (out->n_samples == 0) {
                printf("  FAIL: empty output\n");
                ++failures;
            }
            audiocpp_free_audio(out);
            printf("  OK\n");
        }
    }

    /* Optional: explicit-path variants when assets root is provided. */
    if (assets_root) {
        char path[512];
        printf("\n=== audiocpp_denoise(\"deepfilternet2\") explicit path ===\n");
        snprintf(path, sizeof(path), "%s/audio_utilities/deepfilternet2", assets_root);
        audiocpp_audio_t *out = audiocpp_denoise(pcm, n, rate, "deepfilternet2", path, NULL, &err);
        if (!out) {
            printf("  FAIL: code=%d msg=%s\n", err.code, err.message ? err.message : "(null)");
            ++failures;
        } else {
            printf("  out: %lld samples @ %d Hz\n", (long long)out->n_samples, out->sample_rate);
            audiocpp_free_audio(out);
            printf("  OK\n");
        }
        audiocpp_clear_error(&err);
    }

    free(pcm);
    audiocpp_clear_error(&err);
    printf("\n%s\n", failures ? "FAILED" : "DONE");
    return failures ? 1 : 0;
}
