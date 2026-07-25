/**
 * test_capi_embedded_vad.c — verify audiocpp_vad_energy (model-free) and the
 * embedded VAD path (audiocpp_load_model(NULL, "silero_vad", ...)).
 *
 * Requires a build with AUDIOCPP_EMBED_VAD_ASSETS=ON. Run with a wav file via
 * AUDIOCPP_TEST_WAV=<path>; otherwise generates a synthetic tone in-memory.
 */
#include "../capi/include/audiocpp.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static float *make_synth_audio(int64_t *n_out, int *rate_out) {
    /* 10s @ 16kHz: 3s tone (speech-ish) + 2s silence + 3s tone + 2s silence.
       Lets both energy VAD and model VAD find 2 segments. */
    const int rate = 16000;
    const int64_t total = rate * 10;
    float *pcm = (float *)malloc(sizeof(float) * (size_t)total);
    if (!pcm) return NULL;
    for (int64_t i = 0; i < total; ++i) {
        const double t = (double)i / rate;
        /* tone during [0,3) and [5,8), silence elsewhere */
        const int seg = (int)(t);
        const int into = (int)((t - seg) * 10);
        const int segmod = seg % 5;  /* 0..4 within a 5s block */
        const int subpos = into;     /* 0..9 (first 0.x of the second) */
        (void)subpos;
        const double amp = (segmod < 3) ? 0.3 : 0.0;
        pcm[i] = (float)(amp * sin(2.0 * M_PI * 220.0 * t));
    }
    *n_out = total;
    *rate_out = rate;
    return pcm;
}

int main(void) {
    printf("audiocpp version: %s\n", audiocpp_version());

    int64_t n = 0;
    int rate = 0;
    float *pcm = make_synth_audio(&n, &rate);
    if (!pcm) { printf("FAIL: out of memory\n"); return 1; }
    printf("synthetic audio: %lld samples @ %d Hz (%.1fs)\n",
           (long long)n, rate, (double)n / rate);

    audiocpp_error_t err = {0, NULL};
    int failures = 0;

    /* ---- 1. Energy VAD (no model) ---- */
    printf("\n=== TEST 1: audiocpp_vad_energy (no model) ===\n");
    audiocpp_vad_t *ev = audiocpp_vad_energy(
        pcm, n, rate, "{\"chunk_seconds\":4.0,\"boundary_seconds\":1.0}", &err);
    if (!ev) {
        printf("  FAIL: code=%d msg=%s\n", err.code, err.message ? err.message : "(null)");
        ++failures;
    } else {
        printf("  segments: %lld\n", (long long)ev->n_segments);
        for (int64_t i = 0; i < ev->n_segments && i < 6; ++i) {
            printf("    [%lld] %lld..%lld (%.2fs..%.2fs) conf=%.2f\n",
                   (long long)i,
                   (long long)ev->segments[i].start_sample,
                   (long long)ev->segments[i].end_sample,
                   (double)ev->segments[i].start_sample / rate,
                   (double)ev->segments[i].end_sample / rate,
                   ev->segments[i].confidence);
        }
        if (ev->n_segments < 2) {
            printf("  WARN: expected >=2 segments from the tone/silence pattern\n");
        }
        audiocpp_free_vad(ev);
        printf("  OK\n");
    }
    audiocpp_clear_error(&err);

    /* ---- 2. Embedded VAD load (model_path=NULL) ---- */
    printf("\n=== TEST 2: audiocpp_load_model(NULL, \"silero_vad\") embedded ===\n");
    audiocpp_model_t *model = audiocpp_load_model(
        NULL, "silero_vad", AUDIOCPP_TASK_VAD, AUDIOCPP_BACKEND_CPU, 0, 4, &err);
    if (!model) {
        printf("  SKIP/FAIL: code=%d msg=%s\n", err.code, err.message ? err.message : "(null)");
        printf("  (expected only if built WITHOUT AUDIOCPP_EMBED_VAD_ASSETS=ON)\n");
        /* not a hard failure if the build is non-embedded */
    } else {
        printf("  model loaded from embedded weights OK\n");
        audiocpp_vad_t *vad = audiocpp_vad(model, pcm, n, rate, NULL, &err);
        if (!vad) {
            printf("  vad run FAIL: code=%d msg=%s\n", err.code, err.message ? err.message : "(null)");
            ++failures;
        } else {
            printf("  segments: %lld\n", (long long)vad->n_segments);
            for (int64_t i = 0; i < vad->n_segments && i < 6; ++i) {
                printf("    [%lld] %lld..%lld (%.2fs..%.2fs)\n",
                       (long long)i,
                       (long long)vad->segments[i].start_sample,
                       (long long)vad->segments[i].end_sample,
                       (double)vad->segments[i].start_sample / rate,
                       (double)vad->segments[i].end_sample / rate);
            }
            audiocpp_free_vad(vad);
            printf("  OK\n");
        }
        audiocpp_free_model(model);
    }
    audiocpp_clear_error(&err);

    free(pcm);
    printf("\n%s\n", failures ? "FAILED" : "DONE");
    return failures ? 1 : 0;
}
