/**
 * test_capi_batch.c — verify audiocpp_tts_batch: request-level progress,
 * independent + concat merge modes. Run with a Qwen3-TTS model + voice ref.
 *
 *   set AUDIOCPP_TEST_VOICE_REF=<wav> AUDIOCPP_TEST_MODEL=<path>
 *   test_capi_batch.exe
 */
#include "../capi/include/audiocpp.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_calls = 0;
static int g_last_completed = -1;

static int progress_cb(float p, const char *stage,
                       int64_t completed, int64_t total, void *user) {
    (void)user;
    g_calls++;
    if (completed < g_last_completed) {
        printf("  PROGRESS REGRESSION: %lld after %lld\n",
               (long long)completed, (long long)g_last_completed);
    }
    if (strcmp(stage ? stage : "", "batch_tts") != 0) {
        printf("  UNEXPECTED STAGE: '%s' (wanted 'batch_tts')\n", stage ? stage : "NULL");
    }
    g_last_completed = (int)completed;
    printf("  [batch] %d%% (%lld/%lld)\n", (int)(p * 100),
           (long long)completed, (long long)total);
    return 0;  // continue
}

int main(void) {
    const char *model_path = getenv("AUDIOCPP_TEST_MODEL");
    const char *voice_ref = getenv("AUDIOCPP_TEST_VOICE_REF");
    const char *ref_text = getenv("AUDIOCPP_TEST_REF_TEXT");
    if (!model_path || !voice_ref) {
        printf("SKIP: set AUDIOCPP_TEST_MODEL and AUDIOCPP_TEST_VOICE_REF\n");
        return 0;
    }

    printf("audiocpp version: %s\n", audiocpp_version());
    audiocpp_error_t err = {0, NULL};

    audiocpp_model_t *model = audiocpp_load_model(
        model_path, "qwen3_tts", AUDIOCPP_TASK_TTS, AUDIOCPP_BACKEND_CUDA, 0, 4, &err);
    if (!model) {
        printf("FAIL load: code=%d msg=%s\n", err.code, err.message ? err.message : "(null)");
        audiocpp_clear_error(&err);
        return 1;
    }
    audiocpp_set_progress_callback(model, progress_cb, NULL);

    // Build options with voice ref + reference text.
    char opts[1024];
    snprintf(opts, sizeof(opts),
             "{\"voice_ref\":\"%s\",\"reference_text\":\"%s\"}",
             voice_ref, ref_text ? ref_text : "hello world test");

    const char *texts[] = {
        "First sentence for the batch test.",
        "Second text item, also synthesized.",
        "Third and final item in the batch.",
    };
    const int n = 3;

    // ---- Mode 1: independent (N separate audio) ----
    printf("\n=== BATCH independent (n=%d) ===\n", n);
    g_calls = 0; g_last_completed = -1;
    audiocpp_audio_batch_t *batch = audiocpp_tts_batch(
        model, texts, n, opts, AUDIOCPP_BATCH_MERGE_NONE, &err);
    if (!batch) {
        printf("FAIL batch NONE: code=%d msg=%s\n", err.code, err.message ? err.message : "(null)");
        audiocpp_clear_error(&err);
        audiocpp_free_model(model);
        return 1;
    }
    printf("  n_items=%d, progress calls=%d\n", batch->n_items, g_calls);
    int produced = 0;
    for (int i = 0; i < batch->n_items; ++i) {
        if (batch->items[i].samples && batch->items[i].n_samples > 0) {
            ++produced;
            printf("  item[%d]: %lld samples @ %d Hz\n",
                   i, (long long)batch->items[i].n_samples, batch->items[i].sample_rate);
        } else {
            printf("  item[%d]: (empty)\n", i);
        }
    }
    // Expect progress fired at least n+1 times (before each + final).
    printf("  produced=%d/%d, progress monotonic=%s\n",
           produced, n, g_last_completed == n ? "OK" : "CHECK");
    audiocpp_free_audio_batch(batch);

    // ---- Mode 2: concat (single merged audio + chapters) ----
    printf("\n=== BATCH concat (n=%d) ===\n", n);
    g_calls = 0; g_last_completed = -1;
    batch = audiocpp_tts_batch(
        model, texts, n, opts, AUDIOCPP_BATCH_MERGE_CONCAT, &err);
    if (!batch) {
        printf("FAIL batch CONCAT: code=%d msg=%s\n", err.code, err.message ? err.message : "(null)");
        audiocpp_clear_error(&err);
        audiocpp_free_model(model);
        return 1;
    }
    printf("  n_items=%d, progress calls=%d\n", batch->n_items, g_calls);
    printf("  merged[0]: %lld samples @ %d Hz\n",
           (long long)batch->items[0].n_samples, batch->items[0].sample_rate);
    if (batch->chapter_starts && batch->chapter_ends) {
        for (int i = 0; i < batch->n_items; ++i) {
            printf("  chapter[%d]: [%lld, %lld)\n", i,
                   (long long)batch->chapter_starts[i],
                   (long long)batch->chapter_ends[i]);
        }
    }
    audiocpp_free_audio_batch(batch);

    audiocpp_set_progress_callback(model, NULL, NULL);
    audiocpp_free_model(model);
    audiocpp_clear_error(&err);
    printf("\nDONE\n");
    return 0;
}
