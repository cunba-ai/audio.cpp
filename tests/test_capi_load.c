/**
 * test_capi_load.c — 决定性验证:静态链接 engine_runtime(同 CLI),
 * 直接调用 audiocpp_load_model,排除 DLL/ABI 因素。
 */
#include "../capi/include/audiocpp.h"
#include <stdio.h>

#define MODEL_PATH "models/Qwen3-ASR-0.6B"

static int try_load(const char *label, const char *family_hint) {
    audiocpp_error_t err = {0, NULL};
    printf("\n=== %s (family_hint=%s) ===\n", label, family_hint ? family_hint : "NULL");
    fflush(stdout);
    audiocpp_model_t *model = audiocpp_load_model(
        MODEL_PATH,
        family_hint,
        AUDIOCPP_TASK_ASR,
        AUDIOCPP_BACKEND_CPU,
        0,
        4,
        &err);
    if (model) {
        printf("  SUCCESS: model loaded\n");
        audiocpp_free_model(model);
        audiocpp_clear_error(&err);
        return 0;
    }
    printf("  FAILED: code=%d msg=%s\n", err.code, err.message ? err.message : "(null)");
    audiocpp_clear_error(&err);
    return 1;
}

int main(void) {
    printf("audiocpp version: %s\n", audiocpp_version());
    int r1 = try_load("TEST 1: with family_hint", "qwen3_asr");
    int r2 = try_load("TEST 2: family_hint=NULL", NULL);
    printf("\n=== SUMMARY ===\n  with hint: %s\n  without: %s\n",
           r1 ? "FAIL" : "OK", r2 ? "FAIL" : "OK");
    return (r1 || r2) ? 1 : 0;
}
