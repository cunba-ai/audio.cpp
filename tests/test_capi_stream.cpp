/**
 * test_capi_stream.c — CAPI 流式 TTS 验证(动态加载 audiocpp.dll)。
 * 复现客户端通过 audiocpp_stream_start/pull/finish 调用 supertonic-3 流式的路径。
 *
 * 用法: test_capi_stream.exe <model_path> <text> [backend] [out.wav]
 *   backend: cpu | cuda (默认 cpu)
 *
 * 流程:load_model(TTS,STREAMING) → stream_start({"text":...}) → 循环 stream_pull
 *      累积 audio_samples → stream_finish → write_wav
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#ifdef _WIN32
  #include <windows.h>
  #define LOAD_LIB(path)    LoadLibraryA(path)
  #define GET_SYM(h, name)  GetProcAddress((HMODULE)(h), name)
  #define FREE_LIB(h)       FreeLibrary((HMODULE)(h))
  typedef void * module_handle_t;
#else
  #include <dlfcn.h>
  #define LOAD_LIB(path)    dlopen(path, RTLD_NOW)
  #define GET_SYM(h, name)  dlsym((h), name)
  #define FREE_LIB(h)       dlclose(h)
  typedef void * module_handle_t;
#endif

#include "../capi/include/audiocpp.h"

#define TASK_TTS     5
#define BACKEND_CPU  0
#define BACKEND_CUDA 1

typedef const char * (*fn_version_t)(void);
typedef audiocpp_model_t * (*fn_load_model_t)(const char *, const char *, int, int, int, int, audiocpp_error_t *);
typedef void (*fn_free_model_t)(audiocpp_model_t *);
typedef audiocpp_stream_t * (*fn_stream_start_t)(const audiocpp_model_t *, int, const char *, int64_t, audiocpp_error_t *);
typedef audiocpp_stream_event_t * (*fn_stream_pull_t)(audiocpp_stream_t *, int, audiocpp_error_t *);
typedef int (*fn_stream_finish_t)(audiocpp_stream_t *, audiocpp_text_t *, audiocpp_error_t *);
typedef void (*fn_free_stream_event_t)(audiocpp_stream_event_t *);
typedef void (*fn_stream_free_t)(audiocpp_stream_t *);
typedef int (*fn_write_wav_t)(const char *, const float *, int64_t, int);
typedef void (*fn_clear_error_t)(audiocpp_error_t *);

static fn_version_t            p_version;
static fn_load_model_t         p_load_model;
static fn_free_model_t         p_free_model;
static fn_stream_start_t       p_stream_start;
static fn_stream_pull_t        p_stream_pull;
static fn_stream_finish_t      p_stream_finish;
static fn_free_stream_event_t  p_free_stream_event;
static fn_stream_free_t        p_stream_free;
static fn_write_wav_t          p_write_wav;
static fn_clear_error_t        p_clear_error;

#define BIND(name, type, storage) \
    storage = (type)GET_SYM(handle, name); \
    if (!storage) { fprintf(stderr, "FATAL: dll 缺少导出 %s\n", name); FREE_LIB(handle); return 2; }

static int resolve_symbols(module_handle_t handle) {
    BIND("audiocpp_version",            fn_version_t,            p_version);
    BIND("audiocpp_load_model",         fn_load_model_t,         p_load_model);
    BIND("audiocpp_free_model",         fn_free_model_t,         p_free_model);
    BIND("audiocpp_stream_start",       fn_stream_start_t,       p_stream_start);
    BIND("audiocpp_stream_pull",        fn_stream_pull_t,        p_stream_pull);
    BIND("audiocpp_stream_finish",      fn_stream_finish_t,      p_stream_finish);
    BIND("audiocpp_free_stream_event",  fn_free_stream_event_t,  p_free_stream_event);
    BIND("audiocpp_stream_free",        fn_stream_free_t,        p_stream_free);
    BIND("audiocpp_write_wav",          fn_write_wav_t,          p_write_wav);
    BIND("audiocpp_clear_error",        fn_clear_error_t,        p_clear_error);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "用法: %s <model_path> <text> [backend] [out.wav]\n", argv[0]);
        return 1;
    }
    const char *model_path = argv[1];
    const char *text       = argv[2];
    const char *backend_str= argc > 3 ? argv[3] : "cpu";
    const char *out_wav    = argc > 4 ? argv[4] : "stream_out.wav";
    int backend = (strcmp(backend_str, "cpu") == 0) ? BACKEND_CPU : BACKEND_CUDA;

    const char *dll_path = getenv("AUDIOCPP_DLL");
    module_handle_t handle = dll_path ? LOAD_LIB(dll_path) : NULL;
    if (!handle) handle = LOAD_LIB("audiocpp.dll");
    if (!handle) { fprintf(stderr, "FATAL: 无法加载 audiocpp.dll\n"); return 2; }
    if (resolve_symbols(handle) != 0) return 2;
    printf("[stream] audiocpp version: %s\n", p_version());

    audiocpp_error_t err = {0, NULL};
    // __zh__ => 用源码内写死的 UTF-8 中文,绕过 Windows 命令行编码(codepage)问题,
    // 确保 text 进 CAPI 时是合法 UTF-8。
    const char *zh_text = u8"你好世界,这是一个测试。";
    if (strcmp(text, "__zh__") == 0) {
        text = zh_text;
    }
    printf("[stream] 加载模型: %s (backend=%s, task=TTS)\n", model_path, backend_str);
    audiocpp_model_t *model = p_load_model(model_path, NULL, TASK_TTS, backend, 0, 4, &err);
    if (!model) {
        fprintf(stderr, "FAIL[load]: code=%d msg=%s\n", err.code, err.message ? err.message : "(null)");
        p_clear_error(&err); FREE_LIB(handle); return 3;
    }
    printf("[stream] 模型加载成功\n");

    /* 关键:text 放 options_json(文档要求 {"text":"..."}) */
    char options[2048];
    snprintf(options, sizeof(options), "{\"text\": \"%s\"}", text);
    printf("[stream] stream_start: options=%s\n", options);
    audiocpp_stream_t *stream = p_stream_start(model, TASK_TTS, options, 0, &err);
    if (!stream) {
        fprintf(stderr, "FAIL[stream_start]: code=%d msg=%s\n", err.code, err.message ? err.message : "(null)");
        p_clear_error(&err); p_free_model(model); FREE_LIB(handle); return 4;
    }
    printf("[stream] stream_start 成功,开始 pull\n");

    /* 循环 pull,累积音频 */
    std::vector<float> all_samples;
    int sr = 0;
    int pull_count = 0;
    int total_samples = 0;
    while (1) {
        audiocpp_stream_event_t *ev = p_stream_pull(stream, -1, &err);
        if (err.code != 0 && err.message) {
            fprintf(stderr, "FAIL[pull#%d]: code=%d msg=%s\n", pull_count, err.code, err.message);
            p_clear_error(&err);
            break;
        }
        if (!ev) {
            printf("[stream] pull 返回 NULL(流结束)\n");
            break;
        }
        pull_count++;
        if (ev->n_audio_samples > 0) {
            sr = ev->audio_sample_rate;
            total_samples += (int)ev->n_audio_samples;
            for (int64_t i = 0; i < ev->n_audio_samples; ++i)
                all_samples.push_back(ev->audio_samples[i]);
            printf("[stream] pull#%d: audio=%lld samples @ %d Hz (累计 %d)\n",
                   pull_count, (long long)ev->n_audio_samples, sr, total_samples);
        } else {
            printf("[stream] pull#%d: 无音频 (is_final=%d)\n", pull_count, ev->is_final);
        }
        p_free_stream_event(ev);
    }

    printf("[stream] finish\n");
    int rc = p_stream_finish(stream, NULL, &err);
    if (rc != 0) {
        fprintf(stderr, "FAIL[finish]: code=%d msg=%s\n", err.code, err.message ? err.message : "(null)");
        p_clear_error(&err);
    }
    p_stream_free(stream);

    printf("[stream] 总计: %d 次 pull, %d samples @ %d Hz (%.2fs)\n",
           pull_count, total_samples, sr, total_samples > 0 ? total_samples / (float)sr : 0.0f);

    if (total_samples > 0 && sr > 0) {
        if (p_write_wav(out_wav, all_samples.data(), all_samples.size(), sr) != 0) {
            fprintf(stderr, "WARN: 写出 %s 失败\n", out_wav);
        } else {
            printf("[stream] 已写出 %s\n", out_wav);
        }
    } else {
        printf("[stream] 无音频输出!\n");
    }

    p_free_model(model);
    p_clear_error(&err);
    FREE_LIB(handle);
    return 0;
}
