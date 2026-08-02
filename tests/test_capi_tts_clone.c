/**
 * test_capi_tts_clone.c — 独立 CAPI 验证工具(动态加载 audiocpp.dll)。
 *
 * 目的:模拟 sound-server 的真实 FFI 路径,对 Qwen3-TTS 做最小化 voice-clone
 * 请求,确认崩溃(bad_alloc / 挂死)是否发生在引擎层而非包装层。
 *
 * 与 audiocpp_cli 的区别:
 *   - 本工具 *动态加载* audiocpp.dll(LoadLibrary + GetProcAddress),不走
 *     静态链接,完全复现外部调用方的 ABI 路径。
 *   - 只做一件事:read_wav(voice_ref) -> load_model -> tts_with_voice_ref
 *     -> write_wav(out),便于在崩溃时精确定位是 load 还是 run 阶段。
 *
 * 用法:
 *   test_capi_tts_clone.exe <model_path> <voice_ref.wav> \
 *                           [reference_text] [text] [backend] [out.wav]
 *
 *   backend: cpu | cuda (默认 cuda)
 *
 * 不依赖 audiocpp.lib:函数指针在运行时从 dll 解析,因此只需与 audiocpp.dll
 * 同目录即可运行(和 sound-server 部署形态一致)。
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
  #include <windows.h>
  #define LOAD_LIB(path)       LoadLibraryA(path)
  #define GET_SYM(h, name)    GetProcAddress((HMODULE)(h), name)
  #define FREE_LIB(h)          FreeLibrary((HMODULE)(h))
  typedef void * module_handle_t;
#else
  #include <dlfcn.h>
  #define LOAD_LIB(path)       dlopen(path, RTLD_NOW)
  #define GET_SYM(h, name)    dlsym((h), name)
  #define FREE_LIB(h)          dlclose(h)
  typedef void * module_handle_t;
#endif

#include "../capi/include/audiocpp.h"

/* ---- engine constants (mirror audiocpp.h enums) ---- */
#define TASK_TTS    5
#define TASK_CLON   8
#define BACKEND_CPU 0
#define BACKEND_CUDA 1

/* ---- function pointer typedefs for the CAPI surface we use ---- */
typedef const char * (*fn_version_t)(void);
typedef audiocpp_model_t * (*fn_load_model_t)(
    const char *, const char *, int, int, int, int, audiocpp_error_t *);
typedef void (*fn_free_model_t)(audiocpp_model_t *);
typedef audiocpp_audio_t * (*fn_tts_with_voice_ref_t)(
    const audiocpp_model_t *, const char *, const char *,
    const float *, int64_t, int, audiocpp_error_t *);
typedef void (*fn_free_audio_t)(audiocpp_audio_t *);
typedef int (*fn_read_wav_t)(const char *, float **, int64_t *, int *);
typedef int (*fn_write_wav_t)(const char *, const float *, int64_t, int);
typedef void (*fn_clear_error_t)(audiocpp_error_t *);

/* resolved pointers */
static fn_version_t              p_version;
static fn_load_model_t           p_load_model;
static fn_free_model_t           p_free_model;
static fn_tts_with_voice_ref_t   p_tts_with_voice_ref;
static fn_free_audio_t           p_free_audio;
static fn_read_wav_t             p_read_wav;
static fn_write_wav_t            p_write_wav;
static fn_clear_error_t          p_clear_error;

#define BIND(name, type, storage) \
    storage = (type)GET_SYM(handle, name); \
    if (!storage) { fprintf(stderr, "FATAL: dll 缺少导出符号 %s\n", name); FREE_LIB(handle); return 2; }

static int resolve_symbols(module_handle_t handle) {
    BIND("audiocpp_version",            fn_version_t,            p_version);
    BIND("audiocpp_load_model",         fn_load_model_t,         p_load_model);
    BIND("audiocpp_free_model",         fn_free_model_t,         p_free_model);
    BIND("audiocpp_tts_with_voice_ref", fn_tts_with_voice_ref_t, p_tts_with_voice_ref);
    BIND("audiocpp_free_audio",         fn_free_audio_t,         p_free_audio);
    BIND("audiocpp_read_wav",           fn_read_wav_t,           p_read_wav);
    BIND("audiocpp_write_wav",          fn_write_wav_t,          p_write_wav);
    BIND("audiocpp_clear_error",        fn_clear_error_t,        p_clear_error);
    return 0;
}

/* portable monotonic-ish timer in seconds */
static double now_seconds(void) {
#ifdef _WIN32
    static LARGE_INTEGER freq = {0};
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
#endif
}

static void print_usage(const char *argv0) {
    fprintf(stderr,
        "用法: %s <model_path> <voice_ref.wav> [reference_text] [text] [backend] [out.wav]\n"
        "  model_path     模型目录或 GGUF 路径\n"
        "  voice_ref.wav  克隆参考音频\n"
        "  reference_text 参考音频转写(默认 \"test\")\n"
        "  text           要合成的文本(默认 \"Hello world.\")\n"
        "  backend        cpu | cuda (默认 cuda)\n"
        "  out.wav        输出文件(默认 out.wav)\n"
        "\n环境变量:\n"
        "  AUDIOCPP_DLL         指定 audiocpp.dll 路径(默认同目录或 PATH)\n"
        "  AUDIOCPP_SEED        固定采样种子(复现失控用)\n"
        "  AUDIOCPP_DO_SAMPLE   0=argmax / 1=sampling\n"
        "  AUDIOCPP_TEMPERATURE 覆盖 temperature\n"
        "  AUDIOCPP_TOP_P       覆盖 top_p\n",
        argv0);
}

int main(int argc, char **argv) {
    if (argc < 3) { print_usage(argv[0]); return 1; }
    const char *model_path    = argv[1];
    const char *voice_ref_wav = argv[2];
    const char *reference_text= argc > 3 ? argv[3] : "test";
    const char *text          = argc > 4 ? argv[4] : "Hello world.";
    const char *backend_str   = argc > 5 ? argv[5] : "cuda";
    const char *out_wav       = argc > 6 ? argv[6] : "out.wav";

    int backend = (strcmp(backend_str, "cpu") == 0) ? BACKEND_CPU : BACKEND_CUDA;

    /* ---- 1. 加载 audiocpp.dll ---- */
    const char *dll_path = getenv("AUDIOCPP_DLL");
    module_handle_t handle = dll_path ? LOAD_LIB(dll_path) : NULL;
    if (!handle) handle = LOAD_LIB("audiocpp.dll");
    if (!handle) {
        /* 试同目录 */
        char self[MAX_PATH];
#ifdef _WIN32
        GetModuleFileNameA(NULL, self, MAX_PATH);
        char *slash = strrchr(self, '\\');
#else
        readlink("/proc/self/exe", self, sizeof(self));
        char *slash = strrchr(self, '/');
#endif
        if (slash) { strcpy(slash + 1, "audiocpp.dll"); handle = LOAD_LIB(self); }
    }
    if (!handle) {
        fprintf(stderr, "FATAL: 无法加载 audiocpp.dll (AUDIOCPP_DLL=%s)\n",
                dll_path ? dll_path : "(未设置)");
        return 2;
    }
    if (resolve_symbols(handle) != 0) return 2;
    printf("[capi] audiocpp version: %s\n", p_version());
    printf("[capi] dll 已加载\n");

    double t0, t1, t2, t3, t4;

    /* ---- 2. 读取参考音频 ---- */
    printf("[capi] 读取参考音频: %s\n", voice_ref_wav);
    float *ref_pcm = NULL;
    int64_t ref_n = 0;
    int ref_sr = 0;
    if (p_read_wav(voice_ref_wav, &ref_pcm, &ref_n, &ref_sr) != 0 || ref_n == 0) {
        fprintf(stderr, "FATAL: 无法读取参考音频 %s\n", voice_ref_wav);
        FREE_LIB(handle);
        return 3;
    }
    printf("[capi] 参考音频: %lld samples @ %d Hz (%.2fs)\n",
           (long long)ref_n, ref_sr, (double)ref_n / ref_sr);

    /* ---- 3. 加载模型 ---- */
    audiocpp_error_t err = {0, NULL};
    printf("[capi] 加载模型: %s (backend=%s, task=TTS)\n", model_path, backend_str);
    t0 = now_seconds();
    audiocpp_model_t *model = p_load_model(
        model_path, "qwen3_tts", TASK_TTS, backend, 0, 4, &err);
    t1 = now_seconds();
    if (!model) {
        fprintf(stderr, "FAIL[load]: code=%d msg=%s (耗时 %.3fs)\n",
                err.code, err.message ? err.message : "(null)", t1 - t0);
        p_clear_error(&err);
        free(ref_pcm);
        FREE_LIB(handle);
        return 4;
    }
    printf("[capi] 模型加载成功 (%.3fs)\n", t1 - t0);

    /* ---- 4. voice-clone 合成 ---- */
    /* 构建 options_json:reference_text + 可选采样控制(环境变量)。
     *   AUDIOCPP_SEED       固定采样种子(便于复现失控)
     *   AUDIOCPP_DO_SAMPLE  "0"=argmax / "1"=sampling
     *   AUDIOCPP_TEMPERATURE/AUDIOCPP_TOP_P/AUDIOCPP_TOP_K  覆盖采样参数
     *   AUDIOCPP_MAX_TOKENS  覆盖 max_tokens 上限(诊断失控用)
     * voice_ref 通过 PCM 内联传递。 */
    char options[768];
    char extra[512];
    extra[0] = '\0';
    const char *seed_env = getenv("AUDIOCPP_SEED");
    const char *do_sample_env = getenv("AUDIOCPP_DO_SAMPLE");
    const char *temp_env = getenv("AUDIOCPP_TEMPERATURE");
    const char *topp_env = getenv("AUDIOCPP_TOP_P");
    const char *topk_env = getenv("AUDIOCPP_TOP_K");
    const char *maxtok_env = getenv("AUDIOCPP_MAX_TOKENS");
    const char *reppen_env = getenv("AUDIOCPP_REP_PENALTY");
    {
        size_t off = 0;
        #define APPEND(fmt, ...) do { \
            int _n = snprintf(extra + off, sizeof(extra) - off, fmt, ##__VA_ARGS__); \
            if (_n > 0) off += (size_t)_n; } while (0)
        if (seed_env)      APPEND(",\"seed\": \"%s\"", seed_env);
        if (do_sample_env) APPEND(",\"do_sample\": \"%s\"", do_sample_env);
        if (temp_env)      APPEND(",\"temperature\": \"%s\"", temp_env);
        if (topp_env)      APPEND(",\"top_p\": \"%s\"", topp_env);
        if (topk_env)      APPEND(",\"top_k\": \"%s\"", topk_env);
        if (maxtok_env)    APPEND(",\"max_tokens\": \"%s\"", maxtok_env);
        if (reppen_env)    APPEND(",\"repetition_penalty\": \"%s\"", reppen_env);
        #undef APPEND
    }
    snprintf(options, sizeof(options),
             "{\"reference_text\": \"%s\"%s}", reference_text, extra);
    printf("[capi] 开始合成: text=\"%s\" reference_text=\"%s\" options=%s\n",
           text, reference_text, options);
    printf("[capi] -----> 若此处后无输出且进程挂起/退出,即为崩溃点 <-----\n");
    fflush(stdout);

    t2 = now_seconds();
    audiocpp_audio_t *audio = p_tts_with_voice_ref(
        model, text, options, ref_pcm, ref_n, ref_sr, &err);
    t3 = now_seconds();
    free(ref_pcm);

    if (!audio) {
        fprintf(stderr, "FAIL[tts]: code=%d msg=%s (合成耗时 %.3fs)\n",
                err.code, err.message ? err.message : "(null)", t3 - t2);
        p_clear_error(&err);
        p_free_model(model);
        FREE_LIB(handle);
        return 5;
    }

    /* ---- 5. 写出 ---- */
    printf("[capi] 合成成功: %lld samples @ %d Hz (合成耗时 %.3fs)\n",
           (long long)audio->n_samples, audio->sample_rate, t3 - t2);
    t4 = now_seconds();
    if (p_write_wav(out_wav, audio->samples, audio->n_samples, audio->sample_rate) != 0) {
        fprintf(stderr, "WARN: 写出 %s 失败\n", out_wav);
    } else {
        printf("[capi] 已写出 %s (耗时 %.3fs)\n", out_wav, now_seconds() - t4);
    }

    printf("[capi] 总耗时: load=%.3fs tts=%.3fs\n", t1 - t0, t3 - t2);

    p_free_audio(audio);
    p_free_model(model);
    p_clear_error(&err);
    FREE_LIB(handle);
    printf("[capi] 完成,资源已释放\n");
    return 0;
}
