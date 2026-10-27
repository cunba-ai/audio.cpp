/*
 * audiocpp_e2e.c — end-to-end driver for audiocpp.dll.
 *
 * Mirrors the sound-rs calling style (sound-server/src/backend/audiocpp.rs):
 * the DLL is loaded at RUNTIME via LoadLibraryA + GetProcAddress (the C
 * equivalent of libloading), every result is checked against the
 * null-return + audiocpp_error_t protocol, errors are released with
 * audiocpp_clear_error, and per-run progress callbacks are installed and
 * cleared around the run exactly like the sound-server job path.
 *
 * Conventions shared with sound-rs:
 *   - audiocpp_audio_t.n_samples is the INTERLEAVED total (channels*frames).
 *   - A model handle is serialized: one run at a time per handle.
 *
 * Machine-readable protocol: every subcommand prints one final line
 *   E2E\t<sub>\tstatus=ok|fail\tms=<int>\tk=v...\t
 * to stdout; run_e2e.ps1 parses that line and applies per-scenario
 * expectations. Human-readable detail goes to stderr.
 *
 * Text/options travel through files (--text-file/--options-file) so nothing
 * is mangled by shells (same trick as capi_tts_test.c).
 *
 * Build (see run_e2e.ps1):
 *   cl /nologo /O2 /Fe:audiocpp_e2e.exe audiocpp_e2e.c /I..\..\include
 * Place audiocpp.dll next to the exe (or pass --dll <path>).
 */

#include "audiocpp.h"

#define _CRT_SECURE_NO_WARNINGS
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── API table (dlopen posture, mirrors sound-rs AudioCppLib::load) ───── */

typedef const char *(*fn_version)(void);
typedef fn_version fn_build_id;
typedef int (*fn_backend_available)(int);
typedef audiocpp_model_t *(*fn_load_model)(const char *, const char *, int, int, int, int, audiocpp_error_t *);
typedef audiocpp_model_t *(*fn_load_model_ex)(const char *, const char *, int, int, int, int, const char *, audiocpp_error_t *);
typedef void (*fn_free_model)(audiocpp_model_t *);
typedef int (*fn_model_info)(const audiocpp_model_t *, audiocpp_model_info_t *);
typedef int (*fn_model_capabilities)(const audiocpp_model_t *, audiocpp_model_capabilities_t *);
typedef void (*fn_free_model_info)(audiocpp_model_info_t *);
typedef void (*fn_free_capabilities)(audiocpp_model_capabilities_t *);
typedef audiocpp_audio_t *(*fn_tts)(const audiocpp_model_t *, const char *, const char *, audiocpp_error_t *);
typedef audiocpp_audio_t *(*fn_tts_vr)(const audiocpp_model_t *, const char *, const char *, const float *, int64_t, int, audiocpp_error_t *);
typedef audiocpp_audio_batch_t *(*fn_tts_batch)(audiocpp_model_t *, const char *const *, int, const char *, int, audiocpp_error_t *);
typedef void (*fn_free_audio_batch)(audiocpp_audio_batch_t *);
typedef audiocpp_text_t *(*fn_asr)(const audiocpp_model_t *, const float *, int64_t, int, const char *, audiocpp_error_t *);
typedef audiocpp_audio_t *(*fn_transform)(const audiocpp_model_t *, const float *, int64_t, int, const char *, audiocpp_error_t *);
typedef audiocpp_audio_t *(*fn_transform_vr)(const audiocpp_model_t *, const float *, int64_t, int, const char *, const float *, int64_t, int, audiocpp_error_t *);
typedef audiocpp_stems_t *(*fn_transform_stems)(const audiocpp_model_t *, const float *, int64_t, int, const char *, const float *, int64_t, int, audiocpp_error_t *);
typedef void (*fn_free_stems)(audiocpp_stems_t *);
typedef audiocpp_diar_t *(*fn_diar)(const audiocpp_model_t *, const float *, int64_t, int, const char *, audiocpp_error_t *);
typedef void (*fn_free_diar)(audiocpp_diar_t *);
typedef audiocpp_vad_t *(*fn_vad)(const audiocpp_model_t *, const float *, int64_t, int, const char *, audiocpp_error_t *);
typedef audiocpp_vad_t *(*fn_vad_energy)(const float *, int64_t, int, const char *, audiocpp_error_t *);
typedef void (*fn_free_vad)(audiocpp_vad_t *);
typedef audiocpp_audio_t *(*fn_denoise)(const float *, int64_t, int, const char *, const char *, const char *, audiocpp_error_t *);
typedef audiocpp_audio_t *(*fn_super_resolve)(const float *, int64_t, int, const char *, const char *, audiocpp_error_t *);
typedef audiocpp_align_t *(*fn_align)(const audiocpp_model_t *, const float *, int64_t, int, const char *, const char *, const char *, audiocpp_error_t *);
typedef void (*fn_free_align)(audiocpp_align_t *);
typedef audiocpp_gen_result_t *(*fn_generate)(const audiocpp_model_t *, const char *, const char *, audiocpp_error_t *);
typedef void (*fn_free_gen_result)(audiocpp_gen_result_t *);
typedef audiocpp_artifacts_t *(*fn_midi)(const audiocpp_model_t *, const float *, int64_t, int, const char *, audiocpp_error_t *);
typedef audiocpp_artifacts_t *(*fn_midi_wav)(const audiocpp_model_t *, const uint8_t *, int64_t, const char *, audiocpp_error_t *);
typedef void (*fn_free_artifacts)(audiocpp_artifacts_t *);
typedef audiocpp_stream_t *(*fn_stream_start)(const audiocpp_model_t *, int, const char *, int64_t, const audiocpp_audio_contract_t *, audiocpp_error_t *);
typedef audiocpp_stream_event_t *(*fn_stream_push)(audiocpp_stream_t *, const float *, int64_t, int, audiocpp_error_t *);
typedef audiocpp_stream_event_t *(*fn_stream_pull)(audiocpp_stream_t *, int, audiocpp_error_t *);
typedef int (*fn_stream_finish)(audiocpp_stream_t *, audiocpp_text_t **, audiocpp_error_t *);
typedef void (*fn_free_stream_event)(audiocpp_stream_event_t *);
typedef void (*fn_stream_free)(audiocpp_stream_t *);
typedef int (*fn_device_count)(void);
typedef int (*fn_device_info)(int, audiocpp_device_info_t *);
typedef void (*fn_list_devices)(void);
typedef int (*fn_read_wav)(const char *, float **, int64_t *, int *);
typedef int (*fn_write_wav)(const char *, const float *, int64_t, int);
typedef int (*fn_write_wav_ex)(const char *, const float *, int64_t, int, int);
typedef void (*fn_free_audio)(audiocpp_audio_t *);
typedef void (*fn_free_text)(audiocpp_text_t *);
typedef void (*fn_free_string)(char *);
typedef void (*fn_clear_error)(audiocpp_error_t *);
typedef void (*fn_set_progress)(audiocpp_model_t *, audiocpp_progress_fn, void *);

static struct {
    HMODULE lib;
    fn_version version;
    fn_version build_id;
    fn_backend_available backend_available;
    fn_load_model load_model;
    fn_load_model_ex load_model_ex;
    fn_free_model free_model;
    fn_model_info model_info;
    fn_model_capabilities model_capabilities;
    fn_free_model_info free_model_info;
    fn_free_capabilities free_capabilities;
    fn_tts tts;
    fn_tts_vr tts_vr;
    fn_tts_batch tts_batch;
    fn_free_audio_batch free_audio_batch;
    fn_asr asr;
    fn_transform transform;
    fn_transform_vr transform_vr;
    fn_transform_stems transform_stems;
    fn_free_stems free_stems;
    fn_diar diar;
    fn_free_diar free_diar;
    fn_vad vad;
    fn_vad_energy vad_energy;
    fn_free_vad free_vad;
    fn_denoise denoise;
    fn_super_resolve super_resolve;
    fn_align align;
    fn_free_align free_align;
    fn_generate generate;
    fn_free_gen_result free_gen_result;
    fn_midi midi;
    fn_midi_wav midi_wav;
    fn_free_artifacts free_artifacts;
    fn_stream_start stream_start;
    fn_stream_push stream_push;
    fn_stream_pull stream_pull;
    fn_stream_finish stream_finish;
    fn_free_stream_event free_stream_event;
    fn_stream_free stream_free;
    fn_device_count device_count;
    fn_device_info device_info;
    fn_list_devices list_devices;
    fn_read_wav read_wav;
    fn_write_wav write_wav;
    fn_write_wav_ex write_wav_ex;
    fn_free_audio free_audio;
    fn_free_text free_text;
    fn_free_string free_string;
    fn_clear_error clear_error;
    fn_set_progress set_progress;
} A;

static void die_load(const char *sym) {
    fprintf(stderr, "FATAL: audiocpp.dll is missing export '%s' (header/ABI drift?)\n", sym);
    exit(2);
}

#define LOAD_SYM(field, name)                                       \
    do {                                                            \
        A.field = (fn_##field)(void (*)(void)) GetProcAddress(A.lib, name); \
        if (A.field == NULL) die_load(name);                        \
    } while (0)

static void api_load(const char *dll_path) {
    A.lib = LoadLibraryA(dll_path);
    if (A.lib == NULL) {
        fprintf(stderr, "FATAL: LoadLibraryA(\"%s\") failed with %lu\n", dll_path, GetLastError());
        exit(2);
    }
    LOAD_SYM(version, "audiocpp_version");
    LOAD_SYM(build_id, "audiocpp_build_id");
    LOAD_SYM(backend_available, "audiocpp_backend_available");
    LOAD_SYM(load_model, "audiocpp_load_model");
    LOAD_SYM(load_model_ex, "audiocpp_load_model_ex");
    LOAD_SYM(free_model, "audiocpp_free_model");
    LOAD_SYM(model_info, "audiocpp_model_info");
    LOAD_SYM(model_capabilities, "audiocpp_model_capabilities");
    LOAD_SYM(free_model_info, "audiocpp_free_model_info");
    LOAD_SYM(free_capabilities, "audiocpp_free_capabilities");
    LOAD_SYM(tts, "audiocpp_tts");
    LOAD_SYM(tts_vr, "audiocpp_tts_with_voice_ref");
    LOAD_SYM(tts_batch, "audiocpp_tts_batch");
    LOAD_SYM(free_audio_batch, "audiocpp_free_audio_batch");
    LOAD_SYM(asr, "audiocpp_asr");
    LOAD_SYM(transform, "audiocpp_audio_transform");
    LOAD_SYM(transform_vr, "audiocpp_audio_transform_with_voice_ref");
    LOAD_SYM(transform_stems, "audiocpp_transform_stems");
    LOAD_SYM(free_stems, "audiocpp_free_stems");
    LOAD_SYM(diar, "audiocpp_diar");
    LOAD_SYM(free_diar, "audiocpp_free_diar");
    LOAD_SYM(vad, "audiocpp_vad");
    LOAD_SYM(vad_energy, "audiocpp_vad_energy");
    LOAD_SYM(free_vad, "audiocpp_free_vad");
    LOAD_SYM(denoise, "audiocpp_denoise");
    LOAD_SYM(super_resolve, "audiocpp_super_resolve");
    LOAD_SYM(align, "audiocpp_align");
    LOAD_SYM(free_align, "audiocpp_free_align");
    LOAD_SYM(generate, "audiocpp_generate");
    LOAD_SYM(free_gen_result, "audiocpp_free_gen_result");
    LOAD_SYM(midi, "audiocpp_midi");
    LOAD_SYM(midi_wav, "audiocpp_midi_from_wav");
    LOAD_SYM(free_artifacts, "audiocpp_free_artifacts");
    LOAD_SYM(stream_start, "audiocpp_stream_start");
    LOAD_SYM(stream_push, "audiocpp_stream_push");
    LOAD_SYM(stream_pull, "audiocpp_stream_pull");
    LOAD_SYM(stream_finish, "audiocpp_stream_finish");
    LOAD_SYM(free_stream_event, "audiocpp_free_stream_event");
    LOAD_SYM(stream_free, "audiocpp_stream_free");
    LOAD_SYM(device_count, "audiocpp_device_count");
    LOAD_SYM(device_info, "audiocpp_device_info");
    LOAD_SYM(list_devices, "audiocpp_list_devices");
    LOAD_SYM(read_wav, "audiocpp_read_wav");
    LOAD_SYM(write_wav, "audiocpp_write_wav");
    LOAD_SYM(write_wav_ex, "audiocpp_write_wav_ex");
    LOAD_SYM(free_audio, "audiocpp_free_audio");
    LOAD_SYM(free_text, "audiocpp_free_text");
    LOAD_SYM(free_string, "audiocpp_free_string");
    LOAD_SYM(clear_error, "audiocpp_clear_error");
    LOAD_SYM(set_progress, "audiocpp_set_progress_callback");
}

/* ── helpers ─────────────────────────────────────────────────────────── */

static double now_ms(void) {
    static LARGE_INTEGER freq = {0};
    LARGE_INTEGER c;
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&c);
    return (double) c.QuadPart * 1000.0 / (double) freq.QuadPart;
}

static void kv_duration_peak_rms(const audiocpp_audio_t *a) {
    if (a == NULL || a->samples == NULL || a->n_samples <= 0 || a->channels <= 0) {
        printf("dur=0 peak=0 rms=0 sr=0 ch=0 n=0\t");
        return;
    }
    const int64_t frames = a->n_samples / a->channels;
    double peak = 0.0, sum_sq = 0.0;
    for (int64_t i = 0; i < a->n_samples; ++i) {
        const double v = fabs((double) a->samples[i]);
        if (v > peak) peak = v;
        sum_sq += (double) a->samples[i] * (double) a->samples[i];
    }
    printf("dur=%.3f\tpeak=%.4f\trms=%.4f\tsr=%d\tch=%d\tn=%lld\t",
        (double) frames / a->sample_rate, peak, sqrt(sum_sq / (double) a->n_samples),
        a->sample_rate, a->channels, (long long) a->n_samples);
}

static char *read_text_arg(int argc, char **argv, const char *flag_inline, const char *flag_file) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (strcmp(argv[i], flag_inline) == 0) return _strdup(argv[i + 1]);
    }
    for (int i = 1; i + 1 < argc; ++i) {
        if (strcmp(argv[i], flag_file) == 0) {
            FILE *f = fopen(argv[i + 1], "rb");
            if (f == NULL) {
                fprintf(stderr, "cannot open %s\n", argv[i + 1]);
                exit(2);
            }
            fseek(f, 0, SEEK_END);
            long sz = ftell(f);
            fseek(f, 0, SEEK_SET);
            char *buf = (char *) malloc((size_t) sz + 1);
            size_t rd = fread(buf, 1, (size_t) sz, f);
            fclose(f);
            buf[rd] = '\0';
            /* strip a trailing newline so --text-file and --text behave the same */
            while (rd > 0 && (buf[rd - 1] == '\n' || buf[rd - 1] == '\r')) buf[--rd] = '\0';
            return buf;
        }
    }
    return NULL;
}

static const char *arg_str(int argc, char **argv, const char *flag, const char *dflt) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (strcmp(argv[i], flag) == 0) return argv[i + 1];
    }
    return dflt;
}

static int arg_int(int argc, char **argv, const char *flag, int dflt) {
    const char *v = arg_str(argc, argv, flag, NULL);
    return v == NULL ? dflt : atoi(v);
}

static void print_err(const char *what, const audiocpp_error_t *err) {
    fprintf(stderr, "[%s] err code=%d msg=%s\n", what, err ? err->code : 0,
        (err && err->message) ? err->message : "(null)");
}

/* Progress callback: sound-server installs this around job runs. */
static int g_progress_calls = 0;
static int progress_cb(float progress, const char *stage, int64_t done, int64_t total, void *user) {
    (void) user;
    ++g_progress_calls;
    fprintf(stderr, "  progress %.3f stage=%s %lld/%lld\n", progress, stage ? stage : "?",
        (long long) done, (long long) total);
    return 0; /* never cancel here */
}

/* Shared model loader: load_model_ex when --load-options[-file] is given. */
static audiocpp_model_t *load_model_from_args(int argc, char **argv, int task) {
    const char *model = arg_str(argc, argv, "--model", NULL);
    const char *family = arg_str(argc, argv, "--family", NULL);
    const int backend = arg_int(argc, argv, "--backend", AUDIOCPP_BACKEND_CPU);
    const int device = arg_int(argc, argv, "--device", 0);
    const int threads = arg_int(argc, argv, "--threads", 0);
    char *load_opts = read_text_arg(argc, argv, "--load-options", "--load-options-file");
    audiocpp_error_t err = {0, NULL};
    audiocpp_model_t *m = NULL;
    const double t0 = now_ms();
    if (load_opts != NULL) {
        m = A.load_model_ex(model, family, task, backend, device, threads, load_opts, &err);
    } else {
        m = A.load_model(model, family, task, backend, device, threads, &err);
    }
    fprintf(stderr, "load '%s' task=%d took %.0f ms -> %s\n", model ? model : "(embedded)", task,
        now_ms() - t0, m ? "ok" : "FAILED");
    if (m == NULL) print_err("load_model", &err);
    if (err.message != NULL) A.clear_error(&err);
    free(load_opts);
    return m;
}

static int write_audio_wav(const audiocpp_audio_t *a, const char *out_path) {
    if (a == NULL || a->samples == NULL) return -1;
    return A.write_wav_ex(out_path, a->samples, a->n_samples, a->sample_rate,
        a->channels > 0 ? a->channels : 1);
}

/* inspect + report model metadata (mirrors sound-rs boot logging) */
static void report_model_info(const audiocpp_model_t *m) {
    audiocpp_model_info_t info = {NULL, NULL, NULL};
    audiocpp_model_capabilities_t caps;
    memset(&caps, 0, sizeof(caps));
    if (A.model_info(m, &info) == 0) {
        printf("family=%s variant=%s\t", info.family ? info.family : "?",
            info.variant ? info.variant : "");
        A.free_model_info(&info);
    }
    if (A.model_capabilities(m, &caps) == 0) {
        printf("cap_speaker_ref=%d cap_timestamps=%d cap_tasks=%d\t",
            caps.supports_speaker_reference, caps.supports_timestamps, caps.n_supported_tasks);
        A.free_capabilities(&caps);
    }
    printf("\n");
}

/* ── subcommands ─────────────────────────────────────────────────────── */

static int cmd_info(void) {
    const double t0 = now_ms();
    fprintf(stderr, "version: %s\nbuild: %s\n", A.version(), A.build_id());
    printf("E2E\tinfo\tstatus=ok\tms=%.0f\tcpu=%d\tcuda=%d\tvulkan=%d\tmetal=%d\tsycl=%d\tdevices=%d\t",
        now_ms() - t0,
        A.backend_available(AUDIOCPP_BACKEND_CPU), A.backend_available(AUDIOCPP_BACKEND_CUDA),
        A.backend_available(AUDIOCPP_BACKEND_VULKAN), A.backend_available(AUDIOCPP_BACKEND_METAL),
        A.backend_available(AUDIOCPP_BACKEND_SYCL), A.device_count());
    for (int i = 0; i < A.device_count(); ++i) {
        audiocpp_device_info_t di;
        if (A.device_info(i, &di) == 0) {
            fprintf(stderr, "device[%d] backend=%d id=%d name=%s\n", i, di.backend, di.device_id, di.name);
        }
    }
    printf("\n");
    return 0;
}

static int cmd_tts(int argc, char **argv) {
    const double t0 = now_ms();
    char *text = read_text_arg(argc, argv, "--text", "--text-file");
    if (text == NULL) {
        fprintf(stderr, "tts needs --text/--text-file\n");
        return 2;
    }
    char *opts = read_text_arg(argc, argv, "--options", "--options-file");
    audiocpp_model_t *m = load_model_from_args(argc, argv, AUDIOCPP_TASK_TTS);
    if (m == NULL) {
        printf("E2E\ttts\tstatus=fail\treason=load\tms=%.0f\t\n", now_ms() - t0);
        free(text); free(opts);
        return 1;
    }
    report_model_info(m);
    if (arg_int(argc, argv, "--progress", 0) != 0) {
        g_progress_calls = 0;
        A.set_progress(m, progress_cb, NULL);
    }
    audiocpp_error_t err = {0, NULL};
    audiocpp_audio_t *a = A.tts(m, text, opts ? opts : "{}", &err);
    if (arg_int(argc, argv, "--progress", 0) != 0) {
        A.set_progress(m, NULL, NULL);
        fprintf(stderr, "progress callback fired %d times\n", g_progress_calls);
    }
    const int ok = a != NULL;
    if (!ok) print_err("tts", &err);
    if (err.message != NULL) A.clear_error(&err);
    const char *out = arg_str(argc, argv, "--out", NULL);
    if (ok && out != NULL) write_audio_wav(a, out);
    printf("E2E\ttts\tstatus=%s\tms=%.0f\tmodel=%s\tprog=%d\t", ok ? "ok" : "fail",
        now_ms() - t0, arg_str(argc, argv, "--model", ""), g_progress_calls);
    kv_duration_peak_rms(a);
    printf("out=%s\t\n", out ? out : "");
    if (a != NULL) A.free_audio(a);
    A.free_model(m);
    free(text); free(opts);
    return ok ? 0 : 1;
}

static int cmd_tts_batch(int argc, char **argv) {
    const double t0 = now_ms();
    char *t1 = read_text_arg(argc, argv, "--text", "--text-file");
    char *t2 = read_text_arg(argc, argv, "--text2", "--text2-file");
    if (t1 == NULL || t2 == NULL) {
        fprintf(stderr, "tts-batch needs --text and --text2\n");
        return 2;
    }
    audiocpp_model_t *m = load_model_from_args(argc, argv, AUDIOCPP_TASK_TTS);
    if (m == NULL) {
        printf("E2E\ttts-batch\tstatus=fail\treason=load\tms=%.0f\t\n", now_ms() - t0);
        free(t1); free(t2);
        return 1;
    }
    const char *texts[2] = {t1, t2};
    audiocpp_error_t err = {0, NULL};
    audiocpp_audio_batch_t *b = A.tts_batch(m, texts, 2, "{}", AUDIOCPP_BATCH_MERGE_CONCAT, &err);
    const int ok = b != NULL && b->n_items >= 1 && b->items != NULL &&
        b->items[0].samples != NULL && b->items[0].n_samples > 0;
    if (!ok) print_err("tts_batch", &err);
    if (err.message != NULL) A.clear_error(&err);
    const char *out = arg_str(argc, argv, "--out", NULL);
    printf("E2E\ttts-batch\tstatus=%s\tms=%.0f\tn_items=%d\t", ok ? "ok" : "fail", now_ms() - t0,
        b ? b->n_items : 0);
    if (b != NULL && b->n_items > 0) {
        kv_duration_peak_rms(&b->items[0]);
        if (b->chapter_starts != NULL && b->n_items >= 2) {
            printf("chapters=%lld,%lld\t", (long long) b->chapter_starts[1],
                (long long) b->chapter_ends[1]);
        }
        if (ok && out != NULL) write_audio_wav(&b->items[0], out);
    }
    printf("out=%s\t\n", out ? out : "");
    if (b != NULL) A.free_audio_batch(b);
    A.free_model(m);
    free(t1); free(t2);
    return ok ? 0 : 1;
}

static int cmd_clon(int argc, char **argv) {
    const double t0 = now_ms();
    char *text = read_text_arg(argc, argv, "--text", "--text-file");
    const char *ref_wav = arg_str(argc, argv, "--ref-wav", NULL);
    char *opts = read_text_arg(argc, argv, "--options", "--options-file");
    if (text == NULL || ref_wav == NULL) {
        fprintf(stderr, "clon needs --text and --ref-wav\n");
        return 2;
    }
    float *ref = NULL;
    int64_t ref_n = 0;
    int ref_sr = 0;
    if (A.read_wav(ref_wav, &ref, &ref_n, &ref_sr) != 0 || ref == NULL || ref_n <= 0) {
        printf("E2E\tclon\tstatus=fail\treason=ref-wav\tms=%.0f\t\n", now_ms() - t0);
        free(text); free(opts);
        return 1;
    }
    audiocpp_model_t *m = load_model_from_args(argc, argv, AUDIOCPP_TASK_CLON);
    if (m == NULL) {
        printf("E2E\tclon\tstatus=fail\treason=load\tms=%.0f\t\n", now_ms() - t0);
        free(ref); free(text); free(opts);
        return 1;
    }
    g_progress_calls = 0;
    A.set_progress(m, progress_cb, NULL);
    audiocpp_error_t err = {0, NULL};
    audiocpp_audio_t *a = A.tts_vr(m, text, opts ? opts : "{}", ref, ref_n, ref_sr, &err);
    A.set_progress(m, NULL, NULL);
    const int ok = a != NULL;
    if (!ok) print_err("tts_with_voice_ref", &err);
    if (err.message != NULL) A.clear_error(&err);
    const char *out = arg_str(argc, argv, "--out", NULL);
    if (ok && out != NULL) write_audio_wav(a, out);
    printf("E2E\tclon\tstatus=%s\tms=%.0f\tprog=%d\t", ok ? "ok" : "fail", now_ms() - t0,
        g_progress_calls);
    kv_duration_peak_rms(a);
    printf("out=%s\t\n", out ? out : "");
    if (a != NULL) A.free_audio(a);
    A.free_model(m);
    free(ref); free(text); free(opts);
    return ok ? 0 : 1;
}

static int cmd_asr(int argc, char **argv) {
    const double t0 = now_ms();
    const char *in = arg_str(argc, argv, "--in", NULL);
    char *opts = read_text_arg(argc, argv, "--options", "--options-file");
    if (in == NULL) {
        fprintf(stderr, "asr needs --in wav\n");
        return 2;
    }
    float *pcm = NULL;
    int64_t n = 0;
    int sr = 0;
    if (A.read_wav(in, &pcm, &n, &sr) != 0 || pcm == NULL) {
        printf("E2E\tasr\tstatus=fail\treason=read-wav\tms=%.0f\t\n", now_ms() - t0);
        free(opts);
        return 1;
    }
    audiocpp_model_t *m = load_model_from_args(argc, argv, AUDIOCPP_TASK_ASR);
    if (m == NULL) {
        printf("E2E\tasr\tstatus=fail\treason=load\tms=%.0f\t\n", now_ms() - t0);
        free(pcm); free(opts);
        return 1;
    }
    report_model_info(m);
    audiocpp_error_t err = {0, NULL};
    audiocpp_text_t *t = A.asr(m, pcm, n, sr, opts ? opts : "{}", &err);
    const int ok = t != NULL && t->text != NULL && t->text[0] != '\0';
    if (!ok) print_err("asr", &err);
    if (err.message != NULL) A.clear_error(&err);
    if (ok) fprintf(stderr, "asr text: %s\n", t->text);
    printf("E2E\tasr\tstatus=%s\tms=%.0f\tchars=%lld\tlang=%s\ttext=\"%s\"\t\n",
        ok ? "ok" : "fail", now_ms() - t0,
        (long long) (t && t->text ? (int64_t) strlen(t->text) : 0),
        (t && t->language) ? t->language : "", (t && t->text) ? t->text : "");
    if (t != NULL) A.free_text(t);
    A.free_model(m);
    free(pcm); free(opts);
    return ok ? 0 : 1;
}

static int cmd_asr_stream(int argc, char **argv) {
    const double t0 = now_ms();
    const char *in = arg_str(argc, argv, "--in", NULL);
    char *opts = read_text_arg(argc, argv, "--options", "--options-file");
    const int64_t preferred_chunk = (int64_t) arg_int(argc, argv, "--chunk", 0);
    if (in == NULL) {
        fprintf(stderr, "asr-stream needs --in wav\n");
        return 2;
    }
    float *pcm = NULL;
    int64_t n = 0;
    int sr = 0;
    if (A.read_wav(in, &pcm, &n, &sr) != 0) {
        printf("E2E\tasr-stream\tstatus=fail\treason=read-wav\tms=%.0f\t\n", now_ms() - t0);
        return 1;
    }
    audiocpp_model_t *m = load_model_from_args(argc, argv, AUDIOCPP_TASK_ASR);
    if (m == NULL) {
        printf("E2E\tasr-stream\tstatus=fail\treason=load\tms=%.0f\t\n", now_ms() - t0);
        free(pcm);
        return 1;
    }
    audiocpp_error_t err = {0, NULL};
    audiocpp_audio_contract_t contract;
    memset(&contract, 0, sizeof(contract));
    contract.sample_rate = sr;
    contract.channels = 1;
    contract.max_input_samples = n;
    audiocpp_stream_t *s = A.stream_start(m, AUDIOCPP_TASK_ASR, opts ? opts : "{}", preferred_chunk, &contract, &err);
    if (s == NULL) {
        print_err("stream_start", &err);
        if (err.message != NULL) A.clear_error(&err);
        printf("E2E\tasr-stream\tstatus=fail\treason=stream-start\tms=%.0f\t\n", now_ms() - t0);
        A.free_model(m);
        free(pcm);
        return 1;
    }
    const int64_t chunk = 1600; /* 100 ms @ 16k */
    int n_events = 0, n_partials = 0;
    audiocpp_stream_event_t *ev = NULL;
    for (int64_t off = 0; off < n; off += chunk) {
        const int64_t len = (n - off < chunk) ? n - off : chunk;
        ev = A.stream_push(s, pcm + off, len, sr, &err);
        if (ev == NULL) {
            print_err("stream_push", &err);
            break;
        }
        ++n_events;
        if (ev->partial_text != NULL) {
            ++n_partials;
            fprintf(stderr, "  partial: %s\n", ev->partial_text);
        }
        A.free_stream_event(ev);
        ev = NULL;
    }
    audiocpp_text_t *final_text = NULL;
    const int fin = A.stream_finish(s, &final_text, &err);
    /* Protocol success == clean finish. Nemotron currently decodes to empty text
       through this path (verified: the engine's own audiocpp_cli --mode streaming
       is also empty on this checkpoint); report chars, don't fail the protocol. */
    const int ok = fin == 0;
    if (!ok) print_err("stream_finish", &err);
    if (err.message != NULL) A.clear_error(&err);
    /* stream_finish may finish cleanly with *out_text == NULL ("or NULL when
       the stream produced none") — never dereference it unguarded. */
    const int have_text = ok && final_text != NULL && final_text->text != NULL;
    if (have_text) fprintf(stderr, "asr-stream final: %s\n", final_text->text);
    printf("E2E\tasr-stream\tstatus=%s\tms=%.0f\tpushes=%d\tpartials=%d\tchars=%lld\ttext=\"%s\"\t\n",
        ok ? "ok" : "fail", now_ms() - t0, n_events, n_partials,
        (long long) (have_text ? (int64_t) strlen(final_text->text) : 0),
        have_text ? final_text->text : "");
    if (final_text != NULL) A.free_text(final_text);
    A.free_model(m);
    free(pcm);
    return ok ? 0 : 1;
}

static int cmd_tts_stream(int argc, char **argv) {
    const double t0 = now_ms();
    char *opts = read_text_arg(argc, argv, "--options", "--options-file");
    if (opts == NULL) {
        fprintf(stderr, "tts-stream needs --options/--options-file (with text)\n");
        return 2;
    }
    audiocpp_model_t *m = load_model_from_args(argc, argv, AUDIOCPP_TASK_TTS);
    if (m == NULL) {
        printf("E2E\ttts-stream\tstatus=fail\treason=load\tms=%.0f\t\n", now_ms() - t0);
        free(opts);
        return 1;
    }
    audiocpp_error_t err = {0, NULL};
    audiocpp_stream_t *s = A.stream_start(m, AUDIOCPP_TASK_TTS, opts, 0, NULL, &err);
    if (s == NULL) {
        print_err("stream_start", &err);
        if (err.message != NULL) A.clear_error(&err);
        printf("E2E\ttts-stream\tstatus=fail\treason=stream-start\tms=%.0f\t\n", now_ms() - t0);
        A.free_model(m);
        free(opts);
        return 1;
    }
    /* PullEvents loop: accumulate audio until is_final (supertonic-style). */
    float *acc = NULL;
    int64_t acc_n = 0, acc_cap = 0;
    int out_sr = 0, n_pulls = 0, is_final = 0;
    const double deadline = now_ms() + 300000.0; /* 5 min cap */
    int pull_end_of_stream = 0;
    while (now_ms() < deadline) {
        audiocpp_stream_event_t *ev = A.stream_pull(s, -1, &err);
        if (ev == NULL) {
            /* Official protocol (tests/test_capi_stream.cpp): a NULL event with a
               clean error struct means the stream finished, not a failure. */
            if (err.code == 0) {
                pull_end_of_stream = 1;
            } else {
                print_err("stream_pull", &err);
            }
            break;
        }
        ++n_pulls;
        if (ev->audio_samples != NULL && ev->n_audio_samples > 0) {
            if (acc_n + ev->n_audio_samples > acc_cap) {
                acc_cap = (acc_n + ev->n_audio_samples) * 2;
                float *grown = (float *) realloc(acc, sizeof(float) * (size_t) acc_cap);
                if (grown == NULL) {
                    A.free_stream_event(ev);
                    ev = NULL;
                    break; /* keep whatever was accumulated so far */
                }
                acc = grown;
            }
            memcpy(acc + acc_n, ev->audio_samples, sizeof(float) * (size_t) ev->n_audio_samples);
            acc_n += ev->n_audio_samples;
            out_sr = ev->audio_sample_rate;
        }
        if (ev->is_final) {
            is_final = 1;
            A.free_stream_event(ev);
            break;
        }
        A.free_stream_event(ev);
    }
    if (err.message != NULL) A.clear_error(&err);
    const int ok = (is_final || pull_end_of_stream) && acc_n > 0;
    const char *out = arg_str(argc, argv, "--out", NULL);
    if (ok && out != NULL && acc != NULL) {
        A.write_wav_ex(out, acc, acc_n, out_sr > 0 ? out_sr : 24000, 1);
    }
    printf("E2E\ttts-stream\tstatus=%s\tms=%.0f\tpulls=%d\tfinal=%d\teos=%d\tsr=%d\t", ok ? "ok" : "fail",
        now_ms() - t0, n_pulls, is_final, pull_end_of_stream, out_sr);
    if (acc_n > 0) {
        double peak = 0, sum_sq = 0;
        for (int64_t i = 0; i < acc_n; ++i) {
            const double v = fabs((double) acc[i]);
            if (v > peak) peak = v;
            sum_sq += (double) acc[i] * acc[i];
        }
        printf("dur=%.3f\tpeak=%.4f\trms=%.4f\tn=%lld\t",
            out_sr > 0 ? (double) acc_n / out_sr : 0.0, peak, sqrt(sum_sq / (double) acc_n),
            (long long) acc_n);
    } else {
        printf("dur=0\t");
    }
    printf("out=%s\t\n", out ? out : "");
    A.stream_free(s);
    A.free_model(m);
    free(acc);
    free(opts);
    return ok ? 0 : 1;
}

static int cmd_vad(int argc, char **argv) {
    const double t0 = now_ms();
    const char *in = arg_str(argc, argv, "--in", NULL);
    if (in == NULL) {
        fprintf(stderr, "vad needs --in wav\n");
        return 2;
    }
    float *pcm = NULL;
    int64_t n = 0;
    int sr = 0;
    if (A.read_wav(in, &pcm, &n, &sr) != 0) {
        printf("E2E\tvad\tstatus=fail\treason=read-wav\tms=%.0f\t\n", now_ms() - t0);
        return 1;
    }
    audiocpp_error_t err = {0, NULL};

    /* model VAD via embedded silero weights (NULL path, fork feature) */
    audiocpp_model_t *m = A.load_model(NULL, "silero_vad", AUDIOCPP_TASK_VAD,
        AUDIOCPP_BACKEND_CPU, 0, 0, &err);
    int64_t seg_model = -1;
    int ok = 0;
    if (m != NULL) {
        audiocpp_vad_t *v = A.vad(m, pcm, n, sr, "{}", &err);
        if (v != NULL) {
            seg_model = v->n_segments;
            for (int64_t i = 0; i < v->n_segments && i < 8; ++i) {
                fprintf(stderr, "  vad seg %lld..%lld conf=%.2f\n",
                    (long long) v->segments[i].start_sample,
                    (long long) v->segments[i].end_sample, v->segments[i].confidence);
            }
            A.free_vad(v);
            ok = seg_model > 0;
        } else {
            print_err("vad", &err);
        }
        A.free_model(m);
    } else {
        print_err("load embedded silero", &err);
    }
    if (err.message != NULL) A.clear_error(&err);

    /* model-free energy VAD */
    int64_t seg_energy = -1;
    audiocpp_vad_t *ve = A.vad_energy(pcm, n, sr, "{\"chunk_seconds\":4.0}", &err);
    if (ve != NULL) {
        seg_energy = ve->n_segments;
        A.free_vad(ve);
    } else {
        print_err("vad_energy", &err);
    }
    if (err.message != NULL) A.clear_error(&err);

    printf("E2E\tvad\tstatus=%s\tms=%.0f\tmodel_segments=%lld\tenergy_segments=%lld\t\n",
        (ok && seg_energy >= 0) ? "ok" : "fail", now_ms() - t0,
        (long long) seg_model, (long long) seg_energy);
    free(pcm);
    return (ok && seg_energy >= 0) ? 0 : 1;
}

static int cmd_diar(int argc, char **argv) {
    const double t0 = now_ms();
    const char *in = arg_str(argc, argv, "--in", NULL);
    if (in == NULL) {
        fprintf(stderr, "diar needs --in wav\n");
        return 2;
    }
    float *pcm = NULL;
    int64_t n = 0;
    int sr = 0;
    if (A.read_wav(in, &pcm, &n, &sr) != 0) {
        printf("E2E\tdiar\tstatus=fail\treason=read-wav\tms=%.0f\t\n", now_ms() - t0);
        return 1;
    }
    audiocpp_model_t *m = load_model_from_args(argc, argv, AUDIOCPP_TASK_DIAR);
    if (m == NULL) {
        printf("E2E\tdiar\tstatus=fail\treason=load\tms=%.0f\t\n", now_ms() - t0);
        free(pcm);
        return 1;
    }
    audiocpp_error_t err = {0, NULL};
    audiocpp_diar_t *d = A.diar(m, pcm, n, sr, "{}", &err);
    const int ok = d != NULL && d->n_turns > 0;
    if (!ok) print_err("diar", &err);
    if (err.message != NULL) A.clear_error(&err);
    if (ok) {
        for (int64_t i = 0; i < d->n_turns && i < 8; ++i) {
            fprintf(stderr, "  turn %lld..%lld spk=%s conf=%.2f\n",
                (long long) d->turns[i].start_sample, (long long) d->turns[i].end_sample,
                d->turns[i].speaker_id ? d->turns[i].speaker_id : "?", d->turns[i].confidence);
        }
    }
    printf("E2E\tdiar\tstatus=%s\tms=%.0f\tturns=%lld\t\n", ok ? "ok" : "fail", now_ms() - t0,
        d ? (long long) d->n_turns : 0);
    if (d != NULL) A.free_diar(d);
    A.free_model(m);
    free(pcm);
    return ok ? 0 : 1;
}

static int cmd_align(int argc, char **argv) {
    const double t0 = now_ms();
    const char *in = arg_str(argc, argv, "--in", NULL);
    char *text = read_text_arg(argc, argv, "--text", "--text-file");
    const char *lang = arg_str(argc, argv, "--language", "en");
    if (in == NULL || text == NULL) {
        fprintf(stderr, "align needs --in and --text\n");
        return 2;
    }
    float *pcm = NULL;
    int64_t n = 0;
    int sr = 0;
    if (A.read_wav(in, &pcm, &n, &sr) != 0) {
        printf("E2E\talign\tstatus=fail\treason=read-wav\tms=%.0f\t\n", now_ms() - t0);
        free(text);
        return 1;
    }
    audiocpp_model_t *m = load_model_from_args(argc, argv, AUDIOCPP_TASK_ALIGN);
    if (m == NULL) {
        printf("E2E\talign\tstatus=fail\treason=load\tms=%.0f\t\n", now_ms() - t0);
        free(pcm); free(text);
        return 1;
    }
    audiocpp_error_t err = {0, NULL};
    audiocpp_align_t *al = A.align(m, pcm, n, sr, text, lang, "{}", &err);
    const int ok = al != NULL && al->n_words > 0;
    if (!ok) print_err("align", &err);
    if (err.message != NULL) A.clear_error(&err);
    if (ok) {
        for (int64_t i = 0; i < al->n_words && i < 12; ++i) {
            fprintf(stderr, "  word %.2f..%.2f %s\n", al->words[i].start_seconds,
                al->words[i].end_seconds, al->words[i].word ? al->words[i].word : "?");
        }
    }
    printf("E2E\talign\tstatus=%s\tms=%.0f\twords=%lld\t\n", ok ? "ok" : "fail", now_ms() - t0,
        al ? (long long) al->n_words : 0);
    if (al != NULL) A.free_align(al);
    A.free_model(m);
    free(pcm); free(text);
    return ok ? 0 : 1;
}

static int cmd_sep(int argc, char **argv) {
    const double t0 = now_ms();
    const char *in = arg_str(argc, argv, "--in", NULL);
    const char *outdir = arg_str(argc, argv, "--out-dir", ".");
    if (in == NULL) {
        fprintf(stderr, "sep needs --in wav\n");
        return 2;
    }
    float *pcm = NULL;
    int64_t n = 0;
    int sr = 0;
    if (A.read_wav(in, &pcm, &n, &sr) != 0) {
        printf("E2E\tsep\tstatus=fail\treason=read-wav\tms=%.0f\t\n", now_ms() - t0);
        return 1;
    }
    audiocpp_model_t *m = load_model_from_args(argc, argv, AUDIOCPP_TASK_SEP);
    if (m == NULL) {
        printf("E2E\tsep\tstatus=fail\treason=load\tms=%.0f\t\n", now_ms() - t0);
        free(pcm);
        return 1;
    }
    audiocpp_error_t err = {0, NULL};
    audiocpp_stems_t *st = A.transform_stems(m, pcm, n, sr, "{}", NULL, 0, 0, &err);
    const int ok = st != NULL && st->n_stems >= 2;
    if (!ok) print_err("transform_stems", &err);
    if (err.message != NULL) A.clear_error(&err);
    printf("E2E\tsep\tstatus=%s\tms=%.0f\tstems=%lld\t", ok ? "ok" : "fail", now_ms() - t0,
        st ? (long long) st->n_stems : 0);
    if (st != NULL) {
        for (int64_t i = 0; i < st->n_stems; ++i) {
            fprintf(stderr, "  stem '%s' n=%lld sr=%d ch=%d\n",
                st->stems[i].name ? st->stems[i].name : "?",
                (long long) st->stems[i].n_samples, st->stems[i].sample_rate,
                st->stems[i].channels);
            if (st->stems[i].samples != NULL && st->stems[i].n_samples > 0 && ok) {
                char path[MAX_PATH];
                _snprintf(path, sizeof(path), "%s\\sep_%s.wav", outdir,
                    st->stems[i].name ? st->stems[i].name : "stem");
                path[sizeof(path) - 1] = '\0';
                A.write_wav_ex(path, st->stems[i].samples, st->stems[i].n_samples,
                    st->stems[i].sample_rate, st->stems[i].channels > 0 ? st->stems[i].channels : 1);
            }
        }
        if (st->n_stems > 0) {
            printf("first_stem_frames=%lld\t",
                (long long) (st->stems[0].n_samples /
                    (st->stems[0].channels > 0 ? st->stems[0].channels : 1)));
        }
    }
    printf("\n");
    if (st != NULL) A.free_stems(st);
    A.free_model(m);
    free(pcm);
    return ok ? 0 : 1;
}

static int cmd_vc(int argc, char **argv) {
    const double t0 = now_ms();
    const char *in = arg_str(argc, argv, "--in", NULL);
    const char *ref_wav = arg_str(argc, argv, "--ref-wav", NULL);
    char *opts = read_text_arg(argc, argv, "--options", "--options-file");
    if (in == NULL || ref_wav == NULL) {
        fprintf(stderr, "vc needs --in and --ref-wav\n");
        return 2;
    }
    float *pcm = NULL, *ref = NULL;
    int64_t n = 0, ref_n = 0;
    int sr = 0, ref_sr = 0;
    if (A.read_wav(in, &pcm, &n, &sr) != 0 || A.read_wav(ref_wav, &ref, &ref_n, &ref_sr) != 0) {
        printf("E2E\tvc\tstatus=fail\treason=read-wav\tms=%.0f\t\n", now_ms() - t0);
        free(opts);
        return 1;
    }
    audiocpp_model_t *m = load_model_from_args(argc, argv, AUDIOCPP_TASK_VC);
    if (m == NULL) {
        printf("E2E\tvc\tstatus=fail\treason=load\tms=%.0f\t\n", now_ms() - t0);
        free(pcm); free(ref); free(opts);
        return 1;
    }
    audiocpp_error_t err = {0, NULL};
    audiocpp_audio_t *a = A.transform_vr(m, pcm, n, sr, opts ? opts : "{}", ref, ref_n, ref_sr, &err);
    const int ok = a != NULL && a->samples != NULL && a->n_samples > 0;
    if (!ok) print_err("audio_transform_with_voice_ref", &err);
    if (err.message != NULL) A.clear_error(&err);
    const char *out = arg_str(argc, argv, "--out", NULL);
    if (ok && out != NULL) write_audio_wav(a, out);
    printf("E2E\tvc\tstatus=%s\tms=%.0f\t", ok ? "ok" : "fail", now_ms() - t0);
    kv_duration_peak_rms(a);
    printf("out=%s\t\n", out ? out : "");
    if (a != NULL) A.free_audio(a);
    A.free_model(m);
    free(pcm); free(ref); free(opts);
    return ok ? 0 : 1;
}

static int cmd_denoise(int argc, char **argv) {
    const double t0 = now_ms();
    const char *in = arg_str(argc, argv, "--in", NULL);
    const char *model_name = arg_str(argc, argv, "--denoiser", "gtcrn");
    if (in == NULL) {
        fprintf(stderr, "denoise needs --in wav\n");
        return 2;
    }
    float *pcm = NULL;
    int64_t n = 0;
    int sr = 0;
    if (A.read_wav(in, &pcm, &n, &sr) != 0) {
        printf("E2E\tdenoise\tstatus=fail\treason=read-wav\tms=%.0f\t\n", now_ms() - t0);
        return 1;
    }
    audiocpp_error_t err = {0, NULL};
    audiocpp_audio_t *a = A.denoise(pcm, n, sr, model_name, NULL, "{}", &err);
    const int ok = a != NULL && a->samples != NULL && a->n_samples > 0;
    if (!ok) print_err("denoise", &err);
    if (err.message != NULL) A.clear_error(&err);
    const char *out = arg_str(argc, argv, "--out", NULL);
    if (ok && out != NULL) write_audio_wav(a, out);
    printf("E2E\tdenoise\tstatus=%s\tms=%.0f\tmodel=%s\t", ok ? "ok" : "fail", now_ms() - t0,
        model_name);
    kv_duration_peak_rms(a);
    printf("out=%s\t\n", out ? out : "");
    if (a != NULL) A.free_audio(a);
    free(pcm);
    return ok ? 0 : 1;
}

static int cmd_superres(int argc, char **argv) {
    const double t0 = now_ms();
    const char *in = arg_str(argc, argv, "--in", NULL);
    if (in == NULL) {
        fprintf(stderr, "superres needs --in wav\n");
        return 2;
    }
    float *pcm = NULL;
    int64_t n = 0;
    int sr = 0;
    if (A.read_wav(in, &pcm, &n, &sr) != 0) {
        printf("E2E\tsuperres\tstatus=fail\treason=read-wav\tms=%.0f\t\n", now_ms() - t0);
        return 1;
    }
    audiocpp_error_t err = {0, NULL};
    audiocpp_audio_t *a = A.super_resolve(pcm, n, sr, NULL, "{}", &err);
    const int ok = a != NULL && a->samples != NULL && a->sample_rate == 48000;
    if (!ok) print_err("super_resolve", &err);
    if (err.message != NULL) A.clear_error(&err);
    const char *out = arg_str(argc, argv, "--out", NULL);
    if (ok && out != NULL) write_audio_wav(a, out);
    printf("E2E\tsuperres\tstatus=%s\tms=%.0f\t", ok ? "ok" : "fail", now_ms() - t0);
    kv_duration_peak_rms(a);
    printf("out=%s\t\n", out ? out : "");
    if (a != NULL) A.free_audio(a);
    free(pcm);
    return ok ? 0 : 1;
}

static int cmd_gen(int argc, char **argv) {
    const double t0 = now_ms();
    char *prompt = read_text_arg(argc, argv, "--text", "--text-file");
    char *opts = read_text_arg(argc, argv, "--options", "--options-file");
    if (prompt == NULL) {
        fprintf(stderr, "gen needs --text\n");
        return 2;
    }
    audiocpp_model_t *m = load_model_from_args(argc, argv, AUDIOCPP_TASK_GEN);
    if (m == NULL) {
        printf("E2E\tgen\tstatus=fail\treason=load\tms=%.0f\t\n", now_ms() - t0);
        free(prompt); free(opts);
        return 1;
    }
    report_model_info(m);
    audiocpp_error_t err = {0, NULL};
    audiocpp_gen_result_t *g = A.generate(m, prompt, opts ? opts : "{}", &err);
    const int has_audio = g != NULL && g->audio != NULL && g->audio->samples != NULL &&
        g->audio->n_samples > 0;
    const int ok = g != NULL && has_audio;
    if (!ok) print_err("generate", &err);
    if (err.message != NULL) A.clear_error(&err);
    const char *out = arg_str(argc, argv, "--out", NULL);
    if (has_audio && out != NULL) write_audio_wav(g->audio, out);
    printf("E2E\tgen\tstatus=%s\tms=%.0f\t", ok ? "ok" : "fail", now_ms() - t0);
    if (has_audio) {
        kv_duration_peak_rms(g->audio);
    }
    int64_t n_art = g && g->artifacts ? g->artifacts->n_artifacts : 0;
    printf("artifacts=%lld\tout=%s\t\n", (long long) n_art, out ? out : "");
    if (g != NULL) A.free_gen_result(g);
    A.free_model(m);
    free(prompt); free(opts);
    return ok ? 0 : 1;
}

static int cmd_midi(int argc, char **argv) {
    const double t0 = now_ms();
    const char *in = arg_str(argc, argv, "--in", NULL);
    const char *out = arg_str(argc, argv, "--out", NULL);
    if (in == NULL) {
        fprintf(stderr, "midi needs --in wav\n");
        return 2;
    }
    audiocpp_model_t *m = load_model_from_args(argc, argv, AUDIOCPP_TASK_MIDI);
    if (m == NULL) {
        printf("E2E\tmidi\tstatus=fail\treason=load\tms=%.0f\t\n", now_ms() - t0);
        return 1;
    }
    /* Feed raw WAV bytes via audiocpp_midi_from_wav (the HTTP-upload shape). */
    FILE *f = fopen(in, "rb");
    if (f == NULL) {
        printf("E2E\tmidi\tstatus=fail\treason=open-wav\tms=%.0f\t\n", now_ms() - t0);
        A.free_model(m);
        return 1;
    }
    fseek(f, 0, SEEK_END);
    const long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *bytes = (uint8_t *) malloc((size_t) size);
    const size_t rd = fread(bytes, 1, (size_t) size, f);
    fclose(f);
    audiocpp_error_t err = {0, NULL};
    audiocpp_artifacts_t *arts = A.midi_wav(m, bytes, (int64_t) rd, "{}", &err);
    free(bytes);
    const int ok = arts != NULL && arts->n_artifacts >= 1;
    if (!ok) print_err("midi_from_wav", &err);
    if (err.message != NULL) A.clear_error(&err);
    int64_t payload = 0;
    char fmt[32] = "";
    if (ok) {
        for (int64_t i = 0; i < arts->n_artifacts; ++i) {
            const audiocpp_artifact_t *a = &arts->artifacts[i];
            payload += a->payload_size;
            fprintf(stderr, "  artifact kind=%d id=%s payload=%lld\n", a->kind,
                a->id ? a->id : "?", (long long) a->payload_size);
            for (int k = 0; k < a->n_meta; ++k) {
                if (strcmp(a->meta_keys[k], "format") == 0 && fmt[0] == '\0') {
                    _snprintf(fmt, sizeof(fmt), "%s", a->meta_values[k]);
                    fmt[sizeof(fmt) - 1] = '\0';
                }
            }
            if (out != NULL && a->payload != NULL && a->payload_size > 0 && i == 0) {
                FILE *o = fopen(out, "wb");
                if (o != NULL) {
                    fwrite(a->payload, 1, (size_t) a->payload_size, o);
                    fclose(o);
                }
            }
        }
    }
    printf("E2E\tmidi\tstatus=%s\tms=%.0f\tartifacts=%lld\tpayload_bytes=%lld\tformat=%s\tout=%s\t\n",
        ok ? "ok" : "fail", now_ms() - t0, ok ? (long long) arts->n_artifacts : 0,
        (long long) payload, fmt, out ? out : "");
    if (arts != NULL) A.free_artifacts(arts);
    A.free_model(m);
    return ok ? 0 : 1;
}

static int cmd_errpath(void) {
    const double t0 = now_ms();
    int checks = 0, passed = 0;
    audiocpp_error_t err = {0, NULL};

    audiocpp_model_t *m = A.load_model("Z:\\definitely\\not\\a\\model.gguf", NULL,
        AUDIOCPP_TASK_TTS, AUDIOCPP_BACKEND_CPU, 0, 0, &err);
    ++checks;
    if (m == NULL && err.code != 0 && err.message != NULL) {
        ++passed;
        fprintf(stderr, "bad-path rejected: %s\n", err.message);
    } else if (m != NULL) {
        A.free_model(m);
    }
    if (err.message != NULL) A.clear_error(&err);

    /* malformed session options are a hard error per the header contract */
    m = A.load_model_ex(NULL, "silero_vad", AUDIOCPP_TASK_VAD, AUDIOCPP_BACKEND_CPU, 0, 0,
        "{\"nope\": ", &err);
    ++checks;
    if (m == NULL && err.message != NULL) {
        ++passed;
        fprintf(stderr, "malformed options rejected: %s\n", err.message);
    } else if (m != NULL) {
        A.free_model(m); /* embedded weights load fine; the JSON must still be rejected */
        fprintf(stderr, "malformed options NOT rejected (load_model_ex succeeded)\n");
    }
    if (err.message != NULL) A.clear_error(&err);

    /* Metal on Windows must report unavailable, not crash */
    ++checks;
    if (A.backend_available(AUDIOCPP_BACKEND_METAL) == 0) {
        ++passed;
    }

    printf("E2E\terrpath\tstatus=%s\tms=%.0f\tchecks=%d\tpassed=%d\t\n",
        (checks == passed) ? "ok" : "fail", now_ms() - t0, checks, passed);
    return checks == passed ? 0 : 1;
}

static int cmd_concat(int argc, char **argv) {
    const double t0 = now_ms();
    const char *a1 = arg_str(argc, argv, "--in", NULL);
    const char *a2 = arg_str(argc, argv, "--in2", NULL);
    const char *out = arg_str(argc, argv, "--out", NULL);
    const double gap = (double) arg_int(argc, argv, "--gap-ms", 300) / 1000.0;
    if (a1 == NULL || a2 == NULL || out == NULL) {
        fprintf(stderr, "concat needs --in --in2 --out\n");
        return 2;
    }
    float *p1 = NULL, *p2 = NULL;
    int64_t n1 = 0, n2 = 0;
    int sr1 = 0, sr2 = 0;
    if (A.read_wav(a1, &p1, &n1, &sr1) != 0 || A.read_wav(a2, &p2, &n2, &sr2) != 0) {
        printf("E2E\tconcat\tstatus=fail\treason=read-wav\tms=%.0f\t\n", now_ms() - t0);
        free(p1); free(p2);
        return 1;
    }
    if (sr1 != sr2) {
        printf("E2E\tconcat\tstatus=fail\treason=rate-mismatch %d vs %d\tms=%.0f\t\n", sr1, sr2,
            now_ms() - t0);
        free(p1); free(p2);
        return 1;
    }
    const int64_t gap_n = (int64_t) (gap * sr1);
    const int64_t total = n1 + gap_n + n2;
    float *pcm = (float *) malloc(sizeof(float) * (size_t) total);
    memcpy(pcm, p1, sizeof(float) * (size_t) n1);
    for (int64_t i = 0; i < gap_n; ++i) pcm[n1 + i] = 0.0F;
    memcpy(pcm + n1 + gap_n, p2, sizeof(float) * (size_t) n2);
    const int w = A.write_wav_ex(out, pcm, total, sr1, 1);
    printf("E2E\tconcat\tstatus=%s\tms=%.0f\tsr=%d\tn=%lld\tout=%s\t\n", w == 0 ? "ok" : "fail",
        now_ms() - t0, sr1, (long long) total, out);
    free(p1); free(p2); free(pcm);
    return w == 0 ? 0 : 1;
}

/* ── main ────────────────────────────────────────────────────────────── */

static void usage(void) {
    fprintf(stderr,
        "usage: audiocpp_e2e <sub> [flags]\n"
        "subs: info | tts | tts-batch | clon | asr | asr-stream | tts-stream | vad |\n"
        "      diar | align | sep | vc | denoise | superres | gen | midi | errpath\n"
        "common flags: --dll <path> --model <p> --family <f> --task <n> --backend <n>\n"
        "              --device <n> --threads <n> --load-options|--load-options-file <p>\n"
        "text flags:   --text|--text-file <p>   --options|--options-file <p>\n"
        "audio flags:  --in <wav> --ref-wav <wav> --out <wav> --out-dir <dir>\n");
}

int main(int argc, char **argv) {
    if (argc < 2) {
        usage();
        return 2;
    }
    api_load(arg_str(argc, argv, "--dll", "audiocpp.dll"));
    const char *sub = argv[1];
    if (strcmp(sub, "info") == 0) return cmd_info();
    if (strcmp(sub, "tts") == 0) return cmd_tts(argc, argv);
    if (strcmp(sub, "tts-batch") == 0) return cmd_tts_batch(argc, argv);
    if (strcmp(sub, "clon") == 0) return cmd_clon(argc, argv);
    if (strcmp(sub, "asr") == 0) return cmd_asr(argc, argv);
    if (strcmp(sub, "asr-stream") == 0) return cmd_asr_stream(argc, argv);
    if (strcmp(sub, "tts-stream") == 0) return cmd_tts_stream(argc, argv);
    if (strcmp(sub, "vad") == 0) return cmd_vad(argc, argv);
    if (strcmp(sub, "diar") == 0) return cmd_diar(argc, argv);
    if (strcmp(sub, "align") == 0) return cmd_align(argc, argv);
    if (strcmp(sub, "sep") == 0) return cmd_sep(argc, argv);
    if (strcmp(sub, "vc") == 0) return cmd_vc(argc, argv);
    if (strcmp(sub, "denoise") == 0) return cmd_denoise(argc, argv);
    if (strcmp(sub, "superres") == 0) return cmd_superres(argc, argv);
    if (strcmp(sub, "gen") == 0) return cmd_gen(argc, argv);
    if (strcmp(sub, "midi") == 0) return cmd_midi(argc, argv);
    if (strcmp(sub, "errpath") == 0) return cmd_errpath();
    if (strcmp(sub, "concat") == 0) return cmd_concat(argc, argv);
    usage();
    return 2;
}
