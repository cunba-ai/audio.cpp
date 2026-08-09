/*
 * vram_probe.c — multi-session inference probe for the audiocpp CAPI.
 *
 * Loads N sessions of one model (shared weight buffers across sessions) and
 * runs one inference per session with per-session inputs: TTS/voice-cloning
 * texts, or audio for ASR/VAD/diarization/alignment/transforms. Prints VRAM +
 * host RAM at every step, sanity-checks outputs, saves WAVs.
 *
 * Build (MSVC): cl /O2 vram_probe.c /I..\include /link /LIBPATH:<dll dir> audiocpp.lib
 *
 * Usage:
 *   vram_probe.exe --model <gguf|dir> [--family <hint>] [--codec <miotts codec gguf>]
 *                  [--n <sessions>] [--task <0..12>] [--parallel]
 *                  [--synth-text <t1|t2|...>] [--audio <wav>] [--reference-text <t>]
 *                  [--speaker <wav>] [--options <json>] [--out-prefix <p>]
 *                  [--voice-ref <wav>] [--threads <n>] [--hold]
 */

#include "audiocpp.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")

static const char *arg_value(int argc, char **argv, const char *name) {
    for (int i = 1; i < argc - 1; ++i) {
        if (strcmp(argv[i], name) == 0) return argv[i + 1];
    }
    return NULL;
}

static int arg_flag(int argc, char **argv, const char *name) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], name) == 0) return 1;
    }
    return 0;
}

static long long free_vram(void) {
    audiocpp_device_info_t info;
    if (audiocpp_device_info(0, &info) != 0 || info.memory_total == 0) {
        return -1;
    }
    return (long long)info.memory_free;
}

static long long host_private_bytes(void) {
    PROCESS_MEMORY_COUNTERS_EX pmc;
    memset(&pmc, 0, sizeof(pmc));
    pmc.cb = sizeof(pmc);
    if (!GetProcessMemoryInfo(GetCurrentProcess(),
                              (PROCESS_MEMORY_COUNTERS *)&pmc, sizeof(pmc))) {
        return -1;
    }
    return (long long)pmc.PrivateUsage;
}

static void mark(const char *label) {
    const long long free_mb = free_vram() / (1024 * 1024);
    const long long host_mb = host_private_bytes() / (1024 * 1024);
    printf("MEM  %-44s free=%lld MiB  host=%lld MiB\n", label, free_mb, host_mb);
    fflush(stdout);
}

/* JSON-escape a Windows path (backslashes) into a buffer. */
static void json_escape_path(char *dst, size_t dst_cap, const char *src) {
    size_t j = 0;
    for (size_t i = 0; src[i] != '\0' && j + 3 < dst_cap - 1; ++i) {
        if (src[i] == '\\') dst[j++] = '\\';
        dst[j++] = src[i];
    }
    dst[j] = '\0';
}

/* Build the run options JSON from voice_ref / reference_text / speaker /
 * user-supplied --options. Only non-empty fields are emitted, so strict
 * spec-backed models (htdemucs etc.) do not see unknown empty keys. */
static const char *build_options(const char *voice_ref, const char *reference_text,
                                 const char *speaker, const char *user_json) {
    static char buf[4096];
    char voice_json[512] = "";
    char ref_json[512] = "";
    char spk_json[512] = "";
    if (voice_ref && voice_ref[0]) {
        json_escape_path(voice_json, sizeof(voice_json), voice_ref);
    }
    if (speaker && speaker[0]) {
        json_escape_path(spk_json, sizeof(spk_json), speaker);
    }
    if (reference_text && reference_text[0]) {
        size_t j = 0;
        for (size_t i = 0; reference_text[i] != '\0' && j + 3 < sizeof(ref_json) - 1; ++i) {
            if (reference_text[i] == '"' || reference_text[i] == '\\') ref_json[j++] = '\\';
            ref_json[j++] = reference_text[i];
        }
        ref_json[j] = '\0';
    }
    size_t off = 0;
    buf[0] = '{';
    off = 1;
    int need_comma = 0;
#define EMIT_KEY(key, val)                                       \
    do {                                                         \
        if ((val)[0] != '\0') {                                  \
            if (need_comma) buf[off++] = ',';                    \
            off += (size_t)snprintf(buf + off, sizeof(buf) - off, \
                                    "\"" key "\":\"%s\"", (val)); \
            need_comma = 1;                                      \
        }                                                        \
    } while (0)
    EMIT_KEY("voice_ref", voice_json);
    EMIT_KEY("reference_text", ref_json);
    EMIT_KEY("speaker", spk_json);
#undef EMIT_KEY
    if (user_json && user_json[0] == '{') {
        const char *inner = user_json + 1;
        size_t len = strlen(inner);
        while (len > 0 && (inner[len - 1] == '}' || inner[len - 1] == ' ' ||
                           inner[len - 1] == '\n' || inner[len - 1] == '\t')) {
            --len;
        }
        if (len > 0 && strcmp(inner, "}") != 0) {
            if (need_comma) buf[off++] = ',';
            memcpy(buf + off, inner, len);
            off += len;
            need_comma = 1;
        }
    }
    buf[off++] = '}';
    buf[off] = '\0';
    return buf;
}

typedef struct {
    audiocpp_model_t *session;
    int task;
    const char *text;         /* TTS / align text */
    const float *audio_pcm;   /* ASR / VAD / DIAR / SEP / VC audio */
    int64_t audio_n;
    int audio_sr;
    const char *options;
    const char *language;     /* align language code */
    const char *out_prefix;
    int index;
    int ok;
    char result_text[1024];
    double duration;
    double rms;
    double peak;
    long long nonzero;
    long long n_samples;
    int sample_rate;
    int channels;
    char err_msg[512];
} infer_task;

static void analyze_audio(const audiocpp_audio_t *audio, infer_task *t) {
    t->sample_rate = audio->sample_rate;
    t->channels = audio->channels;
    t->n_samples = audio->n_samples;
    double rms = 0.0;
    double peak = 0.0;
    long long nonzero = 0;
    for (int64_t s = 0; s < audio->n_samples; ++s) {
        const double v = audio->samples[s];
        rms += v * v;
        const double a = v < 0 ? -v : v;
        if (a > peak) peak = a;
        if (v != 0.0) ++nonzero;
    }
    if (audio->n_samples > 0) rms = sqrt(rms / (double)audio->n_samples);
    t->rms = rms;
    t->peak = peak;
    t->nonzero = nonzero;
    t->duration = (double)audio->n_samples / (double)audio->sample_rate;
    if (t->out_prefix) {
        char wav_path[1024];
        snprintf(wav_path, sizeof(wav_path), "%s_%d.wav", t->out_prefix, t->index + 1);
        if (audiocpp_write_wav_ex(wav_path, audio->samples, audio->n_samples,
                                  audio->sample_rate, audio->channels) != 0) {
            printf("  [task %d] failed to write %s\n", t->index + 1, wav_path);
        } else {
            printf("  [task %d] wrote %s\n", t->index + 1, wav_path);
        }
        fflush(stdout);
    }
}

static DWORD WINAPI infer_worker(LPVOID param) {
    infer_task *t = (infer_task *)param;
    audiocpp_error_t err = {0};
    t->result_text[0] = '\0';
    switch (t->task) {
        case 0: {  /* VAD */
            audiocpp_vad_t *vad = audiocpp_vad(t->session, t->audio_pcm, t->audio_n,
                                               t->audio_sr, t->options, &err);
            if (!vad) break;
            t->ok = 1;
            snprintf(t->result_text, sizeof(t->result_text),
                     "%lld VAD segments", (long long)vad->n_segments);
            audiocpp_free_vad(vad);
            return 0;
        }
        case 1: {  /* ASR */
            audiocpp_text_t *text = audiocpp_asr(t->session, t->audio_pcm, t->audio_n,
                                                 t->audio_sr, t->options, &err);
            if (!text) break;
            t->ok = 1;
            snprintf(t->result_text, sizeof(t->result_text), "%s",
                     text->text ? text->text : "(null)");
            audiocpp_free_text(text);
            return 0;
        }
        case 2: {  /* DIAR */
            audiocpp_diar_t *diar = audiocpp_diar(t->session, t->audio_pcm, t->audio_n,
                                                  t->audio_sr, t->options, &err);
            if (!diar) break;
            t->ok = 1;
            snprintf(t->result_text, sizeof(t->result_text), "%lld speaker turns",
                     (long long)diar->n_turns);
            audiocpp_free_diar(diar);
            return 0;
        }
        case 6: {  /* ALIGN */
            audiocpp_align_t *align = audiocpp_align(t->session, t->audio_pcm, t->audio_n,
                                                     t->audio_sr, t->text, t->language, t->options, &err);
            if (!align) break;
            t->ok = 1;
            snprintf(t->result_text, sizeof(t->result_text), "%lld word timestamps",
                     (long long)align->n_words);
            audiocpp_free_align(align);
            return 0;
        }
        case 3:   /* SEP */
        case 7:   /* VC */
        case 9:   /* S2S */
        case 11:  /* SPK */
        case 12:  /* SVC */
        case 13: {  /* GEN via transform */
            audiocpp_audio_t *audio = audiocpp_audio_transform(
                t->session, t->audio_pcm, t->audio_n, t->audio_sr, t->options, &err);
            if (!audio) break;
            t->ok = 1;
            analyze_audio(audio, t);
            audiocpp_free_audio(audio);
            return 0;
        }
        case 5:   /* TTS */
        case 8:   /* CLON */
        case 10:  /* VDES */
        default: {
            audiocpp_audio_t *audio = audiocpp_tts(t->session, t->text, t->options, &err);
            if (!audio) break;
            t->ok = 1;
            analyze_audio(audio, t);
            audiocpp_free_audio(audio);
            return 0;
        }
    }
    t->ok = 0;
    snprintf(t->err_msg, sizeof(t->err_msg), "%s", err.message);
    return 0;
}

int main(int argc, char **argv) {
    const char *model = arg_value(argc, argv, "--model");
    const char *codec = arg_value(argc, argv, "--codec");
    const char *text = arg_value(argc, argv, "--synth-text");
    const char *audio_path = arg_value(argc, argv, "--audio");
    const char *voice_ref = arg_value(argc, argv, "--voice-ref");
    const char *reference_text = arg_value(argc, argv, "--reference-text");
    const char *speaker = arg_value(argc, argv, "--speaker");
    const char *user_json = arg_value(argc, argv, "--options");
    const int task = arg_value(argc, argv, "--task") ? atoi(arg_value(argc, argv, "--task")) : 5;
    const int n_max = arg_value(argc, argv, "--n") ? atoi(arg_value(argc, argv, "--n")) : 1;
    const int n_threads = arg_value(argc, argv, "--threads") ? atoi(arg_value(argc, argv, "--threads")) : 8;
    if (!model) {
        fprintf(stderr,
                "usage: vram_probe.exe --model <gguf|dir> [--family <hint>] [--codec <gguf>]\n"
                "       [--n <sessions>] [--task <0..12>] [--parallel]\n"
                "       [--synth-text <t1|t2|...>] [--audio <wav>] [--reference-text <t>]\n"
                "       [--speaker <wav>] [--options <json>] [--out-prefix <p>]\n"
                "       [--voice-ref <wav>] [--threads <n>] [--hold]\n");
        return 2;
    }
    if (n_max < 1 || n_max > 8) {
        fprintf(stderr, "--n must be 1..8\n");
        return 2;
    }

    /* Session options (miotts codec path) */
    char session_options[2048];
    {
        char codec_json[512] = "";
        char spec_json[512] = "";
        const char *spec = arg_value(argc, argv, "--spec");
        if (codec && codec[0]) {
            json_escape_path(codec_json, sizeof(codec_json), codec);
        }
        if (spec && spec[0]) {
            json_escape_path(spec_json, sizeof(spec_json), spec);
        }
        snprintf(session_options, sizeof(session_options),
                 "{\"miotts.codec_model_path\":\"%s\",\"model_spec_override\":\"%s\"}",
                 codec_json, spec_json);
    }
    const char *family = arg_value(argc, argv, "--family") ? arg_value(argc, argv, "--family") : "miotts";

    printf("model=%s\nfamily=%s\ntask=%d sessions=%d\n", model, family, task, n_max);
    mark("start (no session)");

    enum { kMax = 8 };
    audiocpp_model_t *sessions[kMax] = {0};
    for (int i = 0; i < n_max; ++i) {
        audiocpp_error_t err = {0};
        sessions[i] = audiocpp_load_model_ex(
            model, family, task,
            AUDIOCPP_BACKEND_CUDA, 0, n_threads, session_options, &err);
        if (!sessions[i]) {
            printf("load session %d FAILED: %s\n", i + 1, err.message);
            return 1;
        }
        char label[64];
        snprintf(label, sizeof(label), "after load #%d (%d sessions)", i + 1, i + 1);
        mark(label);
        Sleep(200);
    }

    /* Optional audio input for ASR/VAD/DIAR/SEP/VC/ALIGN */
    float *audio_pcm = NULL;
    int64_t audio_n = 0;
    int audio_sr = 0;
    if (audio_path && audio_path[0]) {
        audiocpp_error_t err = {0};
        if (audiocpp_read_wav(audio_path, &audio_pcm, &audio_n, &audio_sr) != 0) {
            printf("failed to read --audio %s: %s\n", audio_path, err.message);
            return 1;
        }
        printf("audio input: %.2fs %d Hz\n", (double)audio_n / (double)audio_sr, audio_sr);
    }

    const char *options = build_options(voice_ref, reference_text, speaker, user_json);

    /* Split --synth-text on '|' so session i gets its own text */
    char texts_copy[2048];
    char *texts[16] = {0};
    int n_texts = 0;
    if (text) {
        snprintf(texts_copy, sizeof(texts_copy), "%s", text);
        char *tok = texts_copy;
        char *next;
        do {
            next = strchr(tok, '|');
            if (next) *next = '\0';
            texts[n_texts++] = tok;
            if (n_texts >= 16) break;
            tok = next ? next + 1 : NULL;
        } while (tok != NULL && *tok != '\0');
        if (n_texts == 0) texts[n_texts++] = texts_copy;
    }
    const char *out_prefix = arg_value(argc, argv, "--out-prefix");
    const char *language = arg_value(argc, argv, "--language");

    infer_task tasks[16];
    memset(tasks, 0, sizeof(tasks));
    const int n_run = n_max < 16 ? n_max : 16;
    for (int i = 0; i < n_run; ++i) {
        tasks[i].session = sessions[i];
        tasks[i].task = task;
        tasks[i].text = texts[i % (n_texts > 0 ? n_texts : 1)];
        tasks[i].audio_pcm = audio_pcm;
        tasks[i].audio_n = audio_n;
        tasks[i].audio_sr = audio_sr;
        tasks[i].options = options;
        tasks[i].language = language;
        tasks[i].out_prefix = out_prefix;
        tasks[i].index = i;
    }

    const int parallel = arg_flag(argc, argv, "--parallel") && task >= 5;
    const DWORD t0 = GetTickCount();
    if (parallel) {
        HANDLE handles[16];
        DWORD ids[16];
        for (int i = 0; i < n_run; ++i) {
            handles[i] = CreateThread(NULL, 0, infer_worker, &tasks[i], 0, &ids[i]);
            if (handles[i] == NULL) {
                tasks[i].ok = 0;
                snprintf(tasks[i].err_msg, sizeof(tasks[i].err_msg), "thread create failed");
                handles[i] = NULL;
            }
        }
        for (int i = 0; i < n_run; ++i) {
            if (handles[i] != NULL) {
                WaitForSingleObject(handles[i], INFINITE);
                CloseHandle(handles[i]);
            }
        }
    } else {
        for (int i = 0; i < n_run; ++i) {
            infer_worker(&tasks[i]);
        }
    }
    const double wall_s = (double)(GetTickCount() - t0) / 1000.0;

    for (int i = 0; i < n_run; ++i) {
        const infer_task *t = &tasks[i];
        if (!t->ok) {
            printf("run #%d FAILED: %s\n", i + 1, t->err_msg);
            continue;
        }
        if (t->n_samples > 0) {
            printf("run #%d -> %.2fs, %d Hz, %d ch, rms=%.4f peak=%.3f nonzero=%lld/%lld\n",
                   i + 1, t->duration, t->sample_rate, t->channels,
                   t->rms, t->peak, t->nonzero, (long long)t->n_samples);
        } else if (t->result_text[0]) {
            printf("run #%d -> %s\n", i + 1, t->result_text);
        } else {
            printf("run #%d -> (ok)\n", i + 1);
        }
    }
    printf("%s wall time for %d runs: %.2fs\n", parallel ? "parallel" : "serial", n_run, wall_s);
    fflush(stdout);
    char label[64];
    snprintf(label, sizeof(label), "after %d runs (%s, %.1fs wall)",
             n_run, parallel ? "parallel" : "serial", wall_s);
    mark(label);

    if (audio_pcm) free(audio_pcm);
    printf("done; sessions released\n");
    fflush(stdout);
    if (arg_flag(argc, argv, "--hold")) {
        printf("holding sessions; press Ctrl+C to exit...\n");
        fflush(stdout);
        for (;;) Sleep(10000);
    }
    return 0;
}
