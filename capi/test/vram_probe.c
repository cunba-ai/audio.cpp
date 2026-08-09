/*
 * vram_probe.c — measure GPU VRAM growth when N sessions of the same model
 * are loaded in ONE process (the L1 parallel-inference shape: each
 * audiocpp_load_model_ex creates an independent session with its own
 * backend/weight upload/graph arenas).
 *
 * Build (mingw):
 *   gcc -O2 -o vram_probe.exe vram_probe.c -I../include -L. -laudiocpp
 * Run next to audiocpp.dll:
 *   vram_probe.exe --model <miotts gguf> --codec <miocodec gguf>
 *                  [--n 3] [--synth-text "hello"]
 *
 * Prints free VRAM after each load and after each synth, so you can plot
 * V(N) = base + N * per_session.
 */

#include "audiocpp.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")

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

static void mark(const char *label) {
    const long long free_mb = free_vram() / (1024 * 1024);
    const long long host_mb = host_private_bytes() / (1024 * 1024);
    printf("MEM  %-38s free=%lld MiB  host=%lld MiB\n", label, free_mb, host_mb);
    fflush(stdout);
}

typedef struct {
    audiocpp_model_t *session;
    const char *text;
    const char *synth_options;
    const char *out_prefix;
    int index;
    int ok;
    double duration;
    double rms;
    double peak;
    long long nonzero;
    long long n_samples;
    int sample_rate;
    int channels;
    char err_msg[512];
} synth_task;

/* One TTS run: synthesize, sanity-check the PCM, save a WAV. Thread-safe:
 * each task touches only its own session (parallel inference shares the GPU
 * weight buffer but each session has its own graph/KV state). */
static DWORD WINAPI synth_worker(LPVOID param) {
    synth_task *t = (synth_task *)param;
    audiocpp_error_t err = {0};
    audiocpp_audio_t *audio = audiocpp_tts(t->session, t->text, t->synth_options, &err);
    if (!audio) {
        t->ok = 0;
        snprintf(t->err_msg, sizeof(t->err_msg), "%s", err.message);
        return 0;
    }
    t->ok = 1;
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
    audiocpp_free_audio(audio);
    return 0;
}

int main(int argc, char **argv) {
    const char *model = arg_value(argc, argv, "--model");
    const char *codec = arg_value(argc, argv, "--codec");
    const char *text  = arg_value(argc, argv, "--synth-text");
    const int n_max = arg_value(argc, argv, "--n") ? atoi(arg_value(argc, argv, "--n")) : 3;
    const int n_threads = arg_value(argc, argv, "--threads") ? atoi(arg_value(argc, argv, "--threads")) : 8;
    if (!model) {
        fprintf(stderr,
                "usage: vram_probe.exe --model <gguf> [--codec <codec gguf>] "
                "[--family <hint>] [--n <sessions>] [--synth-text <text>] [--threads <n>]\n");
        return 2;
    }
    if (n_max < 1 || n_max > 8) {
        fprintf(stderr, "--n must be 1..8\n");
        return 2;
    }

    /* Backslashes in Windows paths must be escaped to be valid JSON. */
    char options[1024];
    if (codec && codec[0]) {
        char codec_json[512];
        size_t j = 0;
        for (size_t i = 0; codec[i] != '\0' && j + 3 < sizeof(codec_json) - 1; ++i) {
            if (codec[i] == '\\') codec_json[j++] = '\\';
            codec_json[j++] = codec[i];
        }
        codec_json[j] = '\0';
        snprintf(options, sizeof(options),
                 "{\"miotts.codec_model_path\":\"%s\"}", codec_json);
    } else {
        snprintf(options, sizeof(options), "{}");
    }

    printf("model=%s\ncodec=%s\nsessions=%d\n", model, codec ? codec : "(none)", n_max);
    mark("start (no session)");

    enum { kMax = 8 };
    audiocpp_model_t *sessions[kMax] = {0};
    const char *family = arg_value(argc, argv, "--family") ? arg_value(argc, argv, "--family") : "miotts";

    for (int i = 0; i < n_max; ++i) {
        audiocpp_error_t err = {0};
        sessions[i] = audiocpp_load_model_ex(
            model, family, AUDIOCPP_TASK_TTS,
            AUDIOCPP_BACKEND_CUDA, 0, n_threads, options, &err);
        if (!sessions[i]) {
            printf("load session %d FAILED: %s\n", i + 1, err.message);
            return 1;
        }
        char label[64];
        snprintf(label, sizeof(label), "after load #%d (%d sessions)", i + 1, i + 1);
        mark(label);
        Sleep(500);
    }

    if (text) {
        /* --synth-text may hold several texts separated by '|'; session i uses
         * text i (wrapping around if fewer texts than sessions) so each session
         * synthesizes DIFFERENT content — proves results are correct per
         * session, not just that inference runs. */
        char texts_copy[2048];
        snprintf(texts_copy, sizeof(texts_copy), "%s", text);
        char *texts[16] = {0};
        int n_texts = 0;
        {
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

        /* miotts requires a speaker reference; pass one via the run options. */
        const char *voice_ref = arg_value(argc, argv, "--voice-ref") ? arg_value(argc, argv, "--voice-ref") : "";
        char voice_json[512];
        {
            size_t j = 0;
            for (size_t i = 0; voice_ref[i] != '\0' && j + 3 < sizeof(voice_json) - 1; ++i) {
                if (voice_ref[i] == '\\') voice_json[j++] = '\\';
                voice_json[j++] = voice_ref[i];
            }
            voice_json[j] = '\0';
        }
        char synth_options[1024];
        snprintf(synth_options, sizeof(synth_options), "{\"voice_ref\":\"%s\"}", voice_json);

        /* --parallel: run all sessions' synth on separate threads at once
         * (true concurrent inference on the shared weights). */
        const int parallel = arg_flag(argc, argv, "--parallel");
        synth_task tasks[16];
        memset(tasks, 0, sizeof(tasks));
        const int n_synth = n_max < 16 ? n_max : 16;
        for (int i = 0; i < n_synth; ++i) {
            tasks[i].session = sessions[i];
            tasks[i].text = texts[i % n_texts];
            tasks[i].synth_options = synth_options;
            tasks[i].out_prefix = out_prefix;
            tasks[i].index = i;
        }

        const DWORD t0 = GetTickCount();
        if (parallel) {
            HANDLE handles[16];
            DWORD ids[16];
            for (int i = 0; i < n_synth; ++i) {
                handles[i] = CreateThread(NULL, 0, synth_worker, &tasks[i], 0, &ids[i]);
                if (handles[i] == NULL) {
                    printf("failed to create thread for task %d\n", i + 1);
                    tasks[i].ok = 0;
                    snprintf(tasks[i].err_msg, sizeof(tasks[i].err_msg), "thread create failed");
                    handles[i] = NULL;
                }
            }
            for (int i = 0; i < n_synth; ++i) {
                if (handles[i] != NULL) {
                    WaitForSingleObject(handles[i], INFINITE);
                    CloseHandle(handles[i]);
                }
            }
        } else {
            for (int i = 0; i < n_synth; ++i) {
                synth_worker(&tasks[i]);
            }
        }
        const double wall_s = (double)(GetTickCount() - t0) / 1000.0;

        for (int i = 0; i < n_synth; ++i) {
            const synth_task *t = &tasks[i];
            if (!t->ok) {
                printf("synth #%d FAILED: %s\n", i + 1, t->err_msg);
                continue;
            }
            printf("synth #%d text=\"%s\" -> %.2fs, %d Hz, %d ch, rms=%.4f peak=%.3f nonzero=%lld/%lld\n",
                   i + 1, t->text, t->duration, t->sample_rate, t->channels,
                   t->rms, t->peak, t->nonzero, (long long)t->n_samples);
        }
        printf("%s wall time for %d synths: %.2fs\n",
               parallel ? "parallel" : "serial", n_synth, wall_s);
        fflush(stdout);
        char label[64];
        snprintf(label, sizeof(label), "after %d synths (%s, %.1fs wall)",
                 n_synth, parallel ? "parallel" : "serial", wall_s);
        mark(label);
    }

    printf("done; sessions released\n");
    fflush(stdout);
    if (arg_flag(argc, argv, "--hold")) {
        printf("holding sessions; press Ctrl+C to exit...\n");
        fflush(stdout);
        for (;;) Sleep(10000);
    }
    return 0;
}
