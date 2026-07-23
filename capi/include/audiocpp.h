/**
 * audiocpp.h — C ABI for audio.cpp
 *
 * Minimal C interface exposing audio.cpp's model loading, TTS, and ASR
 * capabilities to foreign-language runtimes (Rust FFI, Python ctypes, etc.).
 *
 * Design:
 *   - Opaque handles (audiocpp_model_t*) wrap C++ objects (ILoadedVoiceModel + session).
 *   - All C++ exceptions are caught at the boundary and converted to error codes.
 *   - Returned strings/audio are heap-allocated; caller MUST free with the
 *     matching audiocpp_free_* function.
 *   - The shared library hides all ggml/sentencepiece/etc. symbols; only
 *     audiocpp_* symbols are exported.
 *
 * Thread safety:
 *   - audiocpp_load_model is thread-safe (registry is read-only after init).
 *   - A single model handle is NOT thread-safe for concurrent run calls
 *     (each call mutates internal session state). Use separate model handles
 *     per thread, or serialize calls externally.
 */

#ifndef AUDIOCPP_H
#define AUDIOCPP_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

/* ======================================================================== */
/* Types                                                                     */
/* ======================================================================== */

/** Opaque handle to a loaded model + task session. */
typedef struct audiocpp_model audiocpp_model_t;

/** Audio output (PCM f32 samples). Caller owns; free with audiocpp_free_audio. */
typedef struct {
    float *samples;      /**< PCM samples, mono, [-1.0, 1.0] */
    int64_t n_samples;   /**< Number of samples */
    int sample_rate;     /**< Sample rate in Hz (e.g. 24000) */
} audiocpp_audio_t;

/** Text output. Caller owns; free with audiocpp_free_text. */
typedef struct {
    char *text;          /**< UTF-8 string (null-terminated) */
    char *language;      /**< Detected language code (may be NULL) */
} audiocpp_text_t;

/** Error info. */
typedef struct {
    int code;            /**< 0 = success, negative = error */
    char *message;       /**< Error message (may be NULL on success). Caller frees with audiocpp_free_string. */
} audiocpp_error_t;

/* ======================================================================== */
/* Backend type (mirrors engine::core::BackendType)                          */
/* ======================================================================== */

enum {
    AUDIOCPP_BACKEND_CPU    = 0,
    AUDIOCPP_BACKEND_CUDA   = 1,
    AUDIOCPP_BACKEND_VULKAN = 2,
    AUDIOCPP_BACKEND_METAL  = 3,
    AUDIOCPP_BACKEND_BEST   = 4,  /**< auto-select best available */
};

/* ======================================================================== */
/* Task type (mirrors engine::runtime::VoiceTaskKind)                        */
/* ======================================================================== */

enum {
    AUDIOCPP_TASK_TTS    = 5,  /**< Text-to-Speech */
    AUDIOCPP_TASK_ASR    = 1,  /**< Speech-to-Text */
    AUDIOCPP_TASK_VAD    = 0,  /**< Voice Activity Detection */
    AUDIOCPP_TASK_DIAR   = 2,  /**< Speaker Diarization */
};

/* ======================================================================== */
/* Lifecycle                                                                 */
/* ======================================================================== */

/**
 * Load a model and create a task session.
 *
 * @param model_path   Path to model directory or GGUF file.
 * @param family_hint  Model family hint (e.g. "qwen3_asr", "qwen3_tts"); NULL = auto-detect.
 * @param task         One of AUDIOCPP_TASK_* (TTS, ASR, VAD, etc.).
 * @param backend      One of AUDIOCPP_BACKEND_*.
 * @param device_id    GPU device index (0 for first GPU; ignored for CPU).
 * @param n_threads    Number of CPU threads (0 = auto).
 * @param err          Optional error output (pass NULL to ignore).
 * @return Model handle, or NULL on failure (check err).
 */
audiocpp_model_t *audiocpp_load_model(
    const char *model_path,
    const char *family_hint,
    int task,
    int backend,
    int device_id,
    int n_threads,
    audiocpp_error_t *err
);

/** Free a model handle. Safe to call with NULL. */
void audiocpp_free_model(audiocpp_model_t *model);

/* ======================================================================== */
/* TTS: text → audio                                                         */
/* ======================================================================== */

/**
 * Synthesize speech from text.
 *
 * @param model       Model handle (must be loaded with AUDIOCPP_TASK_TTS).
 * @param text        UTF-8 input text.
 * @param options     JSON string of model-specific options, e.g.:
 *                      {"voice_ref": "/path/to/ref.wav",
 *                       "reference_text": "transcript of ref audio",
 *                       "speed": 1.0,
 *                       "language": "en",
 *                       "emotion": "happy"}
 *                    Pass NULL or "{}" for defaults.
 * @param err         Optional error output.
 * @return Audio output, or NULL on failure. Caller MUST free with audiocpp_free_audio.
 */
audiocpp_audio_t *audiocpp_tts(
    const audiocpp_model_t *model,
    const char *text,
    const char *options_json,
    audiocpp_error_t *err
);

/**
 * Synthesize speech with inline voice reference PCM (no temp file).
 *
 * @param model            Model handle (must be TTS).
 * @param text             UTF-8 input text.
 * @param options_json     Options (reference_text, speed, language, ...).
 *                         voice_ref in options_json is ignored when voice_ref_pcm is non-NULL.
 * @param voice_ref_pcm    Float32 mono PCM samples for voice cloning. NULL = no clone.
 * @param voice_ref_n      Number of samples in voice_ref_pcm (0 if NULL).
 * @param voice_ref_sr     Sample rate of voice_ref_pcm (e.g. 16000).
 * @param err              Optional error output.
 */
audiocpp_audio_t *audiocpp_tts_with_voice_ref(
    const audiocpp_model_t *model,
    const char *text,
    const char *options_json,
    const float *voice_ref_pcm,
    int64_t voice_ref_n,
    int voice_ref_sr,
    audiocpp_error_t *err
);

/* ======================================================================== */
/* ASR: audio → text                                                         */
/* ======================================================================== */

/**
 * Transcribe audio to text.
 *
 * @param model       Model handle (must be loaded with AUDIOCPP_TASK_ASR).
 * @param pcm         PCM samples (mono f32, [-1.0, 1.0]).
 * @param n_samples   Number of PCM samples.
 * @param sample_rate Sample rate (e.g. 16000).
 * @param options     JSON string of model-specific options, e.g.:
 *                      {"language": "zh", "return_timestamps": "true"}
 *                    Pass NULL or "{}" for defaults.
 * @param err         Optional error output.
 * @return Text result, or NULL on failure. Caller MUST free with audiocpp_free_text.
 */
audiocpp_text_t *audiocpp_asr(
    const audiocpp_model_t *model,
    const float *pcm,
    int64_t n_samples,
    int sample_rate,
    const char *options_json,
    audiocpp_error_t *err
);

/* ======================================================================== */
/* Diarization: audio → speaker turns                                       */
/* ======================================================================== */

/** Speaker turn (one segment of one speaker). */
typedef struct {
    int64_t start_sample;   /**< Start position in samples */
    int64_t end_sample;     /**< End position in samples */
    char *speaker_id;       /**< Speaker ID string (may be NULL) */
    float confidence;       /**< Confidence score [0,1] */
} audiocpp_speaker_turn_t;

/** Diarization result. Caller owns; free with audiocpp_free_diar. */
typedef struct {
    audiocpp_speaker_turn_t *turns;  /**< Array of speaker turns */
    int64_t n_turns;                 /**< Number of turns */
} audiocpp_diar_t;

/**
 * Perform speaker diarization on audio.
 *
 * @param model       Model handle (must be loaded with AUDIOCPP_TASK_DIAR).
 * @param pcm         PCM samples (mono f32, [-1.0, 1.0]).
 * @param n_samples   Number of PCM samples.
 * @param sample_rate Sample rate (e.g. 16000).
 * @param options_json JSON string of model-specific options (NULL = defaults).
 * @param err         Optional error output.
 * @return Diarization result, or NULL on failure. Caller MUST free with audiocpp_free_diar.
 */
audiocpp_diar_t *audiocpp_diar(
    const audiocpp_model_t *model,
    const float *pcm,
    int64_t n_samples,
    int sample_rate,
    const char *options_json,
    audiocpp_error_t *err
);

/* ======================================================================== */
/* VAD: audio → speech segments                                              */
/* ======================================================================== */

/** VAD speech segment. */
typedef struct {
    int64_t start_sample;
    int64_t end_sample;
    float confidence;
} audiocpp_vad_segment_t;

/** VAD result. Caller owns; free with audiocpp_free_vad. */
typedef struct {
    audiocpp_vad_segment_t *segments;
    int64_t n_segments;
} audiocpp_vad_t;

/**
 * Detect speech segments in audio.
 *
 * @param model       Model handle (must be loaded with AUDIOCPP_TASK_VAD).
 * @param pcm         PCM samples (mono f32, [-1.0, 1.0]).
 * @param n_samples   Number of PCM samples.
 * @param sample_rate Sample rate (e.g. 16000).
 * @param options_json JSON options (NULL = defaults).
 * @param err         Optional error output.
 * @return VAD result, or NULL on failure. Caller MUST free with audiocpp_free_vad.
 */
audiocpp_vad_t *audiocpp_vad(
    const audiocpp_model_t *model,
    const float *pcm,
    int64_t n_samples,
    int sample_rate,
    const char *options_json,
    audiocpp_error_t *err
);

/* ======================================================================== */
/* Utilities                                                                 */
/* ======================================================================== */

/** Get the audio.cpp version string (static, do NOT free). */
const char *audiocpp_version(void);

/** Free an audio result. Safe to call with NULL. */
void audiocpp_free_audio(audiocpp_audio_t *audio);

/** Free a text result. Safe to call with NULL. */
void audiocpp_free_text(audiocpp_text_t *text);

/** Free a diarization result. Safe to call with NULL. */
void audiocpp_free_diar(audiocpp_diar_t *diar);

/** Free a VAD result. Safe to call with NULL. */
void audiocpp_free_vad(audiocpp_vad_t *vad);

/** Free a string returned in audiocpp_error_t. Safe to call with NULL. */
void audiocpp_free_string(char *str);

/** Clear an error struct (frees message, resets code to 0). */
void audiocpp_clear_error(audiocpp_error_t *err);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* AUDIOCPP_H */
