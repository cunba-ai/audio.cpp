/**
 * audiocpp_capi.cpp — C ABI implementation for audio.cpp
 *
 * Wraps engine::runtime C++ interfaces into extern "C" functions.
 * All C++ exceptions are caught and converted to audiocpp_error_t.
 */

#include "audiocpp.h"

#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/registry.h"
#include "engine/framework/runtime/session.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/audio/wav_writer.h"

#include "ggml-backend.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#ifdef GGML_USE_SYCL
#include "ggml-sycl.h"
#endif
#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif
#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include "cJSON.h"
// cJSON.h is under external/cJSON/ — add include path if needed

#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <filesystem>

/* ======================================================================== */
/* Internal handle                                                           */
/* ======================================================================== */

struct audiocpp_model {
    std::unique_ptr<engine::runtime::ILoadedVoiceModel> loaded_model;
    std::unique_ptr<engine::runtime::IVoiceTaskSession> session;
    engine::runtime::IOfflineVoiceTaskSession *offline = nullptr;
};

/* ======================================================================== */
/* Error helpers                                                             */
/* ======================================================================== */

// Duplicate a string into malloc'd C string (caller frees via audiocpp_free_string)
static char *dup_cstr(const std::string &s) {
    char *p = static_cast<char *>(malloc(s.size() + 1));
    if (p) {
        memcpy(p, s.c_str(), s.size() + 1);
    }
    return p;
}

static void set_error(audiocpp_error_t *err, int code, const char *msg) {
    if (!err) return;
    audiocpp_clear_error(err);
    err->code = code;
    err->message = msg ? dup_cstr(msg) : nullptr;
}

// Catches all C++ exceptions in a lambda; returns false on exception.
#define AUDIOCPP_CATCH(err, body)                                  \
    do {                                                            \
        try {                                                       \
            body;                                                   \
        } catch (const std::exception &e) {                        \
            set_error((err), -1, e.what());                        \
        } catch (...) {                                             \
            set_error((err), -2, "unknown C++ exception");         \
        }                                                           \
    } while (0)

/* ======================================================================== */
/* Enum mapping                                                              */
/* ======================================================================== */

static engine::core::BackendType map_backend(int backend) {
    switch (backend) {
        case AUDIOCPP_BACKEND_CUDA:   return engine::core::BackendType::Cuda;
        case AUDIOCPP_BACKEND_VULKAN: return engine::core::BackendType::Vulkan;
        case AUDIOCPP_BACKEND_METAL:  return engine::core::BackendType::Metal;
        case AUDIOCPP_BACKEND_SYCL:   return engine::core::BackendType::Sycl;
        case AUDIOCPP_BACKEND_BEST:   return engine::core::BackendType::BestAvailable;
        default:                      return engine::core::BackendType::Cpu;
    }
}

static engine::runtime::VoiceTaskKind map_task(int task) {
    switch (task) {
        case AUDIOCPP_TASK_TTS:   return engine::runtime::VoiceTaskKind::Tts;
        case AUDIOCPP_TASK_ASR:   return engine::runtime::VoiceTaskKind::Asr;
        case AUDIOCPP_TASK_VAD:   return engine::runtime::VoiceTaskKind::Vad;
        case AUDIOCPP_TASK_DIAR:  return engine::runtime::VoiceTaskKind::Diarization;
        case AUDIOCPP_TASK_SEP:   return engine::runtime::VoiceTaskKind::SourceSeparation;
        case AUDIOCPP_TASK_GEN:   return engine::runtime::VoiceTaskKind::AudioGeneration;
        case AUDIOCPP_TASK_ALIGN: return engine::runtime::VoiceTaskKind::Alignment;
        case AUDIOCPP_TASK_VC:    return engine::runtime::VoiceTaskKind::VoiceConversion;
        case AUDIOCPP_TASK_CLON:  return engine::runtime::VoiceTaskKind::VoiceCloning;
        case AUDIOCPP_TASK_S2S:   return engine::runtime::VoiceTaskKind::SpeechToSpeech;
        case AUDIOCPP_TASK_VDES:  return engine::runtime::VoiceTaskKind::VoiceDesign;
        case AUDIOCPP_TASK_SPK:   return engine::runtime::VoiceTaskKind::SpeakerRecognition;
        case AUDIOCPP_TASK_SVC:   return engine::runtime::VoiceTaskKind::Svc;
        default:                  return engine::runtime::VoiceTaskKind::Tts;
    }
}

/* ======================================================================== */
/* Lifecycle                                                                 */
/* ======================================================================== */

audiocpp_model_t *audiocpp_load_model(
    const char *model_path,
    const char *family_hint,
    int task,
    int backend,
    int device_id,
    int n_threads,
    audiocpp_error_t *err
) {
    audiocpp_model_t *result = nullptr;
    AUDIOCPP_CATCH(err, {
        if (!model_path || model_path[0] == '\0') {
            throw std::runtime_error("model_path is null or empty");
        }
        auto registry = engine::runtime::make_default_registry();
        engine::runtime::ModelLoadRequest req;
        req.model_path = model_path;
        if (family_hint && family_hint[0] != '\0') {
            req.family_hint = family_hint;
        }
        auto loaded = registry.load(req);
        if (!loaded) {
            throw std::runtime_error("failed to load model: " + std::string(model_path));
        }
        engine::runtime::TaskSpec task_spec;
        task_spec.task = map_task(task);
        task_spec.mode = engine::runtime::RunMode::Offline;
        engine::runtime::SessionOptions opts;
        opts.backend.type = map_backend(backend);
        opts.backend.device = device_id;
        opts.backend.threads = n_threads > 0 ? n_threads : 0;
        auto session = loaded->create_task_session(task_spec, opts);
        if (!session) {
            throw std::runtime_error("failed to create task session");
        }
        result = new audiocpp_model{};
        result->loaded_model = std::move(loaded);
        result->session = std::move(session);
        result->offline = dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(
            result->session.get());
        if (!result->offline) {
            throw std::runtime_error("model does not support offline task execution");
        }
    });
    return result;
}

void audiocpp_free_model(audiocpp_model_t *model) {
    delete model;
}

/* ======================================================================== */
/* TTS                                                                       */
/* ======================================================================== */

// Helper: parse JSON options string into TaskRequest fields.
// Recognized keys:
//   voice_ref: path to reference WAV (loads audio + sets VoiceReference)
//   reference_text: transcript of reference audio
//   speed, language, emotion, speaking_rate, pitch_shift, energy_scale: scalar options
//   Any other key → passed through as option string
static void apply_options(engine::runtime::TaskRequest & req, const char * options_json) {
    if (!options_json || !options_json[0]) return;

    // Parse JSON (use cJSON which is already vendored)
    auto * root = cJSON_Parse(options_json);
    if (!root) return;

    // Voice cloning
    const char * voice_ref = cJSON_GetStringValue(cJSON_GetObjectItem(root, "voice_ref"));
    if (voice_ref && voice_ref[0]) {
        engine::runtime::VoiceCondition voice;
        voice.speaker = engine::runtime::VoiceReference{};
        auto wav = engine::audio::read_wav_f32(std::filesystem::path(voice_ref));
        engine::runtime::AudioBuffer ref_audio;
        ref_audio.sample_rate = wav.sample_rate;
        ref_audio.channels = wav.channels;
        ref_audio.samples = std::move(wav.samples);
        voice.speaker->audio = std::move(ref_audio);
        req.voice = std::move(voice);
    }

    // Transcript text(for forced aligner: audio + transcript contract)
    const char * text = cJSON_GetStringValue(cJSON_GetObjectItem(root, "text"));
    const char * lang = cJSON_GetStringValue(cJSON_GetObjectItem(root, "language"));
    if (text && text[0] && !req.text_input.has_value()) {
        req.text_input = engine::runtime::Transcript{};
        req.text_input->text = text;
        if (lang && lang[0]) {
            req.text_input->language = lang;
        }
    } else if (lang && lang[0] && req.text_input.has_value() && req.text_input->language.empty()) {
        req.text_input->language = lang;
    }

    // Iterate all string/number keys → options map
    for (auto * item = root->child; item; item = item->next) {
        if (!item->string) continue;
        std::string key = item->string;
        // Skip voice_ref (handled above)
        if (key == "voice_ref") continue;
        // String values
        if (cJSON_IsString(item)) {
            req.options[key] = item->valuestring;
        }
        // Number values
        else if (cJSON_IsNumber(item)) {
            req.options[key] = std::to_string(item->valuedouble);
        }
        // Bool values → "true"/"false"
        else if (cJSON_IsBool(item)) {
            req.options[key] = cJSON_IsTrue(item) ? "true" : "false";
        }
    }

    cJSON_Delete(root);
}

audiocpp_audio_t *audiocpp_tts(
    const audiocpp_model_t *model,
    const char *text,
    const char *options_json,
    audiocpp_error_t *err
) {
    audiocpp_audio_t *result = nullptr;
    AUDIOCPP_CATCH(err, {
        if (!model || !model->offline) {
            throw std::runtime_error("invalid model handle");
        }
        if (!text) {
            throw std::runtime_error("text is null");
        }
        engine::runtime::TaskRequest req;
        req.text_input = engine::runtime::Transcript{};
        req.text_input->text = text;
        apply_options(req, options_json);
        // Some models (e.g. Qwen3-ASR/TTS) require prepare() before run()
        model->session->prepare(engine::runtime::build_preparation_request(req));
        auto task_result = model->offline->run(req);
        if (!task_result.audio_output || task_result.audio_output->samples.empty()) {
            throw std::runtime_error("TTS produced no audio output");
        }
        const auto &buf = *task_result.audio_output;
        result = new audiocpp_audio_t{};
        result->n_samples = static_cast<int64_t>(buf.samples.size());
        result->sample_rate = buf.sample_rate;
        result->samples = static_cast<float *>(malloc(buf.samples.size() * sizeof(float)));
        if (!result->samples) {
            delete result;
            throw std::runtime_error("out of memory allocating audio output");
        }
        std::memcpy(result->samples, buf.samples.data(), buf.samples.size() * sizeof(float));
    });
    return result;
}

audiocpp_audio_t *audiocpp_tts_with_voice_ref(
    const audiocpp_model_t *model,
    const char *text,
    const char *options_json,
    const float *voice_ref_pcm,
    int64_t voice_ref_n,
    int voice_ref_sr,
    audiocpp_error_t *err
) {
    audiocpp_audio_t *result = nullptr;
    AUDIOCPP_CATCH(err, {
        if (!model || !model->offline) {
            throw std::runtime_error("invalid model handle");
        }
        if (!text) {
            throw std::runtime_error("text is null");
        }
        engine::runtime::TaskRequest req;
        req.text_input = engine::runtime::Transcript{};
        req.text_input->text = text;
        apply_options(req, options_json);

        // Inline voice reference PCM overrides file-based voice_ref
        if (voice_ref_pcm && voice_ref_n > 0) {
            engine::runtime::VoiceCondition voice;
            voice.speaker = engine::runtime::VoiceReference{};
            engine::runtime::AudioBuffer ref_audio;
            ref_audio.sample_rate = voice_ref_sr;
            ref_audio.channels = 1;
            ref_audio.samples.assign(voice_ref_pcm, voice_ref_pcm + voice_ref_n);
            voice.speaker->audio = std::move(ref_audio);
            req.voice = std::move(voice);
        }

        model->session->prepare(engine::runtime::build_preparation_request(req));
        auto task_result = model->offline->run(req);
        if (!task_result.audio_output || task_result.audio_output->samples.empty()) {
            throw std::runtime_error("TTS produced no audio output");
        }
        const auto &buf = *task_result.audio_output;
        result = new audiocpp_audio_t{};
        result->n_samples = static_cast<int64_t>(buf.samples.size());
        result->sample_rate = buf.sample_rate;
        result->samples = static_cast<float *>(malloc(buf.samples.size() * sizeof(float)));
        if (!result->samples) {
            delete result;
            throw std::runtime_error("out of memory allocating audio output");
        }
        std::memcpy(result->samples, buf.samples.data(), buf.samples.size() * sizeof(float));
    });
    return result;
}

/* ======================================================================== */
/* ASR                                                                       */
/* ======================================================================== */

audiocpp_text_t *audiocpp_asr(
    const audiocpp_model_t *model,
    const float *pcm,
    int64_t n_samples,
    int sample_rate,
    const char *options_json,
    audiocpp_error_t *err
) {
    audiocpp_text_t *result = nullptr;
    AUDIOCPP_CATCH(err, {
        if (!model || !model->offline) {
            throw std::runtime_error("invalid model handle");
        }
        if (!pcm || n_samples <= 0) {
            throw std::runtime_error("invalid audio input");
        }
        engine::runtime::TaskRequest req;
        req.audio_input = engine::runtime::AudioBuffer{};
        req.audio_input->sample_rate = sample_rate;
        req.audio_input->channels = 1;
        req.audio_input->samples.assign(pcm, pcm + n_samples);
        apply_options(req, options_json);
        // Some models (e.g. Qwen3-ASR) require prepare() before run()
        model->session->prepare(engine::runtime::build_preparation_request(req));
        auto task_result = model->offline->run(req);
        result = new audiocpp_text_t{};
        result->text = dup_cstr(
            task_result.text_output ? task_result.text_output->text : "");
        result->language = task_result.text_output && !task_result.text_output->language.empty()
            ? dup_cstr(task_result.text_output->language)
            : nullptr;
    });
    return result;
}

/* ======================================================================== */
/* Audio transform: audio → audio (Source Separation, Voice Conversion)      */
/* ======================================================================== */

audiocpp_audio_t *audiocpp_audio_transform(
    const audiocpp_model_t *model,
    const float *pcm,
    int64_t n_samples,
    int sample_rate,
    const char *options_json,
    audiocpp_error_t *err
) {
    audiocpp_audio_t *result = nullptr;
    AUDIOCPP_CATCH(err, {
        if (!model || !model->offline) {
            throw std::runtime_error("invalid model handle");
        }
        if (!pcm || n_samples <= 0) {
            throw std::runtime_error("invalid audio input");
        }
        engine::runtime::TaskRequest req;
        req.audio_input = engine::runtime::AudioBuffer{};
        req.audio_input->sample_rate = sample_rate;
        req.audio_input->channels = 1;
        req.audio_input->samples.assign(pcm, pcm + n_samples);
        apply_options(req, options_json);
        model->session->prepare(engine::runtime::build_preparation_request(req));
        auto task_result = model->offline->run(req);
        // VC/audio→audio:用 audio_output;SEP(HTDemucs):用 named_audio_outputs[0]
        // (多 stem 输出,返回第一个 — 通常是 "vocals")
        const engine::runtime::AudioBuffer *buf_ptr = nullptr;
        if (task_result.audio_output && !task_result.audio_output->samples.empty()) {
            buf_ptr = &(*task_result.audio_output);
        } else if (!task_result.named_audio_outputs.empty() &&
                   !task_result.named_audio_outputs[0].audio.samples.empty()) {
            buf_ptr = &task_result.named_audio_outputs[0].audio;
        }
        if (!buf_ptr) {
            throw std::runtime_error("audio transform produced no audio output");
        }
        const auto &buf = *buf_ptr;
        result = new audiocpp_audio_t{};
        result->n_samples = static_cast<int64_t>(buf.samples.size());
        result->sample_rate = buf.sample_rate;
        result->samples = static_cast<float *>(malloc(buf.samples.size() * sizeof(float)));
        if (!result->samples) {
            delete result;
            throw std::runtime_error("out of memory allocating audio output");
        }
        std::memcpy(result->samples, buf.samples.data(), buf.samples.size() * sizeof(float));
    });
    return result;
}

audiocpp_audio_t *audiocpp_audio_transform_with_voice_ref(
    const audiocpp_model_t *model,
    const float *pcm,
    int64_t n_samples,
    int sample_rate,
    const char *options_json,
    const float *voice_ref_pcm,
    int64_t voice_ref_n,
    int voice_ref_sr,
    audiocpp_error_t *err
) {
    audiocpp_audio_t *result = nullptr;
    AUDIOCPP_CATCH(err, {
        if (!model || !model->offline) {
            throw std::runtime_error("invalid model handle");
        }
        if (!pcm || n_samples <= 0) {
            throw std::runtime_error("invalid audio input");
        }
        engine::runtime::TaskRequest req;
        req.audio_input = engine::runtime::AudioBuffer{};
        req.audio_input->sample_rate = sample_rate;
        req.audio_input->channels = 1;
        req.audio_input->samples.assign(pcm, pcm + n_samples);
        apply_options(req, options_json);

        // Inline voice reference PCM (target speaker for VC)
        if (voice_ref_pcm && voice_ref_n > 0) {
            engine::runtime::VoiceCondition voice;
            voice.speaker = engine::runtime::VoiceReference{};
            engine::runtime::AudioBuffer ref_audio;
            ref_audio.sample_rate = voice_ref_sr;
            ref_audio.channels = 1;
            ref_audio.samples.assign(voice_ref_pcm, voice_ref_pcm + voice_ref_n);
            voice.speaker->audio = std::move(ref_audio);
            req.voice = std::move(voice);
        }

        model->session->prepare(engine::runtime::build_preparation_request(req));
        auto task_result = model->offline->run(req);
        const engine::runtime::AudioBuffer *buf_ptr = nullptr;
        if (task_result.audio_output && !task_result.audio_output->samples.empty()) {
            buf_ptr = &(*task_result.audio_output);
        } else if (!task_result.named_audio_outputs.empty() &&
                   !task_result.named_audio_outputs[0].audio.samples.empty()) {
            buf_ptr = &task_result.named_audio_outputs[0].audio;
        }
        if (!buf_ptr) {
            throw std::runtime_error("audio transform produced no audio output");
        }
        const auto &buf = *buf_ptr;
        result = new audiocpp_audio_t{};
        result->n_samples = static_cast<int64_t>(buf.samples.size());
        result->sample_rate = buf.sample_rate;
        result->samples = static_cast<float *>(malloc(buf.samples.size() * sizeof(float)));
        if (!result->samples) {
            delete result;
            throw std::runtime_error("out of memory allocating audio output");
        }
        std::memcpy(result->samples, buf.samples.data(), buf.samples.size() * sizeof(float));
    });
    return result;
}

audiocpp_diar_t *audiocpp_diar(
    const audiocpp_model_t *model,
    const float *pcm,
    int64_t n_samples,
    int sample_rate,
    const char *options_json,
    audiocpp_error_t *err
) {
    audiocpp_diar_t *result = nullptr;
    AUDIOCPP_CATCH(err, {
        if (!model || !model->offline) {
            throw std::runtime_error("invalid model handle");
        }
        if (!pcm || n_samples <= 0) {
            throw std::runtime_error("invalid audio input");
        }
        engine::runtime::TaskRequest req;
        req.audio_input = engine::runtime::AudioBuffer{};
        req.audio_input->sample_rate = sample_rate;
        req.audio_input->channels = 1;
        req.audio_input->samples.assign(pcm, pcm + n_samples);
        apply_options(req, options_json);
        model->session->prepare(engine::runtime::build_preparation_request(req));
        auto task_result = model->offline->run(req);

        const auto & turns = task_result.speaker_turns;
        result = new audiocpp_diar_t{};
        result->n_turns = static_cast<int64_t>(turns.size());
        if (turns.empty()) {
            result->turns = nullptr;
        } else {
            result->turns = static_cast<audiocpp_speaker_turn_t *>(
                calloc(turns.size(), sizeof(audiocpp_speaker_turn_t)));
            for (size_t i = 0; i < turns.size(); ++i) {
                result->turns[i].start_sample = turns[i].span.start_sample;
                result->turns[i].end_sample = turns[i].span.end_sample;
                result->turns[i].speaker_id = dup_cstr(turns[i].speaker_id);
                result->turns[i].confidence = turns[i].confidence;
            }
        }
    });
    return result;
}

/* ======================================================================== */
/* VAD                                                                       */
/* ======================================================================== */

audiocpp_vad_t *audiocpp_vad(
    const audiocpp_model_t *model,
    const float *pcm,
    int64_t n_samples,
    int sample_rate,
    const char *options_json,
    audiocpp_error_t *err
) {
    audiocpp_vad_t *result = nullptr;
    AUDIOCPP_CATCH(err, {
        if (!model || !model->offline) {
            throw std::runtime_error("invalid model handle");
        }
        if (!pcm || n_samples <= 0) {
            throw std::runtime_error("invalid audio input");
        }
        engine::runtime::TaskRequest req;
        req.audio_input = engine::runtime::AudioBuffer{};
        req.audio_input->sample_rate = sample_rate;
        req.audio_input->channels = 1;
        req.audio_input->samples.assign(pcm, pcm + n_samples);
        apply_options(req, options_json);
        model->session->prepare(engine::runtime::build_preparation_request(req));
        auto task_result = model->offline->run(req);

        const auto & segs = task_result.speech_segments;
        result = new audiocpp_vad_t{};
        result->n_segments = static_cast<int64_t>(segs.size());
        if (segs.empty()) {
            result->segments = nullptr;
        } else {
            result->segments = static_cast<audiocpp_vad_segment_t *>(
                calloc(segs.size(), sizeof(audiocpp_vad_segment_t)));
            for (size_t i = 0; i < segs.size(); ++i) {
                result->segments[i].start_sample = segs[i].span.start_sample;
                result->segments[i].end_sample = segs[i].span.end_sample;
                result->segments[i].confidence = segs[i].confidence;
            }
        }
    });
    return result;
}

/* ======================================================================== */
/* Utilities                                                                 */
/* ======================================================================== */

const char *audiocpp_version(void) {
    // audio.cpp doesn't expose a version string via the framework headers.
    // Return a static identifier; update per release.
    return "audio.cpp-capi-0.1.0";
}

void audiocpp_free_audio(audiocpp_audio_t *audio) {
    if (!audio) return;
    free(audio->samples);
    delete audio;
}

void audiocpp_free_text(audiocpp_text_t *text) {
    if (!text) return;
    free(text->text);
    free(text->language);
    delete text;
}

void audiocpp_free_diar(audiocpp_diar_t *diar) {
    if (!diar) return;
    if (diar->turns) {
        for (int64_t i = 0; i < diar->n_turns; ++i) {
            free(diar->turns[i].speaker_id);
        }
        free(diar->turns);
    }
    delete diar;
}

void audiocpp_free_vad(audiocpp_vad_t *vad) {
    if (!vad) return;
    free(vad->segments);
    delete vad;
}

/* ======================================================================== */
/* Forced Alignment                                                          */
/* ======================================================================== */

audiocpp_align_t *audiocpp_align(
    const audiocpp_model_t *model,
    const float *pcm,
    int64_t n_samples,
    int sample_rate,
    const char *text,
    const char *language,
    const char *options_json,
    audiocpp_error_t *err
) {
    audiocpp_align_t *result = nullptr;
    AUDIOCPP_CATCH(err, {
        if (!model || !model->offline) {
            throw std::runtime_error("invalid model handle");
        }
        if (!pcm || n_samples <= 0) {
            throw std::runtime_error("invalid audio input");
        }
        if (!text || text[0] == '\0') {
            throw std::runtime_error("align requires non-empty transcript text");
        }
        if (!language || language[0] == '\0') {
            throw std::runtime_error("align requires a language code (e.g. \"en\", \"zh\")");
        }

        engine::runtime::TaskRequest req;
        req.audio_input = engine::runtime::AudioBuffer{};
        req.audio_input->sample_rate = sample_rate;
        req.audio_input->channels = 1;
        req.audio_input->samples.assign(pcm, pcm + n_samples);
        req.text_input = engine::runtime::Transcript{};
        req.text_input->text = text;
        req.text_input->language = language;
        apply_options(req, options_json);

        model->session->prepare(engine::runtime::build_preparation_request(req));
        auto task_result = model->offline->run(req);

        const auto &wt = task_result.word_timestamps;
        const double inv_rate = sample_rate > 0 ? 1.0 / static_cast<double>(sample_rate) : 0.0;

        result = new audiocpp_align_t{};
        result->language = dup_cstr(language);
        result->n_words = static_cast<int64_t>(wt.size());
        if (wt.empty()) {
            result->words = nullptr;
        } else {
            result->words = static_cast<audiocpp_word_t *>(
                calloc(wt.size(), sizeof(audiocpp_word_t)));
            for (size_t i = 0; i < wt.size(); ++i) {
                result->words[i].start_seconds = static_cast<double>(wt[i].span.start_sample) * inv_rate;
                result->words[i].end_seconds = static_cast<double>(wt[i].span.end_sample) * inv_rate;
                result->words[i].word = dup_cstr(wt[i].word);
                result->words[i].confidence = wt[i].confidence;
            }
        }
    });
    return result;
}

void audiocpp_free_align(audiocpp_align_t *align) {
    if (!align) return;
    if (align->words) {
        for (int64_t i = 0; i < align->n_words; ++i) {
            free(align->words[i].word);
        }
        free(align->words);
    }
    free(align->language);
    delete align;
}

/* ======================================================================== */
/* Multi-stem audio transform                                               */
/* ======================================================================== */

audiocpp_stems_t *audiocpp_transform_stems(
    const audiocpp_model_t *model,
    const float *pcm,
    int64_t n_samples,
    int sample_rate,
    const char *options_json,
    const float *voice_ref_pcm,
    int64_t voice_ref_n,
    int voice_ref_sr,
    audiocpp_error_t *err
) {
    audiocpp_stems_t *result = nullptr;
    AUDIOCPP_CATCH(err, {
        if (!model || !model->offline) {
            throw std::runtime_error("invalid model handle");
        }
        if (!pcm || n_samples <= 0) {
            throw std::runtime_error("invalid audio input");
        }
        engine::runtime::TaskRequest req;
        req.audio_input = engine::runtime::AudioBuffer{};
        req.audio_input->sample_rate = sample_rate;
        req.audio_input->channels = 1;
        req.audio_input->samples.assign(pcm, pcm + n_samples);
        apply_options(req, options_json);
        if (voice_ref_pcm && voice_ref_n > 0) {
            engine::runtime::VoiceCondition voice;
            voice.speaker = engine::runtime::VoiceReference{};
            engine::runtime::AudioBuffer ref_audio;
            ref_audio.sample_rate = voice_ref_sr;
            ref_audio.channels = 1;
            ref_audio.samples.assign(voice_ref_pcm, voice_ref_pcm + voice_ref_n);
            voice.speaker->audio = std::move(ref_audio);
            req.voice = std::move(voice);
        }
        model->session->prepare(engine::runtime::build_preparation_request(req));
        auto task_result = model->offline->run(req);

        // Collect ALL named audio outputs as stems
        result = new audiocpp_stems_t{};
        auto collect_stem = [&](const char *name, const engine::runtime::AudioBuffer &buf) {
            audiocpp_stem_t stem;
            stem.name = dup_cstr(name);
            stem.sample_rate = buf.sample_rate;
            stem.n_samples = static_cast<int64_t>(buf.samples.size());
            stem.samples = static_cast<float *>(
                malloc(buf.samples.size() * sizeof(float)));
            if (stem.samples) {
                std::memcpy(stem.samples, buf.samples.data(),
                            buf.samples.size() * sizeof(float));
            }
            // Grow the stems array
            int64_t idx = result->n_stems;
            result->n_stems = idx + 1;
            result->stems = static_cast<audiocpp_stem_t *>(
                realloc(result->stems, result->n_stems * sizeof(audiocpp_stem_t)));
            result->stems[idx] = stem;
        };

        // If there's a primary audio_output, include it as "output"
        if (task_result.audio_output && !task_result.audio_output->samples.empty()) {
            collect_stem("output", *task_result.audio_output);
        }
        // Include all named outputs (vocals, drums, bass, instrumental, etc.)
        for (const auto &named : task_result.named_audio_outputs) {
            if (!named.audio.samples.empty()) {
                collect_stem(named.id.c_str(), named.audio);
            }
        }
    });
    return result;
}

void audiocpp_free_stems(audiocpp_stems_t *stems) {
    if (!stems) return;
    if (stems->stems) {
        for (int64_t i = 0; i < stems->n_stems; ++i) {
            free(stems->stems[i].name);
            free(stems->stems[i].samples);
        }
        free(stems->stems);
    }
    delete stems;
}

/* ======================================================================== */
/* Model inspection                                                         */
/* ======================================================================== */

int audiocpp_model_info(
    const audiocpp_model_t *model,
    audiocpp_model_info_t *out
) {
    if (!model || !model->loaded_model || !out) return -1;
    const auto &meta = model->loaded_model->metadata();
    out->family = dup_cstr(meta.family);
    out->variant = dup_cstr(meta.variant);
    out->description = dup_cstr(meta.description);
    return 0;
}

int audiocpp_model_capabilities(
    const audiocpp_model_t *model,
    audiocpp_model_capabilities_t *out
) {
    if (!model || !model->loaded_model || !out) return -1;
    memset(out, 0, sizeof(*out));
    const auto &caps = model->loaded_model->capabilities();
    out->supports_speaker_reference = caps.supports_speaker_reference ? 1 : 0;
    out->supports_style_condition = caps.supports_style_condition ? 1 : 0;
    out->supports_timestamps = caps.supports_timestamps ? 1 : 0;
    // Supported tasks
    out->n_supported_tasks = static_cast<int>(caps.supported_tasks.size());
    if (out->n_supported_tasks > 0) {
        out->supported_tasks = static_cast<int *>(
            calloc(out->n_supported_tasks, sizeof(int)));
        for (int i = 0; i < out->n_supported_tasks; ++i) {
            out->supported_tasks[i] = static_cast<int>(caps.supported_tasks[i].task);
        }
    }
    // Languages
    out->n_languages = static_cast<int>(caps.languages.size());
    if (out->n_languages > 0) {
        out->languages = static_cast<char **>(
            calloc(out->n_languages, sizeof(char *)));
        for (int i = 0; i < out->n_languages; ++i) {
            out->languages[i] = dup_cstr(caps.languages[i]);
        }
    }
    return 0;
}

void audiocpp_free_model_info(audiocpp_model_info_t *info) {
    if (!info) return;
    free(info->family);
    free(info->variant);
    free(info->description);
    memset(info, 0, sizeof(*info));
}

void audiocpp_free_capabilities(audiocpp_model_capabilities_t *caps) {
    if (!caps) return;
    free(caps->supported_tasks);
    if (caps->languages) {
        for (int i = 0; i < caps->n_languages; ++i) {
            free(caps->languages[i]);
        }
        free(caps->languages);
    }
    memset(caps, 0, sizeof(*caps));
}

/* ======================================================================== */
/* WAV I/O utilities                                                        */
/* ======================================================================== */

int audiocpp_read_wav(
    const char *path,
    float **out_samples,
    int64_t *out_n,
    int *out_rate
) {
    if (!path || !out_samples || !out_n || !out_rate) return -1;
    try {
        auto wav = engine::audio::read_wav_f32(std::filesystem::path(path));
        *out_n = static_cast<int64_t>(wav.samples.size());
        *out_rate = wav.sample_rate;
        *out_samples = static_cast<float *>(malloc(wav.samples.size() * sizeof(float)));
        if (!*out_samples) return -1;
        std::memcpy(*out_samples, wav.samples.data(), wav.samples.size() * sizeof(float));
        return 0;
    } catch (...) {
        return -1;
    }
}

int audiocpp_write_wav(
    const char *path,
    const float *samples,
    int64_t n_samples,
    int sample_rate
) {
    if (!path || !samples || n_samples <= 0) return -1;
    try {
        std::vector<float> vec(samples, samples + n_samples);
        engine::audio::write_pcm16_wav(
            std::filesystem::path(path),
            sample_rate,
            1,
            vec);
        return 0;
    } catch (...) {
        return -1;
    }
}

void audiocpp_free_string(char *str) {
    free(str);
}

void audiocpp_clear_error(audiocpp_error_t *err) {
    if (!err) return;
    if (err->message) {
        free(err->message);
        err->message = nullptr;
    }
    err->code = 0;
}

/* ======================================================================== */
/* Device enumeration                                                        */
/* ======================================================================== */

namespace {

// Map a ggml backend registration to our AUDIOCPP_BACKEND_* enum.
int backend_reg_to_id(ggml_backend_reg_t reg) {
    if (!reg) return AUDIOCPP_BACKEND_CPU;
#ifdef GGML_USE_CUDA
    if (reg == ggml_backend_cuda_reg()) return AUDIOCPP_BACKEND_CUDA;
#endif
#ifdef GGML_USE_SYCL
    if (reg == ggml_backend_sycl_reg()) return AUDIOCPP_BACKEND_SYCL;
#endif
#ifdef GGML_USE_VULKAN
    if (reg == ggml_backend_vk_reg()) return AUDIOCPP_BACKEND_VULKAN;
#endif
#ifdef GGML_USE_METAL
    if (reg == ggml_backend_metal_reg()) return AUDIOCPP_BACKEND_METAL;
#endif
    // CPU and any unknown backend
    return AUDIOCPP_BACKEND_CPU;
}

int dev_type_to_id(enum ggml_backend_dev_type type) {
    switch (type) {
        case GGML_BACKEND_DEVICE_TYPE_GPU:  return AUDIOCPP_DEVICE_GPU;
        case GGML_BACKEND_DEVICE_TYPE_IGPU: return AUDIOCPP_DEVICE_IGPU;
        default:                             return AUDIOCPP_DEVICE_CPU;
    }
}

void copy_cstr(char *dst, size_t dst_size, const char *src) {
    if (!src || dst_size == 0) {
        if (dst_size > 0) dst[0] = '\0';
        return;
    }
    size_t len = strlen(src);
    if (len >= dst_size) len = dst_size - 1;
    memcpy(dst, src, len);
    dst[len] = '\0';
}

}  // namespace

int audiocpp_device_count(void) {
    return static_cast<int>(ggml_backend_dev_count());
}

int audiocpp_device_info(int index, audiocpp_device_info_t *out) {
    if (!out || index < 0 || static_cast<size_t>(index) >= ggml_backend_dev_count()) {
        return -1;
    }
    memset(out, 0, sizeof(*out));

    ggml_backend_dev_t dev = ggml_backend_dev_get(static_cast<size_t>(index));
    if (!dev) return -1;

    ggml_backend_dev_props props;
    ggml_backend_dev_get_props(dev, &props);

    copy_cstr(out->name, sizeof(out->name), props.name);
    copy_cstr(out->description, sizeof(out->description), props.description);
    out->type = dev_type_to_id(props.type);
    out->memory_total = static_cast<uint64_t>(props.memory_total);
    out->memory_free = static_cast<uint64_t>(props.memory_free);

    // Determine which backend this device belongs to.
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    out->backend = backend_reg_to_id(reg);

    // Compute per-backend device index: count how many earlier devices in the
    // global list share the same backend reg. This aligns with
    // ggml_backend_cuda_init(device) / ggml_backend_sycl_init(device) which
    // take backend-relative indices.
    int per_backend_idx = 0;
    for (int i = 0; i < index; ++i) {
        ggml_backend_dev_t earlier = ggml_backend_dev_get(static_cast<size_t>(i));
        if (earlier && ggml_backend_dev_backend_reg(earlier) == reg) {
            ++per_backend_idx;
        }
    }
    out->device_id = per_backend_idx;

    return 0;
}

void audiocpp_list_devices(void) {
    int count = audiocpp_device_count();
    printf("audio.cpp devices (%d):\n", count);
    printf("  %-4s  %-10s  %-6s  %-10s  %-30s\n",
           "idx", "backend", "type", "device_id", "name");
    printf("  %s\n", "---------------------------------------------------------------");
    static const char *backend_names[] = {
        "CPU", "CUDA", "Vulkan", "Metal", "SYCL", "BEST"
    };
    static const char *type_names[] = {"CPU", "GPU", "IGPU"};
    for (int i = 0; i < count; ++i) {
        audiocpp_device_info_t info;
        if (audiocpp_device_info(i, &info) == 0) {
            const char *bn = (info.backend >= 0 && info.backend <= 5)
                ? backend_names[info.backend] : "?";
            const char *tn = (info.type >= 0 && info.type <= 2)
                ? type_names[info.type] : "?";
            printf("  %-4d  %-10s  %-6s  %-10d  %-30s\n",
                   i, bn, tn, info.device_id, info.name);
        }
    }
}

