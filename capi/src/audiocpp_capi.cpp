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
        case AUDIOCPP_BACKEND_BEST:   return engine::core::BackendType::BestAvailable;
        default:                      return engine::core::BackendType::Cpu;
    }
}

static engine::runtime::VoiceTaskKind map_task(int task) {
    switch (task) {
        case AUDIOCPP_TASK_TTS: return engine::runtime::VoiceTaskKind::Tts;
        case AUDIOCPP_TASK_ASR: return engine::runtime::VoiceTaskKind::Asr;
        case AUDIOCPP_TASK_VAD: return engine::runtime::VoiceTaskKind::Vad;
        case AUDIOCPP_TASK_DIAR: return engine::runtime::VoiceTaskKind::Diarization;
        default:                 return engine::runtime::VoiceTaskKind::Tts;
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
/* Diarization                                                               */
/* ======================================================================== */

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
