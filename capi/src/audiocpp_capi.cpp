/**
 * audiocpp_capi.cpp — C ABI implementation for audio.cpp
 *
 * Wraps engine::runtime C++ interfaces into extern "C" functions.
 * All C++ exceptions are caught and converted to audiocpp_error_t.
 */

#include "audiocpp.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/registry.h"
#include "engine/framework/runtime/session.h"

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

audiocpp_audio_t *audiocpp_tts(
    const audiocpp_model_t *model,
    const char *text,
    const char *voice_path,
    float speed,
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
        req.options["speed"] = std::to_string(speed);
        // Voice cloning: load reference audio if provided
        if (voice_path && voice_path[0] != '\0') {
            // Voice reference loading depends on model; pass path as option
            req.options["voice_ref"] = voice_path;
        }
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
    const char *language,
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
        if (language && language[0] != '\0') {
            req.options["language"] = language;
        }
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
