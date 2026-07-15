#include "engine/models/outetts/loader.h"

#include "engine/framework/assets/model_package.h"
#include "engine/models/outetts/session.h"

namespace engine::models::outetts {
namespace {

std::filesystem::path spec_path() {
  return assets::default_model_package_spec_path("outetts");
}

runtime::CapabilitySet make_capabilities() {
  runtime::CapabilitySet out;
  out.supported_tasks = {
      {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline}},
      {runtime::VoiceTaskKind::VoiceCloning, {runtime::RunMode::Offline}},
  };
  out.supports_speaker_reference = true;
  out.languages = {
      "Auto",       "Arabic",  "Belarusian", "Bengali",    "Chinese",
      "Dutch",      "English", "French",     "Georgian",   "German",
      "Hungarian",  "Italian", "Japanese",   "Korean",     "Latvian",
      "Lithuanian", "Persian", "Polish",     "Portuguese", "Russian",
      "Spanish",    "Swahili", "Tamil",      "Ukrainian",
  };
  return out;
}

class Loader final : public runtime::IVoiceModelLoader {
public:
  std::string family() const override { return "outetts"; }
  bool can_load(const runtime::ModelLoadRequest &request) const override {
    if (request.family_hint.has_value() && *request.family_hint != family())
      return false;
    try {
      (void)assets::load_resource_bundle_from_package_spec(request.model_path,
                                                           spec_path());
      return true;
    } catch (...) {
      return false;
    }
  }
  runtime::ModelInspection
  inspect(const runtime::ModelLoadRequest &request) const override {
    const auto model_assets = load_outetts_assets(request.model_path);
    runtime::ModelInspection out;
    out.model_root = model_assets->resources.model_root();
    out.metadata.family = family();
    out.metadata.variant = "1.0-1B";
    out.metadata.description = "OuteTTS 1.0 1B with native IBM DAC speech "
                               "synthesis and voice cloning.";
    out.capabilities = make_capabilities();
    out.cli.request_options = {
        {"max_tokens", "n", "Maximum generated text/audio tokens."},
        {"temperature", "float",
         "Sampling temperature; official cloning default 0.4."},
        {"top_k", "n", "Top-k sampling; official default 40."},
        {"top_p", "float", "Nucleus sampling; official default 0.9."},
        {"min_p", "float",
         "Minimum probability relative to the best token; official default "
         "0.05."},
        {"repetition_penalty", "float",
         "Windowed repetition penalty; official default 1.1."},
        {"repetition_window", "n",
         "Recent-token penalty window; official value 64."},
        {"seed", "n",
         "Sampling seed; cloning defaults to 4099 for native weights and "
         "42 for quantized weights."},
        {"reference_text", "text",
         "Transcript matching the --voice-ref audio for voice cloning."},
        {"reference_language", "code",
         "Language code used to align the reference transcript; default en."},
        {"text_chunk_size", "n",
         "Framework long-form text chunk size; default 2048 characters."},
        {"text_chunk_mode", "default|tag_aware|japanese|endline",
         "Framework long-form text chunking mode."},
    };
    out.cli.session_options = {
        {"outetts.weight_type", "native|f32|f16|bf16|q8_0",
         "Language-model weight storage type. Quantized CUDA voice cloning "
         "is expanded to F32 in memory for generation correctness."},
        {"outetts.llama_weight_context_mb", "n",
         "Language-model weight context size in MiB."},
        {"outetts.constant_context_mb", "n",
         "Language-model constant tensor context size in MiB."},
        {"outetts.dac_weight_context_mb", "n",
         "DAC decoder weight context size in MiB."},
        {"outetts.dac_graph_context_mb", "n",
         "DAC decoder graph context size in MiB."},
        {"outetts.aligner_model_path", "path",
         "Optional Qwen3 Forced Aligner override. Cloning automatically uses "
         "the aligner embedded in a standalone OuteTTS GGUF when present."},
        {"outetts.reference_cache_slots", "n",
         "Prepared reference-profile cache slots; default 1, set 0 to "
         "disable."},
        {"outetts.mem_saver", "true|false",
         "Release cached-step and aligner runtime state after use; default "
         "false."},
    };
    out.discovered_configs = runtime::discover_named_assets_from_package_spec(
        request.model_path, spec_path(),
        assets::ModelPackageResourceKind::Files);
    out.discovered_weights = runtime::discover_named_assets_from_package_spec(
        request.model_path, spec_path(),
        assets::ModelPackageResourceKind::Tensors);
    return out;
  }
  std::unique_ptr<runtime::ILoadedVoiceModel>
  load(const runtime::ModelLoadRequest &request) const override {
    return std::make_unique<OuteTTSLoadedModel>(
        load_outetts_assets(request.model_path));
  }
};

} // namespace

OuteTTSLoadedModel::OuteTTSLoadedModel(
    std::shared_ptr<const OuteTTSAssets> assets)
    : assets_(std::move(assets)), capabilities_(make_capabilities()) {
  metadata_.family = "outetts";
  metadata_.variant = "1.0-1B";
  metadata_.description =
      "OuteTTS 1.0 1B with native IBM DAC speech synthesis and voice cloning.";
}

const runtime::ModelMetadata &OuteTTSLoadedModel::metadata() const noexcept {
  return metadata_;
}
const runtime::CapabilitySet &
OuteTTSLoadedModel::capabilities() const noexcept {
  return capabilities_;
}
std::unique_ptr<runtime::IVoiceTaskSession>
OuteTTSLoadedModel::create_task_session(
    const runtime::TaskSpec &task,
    const runtime::SessionOptions &options) const {
  return std::make_unique<OuteTTSSession>(task, options, assets_);
}

std::shared_ptr<runtime::IVoiceModelLoader> make_outetts_loader() {
  return std::make_shared<Loader>();
}

} // namespace engine::models::outetts
