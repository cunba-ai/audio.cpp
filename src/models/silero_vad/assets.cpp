#include "engine/models/silero_vad/assets.h"

#include "engine/framework/assets/embedded.h"
#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/io/filesystem.h"

#include <cstring>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace engine::models::silero_vad {
namespace {

std::filesystem::path default_weight_candidate(const std::filesystem::path & model_root) {
    return model_root / "silero_vad_16k.safetensors";
}

SileroWeights load_silero_weights(const std::filesystem::path & checkpoint_path) {
    assets::ResourceBundle resources(checkpoint_path.parent_path());
    resources.add_file("weights", checkpoint_path);
    SileroWeights weights;
    weights.source = resources.open_tensor_source("weights");
    return weights;
}

bool file_starts_with_gguf_magic(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary);
    char magic[4] = {};
    return input.read(magic, 4) && std::memcmp(magic, "GGUF", 4) == 0;
}

// Restrict the file branch of resolve_silero_assets to checkpoints that are
// provably silero. silero_vad is the first-registered loader, so with no
// family_hint its can_load used to claim ANY existing file and fail inside
// load() on a silero-only tensor name ("missing tensor: stft_conv.weight"),
// which misdirects debugging for files that belong to other families. Both
// probes are header-only (GGUF init is no_alloc, safetensors parsing reads
// the JSON header), so can_load stays cheap even for multi-GB checkpoints.
bool looks_like_silero_checkpoint(const std::filesystem::path & checkpoint_path) {
    // The magic pre-check keeps ggml from logging "invalid magic" errors for
    // the safetensors checkpoints that go through the tensor probe below.
    if (file_starts_with_gguf_magic(checkpoint_path)) {
        try {
            const auto model_spec = engine::assets::read_gguf_embedded_model_spec(checkpoint_path);
            if (model_spec.has_value()) {
                return model_spec->family == "silero_vad";
            }
        } catch (...) {
            // GGUF with unreadable spec metadata; fall through to the probe.
        }
    }
    try {
        // First weight SileroVADRuntime loads; no other family ships it.
        return engine::assets::open_tensor_source(checkpoint_path)->has_tensor("stft_conv.weight");
    } catch (...) {
        return false;
    }
}

std::string checkpoint_cache_key(const std::filesystem::path & checkpoint_path) {
    std::error_code ec;
    const auto canonical = std::filesystem::weakly_canonical(checkpoint_path, ec);
    return ec ? checkpoint_path.lexically_normal().string() : canonical.string();
}

// Load silero weights from the embedded asset bytes (requires
// AUDIOCPP_EMBED_VAD_ASSETS=ON). Used by load_silero_weights_embedded().
std::shared_ptr<const SileroWeights> load_silero_weights_from_embedded() {
    std::size_t size = 0;
    const auto * data = assets::embedded::embedded_asset_data("silero_vad", &size);
    if (data == nullptr) {
        return nullptr;
    }
    SileroWeights weights;
    weights.source = assets::open_tensor_source_from_bytes(data, size);
    return std::make_shared<const SileroWeights>(std::move(weights));
}

}  // namespace

SileroAssetPaths resolve_silero_assets(const std::filesystem::path & model_path) {
    if (engine::io::is_existing_file(model_path)) {
        const auto checkpoint_path = std::filesystem::weakly_canonical(model_path);
        if (!looks_like_silero_checkpoint(checkpoint_path)) {
            throw std::runtime_error("not a Silero VAD checkpoint: " + checkpoint_path.string());
        }
        SileroAssetPaths paths;
        paths.model_root = checkpoint_path.parent_path();
        paths.checkpoint_path = checkpoint_path;
        return paths;
    }
    if (!engine::io::is_existing_directory(model_path)) {
        throw std::runtime_error("Silero VAD model path does not exist: " + model_path.string());
    }
    const auto checkpoint_path = default_weight_candidate(model_path);
    if (!engine::io::is_existing_file(checkpoint_path)) {
        throw std::runtime_error("Silero VAD weights not found: " + checkpoint_path.string());
    }
    SileroAssetPaths paths;
    paths.model_root = std::filesystem::weakly_canonical(model_path);
    paths.checkpoint_path = std::filesystem::weakly_canonical(checkpoint_path);
    return paths;
}

std::shared_ptr<const SileroWeights> load_silero_weights_cached(const std::filesystem::path & checkpoint_path) {
    static std::mutex cache_mutex;
    static std::unordered_map<std::string, std::weak_ptr<const SileroWeights>> cache;
    const auto key = checkpoint_cache_key(checkpoint_path);
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        if (const auto it = cache.find(key); it != cache.end()) {
            if (auto existing = it->second.lock()) {
                return existing;
            }
        }
    }
    auto loaded = std::make_shared<const SileroWeights>(load_silero_weights(checkpoint_path));
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        cache[key] = loaded;
    }
    return loaded;
}

std::shared_ptr<const SileroWeights> load_silero_weights_embedded() {
    static std::mutex cache_mutex;
    static std::weak_ptr<const SileroWeights> cached;
    std::lock_guard<std::mutex> lock(cache_mutex);
    if (auto existing = cached.lock()) {
        return existing;
    }
    auto loaded = load_silero_weights_from_embedded();
    cached = loaded;
    return loaded;
}

}  // namespace engine::models::silero_vad
