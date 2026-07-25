#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>

namespace engine::assets {
class TensorSource;
}

namespace engine::models::silero_vad {

struct SileroWeights {
    std::shared_ptr<const assets::TensorSource> source;
};

struct SileroAssetPaths {
    std::filesystem::path model_root;
    std::filesystem::path checkpoint_path;
};

SileroAssetPaths resolve_silero_assets(const std::filesystem::path & model_path);
std::shared_ptr<const SileroWeights> load_silero_weights_cached(const std::filesystem::path & checkpoint_path);

// Load silero weights from the embedded asset bytes (requires
// AUDIOCPP_EMBED_VAD_ASSETS=ON at build time). Returns nullptr if no embedded
// silero asset is present in this build.
std::shared_ptr<const SileroWeights> load_silero_weights_embedded();

}  // namespace engine::models::silero_vad
