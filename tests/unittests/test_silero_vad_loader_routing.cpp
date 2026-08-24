// Unit tests for the silero_vad loader's file-claim policy in
// resolve_silero_assets — the gate behind SileroVADLoader::can_load.
//
// Original bug: the file branch of resolve_silero_assets accepted ANY
// existing file as a silero checkpoint. silero_vad is the first-registered
// loader, so every no-hint file-path load was claimed by it and died inside
// load() on "missing tensor: stft_conv.weight" — a tensor no other family
// ships. Two production incidents (qwen3-forced-aligner and
// higgs-audio-v3-stt GGUFs, both loaded without a family hint) were
// misdiagnosed for a long time because of that smoke bomb.
//
// Fork-regression guard: silero must only claim checkpoints that embed a
// silero_vad model spec or carry its stft_conv.weight tensor.

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/io/safetensors.h"
#include "engine/framework/runtime/registry.h"
#include "engine/models/silero_vad/assets.h"
#include "engine/models/silero_vad/session.h"
#include "test_assert.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<unsigned char> f32_bytes(size_t count) {
    const std::vector<float> values(count, 0.25F);
    std::vector<unsigned char> bytes(values.size() * sizeof(float));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
}

engine::runtime::ModelLoadRequest no_hint_request(const std::filesystem::path & path) {
    engine::runtime::ModelLoadRequest request;
    request.model_path = path;
    return request;
}

}  // namespace

int main() {
    namespace fs = std::filesystem;
    const auto root = fs::temp_directory_path() / "audiocpp_silero_loader_routing_test";
    fs::remove_all(root);
    fs::create_directories(root);

    const auto silero_loader = engine::models::silero_vad::make_silero_vad_loader();

    // 1. A foreign safetensors file (no stft_conv.weight) must be rejected by
    //    can_load, and resolve_silero_assets must say so instead of a tensor
    //    error from the silero runtime.
    const auto foreign_safetensors = root / "foreign.safetensors";
    engine::io::write_safetensors_file(foreign_safetensors, {
        {"encoder.weight", "F32", {4, 8}, f32_bytes(32)},
        {"decoder.bias", "F32", {4}, f32_bytes(4)},
    });
    engine::test::require(
        !silero_loader->can_load(no_hint_request(foreign_safetensors)),
        "silero can_load must reject a safetensors file without stft_conv.weight");
    bool rejected = false;
    try {
        (void) engine::models::silero_vad::resolve_silero_assets(foreign_safetensors);
    } catch (const std::exception & error) {
        rejected = true;
        const std::string message = error.what();
        engine::test::require(
            message.find("not a Silero VAD checkpoint") != std::string::npos &&
                message.find("missing tensor") == std::string::npos,
            "resolve_silero_assets must reject with the checkpoint message, got: " + message);
    }
    engine::test::require(rejected, "resolve_silero_assets must throw for a foreign file");

    // 2. The same weights as a spec-less GGUF: the tensor probe must still
    //    reject it (this is the exact shape of both production incidents).
    const auto foreign_gguf = root / "foreign.gguf";
    engine::assets::convert_tensor_source_to_gguf(
        foreign_safetensors, foreign_gguf, engine::assets::TensorStorageType::F32, false, false);
    engine::test::require(
        !silero_loader->can_load(no_hint_request(foreign_gguf)),
        "silero can_load must reject a spec-less GGUF without stft_conv.weight");

    // 3. A GGUF embedding a foreign-family model spec: definitive rejection.
    const auto foreign_spec_gguf = root / "foreign_spec.gguf";
    engine::assets::GgufEmbeddedModelSpec foreign_spec;
    foreign_spec.family = "qwen3_forced_aligner";
    foreign_spec.json = "{}";
    engine::assets::convert_tensor_sources_to_gguf(
        {{foreign_safetensors, ""}}, foreign_spec_gguf, engine::assets::TensorStorageType::F32,
        false, false, {}, {}, foreign_spec);
    engine::test::require(
        !silero_loader->can_load(no_hint_request(foreign_spec_gguf)),
        "silero can_load must reject a GGUF whose embedded spec names another family");

    // 4. A synthetic silero checkpoint (stft_conv.weight present) is claimed,
    //    both as safetensors and as a spec-less GGUF.
    const auto silero_checkpoint = root / "silero_vad_16k.safetensors";
    engine::io::write_safetensors_file(silero_checkpoint, {
        {"stft_conv.weight", "F32", {258, 1, 256}, f32_bytes(static_cast<size_t>(258 * 256))},
    });
    engine::test::require(
        silero_loader->can_load(no_hint_request(silero_checkpoint)),
        "silero can_load must accept a checkpoint carrying stft_conv.weight");
    const auto silero_gguf = root / "silero.gguf";
    engine::assets::convert_tensor_source_to_gguf(
        silero_checkpoint, silero_gguf, engine::assets::TensorStorageType::F32, false, false);
    engine::test::require(
        silero_loader->can_load(no_hint_request(silero_gguf)),
        "silero can_load must accept a spec-less GGUF carrying stft_conv.weight");

    // 5. A GGUF whose embedded spec names silero_vad is claimed via the spec.
    const auto silero_spec_gguf = root / "silero_spec.gguf";
    engine::assets::GgufEmbeddedModelSpec silero_spec;
    silero_spec.family = "silero_vad";
    silero_spec.json = "{}";
    engine::assets::convert_tensor_sources_to_gguf(
        {{silero_checkpoint, ""}}, silero_spec_gguf, engine::assets::TensorStorageType::F32,
        false, false, {}, {}, silero_spec);
    engine::test::require(
        silero_loader->can_load(no_hint_request(silero_spec_gguf)),
        "silero can_load must accept a GGUF whose embedded spec names silero_vad");

    // 6. The directory branch is untouched: a model dir with the canonical
    //    weight name resolves as before.
    const auto silero_dir = root / "silero_dir";
    fs::create_directories(silero_dir);
    fs::copy_file(silero_checkpoint, silero_dir / "silero_vad_16k.safetensors");
    const auto resolved = engine::models::silero_vad::resolve_silero_assets(silero_dir);
    engine::test::require_eq(
        resolved.checkpoint_path.string(),
        fs::weakly_canonical(silero_dir / "silero_vad_16k.safetensors").string(),
        "resolve_silero_assets directory branch checkpoint path");
    engine::test::require(
        silero_loader->can_load(no_hint_request(silero_dir)),
        "silero can_load must accept the canonical model directory");

    // 7. The original production scenario, end to end: a no-hint registry
    //    load of a foreign GGUF must never surface silero's tensor error.
    //    Either the right loader claims the file or the registry reports
    //    that nobody could — both are honest outcomes.
    try {
        (void) engine::runtime::make_default_registry().load(no_hint_request(foreign_gguf));
        std::printf("note: synthetic foreign GGUF was claimed by a loader\n");
    } catch (const std::exception & error) {
        const std::string message = error.what();
        engine::test::require(
            message.find("stft_conv.weight") == std::string::npos,
            "no-hint load of a foreign GGUF must not surface silero's tensor error, got: " + message);
    }

    fs::remove_all(root);
    std::printf("silero_vad_loader_routing_test: all cases passed\n");
    return 0;
}
