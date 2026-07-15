#pragma once

#include "engine/framework/audio/wav_reader.h"

#include <filesystem>

namespace engine::audio {

WavData read_audio_f32(const std::filesystem::path & path);

}  // namespace engine::audio
