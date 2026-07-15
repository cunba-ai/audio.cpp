#include "engine/framework/audio/audio_reader.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wzero-as-null-pointer-constant"
#endif
#define MINIMP3_FLOAT_OUTPUT
#define MINIMP3_IMPLEMENTATION
#include "minimp3_ex.h"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

namespace engine::audio {
namespace {

std::string lower_extension(const std::filesystem::path & path) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return ext;
}

std::vector<uint8_t> read_file_prefix(const std::filesystem::path & path, size_t max_bytes) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("could not open audio input: " + path.string());
    }
    std::vector<uint8_t> data(max_bytes);
    input.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(data.size()));
    data.resize(static_cast<size_t>(input.gcount()));
    if (data.empty() && input.bad()) {
        throw std::runtime_error("failed to read audio input: " + path.string());
    }
    return data;
}

std::vector<uint8_t> read_file_bytes(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("could not open audio input: " + path.string());
    }
    input.seekg(0, std::ios::end);
    const auto size = input.tellg();
    if (size < 0) {
        throw std::runtime_error("failed to size audio input: " + path.string());
    }
    input.seekg(0, std::ios::beg);
    std::vector<uint8_t> data(static_cast<size_t>(size));
    if (!data.empty()) {
        input.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(data.size()));
        if (!input) {
            throw std::runtime_error("failed to read audio input: " + path.string());
        }
    }
    return data;
}

bool has_wav_header(const std::vector<uint8_t> & data) {
    return data.size() >= 12 &&
        std::string(reinterpret_cast<const char *>(data.data()), 4) == "RIFF" &&
        std::string(reinterpret_cast<const char *>(data.data() + 8), 4) == "WAVE";
}

WavData read_mp3_f32(const std::filesystem::path & path, const std::vector<uint8_t> & data) {
    if (data.empty()) {
        throw std::runtime_error("empty MP3 input: " + path.string());
    }

    mp3dec_t decoder;
    mp3dec_init(&decoder);
    mp3dec_file_info_t info{};
    const int rc = mp3dec_load_buf(&decoder, data.data(), data.size(), &info, nullptr, nullptr);
    if (rc != 0) {
        throw std::runtime_error("failed to decode MP3 input: " + path.string());
    }
    if (info.buffer == nullptr || info.samples == 0 || info.hz <= 0 || info.channels <= 0) {
        std::free(info.buffer);
        throw std::runtime_error("decoded MP3 input is empty: " + path.string());
    }

    WavData audio;
    audio.sample_rate = info.hz;
    audio.channels = info.channels;
    audio.samples.assign(info.buffer, info.buffer + info.samples);
    std::free(info.buffer);
    return audio;
}

bool is_mp3_extension(const std::string & ext) {
    return ext == ".mp3" || ext == ".mpa" || ext == ".mpeg";
}

}  // namespace

WavData read_audio_f32(const std::filesystem::path & path) {
    const auto prefix = read_file_prefix(path, 16);
    if (prefix.empty()) {
        throw std::runtime_error("empty audio input: " + path.string());
    }
    if (has_wav_header(prefix)) {
        return read_wav_f32(path);
    }

    const auto ext = lower_extension(path);
    if (ext == ".wav") {
        throw std::runtime_error("invalid WAV RIFF header: " + path.string());
    }
    if (!ext.empty() && !is_mp3_extension(ext)) {
        throw std::runtime_error("unsupported audio input format: " + path.string() + " (supported: WAV, MP3)");
    }

    const auto data = read_file_bytes(path);
    const bool extension_says_mp3 = is_mp3_extension(ext);
    if (extension_says_mp3 || mp3dec_detect_buf(data.data(), data.size()) != 0) {
        return read_mp3_f32(path, data);
    }

    throw std::runtime_error("unsupported audio input format: " + path.string() + " (supported: WAV, MP3)");
}

}  // namespace engine::audio
