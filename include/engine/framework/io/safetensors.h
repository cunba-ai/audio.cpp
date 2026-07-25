#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine::io {

struct SafeTensorInfo {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    size_t data_begin = 0;
    size_t data_end = 0;
};

struct SafeTensorIndex {
    std::filesystem::path source_path;
    std::unordered_map<std::string, std::string> metadata;
    std::unordered_map<std::string, SafeTensorInfo> tensors;
    size_t header_bytes = 0;
};

struct SafeTensorWriteEntry {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    std::vector<unsigned char> data;
};

SafeTensorIndex load_safetensors_index(const std::filesystem::path & path);

// Parse a safetensors index directly from an in-memory byte buffer (the full
// file bytes: 8-byte header-length prefix + header JSON + tensor data). Used by
// the embedded-asset path so a SafeTensorSource can be built without a file on
// disk. source_path is set to "<embedded>" since there is no backing file.
SafeTensorIndex load_safetensors_index_from_bytes(const std::byte * data, size_t size);

void write_safetensors_file(
    const std::filesystem::path & path,
    const std::vector<SafeTensorWriteEntry> & entries);

}  // namespace engine::io
