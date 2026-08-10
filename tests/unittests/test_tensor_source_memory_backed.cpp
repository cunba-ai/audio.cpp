// Embedded (memory-backed) tensor sources have no backing file path: the bytes
// ARE the storage. release_storage() must therefore keep the buffer for them —
// clearing it made the next require_data_range() attempt to open the
// "<embedded>" source_path marker as a file ("failed to open binary file:
// <embedded>"). That surfaced when a loaded VAD model created a second session
// (e.g. audiocpp_stream_start after load_model_ex): both sessions re-upload
// the same shared weights. File-backed sources keep the lazy re-read path.

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/io/binary.h"
#include "test_assert.h"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// Minimal safetensors container in memory: u64 little-endian header length,
// then the JSON header, then the tensor data region (one F32 tensor "w").
std::vector<std::byte> make_memory_blob() {
    std::string header = R"({"w":{"dtype":"F32","shape":[4,4],"data_offsets":[0,64]}})";
    while ((sizeof(uint64_t) + header.size()) % 8 != 0) header.push_back(' ');
    std::vector<std::byte> blob(sizeof(uint64_t) + header.size() + 64);
    const uint64_t header_len = header.size();
    std::memcpy(blob.data(), &header_len, sizeof(header_len));
    std::memcpy(blob.data() + sizeof(header_len), header.data(), header.size());
    return blob;
}

void write_blob(const std::filesystem::path & path, const std::vector<std::byte> & blob) {
    std::ofstream out(path, std::ios::binary);
    engine::test::require(out.good(), "failed to open temp safetensors file");
    out.write(reinterpret_cast<const char *>(blob.data()), static_cast<std::streamsize>(blob.size()));
    engine::test::require(out.good(), "failed to write temp safetensors file");
}

// The regression: a memory-backed source must still serve tensors after
// release_storage() (no backing file to re-read).
void test_memory_backed_source_survives_release() {
    const auto blob = make_memory_blob();
    auto source = engine::assets::open_tensor_source_from_bytes(blob.data(), blob.size());
    const auto first = source->require_tensor_data("w");
    engine::test::require(first.bytes.size() == 64, "initial read should return the tensor");
    source->release_storage();
    const auto second = source->require_tensor_data("w");
    engine::test::require(second.bytes.size() == 64, "memory-backed re-read after release must succeed");
}

// Pins the file-backed contract: after release_storage() the bytes are lazily
// re-read from the backing path, so the source stays usable.
void test_file_backed_source_rereads_after_release() {
    static int counter = 0;
    const auto path = std::filesystem::temp_directory_path() /
                      ("tensor_source_reread_" + std::to_string(counter++) + ".safetensors");
    write_blob(path, make_memory_blob());
    auto source = engine::assets::open_tensor_source(path);
    const auto first = source->require_tensor_data("w");
    engine::test::require(first.bytes.size() == 64, "initial read should return the tensor");
    source->release_storage();
    const auto second = source->require_tensor_data("w");
    engine::test::require(second.bytes.size() == 64, "file-backed re-read after release must succeed");
    std::filesystem::remove(path);
}

}  // namespace

int main() {
    try {
        test_memory_backed_source_survives_release();
        test_file_backed_source_rereads_after_release();
    } catch (const std::exception & error) {
        std::cerr << "tensor source memory-backed test failed: " << error.what() << "\n";
        return 1;
    }
    std::cout << "tensor source memory-backed test passed\n";
    return 0;
}
