// Shared weight buffer tests: two BackendWeightStore instances inside the
// same ScopedWeightShareKey scope must share one backend buffer (same tensor
// data pointers, no second copy), while different keys / conflicting
// fingerprints must stay independent. Also covers reference-counted lifetime
// (buffer freed only when the last user dies) and concurrent first uploads.
//
// The sharing machinery is backend-agnostic (it only binds tensor->data into
// an existing backend buffer); the CPU backend exercises the same code path
// the CUDA/ROCm backends take.

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/shared_weight_registry.h"

#include "test_assert.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

namespace {

using engine::core::BackendType;
using engine::core::BackendWeightStore;
using engine::core::ScopedWeightShareKey;
using engine::core::TensorShape;

constexpr size_t kContextBytes = 8ull * 1024ull * 1024ull;  // 8 MB metadata budget

std::vector<float> read_tensor_f32(const engine::core::TensorValue & value) {
    const auto * data = static_cast<const float *>(value.tensor->data);
    const size_t count = static_cast<size_t>(value.tensor->ne[0]) *
                         (ggml_n_dims(value.tensor) > 1 ? static_cast<size_t>(value.tensor->ne[1]) : 1);
    return std::vector<float>(data, data + count);
}

// Two stores with the same share key must bind to ONE buffer: tensor data
// pointers are identical and values match.
void test_same_key_shares() {
    auto * backend = engine::core::init_backend({BackendType::Cpu, 0, 1});
    engine::test::require(backend != nullptr, "failed to init CPU backend");

    {
        ScopedWeightShareKey key("test-model-a");

        BackendWeightStore first(backend, BackendType::Cpu, "first", kContextBytes);
        auto w1 = first.make_f32(TensorShape::from_dims({4}), {1.0f, 2.0f, 3.0f, 4.0f});
        auto w2 = first.make_f32(TensorShape::from_dims({2, 3}), {1, 2, 3, 4, 5, 6});
        first.upload();

        BackendWeightStore second(backend, BackendType::Cpu, "second", kContextBytes);
        auto v1 = second.make_f32(TensorShape::from_dims({4}), {1.0f, 2.0f, 3.0f, 4.0f});
        auto v2 = second.make_f32(TensorShape::from_dims({2, 3}), {1, 2, 3, 4, 5, 6});
        second.upload();

        // Same physical memory for both stores: no second upload happened.
        engine::test::require(
            w1.tensor->data == v1.tensor->data,
            "same-key stores must share tensor data pointers");
        engine::test::require(
            w2.tensor->data == v2.tensor->data,
            "same-key stores must share tensor data pointers (2nd tensor)");

        // Values are intact through the shared buffer.
        const auto a = read_tensor_f32(w1);
        const auto b = read_tensor_f32(v1);
        engine::test::require(a == std::vector<float>({1, 2, 3, 4}), "provider values corrupted");
        engine::test::require(b == a, "consumer sees different values than provider");
    }

    ggml_backend_free(backend);
}

// Different share keys must not share memory.
void test_different_keys_do_not_share() {
    auto * backend = engine::core::init_backend({BackendType::Cpu, 0, 1});
    engine::test::require(backend != nullptr, "failed to init CPU backend");

    {
        BackendWeightStore first(backend, BackendType::Cpu, "first", kContextBytes);
        ScopedWeightShareKey key_a("test-model-a");
        auto w1 = first.make_f32(TensorShape::from_dims({4}), {1, 2, 3, 4});
        first.upload();

        BackendWeightStore second(backend, BackendType::Cpu, "second", kContextBytes);
        ScopedWeightShareKey key_b("test-model-b");
        auto v1 = second.make_f32(TensorShape::from_dims({4}), {1, 2, 3, 4});
        second.upload();

        engine::test::require(
            w1.tensor->data != v1.tensor->data,
            "different-key stores must allocate separate buffers");
    }

    ggml_backend_free(backend);
}

// No scope at all: sharing disabled, behavior identical to before.
void test_no_scope_is_independent() {
    auto * backend = engine::core::init_backend({BackendType::Cpu, 0, 1});
    engine::test::require(backend != nullptr, "failed to init CPU backend");

    {
        BackendWeightStore first(backend, BackendType::Cpu, "first", kContextBytes);
        auto w1 = first.make_f32(TensorShape::from_dims({4}), {1, 2, 3, 4});
        first.upload();

        BackendWeightStore second(backend, BackendType::Cpu, "second", kContextBytes);
        auto v1 = second.make_f32(TensorShape::from_dims({4}), {1, 2, 3, 4});
        second.upload();

        engine::test::require(
            w1.tensor->data != v1.tensor->data,
            "stores outside a share scope must stay independent");
    }

    ggml_backend_free(backend);
}

// Lifetime: the shared buffer must outlive the provider store (the consumer
// keeps it alive), and must be re-uploaded after the last user dies.
void test_reference_counted_lifetime() {
    auto * backend = engine::core::init_backend({BackendType::Cpu, 0, 1});
    engine::test::require(backend != nullptr, "failed to init CPU backend");

    void * shared_data = nullptr;
    {
        ScopedWeightShareKey key("test-model-lifetime");

        BackendWeightStore * first = new BackendWeightStore(backend, BackendType::Cpu, "first", kContextBytes);
        auto w1 = first->make_f32(TensorShape::from_dims({4}), {7, 8, 9, 10});
        first->upload();
        shared_data = w1.tensor->data;

        BackendWeightStore * second = new BackendWeightStore(backend, BackendType::Cpu, "second", kContextBytes);
        auto v1 = second->make_f32(TensorShape::from_dims({4}), {7, 8, 9, 10});
        second->upload();
        engine::test::require(v1.tensor->data == shared_data, "consumer must bind to provider buffer");

        // Provider dies first; consumer must keep the shared buffer alive.
        delete first;
        const auto values = read_tensor_f32(v1);
        engine::test::require(
            values == std::vector<float>({7, 8, 9, 10}),
            "shared weights must survive the provider store");

        // Consumer dies: last reference gone, buffer freed. A new store for
        // the same key re-uploads into a fresh buffer.
        delete second;
    }

    {
        ScopedWeightShareKey key("test-model-lifetime");
        BackendWeightStore third(backend, BackendType::Cpu, "third", kContextBytes);
        auto w3 = third.make_f32(TensorShape::from_dims({4}), {7, 8, 9, 10});
        third.upload();
        engine::test::require(
            w3.tensor->data != shared_data,
            "after all users die the next store must allocate a fresh buffer");
    }

    ggml_backend_free(backend);
}

// Fingerprint conflict: same key, different tensor set -> no sharing, second
// store allocates independently and its values are its own.
void test_fingerprint_conflict_falls_back() {
    auto * backend = engine::core::init_backend({BackendType::Cpu, 0, 1});
    engine::test::require(backend != nullptr, "failed to init CPU backend");

    {
        ScopedWeightShareKey key("test-model-conflict");

        BackendWeightStore first(backend, BackendType::Cpu, "first", kContextBytes);
        auto w1 = first.make_f32(TensorShape::from_dims({4}), {1, 2, 3, 4});
        first.upload();

        BackendWeightStore second(backend, BackendType::Cpu, "second", kContextBytes);
        auto v1 = second.make_f32(TensorShape::from_dims({8}), {5, 6, 7, 8, 9, 10, 11, 12});
        second.upload();

        engine::test::require(
            w1.tensor->data != v1.tensor->data,
            "fingerprint conflict must not reuse a foreign buffer");
        const auto values = read_tensor_f32(v1);
        engine::test::require(
            values == std::vector<float>({5, 6, 7, 8, 9, 10, 11, 12}),
            "conflict fallback must keep the second store's own values");
    }

    ggml_backend_free(backend);
}

// Concurrent first uploads for the same key: exactly one provider wins; both
// stores end up on the same buffer.
void test_concurrent_first_uploads_share() {
    auto * backend = engine::core::init_backend({BackendType::Cpu, 0, 1});
    engine::test::require(backend != nullptr, "failed to init CPU backend");

    void * a_data = nullptr;
    void * b_data = nullptr;
    std::atomic<bool> a_ready{false};
    std::atomic<bool> b_ready{false};

    std::thread a([&]() {
        ScopedWeightShareKey key("test-model-concurrent");
        BackendWeightStore store(backend, BackendType::Cpu, "a", kContextBytes);
        auto w = store.make_f32(TensorShape::from_dims({4}), {1, 2, 3, 4});
        store.upload();
        a_data = w.tensor->data;
        a_ready.store(true, std::memory_order_release);
        while (!b_ready.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
    });

    std::thread b([&]() {
        ScopedWeightShareKey key("test-model-concurrent");
        BackendWeightStore store(backend, BackendType::Cpu, "b", kContextBytes);
        auto w = store.make_f32(TensorShape::from_dims({4}), {1, 2, 3, 4});
        store.upload();
        b_data = w.tensor->data;
        b_ready.store(true, std::memory_order_release);
        while (!a_ready.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
    });

    a.join();
    b.join();

    engine::test::require(
        a_data != nullptr && a_data == b_data,
        "concurrent same-key uploads must share one buffer");

    ggml_backend_free(backend);
}

}  // namespace

int main() {
    try {
        test_same_key_shares();
        test_different_keys_do_not_share();
        test_no_scope_is_independent();
        test_reference_counted_lifetime();
        test_fingerprint_conflict_falls_back();
        test_concurrent_first_uploads_share();
        std::cout << "test_shared_weight_registry: PASS\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "test_shared_weight_registry: FAIL — " << e.what() << "\n";
        return 1;
    }
}
