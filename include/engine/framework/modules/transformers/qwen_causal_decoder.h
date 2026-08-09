#pragma once

#include "engine/framework/core/module.h"
#include "engine/framework/modules/transformers/qwen_decoder.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/runtime/kv_cache.h"

#include <cstdint>
#include <optional>
#include <vector>

#include <ggml.h>

namespace engine::modules {

enum class QwenCausalDecoderLogitsMode {
    LastStep,
    AllSteps,
};

struct QwenCausalDecoderConfig {
    QwenDecoderStackConfig stack;
    int64_t logits_size = 0;
    QwenCausalDecoderLogitsMode logits_mode = QwenCausalDecoderLogitsMode::LastStep;
    bool use_lm_head_bias = false;
    ggml_prec lm_head_precision = GGML_PREC_DEFAULT;
    std::optional<ggml_type> lm_head_input_type;
};

struct QwenCausalDecoderWeights {
    QwenDecoderStackWeights stack;
    NormWeights final_norm;
    LinearWeights lm_head;
};

struct QwenCausalDecoderOutputs {
    core::TensorValue sequence;
    core::TensorValue hidden;
    core::TensorValue logits;
    QwenDecoderStackState state;
};

struct QwenCausalDecoderStaticCacheOutputs {
    core::TensorValue sequence;
    core::TensorValue hidden;
    core::TensorValue logits;
    runtime::TransformerKVCache cache;
    // Per-layer K/V cache tensors. For the single-sequence path they are the
    // same tensors wrapped by `cache`; the batched path fills them instead
    // (TransformerKVCache stays empty because a batched static cache is owned
    // by the caller's graph).
    std::vector<core::TensorValue> cache_keys;
    std::vector<core::TensorValue> cache_values;
};

class QwenCausalDecoderModule {
public:
    explicit QwenCausalDecoderModule(QwenCausalDecoderConfig config);

    const QwenCausalDecoderConfig & config() const noexcept;

    QwenCausalDecoderOutputs build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const core::TensorValue & positions,
        const QwenCausalDecoderWeights & weights,
        const std::optional<QwenDecoderStackState> & prefix_state = std::nullopt,
        const std::optional<core::TensorValue> & attention_mask = std::nullopt) const;

    QwenCausalDecoderStaticCacheOutputs build_static_cache_tail(
        core::ModuleBuildContext & ctx,
        ggml_cgraph * graph,
        const core::TensorValue & input,
        const core::TensorValue & positions,
        const QwenCausalDecoderWeights & weights,
        int64_t cache_steps,
        const core::TensorValue & attention_mask,
        const std::optional<core::TensorValue> & cache_slot = std::nullopt) const;

    // Batched decode-step variant (n_seqs sequences in lockstep): input is
    // [n_seqs, 1, hidden], positions is [n_seqs], cache_slot is [n_seqs] and
    // attention_mask is [n_seqs, 1, 1, cache_steps] with a per-sequence
    // visible prefix. Each returned cache tensor is [n_seqs, cache_steps,
    // kv_heads, head_dim] and logits are [n_seqs, logits_size]. The output
    // cache is a plain struct (not TransformerKVCache) because a batched
    // static cache is owned by the caller's graph.
    QwenCausalDecoderStaticCacheOutputs build_static_cache_tail_batched(
        core::ModuleBuildContext & ctx,
        ggml_cgraph * graph,
        const core::TensorValue & input,
        const core::TensorValue & positions,
        const QwenCausalDecoderWeights & weights,
        int64_t cache_steps,
        int64_t n_seqs,
        const core::TensorValue & attention_mask,
        const std::optional<core::TensorValue> & cache_slot = std::nullopt) const;

private:
    QwenCausalDecoderConfig config_;
};

std::vector<int32_t> qwen_position_ids(int64_t steps, int64_t offset = 0);

std::vector<ggml_fp16_t> qwen_causal_prefill_mask_values(int64_t batch_size, int64_t steps);

std::vector<ggml_fp16_t> qwen_causal_suffix_mask_values(
    int64_t batch_size,
    int64_t query_steps,
    int64_t prefix_steps);

void write_qwen_causal_prefill_mask(
    ggml_tensor * tensor,
    int64_t batch_size,
    int64_t steps);

void write_qwen_cached_step_mask(
    ggml_tensor * tensor,
    std::vector<ggml_fp16_t> & scratch,
    int64_t mask_steps,
    int64_t visible_prefix_steps,
    int64_t current_slot);

}  // namespace engine::modules
