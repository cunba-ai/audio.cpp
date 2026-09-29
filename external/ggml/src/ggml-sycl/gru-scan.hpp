//
// audio.cpp fork op: fused single-layer GRU scan (RMVPE pitch extractor).
//
#pragma once

#include "common.hpp"

void ggml_sycl_gru_scan(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

bool ggml_sycl_gru_scan_supported(const ggml_tensor * op);
