#include "common.cuh"

#define CUDA_GRU_SCAN_THREADS 256

void ggml_cuda_op_gru_scan(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
