
#ifndef KERNELS_H
#define KERNELS_H

#include <stddef.h>
#include <stdint.h>

// Forward declaration for QuantizedWeight (fully defined in model.h)
// Note: This is only needed if model.h hasn't been included yet
#ifndef QUANTIZED_WEIGHT_DEFINED
struct QuantizedWeight;
#endif

// Quantized matrix multiplication kernels
// Adaptive dispatcher that handles FP32, Q8, and Q4 weights automatically
void matmul_adaptive(const float* A, const float* W_fp32, const void* W_q_ptr,
                     float* C, int M, int N, int K);

#endif
