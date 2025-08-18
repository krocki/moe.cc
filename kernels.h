
#ifndef KERNELS_H
#define KERNELS_H

/**
 * kernels.h - Header for Quantized Matrix Multiplication Kernels
 * 
 * This header provides access to the quantized matrix multiplication
 * functions implemented in kernels.c for use in tests and other modules.
 */

#include <stddef.h>
#include <stdint.h>

// Forward declaration for QuantizedWeight (fully defined in model.h)
// Note: This is only needed if model.h hasn't been included yet
#ifndef QUANTIZED_WEIGHT_DEFINED
struct QuantizedWeight;
#endif

/**
 * FP32 × Q8 quantized matrix multiplication
 * 
 * @param A         Input matrix A [M × K] (FP32)
 * @param B_q       Quantized weights B [N × K] (int8)
 * @param B_s       Scaling factors for B (FP32)
 * @param C         Output matrix C [M × N] (FP32)
 * @param M         Number of rows in A and C
 * @param N         Number of rows in B and columns in C
 * @param K         Number of columns in A and B
 * @param group_size Group size for quantization (0 = rowwise)
 */
void matmul_f32_q8(const float* A, const int8_t* B_q, const float* B_s,
                   float* C, int M, int N, int K, size_t group_size);

/**
 * FP32 × Q4 quantized matrix multiplication
 * 
 * @param A         Input matrix A [M × K] (FP32)
 * @param B_q       Quantized weights B [N × K/2] packed (uint8)
 * @param B_s       Scaling factors for B (FP32)
 * @param C         Output matrix C [M × N] (FP32)
 * @param M         Number of rows in A and C
 * @param N         Number of rows in B and columns in C
 * @param K         Number of columns in A and B (must be even)
 * @param group_size Group size for quantization (0 = rowwise)
 */
void matmul_f32_q4(const float* A, const uint8_t* B_q, const float* B_s,
                   float* C, int M, int N, int K, size_t group_size);

/**
 * Adaptive matrix multiplication with automatic kernel selection
 * 
 * @param A       Input matrix [M × K] (FP32)
 * @param W_fp32  FP32 weights [N × K] or NULL if quantized
 * @param W_q_ptr Quantized weight structure or NULL if FP32
 * @param C       Output matrix [M × N] (FP32)
 * @param M       Number of rows in A and C
 * @param N       Number of rows in W and columns in C
 * @param K       Number of columns in A and W
 */
void matmul_adaptive(const float* A, const float* W_fp32, const void* W_q_ptr,
                     float* C, int M, int N, int K);

#endif
