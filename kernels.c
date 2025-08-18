/**
 * kernels.c - Group-wise Quantized Matrix Multiplication Kernels
 * 
 * This module implements high-performance quantized matrix multiplication kernels
 * for MoE (Mixture of Experts) neural networks using group-wise quantization only.
 * 
 * Architecture Overview:
 * - matmul_f32_q8(): FP32 activations × Q8 weights → FP32 outputs
 * - matmul_f32_q4(): FP32 activations × Q4 weights → FP32 outputs  
 * - matmul_adaptive(): Automatic dispatch based on quantization type
 * 
 * Quantization Support:
 * - Q8: 8-bit signed integers with group-wise scaling factors
 * - Q4: 4-bit values packed 2 per byte with group-wise scaling factors
 * - Group-wise only: Configurable group sizes (32, 64, 128, etc.)
 * 
 * Performance Optimizations:
 * - Direct FP32 computation for better accuracy
 * - Efficient Q4 unpacking and scaling
 * - Cache-friendly memory access patterns
 * - Simplified group-wise scaling only
 * 
 * Mathematical Foundation:
 * For group-wise quantization, the scaling factor is applied per group:
 * result[i,j] = Σ(activation[i,k] * (quantized_weight[j,k] * scale[group(j,k)]))
 * 
 * This approach provides superior accuracy compared to rowwise quantization while
 * maintaining computational efficiency through vectorizable operations.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "model.h"  // For QuantizedWeight definition
#include "quant.h"  // For group quantization utilities

/**
 * Group-wise quantized FP32×Q8 matrix multiplication: C = A × B_quantized
 * 
 * Performs matrix multiplication where:
 * - A is FP32 input matrix [M × K]
 * - B is Q8 quantized weight matrix [N × K] stored as int8 + group scales
 * - C is FP32 output matrix [M × N]
 * 
 * Group-wise quantization provides fine-grained scaling:
 * B_fp32[i,j] ≈ B_q8[i,j] * scale[group_index(i,j)]
 * 
 * @param A         Input matrix A [M × K] in row-major order (FP32)
 * @param B_q       Quantized weights B [N × K] in row-major order (int8)
 * @param B_s       Group-wise scaling factors for B (FP32)
 * @param C         Output matrix C [M × N] in row-major order (FP32)
 * @param M         Number of rows in A and C
 * @param N         Number of rows in B and columns in C
 * @param K         Number of columns in A and B
 * @param group_size Group size for quantization (must be > 0)
 */
void matmul_f32_q8(const float* A, const int8_t* B_q, const float* B_s, 
                   float* C, int M, int N, int K, size_t group_size) {
  if (group_size == 0) {
    fprintf(stderr, "Error: group_size must be > 0 (rowwise quantization removed)\n");
    return;
  }
  
  for (int m = 0; m < M; ++m) {
    const float* a_row = A + m * K;      // Current row of A
    float* c_row = C + m * N;            // Current row of C
    
    for (int n = 0; n < N; ++n) {
      const int8_t* b_row = B_q + n * K;  // Current row of quantized B
      
      float accumulated = 0.0f;
      
      // Group-wise quantization: Each element uses its group's scaling factor
      // This provides better accuracy than rowwise by allowing finer-grained scaling
      for (int k = 0; k < K; ++k) {
        // Calculate which group this weight element belongs to
        // Groups are assigned linearly across the flattened weight matrix
        size_t linear_idx = (size_t)n * (size_t)K + (size_t)k;
        size_t group_idx = linear_idx / group_size;
        float scale = B_s[group_idx];
        
        // Apply group-specific scaling to dequantize the weight
        float a_val = a_row[k];                    // FP32 activation
        accumulated += a_val * (float)b_row[k] * scale;  // Direct computation without intermediate
      }
      
      c_row[n] = accumulated;
    }
  }
}

/**
 * Group-wise quantized FP32×Q4 matrix multiplication: C = A × B_quantized
 * 
 * Performs matrix multiplication where:
 * - A is FP32 input matrix [M × K]
 * - B is Q4 quantized weight matrix [N × K] stored as packed uint8 + group scales
 * - C is FP32 output matrix [M × N]
 * 
 * Q4 packing format:
 * - Each byte contains 2 Q4 values: byte = (val0 & 0xF) | ((val1 & 0xF) << 4)
 * - Q4 values are unsigned [0,15] representing signed [-8,7] via offset: val - 8
 * 
 * Group-wise quantization only for improved accuracy over rowwise methods.
 * 
 * @param A         Input matrix A [M × K] in row-major order (FP32)  
 * @param B_q       Quantized weights B [N × K/2] packed Q4 values (uint8)
 * @param B_s       Group-wise scaling factors for B (FP32)
 * @param C         Output matrix C [M × N] in row-major order (FP32)
 * @param M         Number of rows in A and C
 * @param N         Number of rows in B and columns in C
 * @param K         Number of columns in A and B (must be even for Q4 packing)
 * @param group_size Group size for quantization (must be > 0)
 */
void matmul_f32_q4(const float* A, const uint8_t* B_q, const float* B_s,
                   float* C, int M, int N, int K, size_t group_size) {
  if (group_size == 0) {
    fprintf(stderr, "Error: group_size must be > 0 (rowwise quantization removed)\n");
    return;
  }
  
  for (int m = 0; m < M; ++m) {
    const float* a_row = A + m * K;      // Current row of A
    float* c_row = C + m * N;            // Current row of C
    
    for (int n = 0; n < N; ++n) {
      const uint8_t* b_row = B_q + n * (K / 2);  // Current row of packed B
      
      float accumulated = 0.0f;
      
      // Group-wise quantization: accumulate per-group contributions
      for (int k = 0; k < K; k += 2) {
        // Unpack Q4 values from byte
        uint8_t packed = b_row[k / 2];
        int8_t b0 = (int8_t)((packed & 0xF) - 8);     // First Q4 value
        int8_t b1 = (int8_t)(((packed >> 4) & 0xF) - 8); // Second Q4 value
        
        // Process first Q4 value with group-wise scaling
        size_t linear_idx0 = (size_t)n * (size_t)K + (size_t)k;
        size_t group_idx0 = linear_idx0 / group_size;
        float scale0 = B_s[group_idx0];
        float a_val0 = a_row[k];
        accumulated += a_val0 * (float)b0 * scale0;  // Direct computation without intermediate
        
        // Process second Q4 value with group-wise scaling (if exists)
        if (k + 1 < K) {
          size_t linear_idx1 = (size_t)n * (size_t)K + (size_t)(k + 1);
          size_t group_idx1 = linear_idx1 / group_size;
          float scale1 = B_s[group_idx1];
          float a_val1 = a_row[k + 1];
          accumulated += a_val1 * (float)b1 * scale1;  // Direct computation without intermediate
        }
      }
      
      c_row[n] = accumulated;
    }
  }
}

/**
 * Adaptive matrix multiplication supporting both FP32 and quantized weights
 * 
 * This function automatically dispatches to the appropriate kernel based on
 * the weight format, providing a unified interface for the MoE forward pass.
 * Only group-wise quantization is supported (group_size > 0).
 * 
 * Weight format detection:
 * - If W_q == NULL: use FP32 matrix multiplication
 * - If W_q->dtype == 2: use Q8 quantized multiplication  
 * - If W_q->dtype == 3: use Q4 quantized multiplication
 * 
 * @param A       Input matrix [M × K] (FP32)
 * @param W_fp32  FP32 weights [N × K] or NULL if quantized
 * @param W_q     Quantized weight structure or NULL if FP32
 * @param C       Output matrix [M × N] (FP32)
 * @param M       Number of rows in A and C
 * @param N       Number of rows in W and columns in C  
 * @param K       Number of columns in A and W
 */
void matmul_adaptive(const float* A, const float* W_fp32, const void* W_q_ptr,
                     float* C, int M, int N, int K) {
  const QuantizedWeight* W_q = (const QuantizedWeight*)W_q_ptr;
  
  if (W_q == NULL) {
    // FP32 × FP32 matrix multiplication
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
          sum += A[m * K + k] * W_fp32[n * K + k];
        }
        C[m * N + n] = sum;
      }
    }
  } else if (W_q->dtype == 2) {
    // Q8 quantized multiplication (group-wise only)
    if (W_q->group_size == 0) {
      fprintf(stderr, "Error: Rowwise quantization no longer supported, use group_size > 0\n");
      return;
    }
    matmul_f32_q8(A, W_q->q, W_q->s, C, M, N, K, W_q->group_size);
  } else if (W_q->dtype == 3) {  
    // Q4 quantized multiplication (group-wise only)
    if (W_q->group_size == 0) {
      fprintf(stderr, "Error: Rowwise quantization no longer supported, use group_size > 0\n");
      return;
    }
    matmul_f32_q4(A, (const uint8_t*)W_q->q, W_q->s, C, M, N, K, W_q->group_size);
  } else {
    // Unsupported quantization format - fallback to zero output
    // This should not happen in normal operation
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        C[m * N + n] = 0.0f;
      }
    }
    fprintf(stderr, "Warning: Unsupported quantization dtype %d, zeroing output\n", W_q->dtype);
  }
}