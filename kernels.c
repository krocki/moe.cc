/**
 * kernels.c - Quantized matrix multiplication kernels for MoE expert weights
 * 
 * This module implements efficient quantized matrix multiplication for Q8 and Q4
 * quantized expert weights, following the approach from llama2.c but adapted
 * for rowwise quantization used in the MoE architecture.
 * 
 * Key functions:
 * - matmul_f32_q8(): FP32 input × Q8 weight → FP32 output
 * - matmul_f32_q4(): FP32 input × Q4 weight → FP32 output
 * 
 * Quantization format:
 * - Q8: int8 weights with per-row FP32 scaling factors
 * - Q4: packed int4 weights (2 per byte) with per-row FP32 scaling factors
 *   Q4 values are stored as unsigned bytes with range [0,15] representing [-8,7]
 * 
 * Reference: Based on llama2.c quantization approach but adapted for rowwise
 *           scaling and integrated with existing FP32 matrix multiplication
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "model.h"  // For QuantizedWeight definition

/**
 * Quantized FP32×Q8 matrix multiplication: C = A × B_quantized
 * 
 * Performs matrix multiplication where:
 * - A is FP32 input matrix [M × K]
 * - B is Q8 quantized weight matrix [N × K] stored as int8 + scales
 * - C is FP32 output matrix [M × N]
 * 
 * Quantization formula: B_fp32[i,j] ≈ B_q8[i,j] * scale[i]
 * 
 * Algorithm:
 * 1. For each output element C[m,n]:
 *    - Compute dot product in int32 space: sum(A[m,k] * B_q8[n,k])
 *    - Apply row scaling: result * B_scale[n]
 *    - Store as FP32
 * 
 * @param A       Input matrix A [M × K] in row-major order (FP32)
 * @param B_q     Quantized weights B [N × K] in row-major order (int8)
 * @param B_s     Row scaling factors for B [N] (FP32)
 * @param C       Output matrix C [M × N] in row-major order (FP32)
 * @param M       Number of rows in A and C
 * @param N       Number of rows in B and columns in C
 * @param K       Number of columns in A and B
 */
static void matmul_f32_q8(const float* A, const int8_t* B_q, const float* B_s, 
                          float* C, int M, int N, int K) {
  for (int m = 0; m < M; ++m) {
    const float* a_row = A + m * K;      // Current row of A
    float* c_row = C + m * N;            // Current row of C
    
    for (int n = 0; n < N; ++n) {
      const int8_t* b_row = B_q + n * K;  // Current row of quantized B
      float scale = B_s[n];               // Scale factor for this row of B
      
      // Accumulate dot product in int32 to avoid overflow
      int32_t dot_i32 = 0;
      for (int k = 0; k < K; ++k) {
        // Convert FP32 input to int32 for multiplication
        // Note: This assumes inputs are reasonably bounded
        int32_t a_i32 = (int32_t)(a_row[k] * 127.0f); // Scale to int8 range
        dot_i32 += a_i32 * (int32_t)b_row[k];
      }
      
      // Convert back to FP32 and apply scaling
      c_row[n] = ((float)dot_i32 / 127.0f) * scale;
    }
  }
}

/**
 * Quantized FP32×Q4 matrix multiplication: C = A × B_quantized
 * 
 * Performs matrix multiplication where:
 * - A is FP32 input matrix [M × K]
 * - B is Q4 quantized weight matrix [N × K] stored as packed uint8 + scales
 * - C is FP32 output matrix [M × N]
 * 
 * Q4 packing format:
 * - Each byte contains 2 Q4 values: byte = (val0 & 0xF) | ((val1 & 0xF) << 4)
 * - Q4 values are unsigned [0,15] representing signed [-8,7] via offset: val - 8
 * 
 * @param A       Input matrix A [M × K] in row-major order (FP32)  
 * @param B_q     Quantized weights B [N × K/2] packed Q4 values (uint8)
 * @param B_s     Row scaling factors for B [N] (FP32)
 * @param C       Output matrix C [M × N] in row-major order (FP32)
 * @param M       Number of rows in A and C
 * @param N       Number of rows in B and columns in C
 * @param K       Number of columns in A and B (must be even)
 */
static void matmul_f32_q4(const float* A, const uint8_t* B_q, const float* B_s,
                          float* C, int M, int N, int K) {
  for (int m = 0; m < M; ++m) {
    const float* a_row = A + m * K;      // Current row of A
    float* c_row = C + m * N;            // Current row of C
    
    for (int n = 0; n < N; ++n) {
      const uint8_t* b_row = B_q + n * (K / 2);  // Current row of packed B
      float scale = B_s[n];                      // Scale factor for this row
      
      // Accumulate dot product
      int32_t dot_i32 = 0;
      for (int k = 0; k < K; k += 2) {
        // Unpack Q4 values from byte
        uint8_t packed = b_row[k / 2];
        int8_t b0 = (int8_t)((packed & 0xF) - 8);     // First Q4 value
        int8_t b1 = (int8_t)(((packed >> 4) & 0xF) - 8); // Second Q4 value
        
        // Scale FP32 inputs to match Q4 range
        int32_t a0_i32 = (int32_t)(a_row[k] * 7.0f);
        int32_t a1_i32 = (k + 1 < K) ? (int32_t)(a_row[k + 1] * 7.0f) : 0;
        
        dot_i32 += a0_i32 * (int32_t)b0;
        if (k + 1 < K) {
          dot_i32 += a1_i32 * (int32_t)b1;
        }
      }
      
      // Convert back to FP32 and apply scaling  
      c_row[n] = ((float)dot_i32 / 7.0f) * scale;
    }
  }
}

/**
 * Adaptive matrix multiplication supporting both FP32 and quantized weights
 * 
 * This function automatically dispatches to the appropriate kernel based on
 * the weight format, providing a unified interface for the MoE forward pass.
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
    // Use standard FP32 matrix multiplication
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        const float* a_row = A + m * K;        // Current row of A
        const float* w_row = W_fp32 + n * K;   // Current row of W
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
          sum += a_row[k] * w_row[k];
        }
        C[m * N + n] = sum;
      }
    }
  } else if (W_q->dtype == 2) {
    // Q8 quantized multiplication
    matmul_f32_q8(A, W_q->q, W_q->s, C, M, N, K);
  } else if (W_q->dtype == 3) {  
    // Q4 quantized multiplication
    matmul_f32_q4(A, (const uint8_t*)W_q->q, W_q->s, C, M, N, K);
  } else {
    // Unsupported quantization format - fallback to zero output
    // This should not happen in normal operation
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        C[m * N + n] = 0.0f; // Zero output for safety
      }
    }
  }
}
