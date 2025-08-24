#ifndef QUANT_H
#define QUANT_H

/**
 * quant.h - Cleaned Up Quantization Library 
 * 
 * ONLY the fastest and most accurate functions:
 * - Q8: Standard symmetric quantization
 * - Q4: ONLY asymmetric with zero points (35.8% more accurate than symmetric)
 * - matmul_q8_q8_f32: Fastest Q8×Q8 multiplication
 * - matmul_q8_q4_f32: Asymmetric Q4 multiplication (renamed from matmul_q8_q4_opt)
 * 
 * Removed all symmetric Q4 functions - too inaccurate!
 */

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include "io.h"  // Needed for Tensor type in conversion functions

/**
 * Quantization types
 */
typedef enum {
    QUANT_NONE = 0,  // No quantization (keep original f32)
    QUANT_Q8   = 1,  // 8-bit signed integer quantization  
    QUANT_Q4   = 2   // 4-bit asymmetric quantization with zero points
} QuantType;

/**
 * Quantized tensor structure
 */
typedef struct {
    void*   q_data;     // Quantized data (int8* for Q8, uint8* for Q4)
    float*  scales;     // Group-wise scaling factors
    int8_t* zero_points; // Zero points (only for Q4)
    size_t  num_rows;   
    size_t  row_size;   
    size_t  group_size; 
    size_t  num_groups; 
    QuantType qtype;    
} QuantizedTensor;

/**
 * Q8 QUANTIZATION (Symmetric, Standard)
 */
void quantize_q8(const float* x, int8_t* qx_q, float* qx_s, int n, int group_size);
void dequantize_q8(const int8_t* qx_q, const float* qx_s, float* x, int n, int group_size);

/**
 * Q4 QUANTIZATION (With Zero Points - THE ONLY Q4 functions)
 * 35.8% more accurate than symmetric Q4 (which we removed)
 */
void quantize_q4(const float* input, size_t rows, size_t cols, size_t group_size,
                 float* out_scales, int8_t* out_zero_points, uint8_t* out_quantized);

void dequantize_q4(const uint8_t* qdata, const float* scales, const int8_t* zero_points,
                   float* out_fp32, size_t rows, size_t cols, size_t group_size);

/**
 * MATRIX MULTIPLICATION FUNCTIONS (Only the best ones)
 */

/**
 * Fastest Q8×Q8 matrix multiplication
 */
void matmul_q8_q8_f32(const int8_t* A_q8, const float* A_scales,
                      const int8_t* B_q8, const float* B_scales,
                      float* C, int M, int N, int K, int group_size);

/**
 * Q8×Q4 matrix multiplication with asymmetric Q4 (35.8% accuracy improvement)
 * This is the ONLY Q4 matmul function - renamed from matmul_q8_q4_opt
 * Zero points provide much better accuracy than symmetric Q4
 */
void matmul_q8_q4_f32(const int8_t* A, const float* A_scales,
                      const uint8_t* B_q4, const float* B_scales, const int8_t* B_zps,
                      float* C, int M, int N, int K, size_t group_size);

/**
 * INFERENCE FUNCTIONS (Quantize activations on-the-fly)
 */

/**
 * FP32 × Q8 matrix multiplication (quantizes A on-the-fly)
 */
void matmul_f32_q8_f32(const float* A_fp32, const int8_t* B_q8, const float* B_scales,
                       float* C, int M, int N, int K, int group_size,
                       int8_t* qx_q_scratch, float* qx_s_scratch);

/**
 * FP32 × Q4 matrix multiplication with zero points (new optimized version)
 */
void matmul_f32_q4_f32_with_zeros(const float* A_fp32, const uint8_t* B_q4, const float* B_scales, const int8_t* B_zeros,
                                  float* C, int M, int N, int K, int group_size,
                                  int8_t* qx_q_scratch, float* qx_s_scratch);

/**
 * FP32 × Q4 matrix multiplication (backward compatible - no zero points parameter)
 * This maintains the old API that existing code expects
 */
void matmul_f32_q4_f32(const float* A_fp32, const uint8_t* B_q4, const float* B_scales,
                       float* C, int M, int N, int K, int group_size,
                       int8_t* qx_q_scratch, float* qx_s_scratch);

/**
 * BACKWARD COMPATIBILITY FUNCTIONS for existing code (like run.c)
 */

/**
 * Backward compatible matmul_q8_q4_f32_opt (old name)
 */
void matmul_q8_q4_f32_opt(const int8_t* restrict A_q8, const float* restrict A_scales,
                          const uint8_t* restrict B_q4, const float* restrict B_scales,
                          float* restrict C, int M, int N, int K, size_t group_size);

/**
 * Backward compatible matmul_q8_q4_opt (old asymmetric function name)
 */
void matmul_q8_q4_opt(const int8_t* A, const float* A_scales,
                      const uint8_t* B_q4, const float* B_scales, const int8_t* B_zps,
                      float* C, int M, int N, int K, size_t group_size);

/**
 * UTILITY FUNCTIONS FOR CONVERSION TOOL
 */
bool should_quantize_tensor(const char* tensor_name);
size_t get_quantized_data_size(size_t rows, size_t cols, QuantType qtype);
size_t get_scales_size(size_t rows, size_t cols, size_t group_size);
QuantizedTensor* quantize_tensor(TensorBin* input_tensor, QuantType qtype, size_t group_size);
void quantized_tensor_free(QuantizedTensor* qt);

#endif // QUANT_H