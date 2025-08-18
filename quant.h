#ifndef QUANT_H
#define QUANT_H

/**
 * quant.h - Group-wise Quantization Library for MoE Expert Weights
 * 
 * This module provides group-wise quantization functionality for MoE (Mixture of Experts)
 * neural network weights. Only group-wise quantization is supported for optimal accuracy.
 * 
 * Key Features:
 * - Q8 (8-bit) and Q4 (4-bit) quantization formats
 * - Group-wise quantization with configurable group sizes (32, 64, 128, etc.)
 * - Memory-efficient packed Q4 representation
 * - Compatible with kernels.c for high-performance inference
 * 
 * Quantization Scheme:
 * - Group-wise: Tensor is divided into groups, each with shared scaling factor
 * - Supports common group sizes: 32, 64, 128, 256
 * - Each group maintains high precision through dedicated scaling
 * 
 * Data Flow:
 * FP32 Tensor -> quantize_tensor() -> QuantizedTensor -> kernels.c
 */

#include <stddef.h>
#include <stdint.h>
#include "io.h"

/**
 * Quantization types supported by the conversion tool
 */
typedef enum {
    QUANT_NONE = 0,  // No quantization (keep original f32)
    QUANT_Q8   = 1,  // 8-bit signed integer quantization  
    QUANT_Q4   = 2   // 4-bit quantization (packed)
} QuantType;

/**
 * Structure to hold quantized tensor data with group-wise scaling factors
 * All quantization uses group-wise scheme for optimal accuracy
 */
typedef struct {
    void*   q_data;     // Quantized data (int8* for Q8, uint8* for Q4)
    float*  scales;     // Group-wise scaling factors
    size_t  num_rows;   // Number of rows (for validation)
    size_t  row_size;   // Size of each row in elements
    size_t  group_size; // Group size for quantization (required > 0)
    size_t  num_groups; // Total number of groups
    QuantType qtype;    // Quantization type used
} QuantizedTensor;

/**
 * Group-wise Q8 quantization: quantize in groups of specified size to int8 [-127, 127]
 * 
 * @param input: Input tensor data (float32)
 * @param rows: Number of rows in the tensor
 * @param cols: Number of columns in the tensor
 * @param group_size: Number of elements per group (must be > 0)
 * @param out_scales: Output buffer for per-group scaling factors [num_groups]
 * @param out_quantized: Output buffer for quantized data [rows * cols]
 * @param out_num_groups: Output parameter for total number of groups
 */
void quantize_groupwise_q8(const float* input, size_t rows, size_t cols, size_t group_size,
                          float* out_scales, int8_t* out_quantized, size_t* out_num_groups);

/**
 * Group-wise Q4 quantization: quantize in groups of specified size to 4-bit [-8, 7] then pack
 * 
 * @param input: Input tensor data (float32)
 * @param rows: Number of rows in the tensor
 * @param cols: Number of columns in the tensor
 * @param group_size: Number of elements per group (must be > 0)
 * @param out_scales: Output buffer for per-group scaling factors [num_groups]  
 * @param out_quantized: Output buffer for packed quantized data [rows * (cols/2)]
 * @param out_num_groups: Output parameter for total number of groups
 */
void quantize_groupwise_q4(const float* input, size_t rows, size_t cols, size_t group_size,
                          float* out_scales, uint8_t* out_quantized, size_t* out_num_groups);

/**
 * Quantize a single tensor from FP32 to the specified quantization type using group-wise scheme
 * Creates quantized data + scaling factors with improved accuracy over rowwise methods
 * 
 * @param tensor: Input tensor to quantize (must be 2D for expert weights)
 * @param qtype: Quantization type (Q8 or Q4)
 * @param group_size: Group size for quantization (must be > 0)
 * @return: QuantizedTensor structure with quantized data and scales
 *          Caller must free returned structure with quantized_tensor_free()
 */
QuantizedTensor* quantize_tensor(const TensorBin* tensor, QuantType qtype, size_t group_size);

/**
 * Free memory allocated for a QuantizedTensor structure
 * 
 * @param qt: QuantizedTensor to free (safe to pass NULL)
 */
void quantized_tensor_free(QuantizedTensor* qt);

/**
 * Check if a tensor name represents an expert weight that should be quantized
 * Based on export.py logic: only expert weight matrices (gate_proj, up_proj, down_proj)
 * 
 * @param name: Tensor name to check
 * @return: true if tensor should be quantized, false otherwise
 */
bool should_quantize_tensor(const char* name);

/**
 * Get the size in bytes needed to store quantized data
 * 
 * @param rows: Number of rows
 * @param cols: Number of columns
 * @param qtype: Quantization type
 * @return: Size in bytes needed for quantized data
 */
size_t get_quantized_data_size(size_t rows, size_t cols, QuantType qtype);

/**
 * Get the size in bytes needed to store group-wise scaling factors
 * 
 * @param rows: Number of rows
 * @param cols: Number of columns
 * @param group_size: Group size (must be > 0)
 * @return: Size in bytes for scaling factors (always float32)
 */
size_t get_scales_size(size_t rows, size_t cols, size_t group_size);

/**
 * Calculate the number of groups for given tensor dimensions and group size
 * 
 * @param rows: Number of rows
 * @param cols: Number of columns  
 * @param group_size: Group size (must be > 0)
 * @return: Number of groups needed
 */
size_t get_num_groups(size_t rows, size_t cols, size_t group_size);

/**
 * Q8_0 × Q8_0 -> FP32 matrix multiplication (llama2.c style)
 * Both input matrices are already pre-quantized to Q8
 * 
 * @param A_q8: Pre-quantized matrix A [M × K] (Q8, int8_t)
 * @param A_scales: Scaling factors for A [num_groups_A]
 * @param B_q8: Pre-quantized matrix B [K × N] (Q8, int8_t)
 * @param B_scales: Scaling factors for B [num_groups_B]
 * @param C: Output matrix C [M × N] (FP32)
 * @param M: Number of rows in A and C
 * @param N: Number of columns in B and C
 * @param K: Number of columns in A and rows in B
 * @param group_size: Group size used for quantization
 */
void matmul_q8_q8_f32(const int8_t* A_q8, const float* A_scales,
                      const int8_t* B_q8, const float* B_scales,
                      float* C, int M, int N, int K, size_t group_size);

/**
 * Q8 × Q4 -> FP32 matrix multiplication (llama2.c style)
 * Both input matrices are already pre-quantized
 * 
 * @param A_q8: Pre-quantized matrix A [M × K] (Q8, int8_t)
 * @param A_scales: Scaling factors for A [num_groups_A]
 * @param B_q4: Pre-quantized matrix B [K × N/2] (Q4 packed, uint8_t)
 * @param B_scales: Scaling factors for B [num_groups_B]
 * @param C: Output matrix C [M × N] (FP32)
 * @param M: Number of rows in A and C
 * @param N: Number of columns in B and C
 * @param K: Number of columns in A and rows in B
 * @param group_size: Group size used for quantization
 */
void matmul_q8_q4_f32(const int8_t* A_q8, const float* A_scales,
                      const uint8_t* B_q4, const float* B_scales,
                      float* C, int M, int N, int K, size_t group_size);

#endif // QUANT_H