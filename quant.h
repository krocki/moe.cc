#ifndef QUANT_H
#define QUANT_H

/**
 * quant.h - Quantization Library for MoE Expert Weights
 * 
 * This module provides comprehensive quantization functionality for MoE (Mixture of Experts)
 * neural network weights. It supports both traditional rowwise quantization and modern
 * group-wise quantization schemes.
 * 
 * Key Features:
 * - Q8 (8-bit) and Q4 (4-bit) quantization formats
 * - Rowwise quantization (one scale per matrix row)
 * - Group-wise quantization (configurable group sizes)
 * - Memory-efficient packed Q4 representation
 * - Compatible with export.py and kernels.c
 * 
 * Quantization Schemes:
 * - Rowwise: Each matrix row has its own scaling factor
 * - Group-wise: Tensor is divided into groups, each with shared scaling
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
 * Structure to hold quantized tensor data with scaling factors
 * Supports both rowwise (group_size=0) and group-wise quantization
 */
typedef struct {
    void*   q_data;     // Quantized data (int8* for Q8, uint8* for Q4)
    float*  scales;     // Scaling factors (per-row or per-group)
    size_t  num_rows;   // Number of rows (for validation)
    size_t  row_size;   // Size of each row in elements
    size_t  group_size; // Group size for quantization (0 = rowwise)
    size_t  num_groups; // Total number of groups
    QuantType qtype;    // Quantization type used
} QuantizedTensor;

/**
 * Row-wise Q8 quantization: quantize each row independently to int8 [-127, 127]
 * 
 * @param input: Input tensor data (float32)
 * @param rows: Number of rows in the tensor
 * @param cols: Number of columns in the tensor  
 * @param out_scales: Output buffer for per-row scaling factors [rows]
 * @param out_quantized: Output buffer for quantized data [rows * cols]
 */
void quantize_rowwise_q8(const float* input, size_t rows, size_t cols,
                        float* out_scales, int8_t* out_quantized);

/**
 * Group-wise Q8 quantization: quantize in groups of specified size to int8 [-127, 127]
 * 
 * @param input: Input tensor data (float32)
 * @param rows: Number of rows in the tensor
 * @param cols: Number of columns in the tensor
 * @param group_size: Number of elements per group
 * @param out_scales: Output buffer for per-group scaling factors [num_groups]
 * @param out_quantized: Output buffer for quantized data [rows * cols]
 * @param out_num_groups: Output parameter for total number of groups
 */
void quantize_groupwise_q8(const float* input, size_t rows, size_t cols, size_t group_size,
                          float* out_scales, int8_t* out_quantized, size_t* out_num_groups);

/**
 * Row-wise Q4 quantization: quantize each row independently to 4-bit [-8, 7] then pack
 * 
 * @param input: Input tensor data (float32)
 * @param rows: Number of rows in the tensor
 * @param cols: Number of columns in the tensor
 * @param out_scales: Output buffer for per-row scaling factors [rows]  
 * @param out_quantized: Output buffer for packed quantized data [rows * (cols/2)]
 *                      Each byte contains two 4-bit values: low_nibble | (high_nibble << 4)
 */
void quantize_rowwise_q4(const float* input, size_t rows, size_t cols,
                        float* out_scales, uint8_t* out_quantized);

/**
 * Group-wise Q4 quantization: quantize in groups of specified size to 4-bit [-8, 7] then pack
 * 
 * @param input: Input tensor data (float32)
 * @param rows: Number of rows in the tensor
 * @param cols: Number of columns in the tensor
 * @param group_size: Number of elements per group
 * @param out_scales: Output buffer for per-group scaling factors [num_groups]  
 * @param out_quantized: Output buffer for packed quantized data [rows * (cols/2)]
 * @param out_num_groups: Output parameter for total number of groups
 */
void quantize_groupwise_q4(const float* input, size_t rows, size_t cols, size_t group_size,
                          float* out_scales, uint8_t* out_quantized, size_t* out_num_groups);

/**
 * Quantize a single tensor from FP32 to the specified quantization type
 * Creates quantized data + scaling factors following export.py conventions
 * 
 * @param tensor: Input tensor to quantize (must be 2D for expert weights)
 * @param qtype: Quantization type (Q8 or Q4)
 * @param group_size: Group size for quantization (0 = rowwise)
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
 * Get the size in bytes needed to store scaling factors
 * 
 * @param rows: Number of rows
 * @param cols: Number of columns
 * @param group_size: Group size (0 = rowwise)
 * @return: Size in bytes for scaling factors (always float32)
 */
size_t get_scales_size(size_t rows, size_t cols, size_t group_size);

/**
 * Calculate the number of groups for given tensor dimensions and group size
 * 
 * @param rows: Number of rows
 * @param cols: Number of columns  
 * @param group_size: Group size (0 = rowwise)
 * @return: Number of groups needed
 */
size_t get_num_groups(size_t rows, size_t cols, size_t group_size);

#endif // QUANT_H