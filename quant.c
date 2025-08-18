/**
 * quant.c - Implementation of Quantization Algorithms
 * 
 * This file implements the core quantization algorithms for MoE expert weights.
 * It provides both rowwise and group-wise quantization for Q8 and Q4 formats.
 * 
 * Quantization Process:
 * 1. Analyze input tensor to determine optimal scaling factors
 * 2. Apply quantization formula to convert FP32 -> int8/int4
 * 3. Pack Q4 values efficiently (2 values per byte)
 * 4. Store scaling factors separately for dequantization
 * 
 * Mathematical Foundation:
 * - Q8: value_fp32 ≈ value_int8 * scale, where scale = max_abs_value / 127
 * - Q4: value_fp32 ≈ (value_uint4 - 8) * scale, where scale = max_abs_value / 7
 * 
 * Group-wise quantization divides the tensor into groups of configurable size,
 * with each group sharing a scaling factor. This provides better compression
 * and accuracy trade-offs compared to rowwise quantization.
 */

#include "quant.h"
#include "io.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/**
 * Row-wise Q8 quantization implementation
 * Each row is quantized independently with its own scaling factor
 * Follows export.py _rowwise_q8() logic: scale = max_abs_val/127, q = round(val/scale)
 */
void quantize_rowwise_q8(const float* input, size_t rows, size_t cols,
                        float* out_scales, int8_t* out_quantized) {
    for (size_t row = 0; row < rows; row++) {
        const float* row_data = input + row * cols;
        
        // Find maximum absolute value in this row
        float max_abs = 0.0f;
        for (size_t col = 0; col < cols; col++) {
            float abs_val = fabsf(row_data[col]);
            if (abs_val > max_abs) {
                max_abs = abs_val;
            }
        }
        
        // Calculate scaling factor (clamp to avoid division by zero)
        float scale = fmaxf(max_abs / 127.0f, 1e-8f);
        out_scales[row] = scale;
        
        // Quantize values in this row
        for (size_t col = 0; col < cols; col++) {
            float normalized = row_data[col] / scale;
            int8_t quantized = (int8_t)roundf(normalized);
            out_quantized[row * cols + col] = quantized;
        }
    }
}

/**
 * Row-wise Q4 quantization implementation  
 * Each row is quantized independently, then packed 2 values per byte
 * Follows export.py _rowwise_q4() logic: scale = max_abs_val/7, q = clamp(round(val/scale), -8, 7)
 * Values are shifted to [0,15] range and packed as nibbles
 */
void quantize_rowwise_q4(const float* input, size_t rows, size_t cols,
                        float* out_scales, uint8_t* out_quantized) {
    for (size_t row = 0; row < rows; row++) {
        const float* row_data = input + row * cols;
        
        // Find maximum absolute value in this row
        float max_abs = 0.0f;
        for (size_t col = 0; col < cols; col++) {
            float abs_val = fabsf(row_data[col]);
            if (abs_val > max_abs) {
                max_abs = abs_val;
            }
        }
        
        // Calculate scaling factor for 4-bit range [-8, 7] -> [0, 15]
        float scale = fmaxf(max_abs / 7.0f, 1e-8f);
        out_scales[row] = scale;
        
        // Quantize and pack values in this row
        size_t packed_cols = (cols + 1) / 2; // Round up for odd number of columns
        for (size_t pack_idx = 0; pack_idx < packed_cols; pack_idx++) {
            size_t col_lo = pack_idx * 2;
            size_t col_hi = col_lo + 1;
            
            // Quantize low nibble value
            float val_lo = row_data[col_lo] / scale;
            int8_t q_lo = (int8_t)fmaxf(-8.0f, fminf(7.0f, roundf(val_lo)));
            uint8_t packed_lo = (uint8_t)(q_lo + 8); // Shift to [0, 15]
            
            // Quantize high nibble value (if it exists)
            uint8_t packed_hi = 0;
            if (col_hi < cols) {
                float val_hi = row_data[col_hi] / scale;
                int8_t q_hi = (int8_t)fmaxf(-8.0f, fminf(7.0f, roundf(val_hi)));
                packed_hi = (uint8_t)(q_hi + 8); // Shift to [0, 15]
            }
            
            // Pack two 4-bit values into one byte: low_nibble | (high_nibble << 4)
            out_quantized[row * packed_cols + pack_idx] = packed_lo | (packed_hi << 4);
        }
    }
}

/**
 * Check if tensor should be quantized based on its name
 * Only expert weight matrices should be quantized according to export.py logic
 */
bool should_quantize_tensor(const char* name) {
    if (!name) return false;
    
    // Check if this is an expert weight tensor
    return (strstr(name, ".experts.") != NULL &&
            (strstr(name, ".gate_proj.weight") != NULL ||
             strstr(name, ".up_proj.weight") != NULL ||
             strstr(name, ".down_proj.weight") != NULL));
}

/**
 * Calculate size needed for quantized data storage
 */
size_t get_quantized_data_size(size_t rows, size_t cols, QuantType qtype) {
    switch (qtype) {
        case QUANT_Q8:
            return rows * cols * sizeof(int8_t);
        case QUANT_Q4:
            return rows * ((cols + 1) / 2) * sizeof(uint8_t); // Packed 2 values per byte
        default:
            return 0;
    }
}

/**
 * Calculate the number of groups for given tensor dimensions and group size
 */
size_t get_num_groups(size_t rows, size_t cols, size_t group_size) {
    if (group_size == 0) {
        // Rowwise quantization: one scale per row
        return rows;
    } else {
        // Group-wise quantization: groups span across the tensor
        size_t total_elements = rows * cols;
        return (total_elements + group_size - 1) / group_size; // Ceiling division
    }
}

/**
 * Calculate size needed for scaling factors
 */
size_t get_scales_size(size_t rows, size_t cols, size_t group_size) {
    return get_num_groups(rows, cols, group_size) * sizeof(float);
}

/**
 * Group-wise Q8 quantization implementation
 * Quantizes tensor in groups of specified size with shared scaling factors per group
 */
void quantize_groupwise_q8(const float* input, size_t rows, size_t cols, size_t group_size,
                          float* out_scales, int8_t* out_quantized, size_t* out_num_groups) {
    size_t total_elements = rows * cols;
    size_t num_groups = get_num_groups(rows, cols, group_size);
    *out_num_groups = num_groups;
    
    for (size_t group_idx = 0; group_idx < num_groups; group_idx++) {
        size_t start_idx = group_idx * group_size;
        size_t end_idx = start_idx + group_size;
        if (end_idx > total_elements) end_idx = total_elements;
        
        // Find maximum absolute value in this group
        float max_abs = 0.0f;
        for (size_t idx = start_idx; idx < end_idx; idx++) {
            float abs_val = fabsf(input[idx]);
            if (abs_val > max_abs) {
                max_abs = abs_val;
            }
        }
        
        // Calculate scaling factor
        float scale = fmaxf(max_abs / 127.0f, 1e-8f);
        out_scales[group_idx] = scale;
        
        // Quantize values in this group
        for (size_t idx = start_idx; idx < end_idx; idx++) {
            float normalized = input[idx] / scale;
            int8_t quantized = (int8_t)roundf(normalized);
            out_quantized[idx] = quantized;
        }
    }
}

/**
 * Group-wise Q4 quantization implementation
 * Quantizes tensor in groups of specified size with shared scaling factors per group
 */
void quantize_groupwise_q4(const float* input, size_t rows, size_t cols, size_t group_size,
                          float* out_scales, uint8_t* out_quantized, size_t* out_num_groups) {
    size_t total_elements = rows * cols;
    size_t num_groups = get_num_groups(rows, cols, group_size);
    *out_num_groups = num_groups;
    
    for (size_t group_idx = 0; group_idx < num_groups; group_idx++) {
        size_t start_idx = group_idx * group_size;
        size_t end_idx = start_idx + group_size;
        if (end_idx > total_elements) end_idx = total_elements;
        
        // Find maximum absolute value in this group
        float max_abs = 0.0f;
        for (size_t idx = start_idx; idx < end_idx; idx++) {
            float abs_val = fabsf(input[idx]);
            if (abs_val > max_abs) {
                max_abs = abs_val;
            }
        }
        
        // Calculate scaling factor for 4-bit range [-8, 7]
        float scale = fmaxf(max_abs / 7.0f, 1e-8f);
        out_scales[group_idx] = scale;
        
        // Quantize and pack values in this group
        for (size_t idx = start_idx; idx < end_idx; idx++) {
            float normalized = input[idx] / scale;
            // Clamp to Q4 range [-8, 7] and round to nearest integer
            int8_t quantized = (int8_t)fmaxf(-8.0f, fminf(7.0f, roundf(normalized)));
            
            // Q4 packing: Store 2 values per byte in nibble format
            // Challenge: Groups may not align with byte boundaries, so we pack
            // values back into their original matrix positions for row-major layout
            
            // Convert linear index back to 2D coordinates
            size_t row = idx / cols;
            size_t col = idx % cols;
            size_t packed_cols = (cols + 1) / 2;  // Number of bytes needed per row
            size_t pack_idx = col / 2;            // Which byte in the row
            
            // Shift signed [-8,7] to unsigned [0,15] for storage
            uint8_t packed_val = (uint8_t)(quantized + 8);
            
            // Pack into appropriate nibble (4-bit half) of the byte
            if (col % 2 == 0) {
                // Store in low nibble (bits 0-3)
                out_quantized[row * packed_cols + pack_idx] = 
                    (out_quantized[row * packed_cols + pack_idx] & 0xF0) | (packed_val & 0x0F);
            } else {
                // Store in high nibble (bits 4-7)
                out_quantized[row * packed_cols + pack_idx] = 
                    (out_quantized[row * packed_cols + pack_idx] & 0x0F) | ((packed_val & 0x0F) << 4);
            }
        }
    }
}

/**
 * Quantize a tensor using the specified quantization type
 * Creates separate quantized data and scaling factor arrays
 */
QuantizedTensor* quantize_tensor(const TensorBin* tensor, QuantType qtype, size_t group_size) {
    if (!tensor || !tensor->data || tensor->ndim != 2) {
        fprintf(stderr, "Error: quantize_tensor requires 2D tensor\n");
        return NULL;
    }
    
    if (tensor->dtype != 0) { // Only support f32 input
        fprintf(stderr, "Error: quantize_tensor only supports f32 input tensors\n");
        return NULL;
    }
    
    size_t rows = (size_t)tensor->shape[0];
    size_t cols = (size_t)tensor->shape[1];
    const float* input_data = (const float*)tensor->data;
    
    // Allocate QuantizedTensor structure
    QuantizedTensor* qt = (QuantizedTensor*)malloc(sizeof(QuantizedTensor));
    if (!qt) {
        fprintf(stderr, "Error: Failed to allocate QuantizedTensor\n");
        return NULL;
    }
    
    qt->num_rows = rows;
    qt->row_size = cols;
    qt->group_size = group_size;
    qt->num_groups = get_num_groups(rows, cols, group_size);
    qt->qtype = qtype;
    
    // Allocate scaling factors
    qt->scales = (float*)malloc(get_scales_size(rows, cols, group_size));
    if (!qt->scales) {
        fprintf(stderr, "Error: Failed to allocate scaling factors\n");
        free(qt);
        return NULL;
    }
    
    // Allocate quantized data
    size_t q_data_size = get_quantized_data_size(rows, cols, qtype);
    qt->q_data = malloc(q_data_size);
    if (!qt->q_data) {
        fprintf(stderr, "Error: Failed to allocate quantized data\n");
        free(qt->scales);
        free(qt);
        return NULL;
    }
    
    // Perform quantization
    if (group_size == 0) {
        // Use rowwise quantization (backward compatibility)
        switch (qtype) {
            case QUANT_Q8:
                quantize_rowwise_q8(input_data, rows, cols, qt->scales, (int8_t*)qt->q_data);
                break;
            case QUANT_Q4:
                quantize_rowwise_q4(input_data, rows, cols, qt->scales, (uint8_t*)qt->q_data);
                break;
            default:
                fprintf(stderr, "Error: Unsupported quantization type %d\n", qtype);
                free(qt->q_data);
                free(qt->scales);
                free(qt);
                return NULL;
        }
    } else {
        // Use group-wise quantization
        size_t num_groups_out;
        switch (qtype) {
            case QUANT_Q8:
                quantize_groupwise_q8(input_data, rows, cols, group_size, 
                                     qt->scales, (int8_t*)qt->q_data, &num_groups_out);
                break;
            case QUANT_Q4:
                // Initialize packed data to zero first
                memset(qt->q_data, 0, q_data_size);
                quantize_groupwise_q4(input_data, rows, cols, group_size,
                                     qt->scales, (uint8_t*)qt->q_data, &num_groups_out);
                break;
            default:
                fprintf(stderr, "Error: Unsupported quantization type %d\n", qtype);
                free(qt->q_data);
                free(qt->scales);
                free(qt);
                return NULL;
        }
        qt->num_groups = num_groups_out;
    }
    
    return qt;
}

/**
 * Free memory allocated for QuantizedTensor
 */
void quantized_tensor_free(QuantizedTensor* qt) {
    if (!qt) return;
    
    free(qt->q_data);
    free(qt->scales);
    free(qt);
}