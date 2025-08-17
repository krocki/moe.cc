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
 * Calculate size needed for scaling factors (one float per row)
 */
size_t get_scales_size(size_t rows) {
    return rows * sizeof(float);
}

/**
 * Quantize a tensor using the specified quantization type
 * Creates separate quantized data and scaling factor arrays
 */
QuantizedTensor* quantize_tensor(const TensorBin* tensor, QuantType qtype) {
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
    qt->qtype = qtype;
    
    // Allocate scaling factors (one per row)
    qt->scales = (float*)malloc(get_scales_size(rows));
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