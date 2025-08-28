#include "tensor.h"
#include "quant.h"  // For quantization functions
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/**
 * Calculate size needed for quantized data storage
 */
static size_t get_quantized_data_size(size_t rows, size_t cols, QuantType qtype) {
    switch (qtype) {
        case QUANT_Q8:
            return rows * cols * sizeof(int8_t);
        case QUANT_Q4:
            return (rows * cols + 1) / 2;  // Packed 4-bit
        default:
            return rows * cols * sizeof(float);  // FP32
    }
}

/**
 * Calculate size needed for scaling factors
 */
static size_t get_scales_size(size_t rows, size_t cols, size_t group_size) {
    size_t total_elements = rows * cols;
    size_t num_groups = (total_elements + group_size - 1) / group_size;  // Ceiling division
    return num_groups * sizeof(float);
}

/**
 * Create a new tensor with the given parameters
 * Allocates memory and copies the data
 */
Tensor* tensor_create(const char* name, int dtype, int ndim, const int* shape, const void* data) {
    if (!name || !shape || !data || ndim <= 0) return NULL;
    
    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    if (!tensor) return NULL;
    
    // Copy name
    size_t name_len = strlen(name);
    tensor->name = (char*)malloc(name_len + 1);
    if (!tensor->name) {
        free(tensor);
        return NULL;
    }
    strcpy(tensor->name, name);
    
    // Copy shape
    tensor->shape = (int*)malloc(sizeof(int) * ndim);
    if (!tensor->shape) {
        free(tensor->name);
        free(tensor);
        return NULL;
    }
    memcpy(tensor->shape, shape, sizeof(int) * ndim);
    
    // Calculate data size based on dtype and shape
    size_t elements = 1;
    for (int i = 0; i < ndim; i++) {
        elements *= (size_t)shape[i];
    }
    
    size_t bytes_per_element;
    switch (dtype) {
        case 0: bytes_per_element = 4; break; // f32
        case 1: bytes_per_element = 2; break; // f16
        case 2: bytes_per_element = 1; break; // i8
        case 3: bytes_per_element = 1; break; // i4 (packed)
        default:
            free(tensor->shape);
            free(tensor->name);
            free(tensor);
            return NULL;
    }
    
    size_t data_size = elements * bytes_per_element;
    
    // Copy data
    tensor->data = malloc(data_size);
    if (!tensor->data) {
        free(tensor->shape);
        free(tensor->name);
        free(tensor);
        return NULL;
    }
    memcpy(tensor->data, data, data_size);
    
    tensor->dtype = dtype;
    tensor->ndim = ndim;
    tensor->nbytes = data_size;
    tensor->group_size = 0; // Default to 0 (rowwise or non-quantized)
    
    return tensor;
}

/**
 * Create a Tensor with specified group_size metadata
 * Used for quantized tensors that need group_size information
 */
Tensor* tensor_create_with_group_size(const char* name, int dtype, int ndim, const int* shape, const void* data, size_t group_size) {
    Tensor* tensor = tensor_create(name, dtype, ndim, shape, data);
    if (tensor) {
        tensor->group_size = group_size;
    }
    return tensor;
}

/**
 * Free a single tensor (used when tensor is not part of a BinFile array)
 */
void tensor_free_single(Tensor* tensor) {
    if (!tensor) return;
    
    free(tensor->name);
    free(tensor->shape);
    free(tensor->data);
    free(tensor);
}

/**
 * Quantize a tensor for the conversion tool
 * Expects a Tensor* from tensor.h
 */
QuantizedTensor* quantize_tensor(Tensor* input_tensor, QuantType qtype, size_t group_size) {
    if (!input_tensor || !input_tensor->data) return NULL;
    
    // Get dimensions
    size_t rows, cols;
    if (input_tensor->ndim == 2) {
        rows = input_tensor->shape[0];
        cols = input_tensor->shape[1];
    } else if (input_tensor->ndim == 1) {
        rows = 1;
        cols = input_tensor->shape[0];
    } else {
        // Flatten higher dimensions
        rows = 1;
        cols = 1;
        for (int i = 0; i < input_tensor->ndim; i++) {
            cols *= input_tensor->shape[i];
        }
    }
    
    QuantizedTensor* qt = (QuantizedTensor*)malloc(sizeof(QuantizedTensor));
    if (!qt) return NULL;
    
    qt->num_rows = rows;
    qt->row_size = cols;
    qt->group_size = group_size;
    qt->qtype = qtype;
    qt->num_groups = (rows * cols + group_size - 1) / group_size;
    
    // Allocate scaling factors
    qt->scales = (float*)malloc(get_scales_size(rows, cols, group_size));
    if (!qt->scales) {
        free(qt);
        return NULL;
    }
    
    // Allocate quantized data
    size_t q_data_size = get_quantized_data_size(rows, cols, qtype);
    qt->q_data = malloc(q_data_size);
    if (!qt->q_data) {
        free(qt->scales);
        free(qt);
        return NULL;
    }
    
    if (qtype == QUANT_Q4) {
        // Allocate zero points for Q4
        qt->zero_points = (float*)malloc(qt->num_groups * sizeof(float));
        if (!qt->zero_points) {
            free(qt->q_data);
            free(qt->scales);
            free(qt);
            return NULL;
        }
        
        // Quantize with asymmetric Q4 (the only Q4 we support)
        quantize_q4(input_tensor->data, rows, cols, group_size, qt->scales, qt->zero_points, (uint8_t*)qt->q_data);
    } else if (qtype == QUANT_Q8) {
        qt->zero_points = NULL;  // Q8 doesn't use zero points
        
        // Quantize with symmetric Q8
        quantize_q8(input_tensor->data, (int8_t*)qt->q_data, qt->scales, rows * cols, group_size);
    } else {
        // No quantization - just copy the data
        qt->zero_points = NULL;
        memcpy(qt->q_data, input_tensor->data, rows * cols * sizeof(float));
        for (size_t i = 0; i < qt->num_groups; i++) {
            qt->scales[i] = 1.0f;  // Identity scaling
        }
    }
    
    return qt;
}

/**
 * Free a quantized tensor
 */
void quantized_tensor_free(QuantizedTensor* qt) {
    if (!qt) return;
    
    if (qt->q_data) free(qt->q_data);
    if (qt->scales) free(qt->scales);
    if (qt->zero_points) free(qt->zero_points);
    free(qt);
}

/**
 * Create a QuantizedTensor that points to data in binary file (non-owning)
 * Used by model loading to wrap binary file data without copying
 */
QuantizedTensor* quantized_tensor_from_binary(const void* q_data, const float* scales, const float* zero_points,
                                             size_t rows, size_t cols, QuantType qtype, size_t group_size) {
    if (!q_data || !scales) return NULL;
    
    QuantizedTensor* qt = (QuantizedTensor*)malloc(sizeof(QuantizedTensor));
    if (!qt) return NULL;
    
    qt->q_data = (void*)q_data;  // Cast away const for unified interface
    qt->scales = (float*)scales;
    qt->zero_points = (float*)zero_points;  // May be NULL for Q8
    qt->num_rows = rows;
    qt->row_size = cols;
    qt->group_size = group_size;
    qt->qtype = qtype;
    qt->num_groups = (rows * cols + group_size - 1) / group_size;
    
    return qt;
}

/**
 * Free a non-owning QuantizedTensor (doesn't free the data pointers)
 */
void quantized_tensor_free_non_owning(QuantizedTensor* qt) {
    if (!qt) return;
    free(qt);  // Only free the struct, not the data it points to
}

