#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

/**
 * Tensor structure (renamed from Tensor)
 */
typedef struct {
  char*  name;
  int    dtype;   // 0=f32, 1=f16, 2=i8, 3=i4
  int    ndim;
  int*   shape;   // length=ndim
  void*  data;    // raw pointer (malloc'ed)
  size_t nbytes;  // size of data in bytes
  size_t group_size; // Group size for quantized tensors (0 = rowwise or non-quantized)
} Tensor;

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
    float* zero_points;  // Zero points (only for Q4)
    size_t  num_rows;   
    size_t  row_size;   
    size_t  group_size; 
    size_t  num_groups; 
    QuantType qtype;    
} QuantizedTensor;

/**
 * Tensor allocation and management functions
 */
Tensor* tensor_create(const char* name, int dtype, int ndim, const int* shape, const void* data);
Tensor* tensor_create_with_group_size(const char* name, int dtype, int ndim, const int* shape, const void* data, size_t group_size);
void tensor_free_single(Tensor* tensor);

/**
 * QuantizedTensor allocation and management functions
 */
QuantizedTensor* quantize_tensor(Tensor* input_tensor, QuantType qtype, size_t group_size);
void quantized_tensor_free(QuantizedTensor* qt);

/**
 * Create a QuantizedTensor that points to data in binary file (non-owning)
 * Used by model loading to wrap binary file data without copying
 */
QuantizedTensor* quantized_tensor_from_binary(const void* q_data, const float* scales, const float* zero_points,
                                             size_t rows, size_t cols, QuantType qtype, size_t group_size);

/**
 * Free a non-owning QuantizedTensor (doesn't free the data pointers)
 */
void quantized_tensor_free_non_owning(QuantizedTensor* qt);



#endif // TENSOR_H