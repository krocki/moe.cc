#ifndef QUANT_H
#define QUANT_H

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
 * Used for rowwise quantization where each row has its own scale
 */
typedef struct {
    void*   q_data;     // Quantized data (int8* for Q8, uint8* for Q4)
    float*  scales;     // Per-row scaling factors
    size_t  num_rows;   // Number of rows (for validation)
    size_t  row_size;   // Size of each row in elements
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
 * Quantize a single tensor from FP32 to the specified quantization type
 * Creates quantized data + scaling factors following export.py conventions
 * 
 * @param tensor: Input tensor to quantize (must be 2D for expert weights)
 * @param qtype: Quantization type (Q8 or Q4)
 * @return: QuantizedTensor structure with quantized data and scales
 *          Caller must free returned structure with quantized_tensor_free()
 */
QuantizedTensor* quantize_tensor(const TensorBin* tensor, QuantType qtype);

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
 * @param rows: Number of rows (one scale per row)
 * @return: Size in bytes for scaling factors (always float32)
 */
size_t get_scales_size(size_t rows);

#endif // QUANT_H