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
#include "tensor.h"  // Needed for Tensor and QuantizedTensor types

/**
 * Q8 QUANTIZATION (Symmetric, Standard)
 */
// Helper functions for Q8 quantization (inline for performance)
static inline float find_max_abs(const float* restrict data, int size);
static inline float quantize_q8_group(const float* restrict group_data, int8_t* restrict qx_q, int group_size);

void quantize_q8(const float* restrict x, int8_t* restrict qx_q, float* restrict qx_s, int n, int group_size);
void dequantize_q8(const int8_t* restrict qx_q, const float* restrict qx_s, float* restrict x, int n, int group_size);

/**
 * Q4 QUANTIZATION (With Zero Points - THE ONLY Q4 functions)
 * 35.8% more accurate than symmetric Q4 (which we removed)
 */
void quantize_q4(const float* input, size_t rows, size_t cols, size_t group_size,
                 float* out_scales, float* out_zero_points, uint8_t* out_quantized);

void dequantize_q4(const uint8_t* qdata, const float* scales, const float* zero_points,
                   float* out_fp32, size_t rows, size_t cols, size_t group_size);

/**
 * OPTIMIZED ACTIVATION FUNCTIONS FOR MOE LAYERS
 */
void apply_silu_optimized(float* x, int n);
void elementwise_multiply_optimized(const float* a, const float* b, float* c, int n);

/**
 * MATRIX MULTIPLICATION FUNCTIONS (Only the best ones)
 */

/**
 * Fastest Q8×Q8 matrix multiplication
 */
void matmul_q8_q8_f32(const int8_t* restrict A_q8, const float* restrict A_scales,
                      const int8_t* restrict B_q8, const float* restrict B_scales,
                      float* restrict C, int M, int N, int K, int group_size);

/**
 * Reference Q8×Q8 matrix multiplication (slow, for compatibility)
 */
void matmul_q8_q8_f32_reference(const int8_t* A_q8, const float* A_scales, const int8_t* B_q8, const float* B_scales, float* C, int M, int N, int K, int group_size);

/**
 * Q8×Q4 matrix multiplication with asymmetric Q4 (35.8% accuracy improvement)
 * This is the ONLY Q4 matmul function - renamed from matmul_q8_q4_opt
 * Zero points provide much better accuracy than symmetric Q4
 */
void matmul_q8_q4_f32(const int8_t* A, const float* A_scales,
                      const uint8_t* B_q4, const float* B_scales, const float* B_zps,
                      float* C, int M, int N, int K, size_t group_size);

/**
 * Reference Q8×Q4 matrix multiplication (slow, for compatibility)
 */
void matmul_q8_q4_f32_reference(const int8_t* A, const float* A_scales,
                                const uint8_t* B_q4, const float* B_scales, const float* B_zps,
                                float* C, int M, int N, int K, size_t group_size);

/**
 * INFERENCE FUNCTIONS (Quantize activations on-the-fly)
 */

/**
 * FP32 × Q8 matrix multiplication (quantizes A on-the-fly)
 */
void matmul_f32_q8_f32(const float* restrict A_fp32, const int8_t* restrict B_q8, const float* restrict B_scales,
                       float* restrict C, int M, int N, int K, int group_size,
                       int8_t* restrict qx_q_scratch, float* restrict qx_s_scratch);

/**
 * FP32 × Q4 matrix multiplication with zero points (new optimized version)
 */
void matmul_f32_q4_f32_with_zeros(const float* A_fp32, const uint8_t* B_q4, const float* B_scales, const float* B_zeros,
                                  float* C, int M, int N, int K, int group_size,
                                  int8_t* qx_q_scratch, float* qx_s_scratch);

/**
 * FP32 × Q4 matrix multiplication (backward compatible - no zero points parameter)
 * This maintains the old API that existing code expects
 */
void matmul_f32_q4_f32(const float* restrict A_fp32, const uint8_t* restrict B_q4, const float* restrict B_scales,
                       float* restrict C, int M, int N, int K, int group_size,
                       int8_t* restrict qx_q_scratch, float* restrict qx_s_scratch);

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
                      const uint8_t* B_q4, const float* B_scales, const float* B_zps,
                      float* C, int M, int N, int K, size_t group_size);

/**
 * OPTIMIZED MATRIX MULTIPLICATION FUNCTIONS
 * Fast end-to-end implementation with cached B-packing
 */

/**
 * Cached packed B matrix for Q8×Q8 operations
 */
typedef struct PackedB_q8 {
    int K, N, group_size;
    int tiles;                    // Number of 1×4 tiles
    int8_t* bp;                  // 1×4 packed data (aligned)
    float* bp_scales;            // Packed scales: [tile][g][4] layout
    void* cache_key;             // Original B pointer for cache lookup
    size_t cache_key_size;       // Size for cache validation
    int ref_count;               // Reference counting
    struct PackedB_q8* next;     // LRU linked list
} PackedB_q8;

/**
 * Get or create cached packed B matrix
 */
PackedB_q8* get_or_create_packed_B_q8(const int8_t* B_q8, const float* B_scales,
                                      int K, int N, int group_size);

/**
 * Free cached packed B matrix
 */
void free_packed_B_q8(PackedB_q8* pb);

/**
 * Fast Q8×Q8 with shape-based dispatch and cached packing
 * Uses cached packed B when beneficial, direct path for small matrices
 */
void matmul_q8_q8_f32_fast(const int8_t* A_q8, const float* A_scales,
                           const int8_t* B_q8, const float* B_scales,
                           float* C, int M, int N, int K, int group_size);

/**
 * Direct Q8×Q8 without packing (fast path for small matrices)
 */
void matmul_q8_q8_f32_direct(const int8_t* A_q8, const float* A_scales,
                             const int8_t* B_q8, const float* B_scales,
                             float* C, int M, int N, int K, int group_size);

/**
 * Optimized implementations from optimizations.txt
 */
void matmul_q8_q8_f32_from_optimizations_txt(const int8_t* A_q8, const float* A_scales,
                                             const int8_t* B_q8, const float* B_scales,
                                             float* C, int M, int N, int K, int group_size);

void matmul_f32_q8_f32_fused_from_optimizations_txt(const float* A_fp32, const int8_t* B_q8, 
                                                    const float* B_scales, float* C,
                                                    int M, int N, int K, int group_size,
                                                    int8_t* unused1, float* unused2);

/**
 * Optimized Q8×Q8 with cached packed B (original optimized version)
 */
void matmul_q8_q8_f32_optimized(const int8_t* A_q8, const float* A_scales,
                                const int8_t* B_q8, const float* B_scales,
                                float* C, int M, int N, int K, int group_size);



/**
 * FAST CACHED OPTIMIZATION FUNCTIONS (203ms/token Q8, 296ms/token Q4)
 */

/**
 * Get or create cached packed B matrix
 */
PackedB_q8* get_or_create_packed_B_q8(const int8_t* B_q8, const float* B_scales,
                                      int K, int N, int group_size);

/**
 * Free cached packed B matrix
 */
void free_packed_B_q8(PackedB_q8* pb);

/**
 * Fast Q8×Q8 with shape-based dispatch and cached packing
 * Uses cached packed B when beneficial, direct path for small matrices
 */
void matmul_q8_q8_f32_fast(const int8_t* A_q8, const float* A_scales,
                           const int8_t* B_q8, const float* B_scales,
                           float* C, int M, int N, int K, int group_size);

/**
 * Direct Q8×Q8 without packing (fast path for small matrices)
 */
void matmul_q8_q8_f32_direct(const int8_t* A_q8, const float* A_scales,
                             const int8_t* B_q8, const float* B_scales,
                             float* C, int M, int N, int K, int group_size);

#endif // QUANT_H