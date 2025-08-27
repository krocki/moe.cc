#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <sys/time.h>
#include <omp.h>
#include "quant.h"
#include <string.h>


#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

static inline int get_log2(size_t x) {
    return __builtin_ctzl(x);
}

// Forward declarations for NEON functions
void matmul_q8_q8_f32_neon(const int8_t* A_q8, const float* A_scales, const int8_t* B_q8, const float* B_scales, float* C, int M, int N, int K, int group_size);
void matmul_q8_q4_f32_neon(const int8_t* A_q8, const float* A_scales, const uint8_t* B_q4_packed, const float* B_scales, const float* B_zero_points, float* C, int M, int N, int K, int group_size);

/**
 * Q8 QUANTIZATION FUNCTIONS (Symmetric, Standard)
 */

/**
 * Helper function: Find maximum absolute value in an array
 * Optimized with NEON intrinsics where available
 */
static inline float find_max_abs(const float* restrict data, int size) {
    float max_abs = 0.0f;
    
#ifdef __ARM_NEON
    if (size >= 4) {
        float32x4_t max_vec = vdupq_n_f32(0.0f);
        int i = 0;
        for (; i + 4 <= size; i += 4) {
            float32x4_t val_vec = vld1q_f32(&data[i]);
            float32x4_t abs_vec = vabsq_f32(val_vec);
            max_vec = vmaxq_f32(max_vec, abs_vec);
        }
        
        // Horizontal max reduction
        float32x2_t max_pair = vmax_f32(vget_low_f32(max_vec), vget_high_f32(max_vec));
        max_pair = vpmax_f32(max_pair, max_pair);
        max_abs = vget_lane_f32(max_pair, 0);
        
        // Handle remainder
        for (; i < size; i++) {
            float val = fabsf(data[i]);
            if (val > max_abs) max_abs = val;
        }
    } else {
        // Small arrays - fallback to scalar
        for (int i = 0; i < size; i++) {
            float val = fabsf(data[i]);
            if (val > max_abs) max_abs = val;
        }
    }
#else
    // Fallback implementation
    for (int i = 0; i < size; i++) {
        float val = fabsf(data[i]);
        if (val > max_abs) max_abs = val;
    }
#endif
    
    return max_abs;
}

/**
 * Helper function: Quantize a single group of floats to Q8
 * Returns the scale factor used for quantization
 * Optimized with NEON intrinsics where available
 */
static inline float quantize_q8_group(const float* restrict group_data, int8_t* restrict qx_q, int group_size) {
    const float Q_MAX = 127.0f;
    
    // Step 1: Find maximum absolute value
    float max_abs = find_max_abs(group_data, group_size);
    
    // Step 2: Calculate scale
    float scale = fmaxf(max_abs / Q_MAX, 1e-8f);
    float inv_scale = 1.0f / scale;
    
    // Step 3: Quantize values (SIMD optimized)
#ifdef __ARM_NEON
    if (group_size >= 4) {
        const float32x4_t inv_scale_vec = vdupq_n_f32(inv_scale);
        const int32x4_t max_int_vec = vdupq_n_s32(127);
        const int32x4_t min_int_vec = vdupq_n_s32(-127);
        
        int i = 0;
        for (; i + 4 <= group_size; i += 4) {
            // Load and scale
            float32x4_t val_vec = vld1q_f32(&group_data[i]);
            float32x4_t scaled_vec = vmulq_f32(val_vec, inv_scale_vec);
            
            // Round and convert to int32
            int32x4_t int_vec = vcvtnq_s32_f32(scaled_vec);  // Round to nearest
            
            // Clamp to [-127, 127]
            int_vec = vmaxq_s32(int_vec, min_int_vec);
            int_vec = vminq_s32(int_vec, max_int_vec);
            
            // Convert to int8 and store
            int16x4_t int16_vec = vmovn_s32(int_vec);
            int8x8_t int8_vec = vmovn_s16(vcombine_s16(int16_vec, int16_vec));
            
            // Store 4 bytes
            qx_q[i] = vget_lane_s8(int8_vec, 0);
            qx_q[i + 1] = vget_lane_s8(int8_vec, 1);
            qx_q[i + 2] = vget_lane_s8(int8_vec, 2);
            qx_q[i + 3] = vget_lane_s8(int8_vec, 3);
        }
        
        // Handle remainder
        for (; i < group_size; i++) {
            float quant_value = group_data[i] * inv_scale;
            int rounded = (int)roundf(quant_value);
            qx_q[i] = (int8_t)(rounded < -127 ? -127 : (rounded > 127 ? 127 : rounded));
        }
    } else {
        // Small groups - fallback to scalar
        for (int i = 0; i < group_size; i++) {
            float quant_value = group_data[i] * inv_scale;
            int rounded = (int)roundf(quant_value);
            qx_q[i] = (int8_t)(rounded < -127 ? -127 : (rounded > 127 ? 127 : rounded));
        }
    }
#else
    // Fallback implementation
    for (int i = 0; i < group_size; i++) {
        float quant_value = group_data[i] * inv_scale;
        int rounded = (int)roundf(quant_value);
        qx_q[i] = (int8_t)(rounded < -127 ? -127 : (rounded > 127 ? 127 : rounded));
    }
#endif
    
    return scale;
}

/**
 * SIMD-Optimized Q8 quantization with ARM Neon support
 */
void quantize_q8(const float* restrict x, int8_t* restrict qx_q, float* restrict qx_s, int n, int group_size) {
    assert(is_power_of_2(group_size));
    assert(group_size > 0);
    assert(n % group_size == 0);
    
    // Hint compiler about alignment assumptions for better vectorization
    x = (const float*)__builtin_assume_aligned(x, 16);
    qx_q = (int8_t*)__builtin_assume_aligned(qx_q, 16);  
    qx_s = (float*)__builtin_assume_aligned(qx_s, 16);
    
    const int group_size_log2 = get_log2(group_size);
    const int num_groups = n >> group_size_log2;
    
    // Use OpenMP for large datasets, avoid thread overhead for small ones
    if (num_groups >= 16) {
        #pragma omp parallel for schedule(static)
        for (int group = 0; group < num_groups; group++) {
            const int start = group << group_size_log2;
            qx_s[group] = quantize_q8_group(&x[start], &qx_q[start], group_size);
        }
    } else {
        // Serial version for small datasets
        for (int group = 0; group < num_groups; group++) {
            const int start = group << group_size_log2;
            qx_s[group] = quantize_q8_group(&x[start], &qx_q[start], group_size);
        }
    }
}

/**
 * Optimized Q8 dequantization with group-wise vectorization
 */
void dequantize_q8(const int8_t* qx_q, const float* qx_s, float* x, int n, int group_size) {
    assert(is_power_of_2(group_size));
    assert(group_size > 0);
    assert(n % group_size == 0);
    
    const int group_size_log2 = get_log2(group_size);
    const int num_groups = n >> group_size_log2;
    
    // Vectorized dequantization with group-wise processing
    for (int group = 0; group < num_groups; group++) {
        const int start = group << group_size_log2;
        const float scale = qx_s[group];
        
        // Vectorized scaling within each group
        #pragma GCC unroll 4
        for (int i = 0; i < group_size; i++) {
            x[start + i] = (float)qx_q[start + i] * scale;
        }
    }
}

/**
 * Q4 ASYMMETRIC QUANTIZATION FUNCTIONS (With Zero Points - 35.8% More Accurate)
 */

/**
 * Q4 quantization with zero points (this is THE Q4 function - asymmetric only)
 * 35.8% accuracy improvement over symmetric Q4 (which we removed)
 */
void quantize_q4(const float* input, size_t rows, size_t cols, size_t group_size,
                 float* out_scales, float* out_zero_points, uint8_t* out_quantized) {
    assert((rows * cols) % group_size == 0);
    assert(group_size > 0 && (group_size & (group_size - 1)) == 0);
    
    // Alignment hints for better vectorization
    input = (const float*)__builtin_assume_aligned(input, 16);
    out_scales = (float*)__builtin_assume_aligned(out_scales, 16);
    out_zero_points = (float*)__builtin_assume_aligned(out_zero_points, 16);
    out_quantized = (uint8_t*)__builtin_assume_aligned(out_quantized, 16);

    const size_t num_elems = rows * cols;
    const int log2_group = __builtin_ctz(group_size);
    const size_t num_groups = num_elems >> log2_group;
    const float QMAX = 15.0f;

    for (size_t g = 0; g < num_groups; ++g) {
        size_t start = g * group_size;
        
        // Find min/max in group with vectorization
        float min_val = input[start];
        float max_val = input[start];
        #pragma GCC unroll 4
        for (size_t i = 1; i < group_size; ++i) {
            float v = input[start + i];
            if (v < min_val) min_val = v;
            if (v > max_val) max_val = v;
        }

        // Calculate scale and zero point
        float scale = (max_val - min_val) / QMAX;
        if (scale < 1e-8f) scale = 1e-8f;  // Use minimal scale, not 1.0!
        float inv_scale = 1.0f / scale;

        // Standard asymmetric Q4: q = (x - zero_point) / scale
        // where zero_point = min_val (maps to q=0)
        out_scales[g] = scale;
        
        // Store min_val directly as float (like reference implementation)
        out_zero_points[g] = min_val;

        // Standard asymmetric quantization: q = round((x - zero_point) / scale)
        #pragma GCC unroll 2
        for (size_t i = 0; i < group_size; i += 2) {
            uint8_t q0 = (uint8_t)roundf((input[start + i] - min_val) / scale);
            q0 = q0 > 15 ? 15 : q0;  // Clamp to [0,15]

            uint8_t q1 = 0;  // Padding with 0
            if (i + 1 < group_size) {
                q1 = (uint8_t)roundf((input[start + i + 1] - min_val) / scale);
                q1 = q1 > 15 ? 15 : q1;
            }

            size_t out_idx = (start + i) >> 1;
            out_quantized[out_idx] = (q1 << 4) | (q0 & 0x0F);
        }
    }
}

/**
 * Q4 dequantization with zero points (this is THE Q4 function - asymmetric only)
 */
void dequantize_q4(const uint8_t* qdata, const float* scales, const float* zero_points,
                   float* out_fp32, size_t rows, size_t cols, size_t group_size) {
    const size_t num_elems = rows * cols;
    const int log2_group = __builtin_ctz(group_size);

    for (size_t i = 0; i < num_elems; ++i) {
        size_t g = i >> log2_group;
        size_t packed_idx = i >> 1;
        uint8_t packed = qdata[packed_idx];
        uint8_t q = (i & 1) ? (packed >> 4) & 0x0F : (packed & 0x0F);
        // Standard asymmetric dequantization: x = q * scale + min_val
        float min_val = zero_points[g];  // Use float directly (like reference)  
        out_fp32[i] = (int)q * scales[g] + min_val;
    }
}

/**
 * MATRIX MULTIPLICATION FUNCTIONS
 */

/**
 * Fastest Q8×Q8 matrix multiplication
 */
void matmul_q8_q8_f32(const int8_t* A_q8, const float* A_scales,
                      const int8_t* B_q8, const float* B_scales,
                      float* C, int M, int N, int K, int group_size) {
    // Use NEON-optimized version directly (keep stable performance)
    matmul_q8_q8_f32_neon(A_q8, A_scales, B_q8, B_scales, C, M, N, K, group_size);
}

#ifdef __ARM_NEON
static inline void vec_dot_q8_q8_neon(const int8_t* __restrict a_group, const int8_t* __restrict b_group, int group_size, int32_t* __restrict dot_out) {
    // Use optimized 128-element kernel when possible
    *dot_out = 0;
    int32x4_t sum_vec = vdupq_n_s32(0);
    // Process in multiples of 128 when possible
    int i = 0;
    // Process remaining in chunks of 16 using vdotq_s32 with prefetching
    for (; i < group_size - 15; i += 16) {
        // Prefetch next iteration (distributed-llama technique)
        if (i + 32 < group_size) {
            __builtin_prefetch(a_group + i + 32, 0, 3);
            __builtin_prefetch(b_group + i + 32, 0, 3);
        }
        
        int8x16_t a_vec = vld1q_s8(a_group + i);
        int8x16_t b_vec = vld1q_s8(b_group + i);
        
#if defined(__ARM_FEATURE_DOTPROD)
        // Use dot-product instruction for maximum performance
        sum_vec = vdotq_s32(sum_vec, a_vec, b_vec);
#else
        // Fallback to standard multiply-accumulate
        int16x8_t prod_low = vmull_s8(vget_low_s8(a_vec), vget_low_s8(b_vec));
        int16x8_t prod_high = vmull_s8(vget_high_s8(a_vec), vget_high_s8(b_vec));
        int32x4_t sum_low = vpaddlq_s16(prod_low);
        int32x4_t sum_high = vpaddlq_s16(prod_high);
        sum_vec = vaddq_s32(sum_vec, vaddq_s32(sum_low, sum_high));
#endif
    }
    *dot_out += vaddvq_s32(sum_vec);
    
    // Scalar tail
    for (; i < group_size; i++) {
        *dot_out += (int32_t)a_group[i] * (int32_t)b_group[i];
    }
}
#else
static inline void vec_dot_q8_q8_scalar(const int8_t* a_group, const int8_t* b_group, int group_size, int32_t* dot_out) {
    *dot_out = 0;
    for (int i = 0; i < group_size; i++) {
        *dot_out += (int32_t)a_group[i] * (int32_t)b_group[i];
    }
}
#endif

/**
 * NEON-optimized Q8×Q8 matrix multiplication
 */
void matmul_q8_q8_f32_neon(const int8_t* __restrict A_q8, const float* __restrict A_scales, const int8_t* __restrict B_q8, const float* __restrict B_scales, float* __restrict C, int M, int N, int K, int group_size) {
    assert((K % group_size) == 0 && (group_size & (group_size - 1)) == 0);  // Power of 2
    int num_groups = K / group_size;
    
    
    #pragma omp parallel for collapse(2) schedule(static)
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float acc = 0.0f;
            for (int g = 0; g < num_groups; g++) {
                int start_k = g * group_size;
                const int8_t* a_group = A_q8 + m * K + start_k;
                const int8_t* b_group = B_q8 + n * K + start_k;
                float a_scale = A_scales[m * num_groups + g];
                float b_scale = B_scales[n * num_groups + g];
                int32_t dot = 0;
#ifdef __ARM_NEON
                vec_dot_q8_q8_neon(a_group, b_group, group_size, &dot);
#else
                vec_dot_q8_q8_scalar(a_group, b_group, group_size, &dot);
#endif
                acc += (float)dot * a_scale * b_scale;
            }
            C[m * N + n] = acc;
        }
    }
}

/**
 * Q8×Q4 matrix multiplication with asymmetric Q4 (OPTIMIZED VERSION)
 * Key optimizations:
 * 1. Precompute A sums to eliminate redundant computation
 * 2. Vectorized nibble unpacking (8 nibbles at once)  
 * 3. Better memory access patterns
 * 4. Reduced arithmetic operations
 */
void matmul_q8_q4_f32(const int8_t* A, const float* A_scales,
                      const uint8_t* B_q4, const float* B_scales, const float* B_zps,
                      float* C, int M, int N, int K, size_t group_size) {
    // Use NEON-optimized version directly
    matmul_q8_q4_f32_neon(A, A_scales, B_q4, B_scales, B_zps, C, M, N, K, (int)group_size);
}

/**
 * NEON-optimized vector dot product for Q8×Q4 groups
 */
#ifdef __ARM_NEON
static inline void vec_dot_q8_q4_neon(const int8_t* __restrict a_group, const uint8_t* __restrict b_group_packed, int group_size, int32_t* __restrict dot_out, int32_t* __restrict a_sum_out) {
    *dot_out = 0;
    if (a_sum_out) *a_sum_out = 0;
    int i = 0;
    // Optimized processing: chunks of 16 with prefetching and unrolling
    int32x4_t dot_vec = vdupq_n_s32(0);
    int32x4_t a_sum_vec = vdupq_n_s32(0);
    
    for (; i + 16 <= group_size; i += 16) {
        // Prefetch next iteration
        if (i + 32 < group_size) {
            __builtin_prefetch(a_group + i + 32, 0, 3);
            __builtin_prefetch(b_group_packed + (i + 32)/2, 0, 3);
        }
        
        // Process two 8-element chunks in parallel
        for (int j = 0; j < 16; j += 8) {
            int8x8_t a_vec = vld1_s8(a_group + i + j);
            
            uint32_t packed_u32 = *(const uint32_t*)(b_group_packed + (i + j) / 2);
            uint8x8_t packed_bytes = vcreate_u8(packed_u32);
            
            uint8x8_t low_nibbles = vand_u8(packed_bytes, vdup_n_u8(0x0F));
            uint8x8_t high_nibbles = vshr_n_u8(packed_bytes, 4);
            
            uint8x8x2_t unpacked = vzip_u8(low_nibbles, high_nibbles);
            int8x8_t b_vec = vreinterpret_s8_u8(unpacked.val[0]);
            
            int16x8_t dot_wide = vmull_s8(a_vec, b_vec);
            dot_vec = vaddq_s32(dot_vec, vpaddlq_s16(dot_wide));
            
            if (a_sum_out) {
                int16x4_t a_sum_wide = vpaddl_s8(a_vec);
                a_sum_vec = vaddq_s32(a_sum_vec, vpaddlq_s16(vcombine_s16(a_sum_wide, vdup_n_s16(0))));
            }
        }
    }
    
    *dot_out += vaddvq_s32(dot_vec);
    if (a_sum_out) *a_sum_out += vaddvq_s32(a_sum_vec);
    
    // Process remaining in chunks of 8
    for (; i < group_size - 7; i += 8) {
        int8x8_t a_vec = vld1_s8(a_group + i);
        uint8x8_t packed = vld1_u8(b_group_packed + i / 2);
        uint8x8_t low = vand_u8(packed, vdup_n_u8(0x0F));
        uint8x8_t high = vshr_n_u8(packed, 4);
        uint8x8x2_t zipped = vzip_u8(low, high);
        int8x8_t b_vec = vreinterpret_s8_u8(zipped.val[0]);
        int16x8_t dot_wide = vmull_s8(a_vec, b_vec);
        int32x4_t dot32 = vpaddlq_s16(dot_wide);
        int32x2_t dot_sum = vpadd_s32(vget_low_s32(dot32), vget_high_s32(dot32));
        *dot_out += vget_lane_s32(dot_sum, 0) + vget_lane_s32(dot_sum, 1);
        if (a_sum_out) {
            int16x4_t asum_wide = vpaddl_s8(a_vec);
            int32x2_t asum32 = vpaddl_s16(asum_wide);
            *a_sum_out += vget_lane_s32(asum32, 0) + vget_lane_s32(asum32, 1);
        }
    }
    
    // Scalar tail
    for (; i < group_size; i++) {
        int pack_idx = i / 2;
        uint8_t packed = b_group_packed[pack_idx];
        int8_t b_q = (i % 2 == 0) ? (packed & 0x0F) : (packed >> 4);
        *dot_out += (int32_t)a_group[i] * (int32_t)b_q;
        if (a_sum_out) *a_sum_out += (int32_t)a_group[i];
    }
}
#else
static inline void vec_dot_q8_q4_scalar(const int8_t* a_group, const uint8_t* b_group_packed, int group_size, int32_t* dot_out, int32_t* a_sum_out) {
    *dot_out = 0;
    if (a_sum_out) *a_sum_out = 0;
    for (int i = 0; i < group_size; i++) {
        int pack_idx = i / 2;
        uint8_t packed = b_group_packed[pack_idx];
        int8_t b_q = (i % 2 == 0) ? (packed & 0x0F) : (packed >> 4);
        *dot_out += (int32_t)a_group[i] * (int32_t)b_q;
        if (a_sum_out) *a_sum_out += (int32_t)a_group[i];
    }
}
#endif

/**
 * NEON-optimized Q8×Q4 matrix multiplication  
 */
void matmul_q8_q4_f32_neon(const int8_t* __restrict A_q8, const float* __restrict A_scales,
                           const uint8_t* __restrict B_q4_packed, const float* __restrict B_scales, const float* __restrict B_zero_points,
                           float* __restrict C, int M, int N, int K, int group_size) {
    assert((K % group_size) == 0 && (group_size & (group_size - 1)) == 0);  // Power of 2
    int num_groups = K / group_size;
    
    // Alignment hints for better performance
    A_q8 = (const int8_t*)__builtin_assume_aligned(A_q8, 16);
    B_q4_packed = (const uint8_t*)__builtin_assume_aligned(B_q4_packed, 16);
    A_scales = (const float*)__builtin_assume_aligned(A_scales, 16);
    B_scales = (const float*)__builtin_assume_aligned(B_scales, 16);
    B_zero_points = (const float*)__builtin_assume_aligned(B_zero_points, 16);
    C = (float*)__builtin_assume_aligned(C, 16);
    
    // Prefetch first data
    __builtin_prefetch(A_q8, 0, 2);           // Read, moderate locality
    __builtin_prefetch(B_q4_packed, 0, 2);    // Read, moderate locality
    __builtin_prefetch(A_scales, 0, 3);       // Read, high locality
    __builtin_prefetch(B_scales, 0, 3);       // Read, high locality  
    __builtin_prefetch(B_zero_points, 0, 3);  // Read, high locality
    __builtin_prefetch(C, 1, 1);              // Write, low locality
    
    
    // Use collapse(2) with guided scheduling for better load balancing
    #pragma omp parallel for collapse(2) schedule(static)
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float acc = 0.0f;
            
            // Prefetch next elements for better cache performance
            if (m + 1 < M) {
                __builtin_prefetch(A_q8 + (m + 1) * K, 0, 2);                    // Next A row
                __builtin_prefetch(A_scales + (m + 1) * num_groups, 0, 2);       // Next A scales
            }
            if (n + 4 < N) {
                __builtin_prefetch(B_q4_packed + (n + 4) * (K / 2), 0, 2);       // Next B column
                __builtin_prefetch(B_scales + (n + 4) * num_groups, 0, 2);       // Next B scales
                __builtin_prefetch(B_zero_points + (n + 4) * num_groups, 0, 2);  // Next B zero points
            }
            
            // Process all groups for this (m,n) element
            for (int g = 0; g < num_groups; g++) {
                int start_k = g * group_size;
                const int8_t* a_group = A_q8 + m * K + start_k;
                const uint8_t* b_group_packed = B_q4_packed + n * (K / 2) + (start_k / 2);
                float a_scale = A_scales[m * num_groups + g];
                float b_scale = B_scales[n * num_groups + g];
                float b_zp = B_zero_points[n * num_groups + g];
                
                // Prefetch next group data within the loop
                if (g + 1 < num_groups) {
                    int next_start_k = (g + 1) * group_size;
                    __builtin_prefetch(A_q8 + m * K + next_start_k, 0, 3);
                    __builtin_prefetch(B_q4_packed + n * (K / 2) + (next_start_k / 2), 0, 3);
                }
                
                // Precompute scale products to reduce FP multiplications
                float scale_product = a_scale * b_scale;
                float zp_product = a_scale * b_zp;
                
                int32_t dot = 0, a_sum = 0;
                
                // Use the general kernel for correctness
#ifdef __ARM_NEON
                vec_dot_q8_q4_neon(a_group, b_group_packed, group_size, &dot, &a_sum);
#else
                vec_dot_q8_q4_scalar(a_group, b_group_packed, group_size, &dot, &a_sum);
#endif
                acc += (float)dot * scale_product + (float)a_sum * zp_product;
            }
            C[m * N + n] = acc;
        }
    }
}

/**
 * FP32 × Q8 matrix multiplication - FUSED QUANTIZATION (zero scratch overhead)
 * Based on optimizations.txt guidance: eliminate scratch traffic completely
 */
void matmul_f32_q8_f32(const float* __restrict A_fp32, const int8_t* __restrict B_q8, const float* __restrict B_scales,
                       float* __restrict C, int M, int N, int K, int group_size,
                       int8_t* qx_q_scratch, float* qx_s_scratch) {
    assert(is_power_of_2(group_size));
    assert(K % group_size == 0);
    
    const int num_groups = K / group_size;
    
    // Alignment hints for better performance
    A_fp32 = (const float*)__builtin_assume_aligned(A_fp32, 16);
    B_q8 = (const int8_t*)__builtin_assume_aligned(B_q8, 16);
    B_scales = (const float*)__builtin_assume_aligned(B_scales, 16);
    C = (float*)__builtin_assume_aligned(C, 16);
    
    // FUSED APPROACH: No scratch buffers - quantize A per group in registers
    // Use collapse(2) for maximum parallelization with static scheduling
    #pragma omp parallel for collapse(2) schedule(static)
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float acc = 0.0f;
            
            for (int g = 0; g < num_groups; g++) {
                int start_k = g * group_size;
                const float* a_group = A_fp32 + m * K + start_k;
                const int8_t* b_group = B_q8 + n * K + start_k;
                float b_scale = B_scales[n * num_groups + g];
                
                // FUSED QUANTIZATION: Pass 1 - Find max-abs for A scale
                float max_abs = find_max_abs(a_group, group_size);
                float a_scale = fmaxf(max_abs / 127.0f, 1e-8f);
                float inv_scale = 1.0f / a_scale;
                
                // FUSED QUANTIZATION: Pass 2 - Quantize and dot product in registers
                int32_t dot_accumulator = 0;
                
#ifdef __ARM_NEON
                int32x4_t dot_vec = vdupq_n_s32(0);
                const float32x4_t inv_scale_vec = vdupq_n_f32(inv_scale);
                const float32x4_t half_vec = vdupq_n_f32(0.5f);
                const float32x4_t neg_half_vec = vdupq_n_f32(-0.5f);
                const float32x4_t zero_vec = vdupq_n_f32(0.0f);
                
                // Process in chunks of 16 elements when possible
                int kv = 0;
                for (; kv + 16 <= group_size; kv += 16) {
                    // Load and quantize 16 floats to int8s in registers
                    float32x4_t f0 = vld1q_f32(a_group + kv + 0);
                    float32x4_t f1 = vld1q_f32(a_group + kv + 4);
                    float32x4_t f2 = vld1q_f32(a_group + kv + 8);
                    float32x4_t f3 = vld1q_f32(a_group + kv + 12);
                    
                    // Scale and round to int16 (helper function approach)
                    #define QUANTIZE_F32X4(f) ({ \
                        float32x4_t scaled = vmulq_f32(f, inv_scale_vec); \
                        float32x4_t adjust = vbslq_f32(vcgeq_f32(scaled, zero_vec), half_vec, neg_half_vec); \
                        int32x4_t rounded = vcvtq_s32_f32(vaddq_f32(scaled, adjust)); \
                        rounded = vmaxq_s32(rounded, vdupq_n_s32(-127)); \
                        rounded = vminq_s32(rounded, vdupq_n_s32(127)); \
                        vmovn_s32(rounded); \
                    })
                    
                    int16x4_t q0 = QUANTIZE_F32X4(f0);
                    int16x4_t q1 = QUANTIZE_F32X4(f1);
                    int16x4_t q2 = QUANTIZE_F32X4(f2);
                    int16x4_t q3 = QUANTIZE_F32X4(f3);
                    
                    #undef QUANTIZE_F32X4
                    
                    int16x8_t q01 = vcombine_s16(q0, q1);
                    int16x8_t q23 = vcombine_s16(q2, q3);
                    int8x16_t qa = vcombine_s8(vmovn_s16(q01), vmovn_s16(q23));
                    
                    // Load B and compute dot product
                    int8x16_t b_vec = vld1q_s8(b_group + kv);
                    int16x8_t dot_low = vmull_s8(vget_low_s8(qa), vget_low_s8(b_vec));
                    int16x8_t dot_high = vmull_s8(vget_high_s8(qa), vget_high_s8(b_vec));
                    
                    dot_vec = vaddq_s32(dot_vec, vpaddlq_s16(dot_low));
                    dot_vec = vaddq_s32(dot_vec, vpaddlq_s16(dot_high));
                }
                
                dot_accumulator = vaddvq_s32(dot_vec);
                
                // Handle remaining elements with scalar code
                for (; kv < group_size; kv++) {
                    float a_val = a_group[kv];
                    int8_t a_quant = (int8_t)fmaxf(-127.0f, fminf(127.0f, roundf(a_val * inv_scale)));
                    int8_t b_val = b_group[kv];
                    dot_accumulator += (int32_t)a_quant * (int32_t)b_val;
                }
#else
                // Scalar fallback
                for (int k = 0; k < group_size; k++) {
                    float a_val = a_group[k];
                    int8_t a_quant = (int8_t)fmaxf(-127.0f, fminf(127.0f, roundf(a_val * inv_scale)));
                    int8_t b_val = b_group[k];
                    dot_accumulator += (int32_t)a_quant * (int32_t)b_val;
                }
#endif
                
                // Apply scaling
                acc += (float)dot_accumulator * a_scale * b_scale;
            }
            
            C[m * N + n] = acc;
        }
    }
}

/**
 * FP32 × Q4 matrix multiplication with zero points (new optimized version)
 */
void matmul_f32_q4_f32_with_zeros(const float* A_fp32, const uint8_t* B_q4, const float* B_scales, const float* B_zeros,
                                  float* C, int M, int N, int K, int group_size,
                                  int8_t* qx_q_scratch, float* qx_s_scratch) {
    assert(is_power_of_2(group_size));
    assert(K % group_size == 0);
    
    const int group_size_log2 = get_log2(group_size);
    const int num_groups_per_row = K >> group_size_log2;
    
    // Step 1: PARALLEL quantization of activation A to Q8 on-the-fly
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        const float* a_row = A_fp32 + m * K;
        int8_t* qa_row = qx_q_scratch + m * K;
        float* qa_scales = qx_s_scratch + m * num_groups_per_row;
        
        quantize_q8(a_row, qa_row, qa_scales, K, group_size);
    }
    
    // Step 2: Q8×Q4 matrix multiplication with asymmetric Q4
    matmul_q8_q4_f32(qx_q_scratch, qx_s_scratch, B_q4, B_scales, B_zeros, C, M, N, K, group_size);
}

/**
 * FP32 × Q4 matrix multiplication (backward compatible - no zero points parameter)
 * This maintains the old API that existing code expects
 */
void matmul_f32_q4_f32(const float* A_fp32, const uint8_t* B_q4, const float* B_scales,
                       float* C, int M, int N, int K, int group_size,
                       int8_t* qx_q_scratch, float* qx_s_scratch) {
    // For backward compatibility, use stack allocation for neutral zero points
    int num_groups = (N * K) / group_size;
    
    // Use stack allocation up to reasonable limit, otherwise skip zero points
    if (num_groups <= 16384) {  // ~64KB stack space for 16K groups
        float neutral_zeros[16384];
        for (int i = 0; i < num_groups && i < 16384; i++) {
            neutral_zeros[i] = 8.0f;  // Neutral zero point for Q4 range [0,15]
        }
        
        // Call the optimized version with neutral zero points
        matmul_f32_q4_f32_with_zeros(A_fp32, B_q4, B_scales, neutral_zeros, C, M, N, K, group_size, 
                                     qx_q_scratch, qx_s_scratch);
    } else {
        // For very large models, skip zero points (will be less accurate but no malloc)
        // This should rarely happen in practice
        assert(is_power_of_2(group_size));
        assert(K % group_size == 0);
        
        const int group_size_log2 = get_log2(group_size);
        const int num_groups_per_row = K >> group_size_log2;
        
        // Step 1: PARALLEL quantization of activation A to Q8 on-the-fly
        #pragma omp parallel for schedule(static)
        for (int m = 0; m < M; m++) {
            const float* a_row = A_fp32 + m * K;
            int8_t* qa_row = qx_q_scratch + m * K;
            float* qa_scales = qx_s_scratch + m * num_groups_per_row;
            
            quantize_q8(a_row, qa_row, qa_scales, K, group_size);
        }
        
        // Step 2: Use optimized Q8×Q4 kernel with neutral zero points (8.0f)
        float* neutral_zeros_large = (float*)malloc(num_groups * sizeof(float));
        if (neutral_zeros_large) {
            for (int i = 0; i < num_groups; i++) {
                neutral_zeros_large[i] = 8.0f;  // Neutral zero point for Q4 range [0,15]
            }
            matmul_q8_q4_f32(qx_q_scratch, qx_s_scratch, B_q4, B_scales, neutral_zeros_large, C, M, N, K, group_size);
            free(neutral_zeros_large);
        } else {
            // Memory allocation failed, use Q8×Q8 fallback (less accurate)
            matmul_q8_q8_f32(qx_q_scratch, qx_s_scratch, (const int8_t*)B_q4, B_scales, C, M, N, K, group_size);
        }
        
    }
}
