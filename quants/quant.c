/**
 * Quantization Library - Optimized Q8/Q4 functions with NEON acceleration
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <sys/time.h>
#include <omp.h>
#include <string.h>
#include "quant.h"

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

// ================================================================
// HELPER FUNCTIONS
// ================================================================

static inline bool is_power_of_2(size_t x) {
  return x > 0 && (x & (x - 1)) == 0;
}

static inline int get_log2(size_t x) {
  return __builtin_ctzl(x);
}

// Horizontal max(|x|) over a group (NEON or scalar)
// Algorithm: Find the maximum absolute value in an array using SIMD vectorization
static inline float hmax_abs_f32(const float* x, int n) {
  float m = 0.f;
#ifdef __ARM_NEON
  int i = 0;
  float32x4_t vmax4 = vdupq_n_f32(0.f);                 // Initialize 4-lane vector with zeros
  for (; i + 4 <= n; i += 4) {
    float32x4_t v = vld1q_f32(x + i);                   // Load 4 consecutive floats into vector
    vmax4 = vmaxq_f32(vmax4, vabsq_f32(v));             // Element-wise max(current_max, |v|) across 4 lanes
  }
  float32x2_t vmax2 = vmax_f32(vget_low_f32(vmax4), vget_high_f32(vmax4)); // Reduce 4 lanes to 2
  m = fmaxf(vget_lane_f32(vmax2, 0), vget_lane_f32(vmax2, 1));             // Final reduction to scalar
  for (; i < n; ++i) m = fmaxf(m, fabsf(x[i]));         // Handle remaining elements
#else
  for (int i = 0; i < n; ++i) m = fmaxf(m, fabsf(x[i]));// Scalar fallback
#endif
  return m;
}

#ifdef __ARM_NEON
static inline int16x4_t q_f32x4_to_s8_s16lane(float32x4_t f, float32x4_t inv);
#endif

#ifdef __ARM_NEON
// Quantize 4 floats -> 4 int16 lanes representing clamped int8
// Algorithm: Scale floats by inverse, round to nearest, clamp to [-127,127], narrow to int16
static inline int16x4_t q_f32x4_to_s8_s16lane(float32x4_t f, float32x4_t inv) {
  float32x4_t x = vmulq_f32(f, inv);                         // Scale: x = f * (1/scale)
#if defined(__aarch64__) || defined(__ARM_FEATURE_JCVT)
  int32x4_t q32 = vcvtnq_s32_f32(x);                         // Round-to-nearest (AArch64 native)
#else
  float32x4_t half = vdupq_n_f32(0.5f);                      // Portable rounding: add ±0.5 based on sign
  float32x4_t nhalf = vdupq_n_f32(-0.5f);
  float32x4_t adj = vbslq_f32(vcgeq_f32(x, vdupq_n_f32(0.0f)), half, nhalf); // Select +0.5 or -0.5
  int32x4_t q32 = vcvtq_s32_f32(vaddq_f32(x, adj));          // Convert with rounding adjustment
#endif
  q32 = vmaxq_s32(q32, vdupq_n_s32(-127));                   // Clamp to int8 range (symmetric)
  q32 = vminq_s32(q32, vdupq_n_s32( 127));
  return vmovn_s32(q32);                                     // Narrow 32->16 (prepare for packing)
}
#endif

// Dot product of s8·s8 over n (NEON with dotprod if available)
// Algorithm: Vectorized dot product with optimal instruction selection based on CPU features
static inline int32_t dot_s8s8(const int8_t* a, const int8_t* b, int n) {
  int32_t s = 0;
#ifdef __ARM_NEON
  int i = 0;
  int32x4_t acc = vdupq_n_s32(0);                       // 4-lane 32-bit accumulator
  for (; i + 16 <= n; i += 16) {
    int8x16_t va = vld1q_s8(a + i);                     // Load 16 int8 values from array a
    int8x16_t vb = vld1q_s8(b + i);                     // Load 16 int8 values from array b
#if defined(__ARM_FEATURE_DOTPROD)
    acc = vdotq_s32(acc, va, vb);                       // ARMv8.4-A: 4x dot products in one instruction
#else
    int16x8_t lo = vmull_s8(vget_low_s8(va),  vget_low_s8(vb));  // Multiply low 8 elements -> int16
    int16x8_t hi = vmull_s8(vget_high_s8(va), vget_high_s8(vb)); // Multiply high 8 elements -> int16
    acc = vaddq_s32(acc, vpaddlq_s16(lo));              // Pairwise add 8xint16 -> 4xint32, accumulate
    acc = vaddq_s32(acc, vpaddlq_s16(hi));              // Pairwise add 8xint16 -> 4xint32, accumulate
#endif
  }
  s += vaddvq_s32(acc);                                 // Horizontal sum of 4 int32 lanes
  for (; i < n; ++i) s += (int32_t)a[i] * (int32_t)b[i]; // Handle remaining elements
#else
  for (int i = 0; i < n; ++i) s += (int32_t)a[i] * (int32_t)b[i]; // Scalar fallback
#endif
  return s;
}

// A (s8) · B4 (packed q4) and sum(A) over n (for affine Q4 dequant)
// Algorithm: Compute dot product with 4-bit values while also summing A elements (needed for zero-point correction)
static inline void dot_s8q4_acc(const int8_t* a, const uint8_t* b4, int n,
    int32_t* dot, int32_t* sum_a) {
  int32_t d = 0, sa = 0;
#ifdef __ARM_NEON
  int i = 0;
  int32x4_t accd = vdupq_n_s32(0), acca = vdupq_n_s32(0); // Separate accumulators for dot and sum
  for (; i + 8 <= n; i += 8) {
    int8x8_t va = vld1_s8(a + i);                       // Load 8 int8 values from A
    uint8x8_t pb = vld1_u8(b4 + (i >> 1));              // Load 4 packed bytes (8 nibbles) from B4
    uint8x8_t lo = vand_u8(pb, vdup_n_u8(0x0F));        // Extract low nibbles (0-3 bits)
    uint8x8_t hi = vshr_n_u8(pb, 4);                    // Extract high nibbles (4-7 bits)

    uint8x8x2_t z = vzip_u8(lo, hi);                    // Interleave nibbles: [q0,q1,q2,q3,q4,q5,q6,q7]
    int8x8_t vb = vreinterpret_s8_u8(z.val[0]);         // Reinterpret as signed (0-15 range)

    int16x8_t mul = vmull_s8(va, vb);                   // 8x multiply: va[i] * vb[i] -> int16
    accd = vaddq_s32(accd, vpaddlq_s16(mul));           // Accumulate dot products

    int16x8_t aw = vmovl_s8(va);                        // Widen A values to int16 for sum
    acca = vaddq_s32(acca, vpaddlq_s16(aw));            // Accumulate sum of A values
  }
  d += vaddvq_s32(accd);                               // Horizontal reduction: dot sum
  sa += vaddvq_s32(acca);                              // Horizontal reduction: A sum
  for (; i < n; ++i) {                                 // Handle remaining elements
    uint8_t packed = b4[i >> 1];                       // Get packed byte
    int8_t q = (i & 1) ? (int8_t)((packed >> 4) & 0x0F) : (int8_t)(packed & 0x0F); // Extract nibble
    d  += (int32_t)a[i] * (int32_t)q;
    sa += (int32_t)a[i];
  }
#else
  for (int i = 0; i < n; ++i) {                        // Scalar fallback
    uint8_t packed = b4[i >> 1];
    int8_t q = (i & 1) ? (int8_t)((packed >> 4) & 0x0F) : (int8_t)(packed & 0x0F);
    d  += (int32_t)a[i] * (int32_t)q;
    sa += (int32_t)a[i];
  }
#endif
  *dot = d; if (sum_a) *sum_a = sa;
}

// ================================================================
// QUANTIZATION/DEQUANTIZATION FUNCTIONS
// ================================================================

// Symmetric Q8 quantize one group; returns scale
// Algorithm: Find max absolute value, compute scale, quantize with SIMD vectorization
static inline float q8_group_quant(const float* x, int8_t* q, int gs) {
  const float QMAX = 127.f;                                  // Symmetric range: [-127, 127]
  float maxa = hmax_abs_f32(x, gs);                          // Find max |x| in group
  float s = fmaxf(maxa / QMAX, 1e-8f);                       // Scale = max/127, avoid div-by-zero
  float inv = 1.f / s;                                       // Precompute inverse for efficiency
#ifdef __ARM_NEON
  int i = 0;
  float32x4_t invv = vdupq_n_f32(inv);                       // Broadcast inverse to all lanes
  for (; i + 16 <= gs; i += 16) {                            // Process 16 floats at once
    float32x4_t f0 = vld1q_f32(x + i +  0);                 // Load 4 floats (elements 0-3)
    float32x4_t f1 = vld1q_f32(x + i +  4);                 // Load 4 floats (elements 4-7)
    float32x4_t f2 = vld1q_f32(x + i +  8);                 // Load 4 floats (elements 8-11)
    float32x4_t f3 = vld1q_f32(x + i + 12);                 // Load 4 floats (elements 12-15)
    int16x4_t q0 = q_f32x4_to_s8_s16lane(f0, invv);         // Quantize and narrow to int16
    int16x4_t q1 = q_f32x4_to_s8_s16lane(f1, invv);
    int16x4_t q2 = q_f32x4_to_s8_s16lane(f2, invv);
    int16x4_t q3 = q_f32x4_to_s8_s16lane(f3, invv);
    int8x16_t qs = vcombine_s8(vmovn_s16(vcombine_s16(q0, q1)), // Pack 16 int16 -> 16 int8
        vmovn_s16(vcombine_s16(q2, q3)));
    vst1q_s8(q + i, qs);                                    // Store 16 quantized values
  }
  for (; i < gs; ++i) {                                      // Handle remaining elements
    int r = (int)lrintf(x[i] * inv);                         // Round to nearest integer
    q[i] = (int8_t)(r < -127 ? -127 : r > 127 ? 127 : r);    // Clamp to [-127, 127]
  }
#else
  for (int i = 0; i < gs; ++i) {                             // Scalar fallback
    int r = (int)lrintf(x[i] * inv);
    q[i] = (int8_t)(r < -127 ? -127 : r > 127 ? 127 : r);
  }
#endif
  return s;
}

// Dequant one Q8 group: y = q*s
static inline void q8_group_dequant(const int8_t* q, float s, float* y, int gs) {
  for (int i = 0; i < gs; ++i) y[i] = (float)q[i] * s;
}

// Affine Q4 quantize one group (stores scale+min as zp)
// Algorithm: Find min/max, compute affine mapping to [0,15], pack 2 values per byte
static inline void q4_group_quant(const float* x, uint8_t* q, float* s, float* zp, int gs) {
  float mn = x[0], mx = x[0];                               // Initialize min/max with first element
  int i = 0;
#ifdef __ARM_NEON
  float32x4_t vmin4 = vdupq_n_f32(mn), vmax4 = vdupq_n_f32(mx); // 4-lane min/max accumulators
  for (; i + 4 <= gs; i += 4) {
    float32x4_t v = vld1q_f32(x + i);                      // Load 4 floats
    vmin4 = vminq_f32(vmin4, v);                           // Element-wise minimum
    vmax4 = vmaxq_f32(vmax4, v);                           // Element-wise maximum
  }
  float32x2_t vmin2 = vmin_f32(vget_low_f32(vmin4), vget_high_f32(vmin4)); // Reduce 4->2 lanes
  float32x2_t vmax2 = vmax_f32(vget_low_f32(vmax4), vget_high_f32(vmax4));
  mn = fminf(vget_lane_f32(vmin2, 0), vget_lane_f32(vmin2, 1)); // Final reduction to scalars
  mx = fmaxf(vget_lane_f32(vmax2, 0), vget_lane_f32(vmax2, 1));
  for (; i < gs; ++i) { float v = x[i]; mn = fminf(mn, v); mx = fmaxf(mx, v); } // Handle remaining
#else
  for (int i = 1; i < gs; ++i) { float v = x[i]; if (v < mn) mn = v; if (v > mx) mx = v; } // Scalar fallback
#endif
  float sc = (mx - mn) / 15.f;                             // Scale for range [0,15]
  if (sc < 1e-8f) sc = 1e-8f;                              // Avoid division by zero
  float inv = 1.f / sc;                                    // Precompute inverse
  *s = sc; *zp = mn;                                       // Store scale and zero-point (min)

  i = 0;
  for (; i + 2 <= gs; i += 2) {                            // Pack 2 values per byte
    int q0 = (int)lrintf((x[i + 0] - mn) * inv);           // Quantize: (value - min) / scale
    int q1 = (int)lrintf((x[i + 1] - mn) * inv);
    if (q0 < 0) q0 = 0; else if (q0 > 15) q0 = 15;        // Clamp to [0,15]
    if (q1 < 0) q1 = 0; else if (q1 > 15) q1 = 15;
    q[(i >> 1)] = (uint8_t)((q1 << 4) | (q0 & 0x0F));     // Pack: high nibble | low nibble
  }
  if (i < gs) {                                            // Handle odd tail element
    int q0 = (int)lrintf((x[i] - mn) * inv);
    if (q0 < 0) q0 = 0; else if (q0 > 15) q0 = 15;
    q[(i >> 1)] = (uint8_t)(q0 & 0x0F);                   // Store in low nibble
  }
}

// Affine Q4 dequant one group: y = q*s + zp
static inline void q4_group_dequant(const uint8_t* q, float s, float zp, float* y, int gs) {
  for (int i = 0; i < gs; ++i) {
    uint8_t b = q[i >> 1];
    int qq = (i & 1) ? ((b >> 4) & 0x0F) : (b & 0x0F);
    y[i] = (float)qq * s + zp;
  }
}

void quantize_q8(const float* x, int8_t* q, float* s, int n, int gs) {
  assert(is_power_of_2(gs) && n % gs == 0);
  const int ng = n >> get_log2(gs);
  for (int g = 0; g < ng; ++g) s[g] = q8_group_quant(x + g*gs, q + g*gs, gs);
}

void dequantize_q8(const int8_t* q, const float* s, float* y, int n, int gs) {
  assert(is_power_of_2(gs) && n % gs == 0);
  const int ng = n >> get_log2(gs);
  for (int g = 0; g < ng; ++g) q8_group_dequant(q + g*gs, s[g], y + g*gs, gs);
}

void quantize_q4(const float* x, size_t rows, size_t cols, size_t gs,
    float* scales, float* zps, uint8_t* q) {
  const size_t n = rows * cols;
  assert(is_power_of_2(gs) && (n % gs == 0));
  const size_t ng = n >> get_log2(gs);
  for (size_t g = 0; g < ng; ++g) q4_group_quant(x + g*gs, q + ((g*gs)>>1), &scales[g], &zps[g], (int)gs);
}

void dequantize_q4(const uint8_t* q, const float* s, const float* zp,
    float* y, size_t rows, size_t cols, size_t gs) {
  const size_t n = rows * cols;
  assert(is_power_of_2(gs) && (n % gs == 0));
  const size_t ng = n >> get_log2(gs);
  for (size_t g = 0; g < ng; ++g) q4_group_dequant(q + ((g*gs)>>1), s[g], zp[g], y + g*gs, (int)gs);
}

// ================================================================
// MATRIX MULTIPLICATION FUNCTIONS
// ================================================================

void matmul_q8_q8_f32(const int8_t* A, const float* As,
    const int8_t* B, const float* Bs,
    float* C, int M, int N, int K, int gs) {
  assert(is_power_of_2(gs) && (K % gs == 0));
  const int ng = K / gs;
#pragma omp parallel for collapse(2) schedule(static)
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float acc = 0.f;
      const int8_t* arow = A + m*K;
      const int8_t* bcol = B + n*K;
      const float* as   = As + m*ng;
      const float* bs   = Bs + n*ng;
      for (int g = 0; g < ng; ++g) {
        int32_t d = dot_s8s8(arow + g*gs, bcol + g*gs, gs);
        acc += (float)d * (as[g] * bs[g]);
      }
      C[m*N + n] = acc;
    }
  }
}

void matmul_q8_q4_f32(const int8_t* A, const float* As,
    const uint8_t* B4, const float* Bs, const float* Bzp,
    float* C, int M, int N, int K, size_t gs) {
  assert(is_power_of_2(gs) && (K % gs == 0));
  const int ng = K / gs;
#pragma omp parallel for collapse(2) schedule(static)
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float acc = 0.f;
      const int8_t* arow = A + m*K;
      const uint8_t* bcol4 = B4 + n*(K>>1);
      const float* as  = As + m*ng;
      const float* bs  = Bs + n*ng;
      const float* bzp = Bzp ? (Bzp + n*ng) : NULL;
      for (int g = 0; g < ng; ++g) {
        int32_t d, sa;
        dot_s8q4_acc(arow + g*gs, bcol4 + ((g*gs)>>1), gs, &d, &sa);
        float w = as[g] * bs[g];
        acc += (float)d * w;
        if (bzp) acc += (float)sa * (as[g] * bzp[g]);
      }
      C[m*N + n] = acc;
    }
  }
}

void matmul_f32_q8_f32(const float* A_f32, const int8_t* B, const float* Bs,
    float* C, int M, int N, int K, int gs,
    int8_t* A_q_scratch, float* A_s_scratch) {
  assert(is_power_of_2(gs) && (K % gs == 0));
  const int ng = K / gs;

  const int REUSE_THRESHOLD_N = 4;

  if (A_q_scratch && A_s_scratch && N >= REUSE_THRESHOLD_N) {
    // Pre-quantize A then call q8×q8
#pragma omp parallel for schedule(static)
    for (int m = 0; m < M; ++m) {
      for (int g = 0; g < ng; ++g) {
        A_s_scratch[m*ng + g] =
          q8_group_quant(A_f32 + m*K + g*gs, A_q_scratch + m*K + g*gs, gs);
      }
    }
    matmul_q8_q8_f32(A_q_scratch, A_s_scratch, B, Bs, C, M, N, K, gs);
  } else {
    // Fused quantize-on-the-fly
#pragma omp parallel for collapse(2) schedule(static)
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        float acc = 0.f;
        const int8_t* bcol = B + n*K;
        const float*  bs   = Bs + n*ng;
        for (int g = 0; g < ng; ++g) {
          const float* ag = A_f32 + m*K + g*gs;
          int8_t aq_buf[256];
          assert(gs <= (int)sizeof(aq_buf));
          float as = q8_group_quant(ag, aq_buf, gs);
          int32_t d = dot_s8s8(aq_buf, bcol + g*gs, gs);
          acc += (float)d * (as * bs[g]);
        }
        C[m*N + n] = acc;
      }
    }
  }
}

void matmul_f32_q4_f32(const float* A_fp32, const uint8_t* B_q4, const float* B_scales,
    float* C, int M, int N, int K, int group_size,
    int8_t* qx_q_scratch, float* qx_s_scratch) {
  int num_groups = (N * K) / group_size;

  if (num_groups <= 16384) {
    float neutral_zeros[16384];
    for (int i = 0; i < num_groups; i++) {
      neutral_zeros[i] = 8.0f;
    }
    matmul_f32_q4_f32_with_zeros(A_fp32, B_q4, B_scales, neutral_zeros, C, M, N, K, group_size, qx_q_scratch, qx_s_scratch);
  } else {
    float* neutral_zeros_large = (float*)calloc(num_groups, sizeof(float));
    if (neutral_zeros_large) {
      for (int i = 0; i < num_groups; i++) {
        neutral_zeros_large[i] = 8.0f;
      }
      matmul_f32_q4_f32_with_zeros(A_fp32, B_q4, B_scales, neutral_zeros_large, C, M, N, K, group_size, qx_q_scratch, qx_s_scratch);
      free(neutral_zeros_large);
    } else {
      matmul_q8_q8_f32(qx_q_scratch, qx_s_scratch, (const int8_t*)B_q4, B_scales, C, M, N, K, group_size);
    }
  }
}

void matmul_f32_q4_f32_with_zeros(const float* A_f32, const uint8_t* B4,
    const float* Bs, const float* Bzp,
    float* C, int M, int N, int K, int gs,
    int8_t* A_q_scratch, float* A_s_scratch) {
  assert(is_power_of_2(gs) && (K % gs == 0));
  const int ng = K / gs;

  const int REUSE_THRESHOLD_N = 4;

  if (A_q_scratch && A_s_scratch && N >= REUSE_THRESHOLD_N) {
#pragma omp parallel for schedule(static)
    for (int m = 0; m < M; ++m) {
      for (int g = 0; g < ng; ++g) {
        A_s_scratch[m*ng + g] =
          q8_group_quant(A_f32 + m*K + g*gs, A_q_scratch + m*K + g*gs, gs);
      }
    }
    matmul_q8_q4_f32(A_q_scratch, A_s_scratch, B4, Bs, Bzp, C, M, N, K, gs);
  } else {
#pragma omp parallel for collapse(2) schedule(static)
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        float acc = 0.f;
        const uint8_t* bcol4 = B4 + n*(K>>1);
        const float*    bs   = Bs + n*ng;
        const float*    bzp  = Bzp ? (Bzp + n*ng) : NULL;
        for (int g = 0; g < ng; ++g) {
          const float* ag = A_f32 + m*K + g*gs;
          int8_t aq_buf[256];
          assert(gs <= (int)sizeof(aq_buf));
          float as = q8_group_quant(ag, aq_buf, gs);
          int32_t d, sa;
          dot_s8q4_acc(aq_buf, bcol4 + ((g*gs)>>1), gs, &d, &sa);
          float w = as * bs[g];
          acc += (float)d * w;
          if (bzp) acc += (float)sa * (as * bzp[g]);
        }
        C[m*N + n] = acc;
      }
    }
  }
}
