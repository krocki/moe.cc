/**
 * Quantization Library - Optimized Q8/Q4 functions with NEON acceleration
 * - Fixed-size micro-kernels (K-chunk = 128) for Q8xQ8 and Q8xQ4
 * - Correct Q4 nibble order (low = q0, high = q1) matching quantize_q4()
 * - Fallbacks cover all shapes/dimensions
 * - Fused on-the-fly quantization remains available
 * - Uses #if defined(__ARM_FEATURE_DOTPROD) inside kernels
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <omp.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

// ================================================================
// SMALL UTILITIES
// ================================================================

static inline bool is_power_of_2(size_t x) { return x > 0 && (x & (x - 1)) == 0; }
static inline int  get_log2(size_t x)      { return __builtin_ctzl(x); }

// ================================================================
// NEON HELPERS
// ================================================================

// Horizontal max(|x|) over n floats.
static inline float hmax_abs_f32(const float* x, int n) {
  float m = 0.f;
#ifdef __ARM_NEON
  int i = 0;
  float32x4_t vmax4 = vdupq_n_f32(0.f);               // [0,0,0,0]
  for (; i + 4 <= n; i += 4) {
    float32x4_t v = vld1q_f32(x + i);                 // load 4 f32
    vmax4 = vmaxq_f32(vmax4, vabsq_f32(v));           // max(|v|)
  }
#if defined(__aarch64__)
  m = vmaxvq_f32(vmax4);                              // A64 horizontal max
#else
  float32x2_t t = vmax_f32(vget_low_f32(vmax4), vget_high_f32(vmax4));
  m = fmaxf(vget_lane_f32(t,0), vget_lane_f32(t,1));
#endif
  for (; i < n; ++i) m = fmaxf(m, fabsf(x[i]));
#else
  for (int i = 0; i < n; ++i) m = fmaxf(m, fabsf(x[i]));
#endif
  return m;
}

#ifdef __ARM_NEON
// Quantize 4 floats -> 4x int16 lanes (clamped to int8 domain)
static inline int16x4_t q_f32x4_to_s8_s16lane(float32x4_t f, float32x4_t inv) {
  float32x4_t x = vmulq_f32(f, inv);                  // scale: f * (1/scale)
#if defined(__aarch64__) || defined(__ARM_FEATURE_JCVT)
  int32x4_t q32 = vcvtnq_s32_f32(x);                  // round-to-nearest int
#else
  // Portable rounding: add +/-0.5 based on sign then truncate
  float32x4_t half = vdupq_n_f32(0.5f), nhalf = vdupq_n_f32(-0.5f);
  float32x4_t adj  = vbslq_f32(vcgeq_f32(x, vdupq_n_f32(0.f)), half, nhalf);
  int32x4_t q32 = vcvtq_s32_f32(vaddq_f32(x, adj));
#endif
  q32 = vmaxq_s32(q32, vdupq_n_s32(-127));            // clamp min
  q32 = vminq_s32(q32, vdupq_n_s32( 127));            // clamp max
  return vmovn_s32(q32);                              // narrow to s16 (4 lanes)
}
#endif

// s8·s8 dot product over n
int32_t dot_s8s8(const int8_t* a, const int8_t* b, int n) {
  int32_t s = 0;
#ifdef __ARM_NEON
  int i = 0;
  int32x4_t acc = vdupq_n_s32(0);
  for (; i + 16 <= n; i += 16) {
    int8x16_t va = vld1q_s8(a + i);                   // load 16 s8
    int8x16_t vb = vld1q_s8(b + i);                   // load 16 s8
#if defined(__ARM_FEATURE_DOTPROD)
    acc = vdotq_s32(acc, va, vb);                     // 4-lane s8 dot (AArch64 dotprod)
#else
    int16x8_t lo = vmull_s8(vget_low_s8(va),  vget_low_s8(vb)); // mul low 8 → s16
    int16x8_t hi = vmull_s8(vget_high_s8(va), vget_high_s8(vb));// mul high 8 → s16
    acc = vaddq_s32(acc, vpaddlq_s16(lo));            // pairwise add s16 → s32
    acc = vaddq_s32(acc, vpaddlq_s16(hi));
#endif
  }
#if defined(__aarch64__)
  s += vaddvq_s32(acc);                                // horizontal add
#else
  int32x2_t t = vadd_s32(vget_low_s32(acc), vget_high_s32(acc));
  s += vget_lane_s32(t,0) + vget_lane_s32(t,1);
#endif
  for (; i < n; ++i) s += (int32_t)a[i] * (int32_t)b[i];
#else
  for (int i = 0; i < n; ++i) s += (int32_t)a[i] * (int32_t)b[i];
#endif
  return s;
}

// A(s8)·B4(q4 packed) and sum(A) over n
void dot_s8q4_acc(const int8_t* a, const uint8_t* b4, int n,
                                int32_t* dot, int32_t* sum_a) {
  int32_t d = 0, sa = 0;
#ifdef __ARM_NEON
  int i = 0;
  int32x4_t accd = vdupq_n_s32(0), acca = vdupq_n_s32(0);
  for (; i + 16 <= n; i += 16) {
    int8x16_t va = vld1q_s8(a + i);                   // load 16 A
    // sum(A) lanes
    int16x8_t aw0 = vmovl_s8(vget_low_s8(va));        // widen low 8 → s16
    int16x8_t aw1 = vmovl_s8(vget_high_s8(va));       // widen high 8 → s16
    acca = vaddq_s32(acca, vpaddlq_s16(aw0));         // accumulate
    acca = vaddq_s32(acca, vpaddlq_s16(aw1));

    // Load 8 packed bytes (16 nibbles) → two s8x8 that we combine to s8x16
    uint8x8_t pb = vld1_u8(b4 + (i >> 1));            // load 8 bytes
    uint8x8_t lo = vand_u8(pb, vdup_n_u8(0x0F));      // extract low nibbles
    uint8x8_t hi = vshr_n_u8(pb, 4);                  // extract high nibbles
    uint8x8x2_t z = vzip_u8(lo, hi);                  // interleave -> [q0,q1,...,q7]
    int8x16_t vb = vcombine_s8(vreinterpret_s8_u8(z.val[0]),
                               vreinterpret_s8_u8(z.val[1])); // 16 q4 as s8

#if defined(__ARM_FEATURE_DOTPROD)
    accd = vdotq_s32(accd, va, vb);                   // dot (s8*s8 accumulate)
#else
    int16x8_t loMul = vmull_s8(vget_low_s8(va),  vget_low_s8(vb));
    int16x8_t hiMul = vmull_s8(vget_high_s8(va), vget_high_s8(vb));
    accd = vaddq_s32(accd, vpaddlq_s16(loMul));
    accd = vaddq_s32(accd, vpaddlq_s16(hiMul));
#endif
  }
#if defined(__aarch64__)
  d  += vaddvq_s32(accd);
  sa += vaddvq_s32(acca);
#else
  { int32x2_t t = vadd_s32(vget_low_s32(accd), vget_high_s32(accd));
    d  += vget_lane_s32(t,0) + vget_lane_s32(t,1); }
  { int32x2_t t = vadd_s32(vget_low_s32(acca), vget_high_s32(acca));
    sa += vget_lane_s32(t,0) + vget_lane_s32(t,1); }
#endif
  for (; i < n; ++i) {                                  // scalar tail
    uint8_t packed = b4[i >> 1];
    int8_t  q = (i & 1) ? (int8_t)((packed >> 4) & 0x0F)
                        : (int8_t)( packed       & 0x0F);
    d  += (int32_t)a[i] * (int32_t)q;
    sa += (int32_t)a[i];
  }
#else
  for (int i = 0; i < n; ++i) {
    uint8_t packed = b4[i >> 1];
    int8_t  q = (i & 1) ? (int8_t)((packed >> 4) & 0x0F)
                        : (int8_t)( packed       & 0x0F);
    d  += (int32_t)a[i] * (int32_t)q;
    sa += (int32_t)a[i];
  }
#endif
  *dot = d; if (sum_a) *sum_a = sa;
}

// ================================================================
// GROUP QUANT/DEQUANT
// ================================================================

// Q8 group quantize; returns scale
static inline float q8_group_quant(const float* x, int8_t* q, int gs) {
  const float QMAX = 127.f;
  float maxa = hmax_abs_f32(x, gs);
  float s    = fmaxf(maxa / QMAX, 1e-8f);
  float inv  = 1.f / s;
#ifdef __ARM_NEON
  int i = 0;
  float32x4_t invv = vdupq_n_f32(inv);
  for (; i + 16 <= gs; i += 16) {
    float32x4_t f0 = vld1q_f32(x + i +  0);
    float32x4_t f1 = vld1q_f32(x + i +  4);
    float32x4_t f2 = vld1q_f32(x + i +  8);
    float32x4_t f3 = vld1q_f32(x + i + 12);
    int16x4_t q0 = q_f32x4_to_s8_s16lane(f0, invv);
    int16x4_t q1 = q_f32x4_to_s8_s16lane(f1, invv);
    int16x4_t q2 = q_f32x4_to_s8_s16lane(f2, invv);
    int16x4_t q3 = q_f32x4_to_s8_s16lane(f3, invv);
    int8x16_t qs = vcombine_s8(vmovn_s16(vcombine_s16(q0, q1)),
                               vmovn_s16(vcombine_s16(q2, q3)));
    vst1q_s8(q + i, qs);
  }
  for (; i < gs; ++i) {
    int r = (int)lrintf(x[i] * inv);
    q[i] = (int8_t)(r < -127 ? -127 : (r > 127 ? 127 : r));
  }
#else
  for (int i = 0; i < gs; ++i) {
    int r = (int)lrintf(x[i] * inv);
    q[i] = (int8_t)(r < -127 ? -127 : (r > 127 ? 127 : r));
  }
#endif
  return s;
}

static inline void q8_group_dequant(const int8_t* q, float s, float* y, int gs) {
#ifdef __ARM_NEON
  int i = 0;
  float32x4_t sv = vdupq_n_f32(s);
  for (; i + 16 <= gs; i += 16) {
    int8x16_t vq = vld1q_s8(q + i);                    // load 16 s8
    int16x8_t w0 = vmovl_s8(vget_low_s8(vq));          // widen to s16
    int16x8_t w1 = vmovl_s8(vget_high_s8(vq));
    int32x4_t a0 = vmovl_s16(vget_low_s16(w0));        // widen to s32
    int32x4_t a1 = vmovl_s16(vget_high_s16(w0));
    int32x4_t a2 = vmovl_s16(vget_low_s16(w1));
    int32x4_t a3 = vmovl_s16(vget_high_s16(w1));
    vst1q_f32(y + i +  0, vmulq_f32(vcvtq_f32_s32(a0), sv));
    vst1q_f32(y + i +  4, vmulq_f32(vcvtq_f32_s32(a1), sv));
    vst1q_f32(y + i +  8, vmulq_f32(vcvtq_f32_s32(a2), sv));
    vst1q_f32(y + i + 12, vmulq_f32(vcvtq_f32_s32(a3), sv));
  }
  for (; i < gs; ++i) y[i] = (float)q[i] * s;
#else
  for (int i = 0; i < gs; ++i) y[i] = (float)q[i] * s;
#endif
}

// Affine Q4 group quantize (stores scale and min as zero-point)
static inline void q4_group_quant(const float* x, uint8_t* q, float* s, float* zp, int gs) {
  float mn = x[0], mx = x[0];
#ifdef __ARM_NEON
  int i = 0;
  float32x4_t vmin4 = vdupq_n_f32(mn), vmax4 = vdupq_n_f32(mx);
  for (; i + 4 <= gs; i += 4) {
    float32x4_t v = vld1q_f32(x + i);
    vmin4 = vminq_f32(vmin4, v);
    vmax4 = vmaxq_f32(vmax4, v);
  }
#if defined(__aarch64__)
  mn = vminvq_f32(vmin4);
  mx = vmaxvq_f32(vmax4);
#else
  float32x2_t tmin = vmin_f32(vget_low_f32(vmin4), vget_high_f32(vmin4));
  float32x2_t tmax = vmax_f32(vget_low_f32(vmax4), vget_high_f32(vmax4));
  mn = fminf(vget_lane_f32(tmin,0), vget_lane_f32(tmin,1));
  mx = fmaxf(vget_lane_f32(tmax,0), vget_lane_f32(tmax,1));
#endif
  for (; i < gs; ++i) { float v = x[i]; if (v < mn) mn = v; if (v > mx) mx = v; }
#else
  for (int i = 1; i < gs; ++i) { float v = x[i]; if (v < mn) mn = v; if (v > mx) mx = v; }
#endif
  float sc = (mx - mn) / 15.f; if (sc < 1e-8f) sc = 1e-8f;
  float inv = 1.f / sc;
  *s = sc; *zp = mn;

  i = 0;
  for (; i + 2 <= gs; i += 2) {
    int q0 = (int)lrintf((x[i+0] - mn) * inv); if (q0 < 0) q0 = 0; else if (q0 > 15) q0 = 15;
    int q1 = (int)lrintf((x[i+1] - mn) * inv); if (q1 < 0) q1 = 0; else if (q1 > 15) q1 = 15;
    q[i >> 1] = (uint8_t)((q1 << 4) | (q0 & 0x0F));     // pack (q1|q0)
  }
  if (i < gs) {
    int q0 = (int)lrintf((x[i] - mn) * inv); if (q0 < 0) q0 = 0; else if (q0 > 15) q0 = 15;
    q[i >> 1] = (uint8_t)(q0 & 0x0F);
  }
}

static inline void q4_group_dequant(const uint8_t* q, float s, float zp, float* y, int gs) {
  for (int i = 0; i < gs; ++i) {
    uint8_t b = q[i >> 1];
    int qq = (i & 1) ? ((b >> 4) & 0x0F) : (b & 0x0F);
    y[i] = (float)qq * s + zp;
  }
}

// ================================================================
// PUBLIC QUANT/DEQUANT API
// ================================================================

void quantize_q8(const float* x, int8_t* q, float* s, int n, int gs) {
  assert(is_power_of_2(gs) && n % gs == 0);
  const int ng = n >> get_log2(gs);
#pragma omp parallel for schedule(static)
  for (int g = 0; g < ng; ++g)
    s[g] = q8_group_quant(x + g*gs, q + g*gs, gs);
}

void dequantize_q8(const int8_t* q, const float* s, float* y, int n, int gs) {
  assert(is_power_of_2(gs) && n % gs == 0);
  const int ng = n >> get_log2(gs);
#pragma omp parallel for schedule(static)
  for (int g = 0; g < ng; ++g)
    q8_group_dequant(q + g*gs, s[g], y + g*gs, gs);
}

void quantize_q4(const float* x, size_t rows, size_t cols, size_t gs,
                 float* scales, float* zps, uint8_t* q) {
  const size_t n = rows * cols;
  assert(is_power_of_2(gs) && (n % gs == 0));
  const size_t ng = n >> get_log2(gs);
#pragma omp parallel for schedule(static)
  for (size_t g = 0; g < ng; ++g)
    q4_group_quant(x + g*gs, q + ((g*gs) >> 1), &scales[g], &zps[g], (int)gs);
}

void dequantize_q4(const uint8_t* q, const float* s, const float* zp,
                   float* y, size_t rows, size_t cols, size_t gs) {
  const size_t n = rows * cols;
  assert(is_power_of_2(gs) && (n % gs == 0));
  const size_t ng = n >> get_log2(gs);
#pragma omp parallel for schedule(static)
  for (size_t g = 0; g < ng; ++g)
    q4_group_dequant(q + ((g*gs) >> 1), s[g], zp[g], y + g*gs, (int)gs);
}

// ================================================================
// MICRO-KERNELS (K chunk = 128) + FALLBACKS
// ================================================================

#ifdef __ARM_NEON
// s8x s8 → 4 outputs, K==128 for each output column
static inline void s8s8_dot4_k128(const int8_t* a,
                                  const int8_t* b0, const int8_t* b1,
                                  const int8_t* b2, const int8_t* b3,
                                  int32_t out[4]) {
  int32x4_t acc0 = vdupq_n_s32(0), acc1 = vdupq_n_s32(0);
  int32x4_t acc2 = vdupq_n_s32(0), acc3 = vdupq_n_s32(0);
  for (int i = 0; i < 128; i += 16) {
    int8x16_t va = vld1q_s8(a  + i);
    int8x16_t vb0 = vld1q_s8(b0 + i);
    int8x16_t vb1 = vld1q_s8(b1 + i);
    int8x16_t vb2 = vld1q_s8(b2 + i);
    int8x16_t vb3 = vld1q_s8(b3 + i);
#if defined(__ARM_FEATURE_DOTPROD)
    acc0 = vdotq_s32(acc0, va, vb0);
    acc1 = vdotq_s32(acc1, va, vb1);
    acc2 = vdotq_s32(acc2, va, vb2);
    acc3 = vdotq_s32(acc3, va, vb3);
#else
    int16x8_t lo, hi;
    lo = vmull_s8(vget_low_s8(va), vget_low_s8(vb0)); acc0 = vaddq_s32(acc0, vpaddlq_s16(lo));
    hi = vmull_s8(vget_high_s8(va),vget_high_s8(vb0)); acc0 = vaddq_s32(acc0, vpaddlq_s16(hi));
    lo = vmull_s8(vget_low_s8(va), vget_low_s8(vb1)); acc1 = vaddq_s32(acc1, vpaddlq_s16(lo));
    hi = vmull_s8(vget_high_s8(va),vget_high_s8(vb1)); acc1 = vaddq_s32(acc1, vpaddlq_s16(hi));
    lo = vmull_s8(vget_low_s8(va), vget_low_s8(vb2)); acc2 = vaddq_s32(acc2, vpaddlq_s16(lo));
    hi = vmull_s8(vget_high_s8(va),vget_high_s8(vb2)); acc2 = vaddq_s32(acc2, vpaddlq_s16(hi));
    lo = vmull_s8(vget_low_s8(va), vget_low_s8(vb3)); acc3 = vaddq_s32(acc3, vpaddlq_s16(lo));
    hi = vmull_s8(vget_high_s8(va),vget_high_s8(vb3)); acc3 = vaddq_s32(acc3, vpaddlq_s16(hi));
#endif
  }
#if defined(__aarch64__)
  out[0] = vaddvq_s32(acc0);
  out[1] = vaddvq_s32(acc1);
  out[2] = vaddvq_s32(acc2);
  out[3] = vaddvq_s32(acc3);
#else
  int32x2_t t;
  t = vadd_s32(vget_low_s32(acc0), vget_high_s32(acc0)); out[0] = vget_lane_s32(t,0)+vget_lane_s32(t,1);
  t = vadd_s32(vget_low_s32(acc1), vget_high_s32(acc1)); out[1] = vget_lane_s32(t,0)+vget_lane_s32(t,1);
  t = vadd_s32(vget_low_s32(acc2), vget_high_s32(acc2)); out[2] = vget_lane_s32(t,0)+vget_lane_s32(t,1);
  t = vadd_s32(vget_low_s32(acc3), vget_high_s32(acc3)); out[3] = vget_lane_s32(t,0)+vget_lane_s32(t,1);
#endif
}

// s8 x q4 → 4 outputs, also return sum(a), K==128
static inline void s8q4_dot4_k128(const int8_t* a,
                                  const uint8_t* b40, const uint8_t* b41,
                                  const uint8_t* b42, const uint8_t* b43,
                                  int32_t out[4], int32_t* sum_a) {
  int32x4_t acc0 = vdupq_n_s32(0), acc1 = vdupq_n_s32(0);
  int32x4_t acc2 = vdupq_n_s32(0), acc3 = vdupq_n_s32(0);
  int32x4_t accA = vdupq_n_s32(0);

  for (int i = 0; i < 128; i += 16) {
    int8x16_t va = vld1q_s8(a + i);                   // load 16 A

    // sum(A) lanes
    int16x8_t aw0 = vmovl_s8(vget_low_s8(va));
    int16x8_t aw1 = vmovl_s8(vget_high_s8(va));
    accA = vaddq_s32(accA, vpaddlq_s16(aw0));
    accA = vaddq_s32(accA, vpaddlq_s16(aw1));

    // Helper: unpack 8 bytes → 16 s8 q4 values
    #define UNPACK_Q4_16(ptr) ({ \
      uint8x8_t pb = vld1_u8(ptr);                     /* load 8 bytes */ \
      uint8x8_t lo = vand_u8(pb, vdup_n_u8(0x0F));     /* low nibbles */ \
      uint8x8_t hi = vshr_n_u8(pb, 4);                 /* high nibbles */\
      uint8x8x2_t z = vzip_u8(lo, hi);                 /* interleave */  \
      vcombine_s8(vreinterpret_s8_u8(z.val[0]), vreinterpret_s8_u8(z.val[1])); \
    })

    int8x16_t vb0 = UNPACK_Q4_16(b40 + (i >> 1));
    int8x16_t vb1 = UNPACK_Q4_16(b41 + (i >> 1));
    int8x16_t vb2 = UNPACK_Q4_16(b42 + (i >> 1));
    int8x16_t vb3 = UNPACK_Q4_16(b43 + (i >> 1));
    #undef UNPACK_Q4_16

#if defined(__ARM_FEATURE_DOTPROD)
    acc0 = vdotq_s32(acc0, va, vb0);
    acc1 = vdotq_s32(acc1, va, vb1);
    acc2 = vdotq_s32(acc2, va, vb2);
    acc3 = vdotq_s32(acc3, va, vb3);
#else
    int16x8_t lo, hi;
    lo = vmull_s8(vget_low_s8(va), vget_low_s8(vb0)); acc0 = vaddq_s32(acc0, vpaddlq_s16(lo));
    hi = vmull_s8(vget_high_s8(va),vget_high_s8(vb0)); acc0 = vaddq_s32(acc0, vpaddlq_s16(hi));
    lo = vmull_s8(vget_low_s8(va), vget_low_s8(vb1)); acc1 = vaddq_s32(acc1, vpaddlq_s16(lo));
    hi = vmull_s8(vget_high_s8(va),vget_high_s8(vb1)); acc1 = vaddq_s32(acc1, vpaddlq_s16(hi));
    lo = vmull_s8(vget_low_s8(va), vget_low_s8(vb2)); acc2 = vaddq_s32(acc2, vpaddlq_s16(lo));
    hi = vmull_s8(vget_high_s8(va),vget_high_s8(vb2)); acc2 = vaddq_s32(acc2, vpaddlq_s16(hi));
    lo = vmull_s8(vget_low_s8(va), vget_low_s8(vb3)); acc3 = vaddq_s32(acc3, vpaddlq_s16(lo));
    hi = vmull_s8(vget_high_s8(va),vget_high_s8(vb3)); acc3 = vaddq_s32(acc3, vpaddlq_s16(hi));
#endif
  }

#if defined(__aarch64__)
  out[0] = vaddvq_s32(acc0);
  out[1] = vaddvq_s32(acc1);
  out[2] = vaddvq_s32(acc2);
  out[3] = vaddvq_s32(acc3);
  *sum_a = vaddvq_s32(accA);
#else
  int32x2_t t;
  t = vadd_s32(vget_low_s32(acc0), vget_high_s32(acc0)); out[0] = vget_lane_s32(t,0)+vget_lane_s32(t,1);
  t = vadd_s32(vget_low_s32(acc1), vget_high_s32(acc1)); out[1] = vget_lane_s32(t,0)+vget_lane_s32(t,1);
  t = vadd_s32(vget_low_s32(acc2), vget_high_s32(acc2)); out[2] = vget_lane_s32(t,0)+vget_lane_s32(t,1);
  t = vadd_s32(vget_low_s32(acc3), vget_high_s32(acc3)); out[3] = vget_lane_s32(t,0)+vget_lane_s32(t,1);
  t = vadd_s32(vget_low_s32(accA), vget_high_s32(accA)); *sum_a = vget_lane_s32(t,0)+vget_lane_s32(t,1);
#endif
}
#endif // __ARM_NEON

// ================================================================
// MATRIX MULTIPLICATION
// ================================================================

void matmul_q8_q8_f32(const int8_t* A, const float* As,
                      const int8_t* B, const float* Bs,
                      float* C, int M, int N, int K, int gs) {
  assert(is_power_of_2(gs) && (K % gs == 0));
  const int ng = K / gs;

#pragma omp parallel for collapse(2) schedule(static)
  for (int m = 0; m < M; ++m) {
    for (int n0 = 0; n0 < N; n0 += 4) {
      float acc[4] = {0,0,0,0};
      int nblock = (n0 + 4 <= N) ? 4 : (N - n0);

      const int8_t* arow = A + m*K;
      const float*  as   = As + m*ng;

      for (int g = 0; g < ng; ++g) {
        const int8_t* ag = arow + g*gs;
        float w = as[g]; // A scale for this group

        if (gs == 128 && nblock == 4) {
#ifdef __ARM_NEON
          const int8_t* b0 = B + (n0+0)*K + g*gs;
          const int8_t* b1 = B + (n0+1)*K + g*gs;
          const int8_t* b2 = B + (n0+2)*K + g*gs;
          const int8_t* b3 = B + (n0+3)*K + g*gs;
          int32_t d[4];
          s8s8_dot4_k128(ag, b0,b1,b2,b3, d);
          // scale by A*B for each column
          acc[0] += (float)d[0] * (w * Bs[(n0+0)*ng + g]);
          acc[1] += (float)d[1] * (w * Bs[(n0+1)*ng + g]);
          acc[2] += (float)d[2] * (w * Bs[(n0+2)*ng + g]);
          acc[3] += (float)d[3] * (w * Bs[(n0+3)*ng + g]);
#else
          for (int j = 0; j < 4; ++j) {
            int32_t d = dot_s8s8(ag, B + (n0+j)*K + g*gs, gs);
            acc[j] += (float)d * (w * Bs[(n0+j)*ng + g]);
          }
#endif
        } else { // generic group size or edge columns
          for (int j = 0; j < nblock; ++j) {
            int32_t d = dot_s8s8(ag, B + (n0+j)*K + g*gs, gs);
            acc[j] += (float)d * (w * Bs[(n0+j)*ng + g]);
          }
        }
      }
      for (int j = 0; j < nblock; ++j) C[m*N + (n0+j)] = acc[j];
    }
  }
}

void matmul_q8_q4_f32(const int8_t* A, const float* As,
                      const uint8_t* B4, const float* Bs, const float* Bzp,
                      float* C, int M, int N, int K, size_t gs_) {
  const int gs = (int)gs_;
  assert(is_power_of_2(gs_) && (K % gs_ == 0));
  const int ng = K / gs;

#pragma omp parallel for collapse(2) schedule(static)
  for (int m = 0; m < M; ++m) {
    for (int n0 = 0; n0 < N; n0 += 4) {
      float acc[4] = {0,0,0,0};
      int nblock = (n0 + 4 <= N) ? 4 : (N - n0);

      const int8_t*  arow  = A  + m*K;
      const float*   as    = As + m*ng;
      const uint8_t* bcol0 = B4 + (n0+0)*(K>>1);
      const uint8_t* bcol1 = B4 + (n0+1)*(K>>1);
      const uint8_t* bcol2 = B4 + (n0+2)*(K>>1);
      const uint8_t* bcol3 = B4 + (n0+3)*(K>>1);

      for (int g = 0; g < ng; ++g) {
        const int8_t* ag = arow + g*gs;
        float a_s = as[g];

        if (gs == 128 && nblock == 4) {
#ifdef __ARM_NEON
          int32_t d[4], sa;
          s8q4_dot4_k128(ag,
                         bcol0 + ((g*gs)>>1),
                         bcol1 + ((g*gs)>>1),
                         bcol2 + ((g*gs)>>1),
                         bcol3 + ((g*gs)>>1),
                         d, &sa);
          // Per-column scale products
          acc[0] += (float)d[0] * (a_s * Bs[(n0+0)*ng + g]);
          acc[1] += (float)d[1] * (a_s * Bs[(n0+1)*ng + g]);
          acc[2] += (float)d[2] * (a_s * Bs[(n0+2)*ng + g]);
          acc[3] += (float)d[3] * (a_s * Bs[(n0+3)*ng + g]);
          if (Bzp) {
            float add = (float)sa * a_s;
            acc[0] += add * Bzp[(n0+0)*ng + g];
            acc[1] += add * Bzp[(n0+1)*ng + g];
            acc[2] += add * Bzp[(n0+2)*ng + g];
            acc[3] += add * Bzp[(n0+3)*ng + g];
          }
#else
          for (int j = 0; j < 4; ++j) {
            int32_t d, sa;
            dot_s8q4_acc(ag, B4 + (n0+j)*(K>>1) + ((g*gs)>>1), gs, &d, &sa);
            acc[j] += (float)d * (a_s * Bs[(n0+j)*ng + g]);
            if (Bzp) acc[j] += (float)sa * (a_s * Bzp[(n0+j)*ng + g]);
          }
#endif
        } else {
          for (int j = 0; j < nblock; ++j) {
            int32_t d, sa;
            const uint8_t* b4 = B4 + (n0+j)*(K>>1) + ((g*gs)>>1);
            dot_s8q4_acc(ag, b4, gs, &d, &sa);
            acc[j] += (float)d * (a_s * Bs[(n0+j)*ng + g]);
            if (Bzp) acc[j] += (float)sa * (a_s * Bzp[(n0+j)*ng + g]);
          }
        }
      }
      for (int j = 0; j < nblock; ++j) C[m*N + (n0+j)] = acc[j];
    }
  }
}

// ================================================================
// MIXED (FP32 x Q*)  — keep fused path + scratch path
// ================================================================

void matmul_f32_q8_f32(const float* A_f32, const int8_t* B, const float* Bs,
                       float* C, int M, int N, int K, int gs,
                       int8_t* A_q_scratch, float* A_s_scratch) {
  assert(is_power_of_2(gs) && (K % gs == 0));
  const int ng = K / gs;

  const int REUSE_THRESHOLD_N = 4; // amortize quant if many columns

  if (A_q_scratch && A_s_scratch && N >= REUSE_THRESHOLD_N) {
#pragma omp parallel for schedule(static)
    for (int m = 0; m < M; ++m)
      for (int g = 0; g < ng; ++g)
        A_s_scratch[m*ng + g] =
          q8_group_quant(A_f32 + m*K + g*gs, A_q_scratch + m*K + g*gs, gs);

    matmul_q8_q8_f32(A_q_scratch, A_s_scratch, B, Bs, C, M, N, K, gs);
  } else {
#pragma omp parallel for collapse(2) schedule(static)
    for (int m = 0; m < M; ++m)
      for (int n = 0; n < N; ++n) {
        float acc = 0.f;
        for (int g = 0; g < ng; ++g) {
          const float* ag = A_f32 + m*K + g*gs;
          // stack buf limited to 256 — supports gs up to 256
          int8_t aq_buf[256]; assert(gs <= (int)sizeof(aq_buf));
          float as = q8_group_quant(ag, aq_buf, gs);
          int32_t d = dot_s8s8(aq_buf, B + n*K + g*gs, gs);
          acc += (float)d * (as * Bs[n*ng + g]);
        }
        C[m*N + n] = acc;
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
    for (int m = 0; m < M; ++m)
      for (int g = 0; g < ng; ++g)
        A_s_scratch[m*ng + g] =
          q8_group_quant(A_f32 + m*K + g*gs, A_q_scratch + m*K + g*gs, gs);

    matmul_q8_q4_f32(A_q_scratch, A_s_scratch, B4, Bs, Bzp, C, M, N, K, gs);
  } else {
#pragma omp parallel for collapse(2) schedule(static)
    for (int m = 0; m < M; ++m)
      for (int n = 0; n < N; ++n) {
        float acc = 0.f;
        for (int g = 0; g < ng; ++g) {
          const float* ag = A_f32 + m*K + g*gs;
          int8_t aq_buf[256]; assert(gs <= (int)sizeof(aq_buf));
          float as = q8_group_quant(ag, aq_buf, gs);
          int32_t d, sa;
          dot_s8q4_acc(aq_buf, B4 + n*(K>>1) + ((g*gs)>>1), gs, &d, &sa);
          acc += (float)d * (as * Bs[n*ng + g]);
          if (Bzp) acc += (float)sa * (as * Bzp[n*ng + g]);
        }
        C[m*N + n] = acc;
      }
  }
}

void matmul_f32_q4_f32(const float* A_fp32, const uint8_t* B_q4, const float* B_scales,
                       float* C, int M, int N, int K, int gs,
                       int8_t* A_q_scratch, float* A_s_scratch) {
  const int ng_total = (N * K) / gs;
  // Provide neutral zeros on stack when small enough
  if (ng_total <= 16384) {
    float neutral_zeros[16384];
    for (int i = 0; i < ng_total; ++i) neutral_zeros[i] = 8.0f;
    matmul_f32_q4_f32_with_zeros(A_fp32, B_q4, B_scales, neutral_zeros,
                                 C, M, N, K, gs, A_q_scratch, A_s_scratch);
  } else {
    float* neutral = (float*)calloc((size_t)ng_total, sizeof(float));
    if (!neutral) { fprintf(stderr, "matmul_f32_q4_f32: OOM\n"); exit(1); }
    for (int i = 0; i < ng_total; ++i) neutral[i] = 8.0f;
    matmul_f32_q4_f32_with_zeros(A_fp32, B_q4, B_scales, neutral,
                                 C, M, N, K, gs, A_q_scratch, A_s_scratch);
    free(neutral);
  }
}
