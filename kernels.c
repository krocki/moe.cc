// kernels.c
// Compact, readable kernels for MoE experiments (2-space indentation).
// DEBUG: define to print shapes and key steps.
// BENCH : define to print simple timings for major ops.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "kernels.h"

// -------- timing helpers (BENCH) --------
#ifdef BENCH
  #define TIMER_DECL clock_t __t0, __t1
  #define TIMER_START() do { __t0 = clock(); } while(0)
  #define TIMER_END_MS(ms_out) do { __t1 = clock(); \
    double __ms = (double)(__t1 - __t0) * 1000.0 / (double)CLOCKS_PER_SEC; \
    (ms_out) = __ms; } while(0)
#else
  #define TIMER_DECL
  #define TIMER_START()     do{}while(0)
  #define TIMER_END_MS(x)   do{ (void)(x); }while(0)
#endif

// -------- debug prints (DEBUG) --------
#ifdef DEBUG
  #define DBG(...) do { fprintf(stderr, __VA_ARGS__); } while(0)
#else
  #define DBG(...) do {} while(0)
#endif

// -------- math helpers --------

float silu_scalar(float x) {
  return x / (1.0f + expf(-x));
}

// Y[M,N] = A[M,K] * W[N,K]^T + b[N]
// Row-major everywhere; W is stored [out, in].
void matmul_add_bias_f32(
  const float* A, const float* W, const float* b,
  float* Y, int M, int K, int N)
{
  TIMER_DECL;
  DBG("[matmul] A[%d,%d] * W^T[%d,%d] -> Y[%d,%d]%s\n", M, K, N, K, M, N, b?"+bias":"");
  TIMER_START();
  for (int m = 0; m < M; ++m) {
    const float* a = A + m*K;
    float* y = Y + m*N;
    for (int n = 0; n < N; ++n) {
      const float* w = W + n*K;
      float acc = b ? b[n] : 0.f;
      for (int k = 0; k < K; ++k) acc += a[k] * w[k];
      y[n] = acc;
    }
  }
#ifdef BENCH
  double ms; TIMER_END_MS(ms);
  DBG("[matmul] done in %.3f ms\n", ms);
#else
  DBG("[matmul] done\n");
#endif
}

// Expert forward (SwiGLU): y = down( silu(gate(x)) * up(x) )
void expert_forward_f32(
  const float* x, int T, int d_model, int d_ff,
  const float* Wg, const float* bg,
  const float* Wu, const float* bu,
  const float* Wd, const float* bd,
  float* y, float* tmp_g, float* tmp_u)
{
  DBG("[expert] x[T=%d,d_model=%d] -> d_ff=%d\n", T, d_model, d_ff);
  matmul_add_bias_f32(x, Wg, bg, tmp_g, T, d_model, d_ff);
  matmul_add_bias_f32(x, Wu, bu, tmp_u, T, d_model, d_ff);
  int n = T * d_ff;
  for (int i = 0; i < n; ++i) tmp_g[i] = silu_scalar(tmp_g[i]) * tmp_u[i];
  matmul_add_bias_f32(tmp_g, Wd, bd, y, T, d_ff, d_model);
}

// Softmax over last dim of logits[T,E] -> probs[T,E]
static void softmax_inplace(float* logits, float* out, int T, int E) {
  for (int t = 0; t < T; ++t) {
    float maxv = logits[t*E];
    for (int e = 1; e < E; ++e) {
      float v = logits[t*E + e];
      if (v > maxv) maxv = v;
    }
    float sum = 0.f;
    for (int e = 0; e < E; ++e) {
      float z = expf(logits[t*E + e] - maxv);
      out[t*E + e] = z;
      sum += z;
    }
    float inv = 1.0f / sum;
    for (int e = 0; e < E; ++e) out[t*E + e] *= inv;
  }
}

// Simple top-k per row of probs[T,E]. Returns indices and values (length T*k).
static void topk_per_row(const float* probs, int T, int E, int k,
                         int* top_idx, float* top_p) {
  for (int t = 0; t < T; ++t) {
    for (int j = 0; j < k; ++j) { top_idx[t*k + j] = j; top_p[t*k + j] = probs[t*E + j]; }
    for (int e = k; e < E; ++e) {
      float p = probs[t*E + e];
      int minj = 0;
      for (int j = 1; j < k; ++j) if (top_p[t*k + j] < top_p[t*k + minj]) minj = j;
      if (p > top_p[t*k + minj]) {
        top_p[t*k + minj] = p;
        top_idx[t*k + minj] = e;
      }
    }
    for (int a = 0; a < k; ++a) {
      int best = a;
      for (int b = a+1; b < k; ++b)
        if (top_p[t*k + b] > top_p[t*k + best]) best = b;
      if (best != a) {
        float tp = top_p[t*k + a]; top_p[t*k + a] = top_p[t*k + best]; top_p[t*k + best] = tp;
        int ti = top_idx[t*k + a]; top_idx[t*k + a] = top_idx[t*k + best]; top_idx[t*k + best] = ti;
      }
    }
  }
}

// Top-k over logits[T,E], then softmax over those k only (per token).
void router_topk_softmax_konly(
  const float* logits, int T, int E, int k,
  int* top_idx, float* top_p)
{
  DBG("[router] T=%d E=%d k=%d (topk logits + softmax over k)\n", T, E, k);
  for (int t = 0; t < T; ++t) {
    const float* lt = logits + t*E;

    // 1) find top-k logits (indices + values)
    // initialize with first k
    for (int j = 0; j < k; ++j) { top_idx[t*k + j] = j; top_p[t*k + j] = lt[j]; }
    // insertion for remaining
    for (int e = k; e < E; ++e) {
      float v = lt[e];
      int minj = 0;
      for (int j = 1; j < k; ++j) if (top_p[t*k + j] < top_p[t*k + minj]) minj = j;
      if (v > top_p[t*k + minj]) { top_p[t*k + minj] = v; top_idx[t*k + minj] = e; }
    }

    // 2) stable sort by logit desc (small k)
    for (int a = 0; a < k; ++a) {
      int best = a;
      for (int b = a+1; b < k; ++b)
        if (top_p[t*k + b] > top_p[t*k + best]) best = b;
      if (best != a) {
        float tv = top_p[t*k + a]; top_p[t*k + a] = top_p[t*k + best]; top_p[t*k + best] = tv;
        int ti = top_idx[t*k + a]; top_idx[t*k + a] = top_idx[t*k + best]; top_idx[t*k + best] = ti;
      }
    }

    // 3) softmax over these k logits
    float maxv = top_p[t*k + 0];
    for (int j = 1; j < k; ++j) if (top_p[t*k + j] > maxv) maxv = top_p[t*k + j];
    float sum = 0.f;
    for (int j = 0; j < k; ++j) { float z = expf(top_p[t*k + j] - maxv); top_p[t*k + j] = z; sum += z; }
    float inv = 1.0f / sum;
    for (int j = 0; j < k; ++j) top_p[t*k + j] *= inv;

#ifdef DEBUG
    DBG("[router] t=%d topk:", t);
    for (int j = 0; j < k; ++j) DBG(" (%d:%.4f)", top_idx[t*k + j], top_p[t*k + j]);
    DBG("\n");
#endif
  }
}

// Full MoE forward (naïve, readable)
void moe_forward_f32(
  const float* x, int T, int d_model,
  const float* router_w, const float* router_b,
  int E, int k, int d_ff,
  const float** Wg_arr, const float** bg_arr,
  const float** Wu_arr, const float** bu_arr,
  const float** Wd_arr, const float** bd_arr,
  float* y,
  float* tmp_g, float* tmp_u,
  int* top_idx, float* top_p)
{
  DBG("[moe] T=%d d_model=%d E=%d k=%d d_ff=%d\n", T, d_model, E, k, d_ff);

  // 1) router logits [T,E]
  float* logits = (float*)malloc(sizeof(float) * T * E);
  if (!logits) { fprintf(stderr, "OOM router logits\n"); exit(1); }
  matmul_add_bias_f32(x, router_w, router_b, logits, T, d_model, E);

  // 2) softmax + 3) topk
  router_topk_softmax_konly(logits, T, E, k, top_idx, top_p);

  // 4) per-token expert mix
  for (int i = 0; i < T*d_model; ++i) y[i] = 0.f;

  for (int t = 0; t < T; ++t) {
    const float* xt = x + t*d_model;
    float* yt = y + t*d_model;

    for (int j = 0; j < k; ++j) {
      int e = top_idx[t*k + j];
      float p = top_p[t*k + j];

      matmul_add_bias_f32(xt, Wg_arr[e], bg_arr ? bg_arr[e] : NULL, tmp_g, 1, d_model, d_ff);
      matmul_add_bias_f32(xt, Wu_arr[e], bu_arr ? bu_arr[e] : NULL, tmp_u, 1, d_model, d_ff);
      for (int q = 0; q < d_ff; ++q) tmp_g[q] = silu_scalar(tmp_g[q]) * tmp_u[q];
      float* out_e = tmp_u; // reuse buffer
      matmul_add_bias_f32(tmp_g, Wd_arr[e], bd_arr ? bd_arr[e] : NULL, out_e, 1, d_ff, d_model);

      for (int q = 0; q < d_model; ++q) yt[q] += p * out_e[q];
    }
  }

  free(logits);
}
