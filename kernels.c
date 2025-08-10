
// kernels.c — readable kernels with DEBUG/BENCH and both router modes.
#include "kernels.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef DEBUG
  #define DBG(...) do { fprintf(stderr, __VA_ARGS__); } while(0)
#else
  #define DBG(...)
#endif

#ifdef BENCH
  #define TIMER_DECL struct timespec __t0, __t1
  #define TIMER_START() clock_gettime(CLOCK_MONOTONIC, &__t0)
  #define TIMER_END_MS(ms_out) do { \
      clock_gettime(CLOCK_MONOTONIC, &__t1); \
      double __ms = (__t1.tv_sec - __t0.tv_sec) * 1000.0 + \
                    (__t1.tv_nsec - __t0.tv_nsec) / 1.0e6; \
      (ms_out) = __ms; \
    } while(0)
#else
  #define TIMER_DECL
  #define TIMER_START()     do{}while(0)
  #define TIMER_END_MS(x)   do{ (void)(x); }while(0)
#endif

void matmul_f32(const float* A, const float* B, float* C,
                int M, int N, int K) {
  TIMER_DECL; double ms=0.0;
  DBG("[matmul] A[%d,%d] * W^T[%d,%d] -> Y[%d,%d]\n", M, K, N, K, M, N);
  TIMER_START();
  for (int m = 0; m < M; ++m) {
    const float* a = A + m*K;
    float* c = C + m*N;
    for (int n = 0; n < N; ++n) {
      const float* b = B + n*K;
      float acc = 0.f;
      for (int k = 0; k < K; ++k) acc += a[k] * b[k];
      c[n] = acc;
    }
  }
  TIMER_END_MS(ms);
#ifdef BENCH
  DBG("[matmul] done in %.3f ms\n", ms);
#else
  DBG("[matmul] done\n");
#endif
}

void silu_f32(float* x, int n) {
  for (int i = 0; i < n; ++i) {
    float v = x[i];
    x[i] = v / (1.0f + expf(-v));
  }
}

void expert_forward_f32(const float* x,
                        const float* Wg, const float* bg,
                        const float* Wu, const float* bu,
                        const float* Wd, const float* bd,
                        int T, int d_model, int d_ff,
                        float* tmp_g, float* tmp_u,
                        float* y) {
  DBG("[expert] T=%d d_model=%d d_ff=%d\n", T, d_model, d_ff);
  // gate/up
  matmul_f32(x, Wg, tmp_g, T, d_ff, d_model);
  if (bg) for (int i = 0; i < T*d_ff; ++i) tmp_g[i] += bg[i % d_ff];
  matmul_f32(x, Wu, tmp_u, T, d_ff, d_model);
  if (bu) for (int i = 0; i < T*d_ff; ++i) tmp_u[i] += bu[i % d_ff];
  // SwiGLU
  for (int i = 0; i < T*d_ff; ++i) tmp_g[i] = tmp_g[i] / (1.f + expf(-tmp_g[i])) * tmp_u[i];
  // down
  matmul_f32(tmp_g, Wd, y, T, d_model, d_ff);
  if (bd) for (int i = 0; i < T*d_model; ++i) y[i] += bd[i % d_model];
}

// ---- Router helpers ----

// Top-k over logits[T,E], then softmax over those k only (per token).
void router_topk_softmax_konly(const float* logits, int T, int E, int k,
                               int* top_idx, float* top_p) {
  DBG("[router] T=%d E=%d k=%d (topk logits + softmax over k)\n", T, E, k);
  for (int t = 0; t < T; ++t) {
    const float* lt = logits + t*E;
    // initialize top-k with first k entries
    for (int j = 0; j < k; ++j) { top_idx[t*k + j] = j; top_p[t*k + j] = lt[j]; }
    // replace min in top-k when a larger logit appears
    for (int e = k; e < E; ++e) {
      float v = lt[e];
      int minj = 0;
      for (int j = 1; j < k; ++j) if (top_p[t*k + j] < top_p[t*k + minj]) minj = j;
      if (v > top_p[t*k + minj]) { top_p[t*k + minj] = v; top_idx[t*k + minj] = e; }
    }
    // sort top-k by logit descending
    for (int a = 0; a < k; ++a) {
      int best = a;
      for (int b = a+1; b < k; ++b)
        if (top_p[t*k + b] > top_p[t*k + best]) best = b;
      if (best != a) {
        float tv = top_p[t*k + a]; top_p[t*k + a] = top_p[t*k + best]; top_p[t*k + best] = tv;
        int ti = top_idx[t*k + a]; top_idx[t*k + a] = top_idx[t*k + best]; top_idx[t*k + best] = ti;
      }
    }
    // softmax over these k logits
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

// Softmax over all E first, then pick top-k by probability
void router_softmax_all_topk(const float* logits, int T, int E, int k,
                             int* top_idx, float* top_p) {
  DBG("[router] T=%d E=%d k=%d (softmax over all -> topk)\n", T, E, k);
  float* probs = (float*)malloc(sizeof(float) * T * E);
  if (!probs) { fprintf(stderr, "OOM router probs\n"); exit(1); }
  for (int t = 0; t < T; ++t) {
    const float* lt = logits + t*E;
    float* pt = probs + t*E;
    float maxv = lt[0];
    for (int e = 1; e < E; ++e) if (lt[e] > maxv) maxv = lt[e];
    float sum = 0.f;
    for (int e = 0; e < E; ++e) { float z = expf(lt[e] - maxv); pt[e] = z; sum += z; }
    float inv = 1.0f / sum;
    for (int e = 0; e < E; ++e) pt[e] *= inv;
  }
  // top-k per row
  for (int t = 0; t < T; ++t) {
    const float* pt = probs + t*E;
    // initialize
    for (int j = 0; j < k; ++j) { top_idx[t*k + j] = j; top_p[t*k + j] = pt[j]; }
    for (int e = k; e < E; ++e) {
      float p = pt[e];
      int minj = 0;
      for (int j = 1; j < k; ++j) if (top_p[t*k + j] < top_p[t*k + minj]) minj = j;
      if (p > top_p[t*k + minj]) { top_p[t*k + minj] = p; top_idx[t*k + minj] = e; }
    }
    // sort desc
    for (int a = 0; a < k; ++a) {
      int best = a;
      for (int b = a+1; b < k; ++b)
        if (top_p[t*k + b] > top_p[t*k + best]) best = b;
      if (best != a) {
        float tp = top_p[t*k + a]; top_p[t*k + a] = top_p[t*k + best]; top_p[t*k + best] = tp;
        int ti = top_idx[t*k + a]; top_idx[t*k + a] = top_idx[t*k + best]; top_idx[t*k + best] = ti;
      }
    }
#ifdef DEBUG
    DBG("[router] t=%d topk:", t);
    for (int j = 0; j < k; ++j) DBG(" (%d:%.4f)", top_idx[t*k + j], top_p[t*k + j]);
    DBG("\n");
#endif
  }
  free(probs);
}

// Full MoE forward (per-token loop)
void moe_forward_f32_mode(
    const float* x, int T, int d_model,
    const float* router_w, const float* router_b,
    int E, int k, int d_ff, RouterMode mode,
    const float** Wg_arr, const float** bg_arr,
    const float** Wu_arr, const float** bu_arr,
    const float** Wd_arr, const float** bd_arr,
    float* y, float* tmp_one_g, float* tmp_one_u,
    int* top_idx, float* top_p) {
  DBG("[moe] T=%d d_model=%d E=%d k=%d d_ff=%d mode=%d\n", T, d_model, E, k, d_ff, (int)mode);

  // 1) router logits [T,E] = x @ W_router^T + b
  float* logits = (float*)malloc(sizeof(float) * T * E);
  if (!logits) { fprintf(stderr, "OOM router logits\n"); exit(1); }
  matmul_f32(x, router_w, logits, T, E, d_model);
  if (router_b) {
    for (int t = 0; t < T; ++t) {
      for (int e = 0; e < E; ++e) logits[t*E + e] += router_b[e];
    }
  }

  // 2) routing
  TIMER_DECL; double ms=0.0; TIMER_START();
  if (mode == ROUTER_TOPK_KONLY) {
    router_topk_softmax_konly(logits, T, E, k, top_idx, top_p);
  } else {
    router_softmax_all_topk(logits, T, E, k, top_idx, top_p);
  }
  TIMER_END_MS(ms);
#ifdef BENCH
  DBG("[router] routing done in %.3f ms\n", ms);
#endif

  // 3) expert mix per token
  for (int i = 0; i < T*d_model; ++i) y[i] = 0.f;

  for (int t = 0; t < T; ++t) {
    const float* xt = x + t*d_model;
    float* yt = y + t*d_model;

    for (int j = 0; j < k; ++j) {
      int e = top_idx[t*k + j];
      float p = top_p[t*k + j];
#ifdef DEBUG
      DBG("[moe] t=%d expert=%d prob=%.6f\n", t, e, p);
#endif
      // expert forward for a single token
      matmul_f32(xt, Wg_arr[e], tmp_one_g, 1, d_ff, d_model);
      if (bg_arr && bg_arr[e]) for (int q = 0; q < d_ff; ++q) tmp_one_g[q] += bg_arr[e][q];
      matmul_f32(xt, Wu_arr[e], tmp_one_u, 1, d_ff, d_model);
      if (bu_arr && bu_arr[e]) for (int q = 0; q < d_ff; ++q) tmp_one_u[q] += bu_arr[e][q];
      for (int q = 0; q < d_ff; ++q) tmp_one_g[q] = tmp_one_g[q] / (1.f + expf(-tmp_one_g[q])) * tmp_one_u[q];
      float* out_e = tmp_one_u; // reuse buffer for output
      matmul_f32(tmp_one_g, Wd_arr[e], out_e, 1, d_model, d_ff);
      if (bd_arr && bd_arr[e]) for (int q = 0; q < d_model; ++q) out_e[q] += bd_arr[e][q];
      for (int q = 0; q < d_model; ++q) yt[q] += p * out_e[q];
    }
  }

  free(logits);
}

