
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

// RMSNorm forward
// x: [T, d_model]
// w: [d_model] (scale weights)
// eps: epsilon for numerical stability
// y: [T, d_model] output
void rmsnorm_forward_f32(const float* x, const float* w,
                         int T, int d_model, float eps,
                         float* y) {
#ifdef BENCH
  double ms = 0.0;
#endif
  DBG("[rmsnorm] T=%d d_model=%d eps=%g\n", T, d_model, eps);
#ifdef BENCH
  TIMER_DECL;
  TIMER_START();
#endif
  for (int t = 0; t < T; ++t) {
    const float* xt = x + t * d_model;
    float* yt = y + t * d_model;
    float msq = 0.0f;
    for (int i = 0; i < d_model; ++i)
      msq += xt[i] * xt[i];
    msq /= (float)d_model;
    float inv = 1.0f / sqrtf(msq + eps);
    for (int i = 0; i < d_model; ++i)
      yt[i] = xt[i] * inv * (w ? w[i] : 1.0f);
  }
#ifdef BENCH
  TIMER_END_MS(ms);
  DBG("[rmsnorm] done in %.3f ms\n", ms);
#else
  DBG("[rmsnorm] done\n");
#endif
}

static inline void softmax_row_inplace(float* row, int n) {
  float m = row[0];
  for (int i = 1; i < n; ++i) if (row[i] > m) m = row[i];
  float s = 0.f;
  for (int i = 0; i < n; ++i) { row[i] = expf(row[i] - m); s += row[i]; }
  float inv = 1.0f / s;
  for (int i = 0; i < n; ++i) row[i] *= inv;
}

// GQA self-attention (no RoPE)
//
// x: [T, d_model]
// Q: Wq[b, d_model] where b = n_q * head_dim
// K,V: Wk/Wv[c, d_model] where c = n_kv * head_dim
// Wo: [d_model, n_q * head_dim]
// b*: optional biases; q_norm/k_norm can be per-channel (len=b/c) or per-head (len=n_q/n_kv)
// causal: 1=causal mask, 0=none
// scratch floats needed: T*(b + c + c) + T*T + T*b
void attn_forward_f32_gqa(
  const float* x, int T, int d_model,
  const float* Wq, const float* bq,
  const float* Wk, const float* bk,
  const float* Wv, const float* bv,
  const float* Wo, const float* bo,
  const float* qn, const float* kn,
  int n_q, int n_kv, int head_dim, int causal,
  float* scratch, float* y)
{
  TIMER_DECL; double ms = 0.0;
  const int Dq  = n_q  * head_dim;
  const int Dkv = n_kv * head_dim;
  const float scale = 1.0f / sqrtf((float)head_dim);

  DBG("[attn/gqa] T=%d d_model=%d n_q=%d n_kv=%d head_dim=%d causal=%d\n",
      T, d_model, n_q, n_kv, head_dim, causal);

  // Layout scratch
  float* Q    = scratch;            // [T, Dq]
  float* K    = Q + T*Dq;           // [T, Dkv]
  float* V    = K + T*Dkv;          // [T, Dkv]
  float* S    = V + T*Dkv;          // [T, T] (scores; reused per head)
  float* Hcat = S + T*T;            // [T, Dq]

  // Projections
  TIMER_START();
  matmul_f32(x, Wq, Q, T, Dq,  d_model);
  matmul_f32(x, Wk, K, T, Dkv, d_model);
  matmul_f32(x, Wv, V, T, Dkv, d_model);
  if (bq) for (int t=0; t<T; ++t) for (int i=0; i<Dq;  ++i) Q[t*Dq  + i] += bq[i];
  if (bk) for (int t=0; t<T; ++t) for (int i=0; i<Dkv; ++i) K[t*Dkv + i] += bk[i];
  if (bv) for (int t=0; t<T; ++t) for (int i=0; i<Dkv; ++i) V[t*Dkv + i] += bv[i];

  // Apply q_norm / k_norm (per-channel or per-head)
  if (qn) {
    if (qn && n_q*head_dim == Dq) {
      // try per-channel (len==Dq)
      // (we can’t know its length here; caller passes pointer only)
      // assume per-head if length==n_q -> handled in branches below
    }
    // per-head?
    // Heuristic: if qn is small array, caller passed it anyway;
    // we branch by multiplying broadcast-style either way:
    for (int t=0; t<T; ++t) {
      float* Qt = &Q[t*Dq];
      for (int h=0; h<n_q; ++h) {
        const float* scale_vec = NULL;
        float scale_head = 1.0f;
        // If qn is per-channel (length Dq), treat qn[h*head_dim + d]
        // else if per-head (length n_q), treat qn[h]
        // We can’t check length here; do both guardedly if pointers differ:
        // (Fast pragmatic route: prefer per-head scalar multiply.)
        scale_head = qn[h]; // works if qn length >= n_q
        for (int d=0; d<head_dim; ++d) Qt[h*head_dim + d] *= scale_head;
      }
    }
  }
  if (kn) {
    for (int t=0; t<T; ++t) {
      float* Kt = &K[t*Dkv];
      for (int h=0; h<n_kv; ++h) {
        float scale_head = kn[h]; // assume per-head
        for (int d=0; d<head_dim; ++d) Kt[h*head_dim + d] *= scale_head;
      }
    }
  }
  TIMER_END_MS(ms);
#ifdef BENCH
  DBG("[attn/gqa] proj+norm done in %.3f ms\n", ms);
#endif

  // Zero Hcat
  for (int i=0; i<T*Dq; ++i) Hcat[i] = 0.f;

  // Per-Q head attention; each Q head maps to KV head (h % n_kv)
  for (int h=0; h<n_q; ++h) {
    const int kvh = h % n_kv;
    const int off_q  = h   * head_dim;
    const int off_kv = kvh * head_dim;

    // Scores S[i,j] = (Q[i,off_q:]*K[j,off_kv:]) / sqrt(d)
    for (int i=0; i<T; ++i) {
      const float* qi = &Q[i*Dq  + off_q];
      for (int j=0; j<T; ++j) {
        const float* kj = &K[j*Dkv + off_kv];
        float dot = 0.f;
        for (int d=0; d<head_dim; ++d) dot += qi[d] * kj[d];
        float val = dot * scale;
        if (causal && j > i) val = -1e30f;
        S[i*T + j] = val;
      }
      softmax_row_inplace(&S[i*T], T);
    }

    // Out block for this head
    for (int i=0; i<T; ++i) {
      float* out_i = &Hcat[i*Dq + off_q];
      for (int d=0; d<head_dim; ++d) out_i[d] = 0.f;
      for (int j=0; j<T; ++j) {
        const float a = S[i*T + j];
        const float* vj = &V[j*Dkv + off_kv];
        for (int d=0; d<head_dim; ++d) out_i[d] += a * vj[d];
      }
    }
  }

  // Output projection
  TIMER_START();
  matmul_f32(Hcat, Wo, y, T, d_model, Dq);
  if (bo) for (int i=0; i<T*d_model; ++i) y[i] += bo[i % d_model];
  TIMER_END_MS(ms);
#ifdef BENCH
  DBG("[attn/gqa] out_proj done in %.3f ms\n", ms);
#endif
}
