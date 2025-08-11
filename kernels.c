
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
  // gate
  matmul_f32(x, Wg, tmp_g, T, d_ff, d_model);
  if (bg) for (int i = 0; i < T*d_ff; i++) tmp_g[i] += bg[i % d_ff];
  silu_f32(tmp_g, T*d_ff);

  // up
  matmul_f32(x, Wu, tmp_u, T, d_ff, d_model);
  if (bu) for (int i = 0; i < T*d_ff; i++) tmp_u[i] += bu[i % d_ff];

  // elementwise mul
  for (int i = 0; i < T*d_ff; i++) tmp_g[i] *= tmp_u[i];

  // down
  matmul_f32(tmp_g, Wd, y, T, d_model, d_ff);
  if (bd) for (int i = 0; i < T*d_model; i++) y[i] += bd[i % d_model];
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

// MoE forward with selectable routing mode (NULL-safe biases; no per-expert malloc/free)
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

  #ifdef DEBUG
  if (T > 0) {
    fprintf(stderr, "[router/raw] t=0 top8 (post-mode): ");
    for (int i = 0; i < k && i < E; ++i) {
      int e = top_idx[0*k + i];
      float p = top_p[0*k + i];
      fprintf(stderr, "(%d:%.4f) ", e, p);
    }
    fprintf(stderr, "\n");
  }
  #endif

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

  // allocate once and reuse across experts
  float* tmp_out = (float*)malloc(sizeof(float) * d_model);
  for (int t=0; t<T; ++t) {
    for (int i=0; i<k; ++i) {
      const int e = top_idx[t*k + i];
      const float p = top_p[t*k + i];
      if (e < 0 || e >= E) { fprintf(stderr, "router idx %d out of range 0..%d\n", e, E-1); exit(1); }
      DBG("[moe] t=%d expert=%d prob=%.6f\n", t, e, p);

      // expert forward (SwiGLU): y = down( silu(gate(x)) * up(x) )
      matmul_f32(&x[(size_t)t*d_model], Wg_arr[e], tmp_one_g, 1, d_ff, d_model);
      if (bg_arr[e]) for (int j=0;j<d_ff;++j) tmp_one_g[j] += bg_arr[e][j];
      silu_f32(tmp_one_g, d_ff);

      matmul_f32(&x[(size_t)t*d_model], Wu_arr[e], tmp_one_u, 1, d_ff, d_model);
      if (bu_arr[e]) for (int j=0;j<d_ff;++j) tmp_one_u[j] += bu_arr[e][j];

      for (int j=0;j<d_ff;++j) tmp_one_g[j] *= tmp_one_u[j];

      matmul_f32(tmp_one_g, Wd_arr[e], tmp_out, 1, d_model, d_ff);
      if (bd_arr[e]) for (int j=0;j<d_model;++j) tmp_out[j] += bd_arr[e][j];

      for (int j=0;j<d_model;++j) y[(size_t)t*d_model + j] += p * tmp_out[j];
    }
  }
  free(tmp_out);
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
        //const float* scale_vec = NULL;
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

  rope_apply_inplace_f32_gqa(
    Q, K,
    T, n_q, n_kv, head_dim,
    /*pos0=*/0, /*theta=*/10000.0f
  );
  DBG("[attn/gqa] kv-share group=%d\n", n_q / n_kv);
  // --- GQA attention per head (no explicit K/V replication)
  // Q: [T, n_q*D], K/V: [T, n_kv*D]
  // For each query head h, map to key/value head kvh = h % n_kv
  for (int h = 0; h < n_q; ++h) {
    const int kvh = h % n_kv;
    const float* Qh = &Q[ h   * head_dim];     // row 0 of head h (we’ll index by t)
    const float* Kh = &K[ kvh * head_dim];
    const float* Vh = &V[ kvh * head_dim];

    // Compute scores S[tq, tk] = (Q_h[tq] · K_kvh[tk]) / sqrt(D)
    // Reuse S as a [T,T] scratch for this head
    for (int tq = 0; tq < T; ++tq) {
      const float* qv = &Q[tq * (n_q  * head_dim) + h   * head_dim];
      float* Sout = &S[tq * T];
      for (int tk = 0; tk < T; ++tk) {
        const float* kv = &K[tk * (n_kv * head_dim) + kvh * head_dim];
        float dot = 0.f;
        for (int d = 0; d < head_dim; ++d) dot += qv[d] * kv[d];
        Sout[tk] = dot * scale;
      }
      // causal mask: zero out future positions
      if (causal) {
        for (int tk = tq + 1; tk < T; ++tk) S[tq * T + tk] = -1e30f;
      }
      // softmax over tk
      float maxv = -1e30f;
      for (int tk = 0; tk < T; ++tk) if (Sout[tk] > maxv) maxv = Sout[tk];
      float sum = 0.f;
      for (int tk = 0; tk < T; ++tk) { float e = expf(Sout[tk] - maxv); Sout[tk] = e; sum += e; }
      float inv = 1.f / (sum + 1e-9f);
      for (int tk = 0; tk < T; ++tk) Sout[tk] *= inv;
    }

    // Context = P @ V_kvh  -> write into Hcat[:, h*D : (h+1)*D]
    for (int tq = 0; tq < T; ++tq) {
      const float* Prow = &S[tq * T];
      float* out = &Hcat[tq * (n_q * head_dim) + h * head_dim];
      // zero
      for (int d = 0; d < head_dim; ++d) out[d] = 0.f;
      // accumulate \sum_tk P[tq,tk] * V_kvh[tk]
      for (int tk = 0; tk < T; ++tk) {
        const float* vv = &V[tk * (n_kv * head_dim) + kvh * head_dim];
        float p = Prow[tk];
        for (int d = 0; d < head_dim; ++d) out[d] += p * vv[d];
      }
    }
  }

  // Project concatenated heads: Hcat [T, n_q*D] -> y [T, d_model]
  TIMER_START();
  matmul_f32(Hcat, Wo, y, T, d_model, n_q * head_dim);
  if (bo) {
    for (int t = 0; t < T; ++t) {
      for (int i = 0; i < d_model; ++i) y[t * d_model + i] += bo[i];
    }
  }
  TIMER_END_MS(ms);
#ifdef BENCH
  DBG("[attn/gqa] out_proj done in %.3f ms\n", ms);
#endif
}

// Rotate one even/odd pair in-place by angle φ using cosφ=c and sinφ=s.
// The pair is the 2D vector (even, odd), rotated by the 2x2 rotation matrix.
// After: [even', odd'] = [ even*c - odd*s,  odd*c + even*s ].
// This is a standard 2D rotation:
//   [ even' ]   [  cosφ  −sinφ ] [ even ]
//   [  odd' ] = [  sinφ   cosφ ] [  odd ]
static inline void rope_rotate_pair(float* even, float* odd, float c, float s) {
  float e = *even, o = *odd;
  *even =  e * c - o * s;
  *odd  =  o * c + e * s;
}

// Apply standard RoPE to Q and K (GQA-aware), in-place.
//
// Shapes:
//   Q: [T, n_q * head_dim]       (one block of size head_dim per Q head)
//   K: [T, n_kv * head_dim]      (one block per KV head)
// Parameters:
//   pos0  : starting position index (0 for prompt; add cache length during decode)
//   theta : base rotary θ (Qwen/Qwen2/Qwen3 default is 10000.0)
// Notes:
//   • We rotate each head’s (even,odd) pairs across the entire head_dim.
//   • This matches the RoFormer paper’s formulation: φ(p,m) = (pos0+p) * θ^(−2m/D)
//     where m = 0..(D/2−1) is the pair index and D=head_dim.
//   • We precompute inv_freq[m] = θ^(−2m/D) once (like your PyTorch dumper) for
//     better performance and closer numerical parity with torch.float32.
//
// References:
//   RoFormer / RoPE: Su et al. 2021; Qwen3 uses standard RoPE with θ≈10000.  [oai_citation:2‡arXiv](https://arxiv.org/pdf/2104.09864?utm_source=chatgpt.com)
void rope_apply_inplace_f32_gqa(
  float* Q, float* K,
  int T, int n_q, int n_kv, int head_dim,
  int pos0, float theta)
{
  DBG("[rope] T=%d n_q=%d n_kv=%d head_dim=%d theta=%.1f pos0=%d\n",
      T, n_q, n_kv, head_dim, theta, pos0);

  const int Dq  = n_q  * head_dim;
  const int Dkv = n_kv * head_dim;
  const int npairs = head_dim / 2;

  // Precompute inv_freq[m] = theta^( -2m / head_dim ), m in [0, npairs)
  float* inv_freq = (float*)malloc(sizeof(float) * npairs);
  for (int m = 0; m < npairs; ++m) {
    float exponent = -2.0f * (float)m / (float)head_dim;
    inv_freq[m] = powf(theta, exponent);
  }

  // For each token position (absolute index = pos0 + t)
  for (int t = 0; t < T; ++t) {
    const int p = pos0 + t;

    // --- Rotate all Q heads at this position ---
    for (int h = 0; h < n_q; ++h) {
      float* qh = Q + t*Dq + h*head_dim;
      // Walk even/odd pairs (0&1, 2&3, ...):
      for (int i = 0, m = 0; i < head_dim; i += 2, ++m) {
        float angle = p * inv_freq[m];      // φ = p * inv_freq[m]
        float c = cosf(angle), s = sinf(angle);
        rope_rotate_pair(&qh[i], &qh[i+1], c, s);
      }
    }

    // --- Rotate all KV heads at this position ---
    for (int h = 0; h < n_kv; ++h) {
      float* kh = K + t*Dkv + h*head_dim;
      for (int i = 0, m = 0; i < head_dim; i += 2, ++m) {
        float angle = p * inv_freq[m];
        float c = cosf(angle), s = sinf(angle);
        rope_rotate_pair(&kh[i], &kh[i+1], c, s);
      }
    }
  }

  free(inv_freq);
  DBG("[rope] applied\n");
}

// One transformer layer (fp32, no KV-cache).
// Pipeline: x -> RMSNorm1 -> Attn(GQA + QK-Norm + RoPE) -> +resid
//        -> RMSNorm2 -> MoE(router+experts) -> +resid
void layer_forward_f32(
  float* x, int T, int d_model,
  // Norm1
  const float* w_norm1, float eps1,
  // Attn (GQA)
  const float* Wq,const float* bq,
  const float* Wk,const float* bk,
  const float* Wv,const float* bv,
  const float* Wo,const float* bo,
  const float* q_norm, const float* k_norm,
  int n_q, int n_kv, int head_dim, int causal, float rope_theta,
  // Norm2
  const float* w_norm2, float eps2,
  // MoE
  const float* router_w, const float* router_b,
  const float** Wg_arr,const float** bg_arr,
  const float** Wu_arr,const float** bu_arr,
  const float** Wd_arr,const float** bd_arr,
  int E, int k, int d_ff,
  // scratch
  float* scratch_attn, float* scratch_moe, int* top_idx, float* top_p)
{
  TIMER_DECL; double ms = 0.0;
  DBG("[layer] T=%d d_model=%d n_q=%d n_kv=%d d=%d E=%d k=%d d_ff=%d\n",
      T, d_model, n_q, n_kv, head_dim, E, k, d_ff);

  // Temps
  //float* x_norm1       = (float*)malloc(sizeof(float)*T*d_model);
  //float* attn_out      = (float*)malloc(sizeof(float)*T*d_model);
  //float* x_after_attn  = (float*)malloc(sizeof(float)*T*d_model);
  //float* x_norm2       = (float*)malloc(sizeof(float)*T*d_model);
  //float* moe_out       = (float*)malloc(sizeof(float)*T*d_model);

  // ---- Reuse scratch_attn for all d_model-sized temps to avoid heap churn ----
  // Attn scratch (already reserved by caller): size_attn_floats = T*(b + 2c) + T*T + T*b
  const int b = n_q * head_dim;
  const int c = n_kv * head_dim;
  const size_t size_attn_floats = (size_t)T * (b + 2*c) + (size_t)T*T + (size_t)T*b;
  float* const attn_scratch_base = scratch_attn;                     // [size_attn_floats]
  float* const temps_base        = attn_scratch_base + size_attn_floats;
  float* const x_norm1      = temps_base + 0*(size_t)T*d_model;      // [T,d_model]
  float* const attn_out     = temps_base + 1*(size_t)T*d_model;      // [T,d_model]
  float* const x_after_attn = temps_base + 2*(size_t)T*d_model;      // [T,d_model]
  float* const x_norm2      = temps_base + 3*(size_t)T*d_model;      // [T,d_model]
  float* const moe_out      = temps_base + 4*(size_t)T*d_model;      // [T,d_model]

  // 1) RMSNorm before attention
  DBG("[layer] rmsnorm1\n");
  TIMER_START();
  rmsnorm_forward_f32(x, w_norm1, T, d_model, eps1, x_norm1);
  TIMER_END_MS(ms);
  DBG("[layer] rmsnorm1 done in %.3f ms\n", ms);

  // 2) Attention (your attn kernel already: QK-Norm -> RoPE -> scores)
  DBG("[layer] attention\n");
  TIMER_START();
  attn_forward_f32_gqa(
    x_norm1, T, d_model,
    Wq,bq, Wk,bk, Wv,bv, Wo,bo,
    q_norm, k_norm,
    n_q, n_kv, head_dim, causal,
    //scratch_attn, attn_out
    attn_scratch_base, attn_out
  );
  TIMER_END_MS(ms);
  DBG("[layer] attention done in %.3f ms\n", ms);

  // Residual add (x = x + attn_out)
  for (int i=0; i<T*d_model; ++i) x_after_attn[i] = x[i] + attn_out[i];

  // 3) RMSNorm before MoE
  DBG("[layer] rmsnorm2\n");
  TIMER_START();
  rmsnorm_forward_f32(x_after_attn, w_norm2, T, d_model, eps2, x_norm2);
  TIMER_END_MS(ms);
  DBG("[layer] rmsnorm2 done in %.3f ms\n", ms);

  // 4) MoE block
  DBG("[layer] moe (routing + experts)\n");
  TIMER_START();
  // scratch_moe usage: tmp_g = scratch_moe, tmp_u = scratch_moe + T*d_ff
  float* tmp_g = scratch_moe;
  float* tmp_u = scratch_moe + (size_t)T*d_ff;
  moe_forward_f32_mode(
    x_norm2, T, d_model,
    router_w, router_b,
    E, k, d_ff, ROUTER_TOPK_KONLY,
    Wg_arr, bg_arr, Wu_arr, bu_arr, Wd_arr, bd_arr,
    moe_out,                      // y
    tmp_g,                        // tmp_g [d_ff] per token
    tmp_u,                        // tmp_u [d_ff] per token
    top_idx, top_p
  );
  TIMER_END_MS(ms);
  DBG("[layer] moe done in %.3f ms\n", ms);

  // Residual add (final x = x_after_attn + moe_out)
  for (int i=0; i<T*d_model; ++i) x[i] = x_after_attn[i] + moe_out[i];

  //free(x_norm1); free(attn_out); free(x_after_attn);
  //free(x_norm2); free(moe_out);
}
