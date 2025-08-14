// test_model_trace.c
// End-to-end verification against verify_greedy.py dumps.
// Usage: ./test_model_trace <all.bin> <outbase> <steps>
//
// Changes vs older version:
//  1) No Python-provided top-k routing: routing computed locally (top_k,
//     softmax over top-k scores only), exactly like Raschka's reference.
//  2) No string/file handling inside the forward loop.
//  3) Everything lives in this one file for easy build/run.
//
// RoPE: exact Raschka parity via dumped cos/sin:
//   x' = x * cos + concat(-x[d/2:], x[:d/2]) * sin
// where cos/sin are loaded once as [context, head_dim] (float32).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>

#include "io.h"     // bin_load / bin_find / npy_load_* already available
#include "utils.h"  // max_abs_diff, etc.

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

// -----------------------------
// Minimal model "header" inline
// -----------------------------
typedef struct {
  int d_model;
  int n_layers;
  int head_dim;
  int n_q;
  int n_kv;
  int d_ff;         // per expert
  int n_experts;    // per layer
  int top_k;        // router top-k
  int vocab_size;
  int causal;       // 0/1
  float rope_theta; // (unused when using dumped cos/sin; kept for clarity)
} QwenConfig;

typedef struct {
  // attention
  const float* Wq; const float* bq;
  const float* Wk; const float* bk;
  const float* Wv; const float* bv;
  const float* Wo; const float* bo;
  const float* q_norm; // [head_dim] or NULL
  const float* k_norm; // [head_dim] or NULL
  // norms
  const float* rms1_w; // [d_model]
  const float* rms2_w; // [d_model]
  // router
  const float* router_w; // [E, d_model]
  const float* router_b; // [E] or NULL
  // experts (arrays of E pointers; biases are NULL for Qwen3/A3B)
  const float** Wg; // [E][d_ff, d_model]
  const float** Wu; // [E][d_ff, d_model]
  const float** Wd; // [E][d_model, d_ff]
} QwenLayerWeights;

typedef struct {
  // embedding
  const float* tok_emb;      // [vocab, d_model]
  // final norm
  const float* final_norm_w; // [d_model]
  // output head (NULL => tied with tok_emb)
  const float* lm_head;      // [vocab, d_model] or NULL
  // layers
  QwenLayerWeights* layers;  // [n_layers]
} QwenWeights;

// -----------------------------
// Helpers
// -----------------------------
static void softmax_rows(float* mat, int rows, int cols) {
  for (int r = 0; r < rows; ++r) {
    float* row = &mat[r * cols];
    float maxv = row[0];
    for (int c = 1; c < cols; ++c) if (row[c] > maxv) maxv = row[c];
    float sum = 0.f;
    for (int c = 0; c < cols; ++c) { row[c] = expf(row[c] - maxv); sum += row[c]; }
    float inv = 1.0f / (sum + 1e-9f);
    for (int c = 0; c < cols; ++c) row[c] *= inv;
  }
}

// top-k over a single vector of length E; outputs top_k indices+scores (descending).
// Simple O(E * top_k) selection; fine for test path.
static void topk_desc(const float* x, int E, int top_k, int* out_idx, float* out_val) {
  for (int i = 0; i < top_k; ++i) { out_val[i] = -INFINITY; out_idx[i] = -1; }
  for (int e = 0; e < E; ++e) {
    float v = x[e];
    int pos = -1;
    for (int i = 0; i < top_k; ++i) { if (v > out_val[i]) { pos = i; break; } }
    if (pos >= 0) {
      for (int j = top_k - 1; j > pos; --j) { out_val[j] = out_val[j-1]; out_idx[j] = out_idx[j-1]; }
      out_val[pos] = v; out_idx[pos] = e;
    }
  }
}

// Raschka-consistent RoPE on flat [T, H*d] using dumped cos/sin of shape [ctx, d].
static void apply_rope_from_cos_sin(
  float* X, int T, int n_heads, int head_dim,
  const float* cos, const float* sin, int cos_T, int cos_D)
{
  const int d2 = head_dim / 2;
  if (head_dim % 2) { fprintf(stderr,"[rope] head_dim must be even\n"); exit(1); }
  if (cos_D != head_dim || cos_T < T) {
    fprintf(stderr,"[rope] cos/sin shape mismatch: have [%d,%d], need at least [%d,%d]\n",
            cos_T, cos_D, T, head_dim); exit(1);
  }
  const size_t Hd = (size_t)n_heads * (size_t)head_dim;
  for (int t = 0; t < T; ++t) {
    const float* ct = &cos[(size_t)t * cos_D];
    const float* st = &sin[(size_t)t * cos_D];
    for (int h = 0; h < n_heads; ++h) {
      float* v = &X[(size_t)t * Hd + (size_t)h * head_dim];
      for (int i = 0; i < d2; ++i) {
        float c = ct[i], s = st[i];
        float x_even = v[i], x_odd = v[i + d2];
        float new_even = x_even * c - x_odd * s; // (x * cos + rot * sin) form
        float new_odd  = x_odd  * c + x_even * s;
        v[i]      = new_even;
        v[i+d2]   = new_odd;
      }
    }
  }
}

// -----------------------------
// Attention + MoE (single layer)
// -----------------------------
static void attention_forward_f32(
  const float* x, int T, int d_model,
  const float* Wq, const float* bq,
  const float* Wk, const float* bk,
  const float* Wv, const float* bv,
  const float* Wo, const float* bo,
  const float* qn, const float* kn, // NULL if absent
  int n_q, int n_kv, int head_dim, int causal,
  const float* cos, const float* sin, int cos_T, int cos_D, // dumped RoPE
  float* scratch, float* y_out)
{
  const float scale = 1.0f / sqrtf((float)head_dim);
  const int Dq  = n_q  * head_dim;
  const int Dkv = n_kv * head_dim;

  float* Q    = scratch;            // [T, Dq]
  float* K    = Q + T*Dq;           // [T, Dkv]
  float* V    = K + T*Dkv;          // [T, Dkv]
  float* S    = V + T*Dkv;          // [T, T] (reused row-wise)
  float* Hcat = S + T*T;            // [T, Dq]

  // Projections
  matmul_f32(x, Wq, Q, T, Dq,  d_model);
  matmul_f32(x, Wk, K, T, Dkv, d_model);
  matmul_f32(x, Wv, V, T, Dkv, d_model);
  if (bq) for (int t=0; t<T; ++t) for (int i=0; i<Dq;  ++i) Q[t*Dq  + i] += bq[i];
  if (bk) for (int t=0; t<T; ++t) for (int i=0; i<Dkv; ++i) K[t*Dkv + i] += bk[i];
  if (bv) for (int t=0; t<T; ++t) for (int i=0; i<Dkv; ++i) V[t*Dkv + i] += bv[i];

  // QK RMSNorm per head (Raschka-compatible RMS over last dim, learned scale)
  const float eps = 1e-6f;
  if (qn) {
    for (int t=0; t<T; ++t) {
      float* Qt = &Q[t*Dq];
      for (int h=0; h<n_q; ++h) {
        float* v = &Qt[h*head_dim];
        float msq=0.f; for(int d=0; d<head_dim; ++d){ float z=v[d]; msq+=z*z; }
        float inv = 1.0f / sqrtf(msq/(float)head_dim + eps);
        for (int d=0; d<head_dim; ++d) v[d] = (v[d]*inv) * qn[d];
      }
    }
  }
  if (kn) {
    for (int t=0; t<T; ++t) {
      float* Kt = &K[t*Dkv];
      for (int h=0; h<n_kv; ++h) {
        float* v = &Kt[h*head_dim];
        float msq=0.f; for(int d=0; d<head_dim; ++d){ float z=v[d]; msq+=z*z; }
        float inv = 1.0f / sqrtf(msq/(float)head_dim + eps);
        for (int d=0; d<head_dim; ++d) v[d] = (v[d]*inv) * kn[d];
      }
    }
  }

  // RoPE using dumped cos/sin (exact parity)
  apply_rope_from_cos_sin(Q, T, n_q,  head_dim, cos, sin, cos_T, cos_D);
  apply_rope_from_cos_sin(K, T, n_kv, head_dim, cos, sin, cos_T, cos_D);

  // GQA: each block of (n_q / n_kv) Q-heads uses the same K,V head
  const int group = n_q / n_kv;

  // scores + softmax + context
  for (int h=0; h<n_q; ++h) {
    const int kvh = h / group;
    for (int tq=0; tq<T; ++tq) {
      const float* qv = &Q[tq*Dq + h*head_dim];
      float* Sout = &S[tq*T];
      for (int tk=0; tk<T; ++tk) {
        const float* kv = &K[tk*Dkv + kvh*head_dim];
        float dot=0.f; for (int d=0; d<head_dim; ++d) dot += qv[d]*kv[d];
        Sout[tk] = dot * scale;
      }
      if (causal) for (int tk=tq+1; tk<T; ++tk) S[tq*T + tk] = -INFINITY;
      // softmax row
      float m=Sout[0]; for(int i=1;i<T;++i) if(Sout[i]>m) m=Sout[i];
      float ssum=0.f; for(int i=0;i<T;++i){ float e = expf(Sout[i]-m); Sout[i]=e; ssum+=e; }
      float inv=1.f/(ssum+1e-9f);
      for(int i=0;i<T;++i) Sout[i]*=inv;
    }
    // context mix
    for (int tq=0; tq<T; ++tq) {
      const float* Prow = &S[tq*T];
      float* out = &Hcat[tq*Dq + h*head_dim];
      for (int d=0; d<head_dim; ++d) out[d]=0.f;
      for (int tk=0; tk<T; ++tk) {
        const float* vv = &V[tk*Dkv + kvh*head_dim];
        const float p = Prow[tk];
        for (int d=0; d<head_dim; ++d) out[d] += p * vv[d];
      }
    }
  }

  // output projection
  matmul_f32(Hcat, Wo, y_out, T, d_model, Dq);
  if (bo) for (int t=0; t<T; ++t) for (int i=0; i<d_model; ++i) y_out[t*d_model + i] += bo[i];
}

// One layer: norm1 -> attn -> +res -> norm2 -> MoE -> +res
static void layer_forward_f32(
  float* x, int T, int d_model,
  const QwenLayerWeights* lw,
  int n_q, int n_kv, int head_dim, int causal,
  const float* cos, const float* sin, int cos_T, int cos_D,
  int n_experts, int top_k, int d_ff,
  float* scratch_attn, float* scratch_moe, int* tmp_idx, float* tmp_val)
{
  // temp buffers (reuse scratch_attn tail as temp D vectors)
  float* x_norm1  = scratch_attn + (size_t)T*(n_q*head_dim + 2*n_kv*head_dim) + (size_t)T*T + (size_t)T*(n_q*head_dim);
  float* attn_out = x_norm1 + (size_t)T*d_model;
  float* x_after  = attn_out + (size_t)T*d_model;
  float* x_norm2  = x_after  + (size_t)T*d_model;

  // 1) norm1
  rmsnorm_forward_f32(x, lw->rms1_w, T, d_model, 1e-6f, x_norm1);

  // 2) attention
  attention_forward_f32(
    x_norm1, T, d_model,
    lw->Wq, lw->bq, lw->Wk, lw->bk, lw->Wv, lw->bv, lw->Wo, lw->bo,
    lw->q_norm, lw->k_norm,
    n_q, n_kv, head_dim, causal,
    cos, sin, cos_T, cos_D,
    scratch_attn, attn_out
  );

  // residual
  for (int i=0;i<T*d_model;++i) x_after[i] = x[i] + attn_out[i];

  // 3) norm2
  rmsnorm_forward_f32(x_after, lw->rms2_w, T, d_model, 1e-6f, x_norm2);

  // 4) router logits
  // logits[t,:] = x_norm2[t,:] @ router_w^T + router_b
  float* logits = scratch_moe;                    // [T, E]
  matmul_f32(x_norm2, lw->router_w, logits, T, n_experts, d_model);
  if (lw->router_b) {
    for (int t=0;t<T;++t) for (int e=0;e<n_experts;++e) logits[t*n_experts + e] += lw->router_b[e];
  }

  // 5) expert path with local top-k routing (softmax over top-k only)
  float* moe_out = scratch_moe + (size_t)T * n_experts; // [T, d_model]
  for (int i=0;i<T*d_model;++i) moe_out[i] = 0.f;

  float* tmp_g = (float*)malloc(sizeof(float)*d_ff);
  float* tmp_u = (float*)malloc(sizeof(float)*d_ff);
  float* tmp_y = (float*)malloc(sizeof(float)*d_model);

  for (int t=0; t<T; ++t) {
    const float* log_t = &logits[(size_t)t * n_experts];
    topk_desc(log_t, n_experts, top_k, tmp_idx, tmp_val);
    // softmax over top-k scores
    float maxv = tmp_val[0];
    for (int i=1;i<top_k;++i) if (tmp_val[i] > maxv) maxv = tmp_val[i];
    float sum = 0.f;
    for (int i=0;i<top_k;++i){ tmp_val[i] = expf(tmp_val[i] - maxv); sum += tmp_val[i]; }
    float inv = 1.0f / (sum + 1e-9f);
    for (int i=0;i<top_k;++i) tmp_val[i] *= inv;

    for (int i=0;i<top_k;++i){
      int e = tmp_idx[i];
      float p = tmp_val[i];
      // g = silu( x_norm2[t] @ Wg[e]^T )
      matmul_f32(&x_norm2[(size_t)t*d_model], lw->Wg[e], tmp_g, 1, d_ff, d_model);
      silu_f32(tmp_g, d_ff);
      // u = x_norm2[t] @ Wu[e]^T
      matmul_f32(&x_norm2[(size_t)t*d_model], lw->Wu[e], tmp_u, 1, d_ff, d_model);
      // h = g * u
      for (int q=0;q<d_ff;++q) tmp_g[q] *= tmp_u[q];
      // y = h @ Wd[e]^T
      matmul_f32(tmp_g, lw->Wd[e], tmp_y, 1, d_model, d_ff);
      // accumulate
      for (int q=0;q<d_model;++q) moe_out[(size_t)t*d_model + q] += p * tmp_y[q];
    }
  }

  // 6) residual to x
  for (int i=0;i<T*d_model;++i) x[i] = x_after[i] + moe_out[i];

  free(tmp_g); free(tmp_u); free(tmp_y);
}

// -----------------------------
// Full forward (no file I/O)
// -----------------------------
static void model_forward_f32(
  const int* ids, int T,
  const QwenConfig* cfg,
  const QwenWeights* w,
  const float* cos, const float* sin, int cos_T, int cos_D,
  int apply_softmax,
  float* out_last // [1, vocab] (only last token logits/probs)
){
  const int D = cfg->d_model, V = cfg->vocab_size, L = cfg->n_layers;
  const int Dq0 = cfg->n_q * cfg->head_dim;
  const int Dkv0 = cfg->n_kv * cfg->head_dim;

  size_t attn_f  = (size_t)T*(Dq0 + 2*Dkv0) + (size_t)T*T + (size_t)T*Dq0;
  size_t temps_f = 4ull * (size_t)T * D;
  float* scratch_attn = (float*)malloc(sizeof(float)*(attn_f + temps_f));
  float* scratch_moe  = (float*)malloc(sizeof(float)*((size_t)T*cfg->n_experts + (size_t)T*D));
  int*   tmp_idx      = (int*)  malloc(sizeof(int)*cfg->top_k);
  float* tmp_val      = (float*)malloc(sizeof(float)*cfg->top_k);

  float* x = (float*)malloc(sizeof(float)*(size_t)T*D);

  // embedding lookup
  for (int t=0; t<T; ++t) {
    int id = ids[t];
    memcpy(&x[(size_t)t*D], &w->tok_emb[(size_t)id*D], sizeof(float)*(size_t)D);
  }

  // stack
  for (int l=0; l<L; ++l) {
    layer_forward_f32(
      x, T, D, &w->layers[l],
      cfg->n_q, cfg->n_kv, cfg->head_dim, cfg->causal,
      cos, sin, cos_T, cos_D,
      cfg->n_experts, cfg->top_k, cfg->d_ff,
      scratch_attn, scratch_moe, tmp_idx, tmp_val
    );
  }

  // final norm + head
  float* x_final = (float*)malloc(sizeof(float)*(size_t)T*D);
  rmsnorm_forward_f32(x, w->final_norm_w, T, D, 1e-6f, x_final);

  const float* Wout = w->lm_head ? w->lm_head : w->tok_emb;
  // Only compute last row logits into out_last (saves time/mem)
  // out_last = x_final[T-1,:] @ Wout^T
  // Reuse matmul on a single row for simplicity:
  matmul_f32(&x_final[(size_t)(T-1)*D], Wout, out_last, 1, V, D);

  if (apply_softmax) softmax_rows(out_last, 1, V);

  free(x_final);
  free(x);
  free(scratch_attn); free(scratch_moe); free(tmp_idx); free(tmp_val);
}

// -----------------------------
// Weight loading (one-time)
// -----------------------------
static TensorBin* need(BinFile* b, const char* k){
  TensorBin* t = bin_find(b,k);
  if(!t){ fprintf(stderr,"missing %s\n", k); exit(1); }
  return t;
}
static TensorBin* maybe(BinFile* b, const char* k){ return bin_find(b,k); }

// Fill QwenWeights/QwenConfig from all.bin (Qwen/Qwen3-30B-A3B layout).
// Only strings appear here (one-time), never inside the forward loops.
static void load_all_weights(BinFile* bf, QwenConfig* cfg, QwenWeights* w) {
  // Infer sizes (consistent with your verified setup)
  // d_model, vocab, head_dim, n_q, n_kv, n_layers, d_ff, n_experts, top_k
  TensorBin* Wq0 = need(bf,"model.layers.0.self_attn.q_proj.weight");
  TensorBin* Wk0 = need(bf,"model.layers.0.self_attn.k_proj.weight");
  TensorBin* emb = need(bf,"model.embed_tokens.weight");
  TensorBin* norm = maybe(bf,"model.norm.weight");
  if (!norm) norm = need(bf,"model.final_layernorm.weight");

  cfg->d_model   = Wq0->shape[1];
  cfg->head_dim  = maybe(bf,"model.layers.0.self_attn.q_norm.weight")
                   ? maybe(bf,"model.layers.0.self_attn.q_norm.weight")->shape[0]
                   : (Wq0->shape[0] / 32);
  cfg->n_q       = Wq0->shape[0] / cfg->head_dim;
  cfg->n_kv      = Wk0->shape[0] / cfg->head_dim;
  cfg->vocab_size= emb->shape[0];
  cfg->n_layers  = 0;
  // count layers by probing L until miss
  for (;;) {
    char key[256];
    snprintf(key,sizeof(key),"model.layers.%d.self_attn.q_proj.weight", cfg->n_layers);
    if (!bin_find(bf,key)) break;
    cfg->n_layers++;
  }

  // MoE sizes
  // infer n_experts and d_ff from first layer’s experts
  int E = 0, d_ff = -1;
  for (;;) {
    char k_down[256];
    snprintf(k_down,sizeof(k_down),"model.layers.0.mlp.experts.%d.down_proj.weight", E);
    TensorBin* t = bin_find(bf,k_down);
    if (!t) break;
    d_ff = t->shape[1];
    E++;
  }
  if (E==0 || d_ff<=0) { fprintf(stderr,"infer MoE sizes failed\n"); exit(1); }
  cfg->n_experts = E;
  cfg->d_ff      = d_ff;
  // infer top_k from router weight vs Python dump (use common default 8)
  cfg->top_k     = 8;
  cfg->causal    = 1;
  cfg->rope_theta= 10000000.f; // unused with dumped cos/sin (kept for completeness)

  w->tok_emb      = (const float*)emb->data;
  w->final_norm_w = (const float*)norm->data;
  // head: use lm_head if present, else tie
  TensorBin* head = maybe(bf,"lm_head.weight");
  w->lm_head = head ? (const float*)head->data : NULL;

  // allocate layers
  w->layers = (QwenLayerWeights*)calloc((size_t)cfg->n_layers, sizeof(QwenLayerWeights));

  for (int L=0; L<cfg->n_layers; ++L) {
    QwenLayerWeights* lw = &w->layers[L];
    char k[256];

    // attention weights/biases
    snprintf(k,sizeof(k),"model.layers.%d.self_attn.q_proj.weight",L);
    lw->Wq = (const float*)need(bf,k)->data;
    snprintf(k,sizeof(k),"model.layers.%d.self_attn.k_proj.weight",L);
    lw->Wk = (const float*)need(bf,k)->data;
    snprintf(k,sizeof(k),"model.layers.%d.self_attn.v_proj.weight",L);
    lw->Wv = (const float*)need(bf,k)->data;
    snprintf(k,sizeof(k),"model.layers.%d.self_attn.o_proj.weight",L);
    lw->Wo = (const float*)need(bf,k)->data;

    snprintf(k,sizeof(k),"model.layers.%d.self_attn.q_proj.bias",L);
    lw->bq = (maybe(bf,k) && maybe(bf,k)->dtype==0) ? (const float*)maybe(bf,k)->data : NULL;
    snprintf(k,sizeof(k),"model.layers.%d.self_attn.k_proj.bias",L);
    lw->bk = (maybe(bf,k) && maybe(bf,k)->dtype==0) ? (const float*)maybe(bf,k)->data : NULL;
    snprintf(k,sizeof(k),"model.layers.%d.self_attn.v_proj.bias",L);
    lw->bv = (maybe(bf,k) && maybe(bf,k)->dtype==0) ? (const float*)maybe(bf,k)->data : NULL;
    snprintf(k,sizeof(k),"model.layers.%d.self_attn.o_proj.bias",L);
    lw->bo = (maybe(bf,k) && maybe(bf,k)->dtype==0) ? (const float*)maybe(bf,k)->data : NULL;

    // qk norm (auto-present for Qwen3 A3B)
    TensorBin* qn = maybe(bf, (snprintf(k,sizeof(k),"model.layers.%d.self_attn.q_norm.weight",L), k));
    TensorBin* kn = maybe(bf, (snprintf(k,sizeof(k),"model.layers.%d.self_attn.k_norm.weight",L), k));
    lw->q_norm = qn ? (const float*)qn->data : NULL;
    lw->k_norm = kn ? (const float*)kn->data : NULL;

    // norms
    TensorBin* n1 = maybe(bf, (snprintf(k,sizeof(k),"model.layers.%d.input_layernorm.weight",L), k));
    if (!n1) { snprintf(k, sizeof(k), "model.layers.%d.rms_1.weight", L); n1 = need(bf, k); }
    lw->rms1_w = (const float*)n1->data;

    TensorBin* n2 = maybe(bf, (snprintf(k,sizeof(k),"model.layers.%d.post_attention_layernorm.weight",L), k));
    if (!n2) { snprintf(k, sizeof(k), "model.layers.%d.rms_2.weight", L); n2 = need(bf, k); }
    lw->rms2_w = (const float*)n2->data;

    // router
    TensorBin* RW = maybe(bf, (snprintf(k,sizeof(k),"model.layers.%d.mlp.gate.weight",L), k));
    if (!RW) { snprintf(k, sizeof(k), "model.layers.%d.mlp.router.gate.weight", L); RW = need(bf, k); }
    lw->router_w = (const float*)RW->data;

    TensorBin* RB = maybe(bf, (snprintf(k,sizeof(k),"model.layers.%d.mlp.gate.bias",L), k));
    lw->router_b = (RB && RB->dtype==0) ? (const float*)RB->data : NULL;

    // experts arrays
    lw->Wg = (const float**)calloc((size_t)cfg->n_experts, sizeof(float*));
    lw->Wu = (const float**)calloc((size_t)cfg->n_experts, sizeof(float*));
    lw->Wd = (const float**)calloc((size_t)cfg->n_experts, sizeof(float*));
    for (int e=0;e<cfg->n_experts;++e){
      snprintf(k,sizeof(k),"model.layers.%d.mlp.experts.%d.gate_proj.weight",L,e);
      lw->Wg[e] = (const float*)need(bf,k)->data;
      snprintf(k,sizeof(k),"model.layers.%d.mlp.experts.%d.up_proj.weight",L,e);
      lw->Wu[e] = (const float*)need(bf,k)->data;
      snprintf(k,sizeof(k),"model.layers.%d.mlp.experts.%d.down_proj.weight",L,e);
      lw->Wd[e] = (const float*)need(bf,k)->data;
    }
  }
}

// -----------------------------
// Main
// -----------------------------
int main(int argc, char** argv) {
  if (argc < 4) {
    fprintf(stderr,"Usage: %s <all.bin> <outbase> <steps>\n", argv[0]);
    return 1;
  }
  const char* allfile = argv[1];
  const char* outbase = argv[2];
  int steps = atoi(argv[3]);

  // load weights
  BinFile* bf = bin_load(allfile);
  if (!bf){ fprintf(stderr,"bin load fail\n"); return 1; }

  QwenConfig cfg; QwenWeights w;
  load_all_weights(bf, &cfg, &w);
  printf("[model] d_model=%d n_layers=%d head_dim=%d n_q=%d n_kv=%d vocab=%d\n",
         cfg.d_model, cfg.n_layers, cfg.head_dim, cfg.n_q, cfg.n_kv, cfg.vocab_size);

  // load cos/sin (dumped once by verify_greedy.py)
  char cpath[512], spath[512];
  snprintf(cpath,sizeof(cpath),"%s.cos.npy", outbase);
  snprintf(spath,sizeof(spath),"%s.sin.npy", outbase);
  NpyArray* COS = npy_load_float32(cpath);
  NpyArray* SIN = npy_load_float32(spath);
  if (!COS || !SIN) { fprintf(stderr,"missing cos/sin npy\n"); return 1; }
  const float* cos = COS->data; const float* sin = SIN->data;
  int cos_T = COS->shape[0], cos_D = COS->shape[1];

  // load reference ids/logits/probs
  char ipath[512], lpath[512], ppath[512];
  snprintf(ipath,sizeof(ipath),"%s.ids.npy", outbase);
  snprintf(lpath,sizeof(lpath),"%s.logits.npy", outbase);
  snprintf(ppath,sizeof(ppath),"%s.probs.npy",  outbase);

  NpyArrayI32* IDS = npy_load_int32(ipath);
  NpyArray* LREF   = npy_load_float32(lpath);
  NpyArray* PREF   = npy_load_float32(ppath);
  if (!IDS || !LREF || !PREF) { fprintf(stderr,"missing ids/logits/probs dumps\n"); return 1; }

  // steps sanity
  if (LREF->shape[0] < steps || PREF->shape[0] < steps) {
    fprintf(stderr,"steps=%d exceeds dump steps (%d)\n", steps, (int)LREF->shape[0]); return 1;
  }
  const int V = cfg.vocab_size;
  if (LREF->shape[1] != V || PREF->shape[1] != V) {
    fprintf(stderr,"vocab mismatch: V=%d vs dumps [%d,%d]\n", V, (int)LREF->shape[1], (int)PREF->shape[1]); return 1;
  }

  // run step-by-step
  int T0 = IDS->shape[0] - steps;           // initial prompt length
  if (T0 <= 0) T0 = 1;                      // verify_greedy default
  float* out_last = (float*)malloc(sizeof(float)*V);

  for (int s=0; s<steps; ++s) {
    int T = T0 + s;
    // run forward on the prefix ids[:T]
    model_forward_f32(
      IDS->data, T,
      &cfg, &w,
      cos, sin, cos_T, cos_D,
      /*apply_softmax=*/0,
      out_last
    );

    // compare logits last row vs ref
    double mad_logits = max_abs_diff(out_last, &LREF->data[(size_t)s*V], V);

    // softmax and compare probs
    softmax_rows(out_last, 1, V);
    double mad_probs  = max_abs_diff(out_last, &PREF->data[(size_t)s*V], V);

    // argmax both
    int a_c = 0, a_r = 0; float mv_c = out_last[0], mv_r = PREF->data[(size_t)s*V+0];
    for (int i=1;i<V;++i){ if(out_last[i]>mv_c){mv_c=out_last[i]; a_c=i;} if(PREF->data[(size_t)s*V+i]>mv_r){mv_r=PREF->data[(size_t)s*V+i]; a_r=i;} }

    printf("[step %d] logits MAD=%.8g  probs MAD=%.8g  argmax=%d  ref=%d\n",
           s, mad_logits, mad_probs, a_c, a_r);
  }

  free(out_last);
  npy_free(COS); npy_free(SIN);
  npy_free_i32(IDS); npy_free(LREF); npy_free(PREF);

  // free layer expert arrays
  for (int L=0; L<cfg.n_layers; ++L) {
    free((void*)w.layers[L].Wg);
    free((void*)w.layers[L].Wu);
    free((void*)w.layers[L].Wd);
  }
  free(w.layers);
  bin_free(bf);
  return 0;
}
