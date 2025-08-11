#include <stdlib.h>
#include <string.h>
#include "model.h"
#include "utils.h"

#ifdef DEBUG
#include <stdio.h>
#define DBG(...) do { fprintf(stderr, __VA_ARGS__); } while(0)
#else
#define DBG(...)
#endif

#ifdef BENCH
#include <time.h>
#define TIMER_START() \
  struct timespec t0, t1; \
  clock_gettime(CLOCK_MONOTONIC, &t0);
#define TIMER_END_MS(ms) \
  clock_gettime(CLOCK_MONOTONIC, &t1); \
  ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + \
       (t1.tv_nsec - t0.tv_nsec) / 1.0e6;
#else
#define TIMER_START()
#define TIMER_END_MS(ms)
#endif

// --- tiny helpers using your kernels ---

// Gather embeddings for ids[0..T-1] -> x[T,d_model]
static void embed_lookup_f32(const float* Wemb, const int* ids,
                             int T, int d_model, float* x) {
  double ms=0.0;
  DBG("[emb] T=%d d_model=%d\n", T, d_model);
  TIMER_START();
  for (int t=0; t<T; ++t) {
    const float* row = Wemb + (size_t)ids[t]*d_model;
    memcpy(x + (size_t)t*d_model, row, sizeof(float)*d_model);
  }
  TIMER_END_MS(ms);
  DBG("[emb] done in %.3f ms\n", ms);
}

// logits[T,vocab] = x[T,d_model] @ Wout[vocab,d_model]^T
static void lm_head_forward_f32(const float* x, const float* Wout,
                                int T, int d_model, int vocab,
                                float* logits) {
  double ms=0.0;
  DBG("[lm_head] T=%d d_model=%d vocab=%d\n", T, d_model, vocab);
  TIMER_START();
  // reuse your matmul kernel: C[M,N] = A[M,K] x B[N,K]^T
  matmul_f32(x, Wout, logits, T, vocab, d_model);
  TIMER_END_MS(ms);
  DBG("[lm_head] done in %.3f ms\n", ms);
}

void model_forward_f32(const QwenConfig* cfg, const QwenWeights* W,
                       const int* ids, int T,
                       float* logits_out) {
  const int D = cfg->d_model;
  const int V = cfg->vocab;

  // Scratch
  float* x         = (float*)malloc(sizeof(float)*T*D);
  float* x_final   = (float*)malloc(sizeof(float)*T*D);
  // attn scratch as in your test_attn: T*(Dq + Dkv + Dkv) + T*T + T*Dq
  const int Dq = cfg->n_q * cfg->head_dim;
  const int Dkv= cfg->n_kv * cfg->head_dim;
  float* scratch_attn = (float*)malloc(sizeof(float) * ((size_t)T*(Dq + Dkv + Dkv) + (size_t)T*T + (size_t)T*Dq));
  // moe scratch: per-token tmp_g/tmp_u reused
  float* scratch_moe  = (float*)malloc(sizeof(float) * (2 * T * cfg->d_ff));
  int*   top_idx      = (int*)malloc(sizeof(int) * T * cfg->top_k);
  float* top_p        = (float*)malloc(sizeof(float) * T * cfg->top_k);

  // 0) Embedding
  embed_lookup_f32(W->Wemb, ids, T, D, x);

  // 1) Decoder stack
  for (int L=0; L<cfg->n_layer; ++L) {
    QwenLayerWeights* LW = &W->layers[L];
    layer_forward_f32(
      x, T, D,
      // Norm1
      LW->w_norm1, cfg->eps,
      // Attn (your attn kernel handles QK-Norm + RoPE internally)
      LW->Wq, LW->bq, LW->Wk, LW->bk, LW->Wv, LW->bv, LW->Wo, LW->bo,
      LW->q_norm, LW->k_norm,
      cfg->n_q, cfg->n_kv, cfg->head_dim, /*causal=*/1, cfg->rope_theta,
      // Norm2
      LW->w_norm2, cfg->eps,
      // MoE
      LW->Wroute, LW->broute,
      LW->Wg, LW->bg, LW->Wu, LW->bu, LW->Wd, LW->bd,
      cfg->n_experts, cfg->top_k, cfg->d_ff,
      // scratch
      scratch_attn, scratch_moe, top_idx, top_p
    );
  }

  // 2) Final RMS norm
  rmsnorm_forward_f32(x, W->w_final, T, D, cfg->eps, x_final);

  // 3) LM head (tie to embeddings if Wout==NULL)
  const float* Wout = (W->Wout ? W->Wout : W->Wemb);
  lm_head_forward_f32(x_final, Wout, T, D, V, logits_out);

  free(x); free(x_final);
  free(scratch_attn); free(scratch_moe); free(top_idx); free(top_p);
}
