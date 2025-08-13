#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "kernels.h"
#include "model.h"

// simple per-row softmax (stable)
static void softmax_rows(float* x, int T, int V){
  for (int t=0; t<T; ++t){
    float* r = &x[(size_t)t*V];
    float m = r[0];
    for (int i=1;i<V;++i) if (r[i]>m) m=r[i];
    double s = 0.0;
    for (int i=0;i<V;++i){ r[i] = expf(r[i]-m); s += r[i]; }
    float inv = (float)(1.0 / (s + 1e-9));
    for (int i=0;i<V;++i) r[i] *= inv;
  }
}

void model_forward_f32(
  const int* ids, int T,
  const QwenConfig* cfg,
  const QwenWeights* w,
  int apply_softmax,
  float* out_logits // [T, vocab]
){
  const int D    = cfg->d_model;
  const int V    = cfg->vocab_size;
  const int L    = cfg->n_layers;
  const int d_ff = cfg->d_ff;
  const int n_q  = cfg->n_q;
  const int n_kv = cfg->n_kv;
  const int dh   = cfg->head_dim;

  // scratch for the whole forward (reuse layer_forward scratch layout)
  // we size for layer 0 shapes (constant for Qwen3)
  const int Dq0  = n_q  * dh;
  const int Dkv0 = n_kv * dh;

  size_t attn_f  = (size_t)T*(Dq0 + 2*Dkv0) + (size_t)T*T + (size_t)T*Dq0;
  size_t temps_f = 5ull * (size_t)T * D;
  float* scratch_attn = (float*)malloc(sizeof(float)*(attn_f + temps_f));
  float* scratch_moe  = (float*)malloc(sizeof(float)*(2ull*(size_t)T*d_ff));
  int*   top_idx      = (int*)  malloc(sizeof(int)*T*cfg->top_k);
  float* top_p        = (float*)malloc(sizeof(float)*T*cfg->top_k);

  // hidden buffer
  float* x = (float*)malloc(sizeof(float)*(size_t)T*D);

  // 1) embedding lookup
  for (int t=0; t<T; ++t){
    const int id = ids[t];
    fprintf(stderr, "[model_forward_f32] step %d, token id=%d\n", t,  id);
    if (id < 0 || id >= V){ fprintf(stderr,"token id %d out of range 0..%d\n", id, V-1); exit(1); }
    memcpy(&x[(size_t)t*D], &w->tok_emb[(size_t)id*D], sizeof(float)*(size_t)D);
  }

  // 2) decoder stack
  for (int l=0; l<L; ++l){
    const QwenLayerWeights* lw = &w->layers[l];
    layer_forward_f32(
      x, T, D,
      lw->rms1_w, 1e-6f,
      lw->Wq, lw->bq, lw->Wk, lw->bk, lw->Wv, lw->bv, lw->Wo, lw->bo,
      lw->q_norm, lw->k_norm,
      n_q, n_kv, dh, cfg->causal, cfg->rope_theta,
      lw->rms2_w, 1e-6f,
      lw->router_w, lw->router_b,
      lw->Wg, lw->bg, lw->Wu, lw->bu, lw->Wd, lw->bd,
      cfg->n_experts, cfg->top_k, d_ff,
      scratch_attn, scratch_moe, top_idx, top_p
    );
  }

  // 3) final norm + head
  float* x_final = (float*)malloc(sizeof(float)*(size_t)T*D);
  rmsnorm_forward_f32(x, w->final_norm_w, T, D, 1e-6f, x_final);

  const float* Wout = w->lm_head ? w->lm_head : w->tok_emb; // tied if lm_head==NULL
  matmul_f32(x_final, Wout, out_logits, T, V, D);

  if (apply_softmax) softmax_rows(out_logits, T, V);

  free(x_final);
  free(x);
  free(scratch_attn); free(scratch_moe); free(top_idx); free(top_p);
}
