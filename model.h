#ifndef MODEL_H
#define MODEL_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
  // model sizes
  int d_model;
  int n_layers;
  int head_dim;      // per Q head
  int n_q;           // Q heads per layer (assumed constant across layers)
  int n_kv;          // KV heads per layer
  int d_ff;          // MoE FFN hidden size per expert
  int n_experts;     // total experts per layer
  int top_k;         // top-k routing
  int vocab_size;
  int causal;        // 0/1
  float rope_theta;
} QwenConfig;

typedef struct {
  // attention
  const float* Wq; const float* bq;
  const float* Wk; const float* bk;
  const float* Wv; const float* bv;
  const float* Wo; const float* bo;
  const float* q_norm; // length = head_dim
  const float* k_norm; // length = head_dim
  // norms
  const float* rms1_w;
  const float* rms2_w;
  // router
  const float* router_w; // [E, d_model]
  const float* router_b; // [E] or NULL
  // experts (arrays of E pointers; bias pointers may be NULL)
  const float** Wg; const float** bg; // [E][d_ff, d_model], [E][d_ff]
  const float** Wu; const float** bu; // [E][d_ff, d_model], [E][d_ff]
  const float** Wd; const float** bd; // [E][d_model, d_ff], [E][d_model]
} QwenLayerWeights;

typedef struct {
  // token embedding (required)
  const float* tok_emb;     // [vocab, d_model]
  // final norm (required; Qwen: "model.norm.weight"/"model.final_layernorm.weight")
  const float* final_norm_w; // [d_model]
  // output head (optional, falls back to tok_emb if NULL)
  const float* lm_head;     // [vocab, d_model] or NULL (tied)
  // layers
  QwenLayerWeights* layers; // [n_layers]
} QwenWeights;

// Full forward: ids -> embed -> N layers -> final norm -> logits (optionally softmax)
void model_forward_f32(
  const int* ids, int T,
  const QwenConfig* cfg,
  const QwenWeights* w,
  int apply_softmax,   // 0: logits, 1: softmax probs
  float* out           // [T, vocab]
);

#endif // MODEL_H
