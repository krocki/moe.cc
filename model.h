#ifndef MODEL_H
#define MODEL_H

#include "kernels.h"

// Minimal runtime config we need for a full forward (no KV cache)
typedef struct {
  int d_model;       // e.g., 2048
  int n_layer;       // 48
  int n_q;           // 32
  int n_kv;          // 4
  int head_dim;      // 128 (d_model * n_q / Dq)
  int d_ff;          // 768 (Qwen3-30B-A3B MoE FF dim per expert)
  int n_experts;     // 128
  int top_k;         // 8
  int vocab;         // vocab size for lm head
  float eps;         // 1e-6
  float rope_theta;  // read from HF config (often 10000.0)
} QwenConfig;

// Pointers for one layer
typedef struct {
  // Norms
  const float* w_norm1;
  const float* w_norm2;

  // Attention (GQA + QK-Norm)
  const float* Wq; const float* bq;
  const float* Wk; const float* bk;
  const float* Wv; const float* bv;
  const float* Wo; const float* bo;
  const float* q_norm;
  const float* k_norm;

  // MoE router
  const float* Wroute;
  const float* broute;

  // Experts (arrays of length n_experts)
  const float** Wg; const float** bg;
  const float** Wu; const float** bu;
  const float** Wd; const float** bd;
} QwenLayerWeights;

// Whole-model weights
typedef struct {
  const float* Wemb;        // [vocab, d_model]
  QwenLayerWeights* layers; // array [n_layer]
  const float* w_final;     // final RMS norm
  const float* Wout;        // lm_head.weight (if NULL, tie to Wemb)
} QwenWeights;

// Forward pass for the whole model (fp32, no KV cache)
// ids: [T] token ids
// logits_out: [T, vocab]
void model_forward_f32(const QwenConfig* cfg, const QwenWeights* W,
                       const int* ids, int T,
                       float* logits_out);

#endif // MODEL_H
