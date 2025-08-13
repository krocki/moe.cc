
#ifndef KERNELS_H
#define KERNELS_H

#include <stddef.h>

typedef enum {
  ROUTER_TOPK_KONLY = 0,       // top-k on logits, softmax over k (usual MoE)
  ROUTER_SOFTMAX_ALL_TOPK = 1  // softmax over all E, then top-k by prob
} RouterMode;

// GEMM: C[M,N] = A[M,K] * B[N,K]^T  (row-major; B stored [out,in])
void matmul_f32(const float* A, const float* B, float* C,
                int M, int N, int K);

// Elementwise SiLU on x[0..n-1]
void silu_f32(float* x, int n);

// Expert forward (SwiGLU): y = down( silu(gate(x)) * up(x) )
void expert_forward_f32(const float* x,
                        const float* Wg, const float* bg,
                        const float* Wu, const float* bu,
                        const float* Wd, const float* bd,
                        int T, int d_model, int d_ff,
                        float* tmp_g, float* tmp_u,
                        float* y);

// Router helpers
void router_topk_softmax_konly(const float* logits, int T, int E, int k,
                               int* top_idx, float* top_p);
void router_softmax_all_topk(const float* logits, int T, int E, int k,
                             int* top_idx, float* top_p);

// MoE forward with selectable routing mode
// Signature matches test_moe_block.c call site
void moe_forward_f32_mode(
    const float* x, int T, int d_model,
    const float* router_w, const float* router_b, // [E,d_model], [E] or NULL
    int E, int k, int d_ff, RouterMode mode,
    const float** Wg_arr, const float** bg_arr,
    const float** Wu_arr, const float** bu_arr,
    const float** Wd_arr, const float** bd_arr,
    float* y,              // [T,d_model]
    float* tmp_one_g,      // [d_ff] scratch (per token)
    float* tmp_one_u,      // [d_ff] scratch (per token)
    int* top_idx,          // [T*k]
    float* top_p);         // [T*k]

// RMSNorm forward
// x: [T, d_model]
// w: [d_model] (scale weights)
// eps: epsilon for numerical stability
// y: [T, d_model] output
void rmsnorm_forward_f32(const float* x, const float* w,
                         int T, int d_model, float eps,
                         float* y);

static void rmsnorm_headwise_vec_inplace(float* x, const float* w, int head_dim, float eps);
//
// x: [T, d_model]
// Q: Wq[b, d_model] where b = n_q * head_dim
// K,V: Wk/Wv[c, d_model] where c = n_kv * head_dim
// Wo: [d_model, n_q * head_dim]
// b*: optional biases; q_norm/k_norm can be per-channel (len=b/c) or per-head (len=n_q/n_kv)
// causal: 1=causal mask, 0=none
// scratch floats needed: T*(b + c + c) + T*T + T*b
// NEW: pass qn_len / kn_len and rope_theta through.
void attn_forward_f32_gqa(
  const float* x, int T, int d_model,
  const float* Wq, const float* bq,
  const float* Wk, const float* bk,
  const float* Wv, const float* bv,
  const float* Wo, const float* bo,
  const float* qn, int qn_len,
  const float* kn, int kn_len,
  int n_q, int n_kv, int head_dim, int causal, float rope_theta,
  float* scratch, float* y
);

// Apply RoPE to Q and K in-place (GQA-aware, no weights).
// Q: [T, n_q*head_dim], K: [T, n_kv*head_dim]
// pos0: starting position offset (e.g., for KV-cache decode), usually 0 in tests.
// theta: base rotary θ (Qwen default 10000.0)
// Uses kv head mapping kvh = h % n_kv (Qwen-style GQA).
void rope_apply_inplace_f32_gqa(
  float* Q, float* K,
  int T, int n_q, int n_kv, int head_dim,
  int pos0, float theta);

// One transformer layer (fp32), no KV cache.
// x: [T, d_model] in/out (residual adds done inside)
void layer_forward_f32(
  float* x, int T, int d_model,
  // Norm1
  const float* w_norm1, float eps1,
  // Attn weights/bias/norm (GQA)
  const float* Wq,const float* bq,
  const float* Wk,const float* bk,
  const float* Wv,const float* bv,
  const float* Wo,const float* bo,
  const float* q_norm, const float* k_norm,
  int n_q, int n_kv, int head_dim, int causal, float rope_theta,
  // Norm2
  const float* w_norm2, float eps2,
  // MoE router + experts
  const float* Wroute, const float* broute,
  const float** Wg,const float** bg,
  const float** Wu,const float** bu,
  const float** Wd,const float** bd,
  int E, int k, int d_ff,
  // scratch buffers (reused across subops)
  float* scratch_attn, float* scratch_moe, int* top_idx, float* top_p);

#endif
