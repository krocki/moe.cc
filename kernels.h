// kernels.h
#ifndef KERNELS_H
#define KERNELS_H

// Matmul: A[M,K], W[N,K] (row-major W[out,in]), adds bias if non-null
void matmul_add_bias_f32(
  const float* A, const float* W, const float* b,
  float* Y, int M, int K, int N);

float silu_scalar(float x);

// Single expert forward
void expert_forward_f32(
  const float* x, int T, int d_model, int d_ff,
  const float* Wg, const float* bg,
  const float* Wu, const float* bu,
  const float* Wd, const float* bd,
  float* y, float* tmp_g, float* tmp_u);

// Router: top-k over logits, then softmax over those k (per token)
void router_topk_softmax_konly(
  const float* logits, int T, int E, int k,
  int* top_idx, float* top_p);

// Full MoE forward (naïve per-token loop)
void moe_forward_f32(
  const float* x, int T, int d_model,
  const float* router_w, const float* router_b, // [E, d_model], [E] or NULL
  int E, int k, int d_ff,
  // Expert params: arrays of length E, each pointing to Wg,bg,Wu,bu,Wd,bd
  const float** Wg_arr, const float** bg_arr,
  const float** Wu_arr, const float** bu_arr,
  const float** Wd_arr, const float** bd_arr,
  float* y,                  // [T, d_model]
  float* scratch_g,          // [d_ff] per token (caller can reuse T*d_ff and offset per t)
  float* scratch_u,          // [d_ff] per token
  int* top_idx, float* top_p // [T*k] each (pre-allocated)
);

#endif // KERNELS_H
