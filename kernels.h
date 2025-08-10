
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

#endif
