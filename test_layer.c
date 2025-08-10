#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "io.h"
#include "utils.h"
#include "kernels.h"

static const float* find(BinFile* bf, const char* k) {
  TensorBin* t = bin_find(bf, k);
  if (!t) { fprintf(stderr, "Key not found: %s\n", k); exit(1); }
  return (const float*)t->data;
}
static const float* find_or(BinFile* bf, const char* k) {
  TensorBin* t = bin_find(bf, k);
  return t ? (const float*)t->data : NULL;
}

int main(int argc, char** argv){
  if (argc < 5){
    fprintf(stderr, "Usage: %s <weights.bin> <x.npy> <y.npy> <layer_idx>\n", argv[0]);
    return 1;
  }
  const char* wfile = argv[1];
  const char* xfile = argv[2];
  const char* yfile = argv[3];
  int L = atoi(argv[4]);

  BinFile* bf = bin_load(wfile);
  if (!bf) { fprintf(stderr,"bin load fail\n"); return 1; }

  // Infer shapes
  TensorBin* Wq_t = bin_find(bf, "model.layers.0.self_attn.q_proj.weight");
  TensorBin* Wk_t = bin_find(bf, "model.layers.0.self_attn.k_proj.weight");
  TensorBin* Wo_t = bin_find(bf, "model.layers.0.self_attn.o_proj.weight");
  if (!Wq_t || !Wk_t || !Wo_t){ fprintf(stderr,"attn weights missing\n"); return 1; }
  int d_model = Wq_t->shape[1];
  int Dq = Wq_t->shape[0], Dkv = Wk_t->shape[0];
  int head_dim = 0; int n_q = 0; int n_kv = 0;
  int cand[] = {128, 96, 80, 64, 48, 40, 32};
  for (int i=0;i<(int)(sizeof(cand)/sizeof(cand[0]));++i){
    if (Dq%cand[i]==0 && Dkv%cand[i]==0){ head_dim=cand[i]; break; }
  }
  if (!head_dim) head_dim = Dq;
  n_q = Dq / head_dim; n_kv = Dkv / head_dim;

  // Load tensors (layer 0 keys are fine if your bin only contains that layer)
  const float* w_norm1 = find(bf, "model.layers.0.input_layernorm.weight");
  const float* w_norm2 = find(bf, "model.layers.0.post_attention_layernorm.weight");

  const float* Wq = find(bf, "model.layers.0.self_attn.q_proj.weight");
  const float* Wk = find(bf, "model.layers.0.self_attn.k_proj.weight");
  const float* Wv = find(bf, "model.layers.0.self_attn.v_proj.weight");
  const float* Wo = find(bf, "model.layers.0.self_attn.o_proj.weight");
  const float* bq = find_or(bf, "model.layers.0.self_attn.q_proj.bias");
  const float* bk = find_or(bf, "model.layers.0.self_attn.k_proj.bias");
  const float* bv = find_or(bf, "model.layers.0.self_attn.v_proj.bias");
  const float* bo = find_or(bf, "model.layers.0.self_attn.o_proj.bias");
  const float* qn = find_or(bf, "model.layers.0.self_attn.q_norm.weight");
  const float* kn = find_or(bf, "model.layers.0.self_attn.k_norm.weight");

  const float* Wroute = find(bf, "model.layers.0.mlp.gate.weight");
  const float* broute = find_or(bf, "model.layers.0.mlp.gate.bias");

  // Collect experts
  // Infer d_ff and E
  int E = 0, d_ff = 0;
  for (int e=0; e<2048; ++e){
    char k[256]; snprintf(k,sizeof(k),"model.layers.0.mlp.experts.%d.down_proj.weight", e);
    TensorBin* t = bin_find(bf, k);
    if (!t) break;
    E = e+1; if (!d_ff) d_ff = t->shape[1];
  }
  const float** Wg = (const float**)calloc(E,sizeof(float*));
  const float** bg = (const float**)calloc(E,sizeof(float*));
  const float** Wu = (const float**)calloc(E,sizeof(float*));
  const float** bu = (const float**)calloc(E,sizeof(float*));
  const float** Wd = (const float**)calloc(E,sizeof(float*));
  const float** bd = (const float**)calloc(E,sizeof(float*));
  for (int e=0; e<E; ++e){
    char k1[256],k2[256],k3[256],k1b[256],k2b[256],k3b[256];
    snprintf(k1,sizeof(k1),"model.layers.0.mlp.experts.%d.gate_proj.weight", e);
    snprintf(k2,sizeof(k2),"model.layers.0.mlp.experts.%d.up_proj.weight", e);
    snprintf(k3,sizeof(k3),"model.layers.0.mlp.experts.%d.down_proj.weight", e);
    snprintf(k1b,sizeof(k1b),"model.layers.0.mlp.experts.%d.gate_proj.bias", e);
    snprintf(k2b,sizeof(k2b),"model.layers.0.mlp.experts.%d.up_proj.bias", e);
    snprintf(k3b,sizeof(k3b),"model.layers.0.mlp.experts.%d.down_proj.bias", e);
    TensorBin* t;
    if ((t=bin_find(bf,k1))) Wg[e]=(const float*)t->data;
    if ((t=bin_find(bf,k2))) Wu[e]=(const float*)t->data;
    if ((t=bin_find(bf,k3))) Wd[e]=(const float*)t->data;
    if ((t=bin_find(bf,k1b))) bg[e]=(const float*)t->data;
    if ((t=bin_find(bf,k2b))) bu[e]=(const float*)t->data;
    if ((t=bin_find(bf,k3b))) bd[e]=(const float*)t->data;
  }

  // I/O
  NpyArray* nx = npy_load_float32(xfile);
  NpyArray* ny = npy_load_float32(yfile);
  int T = nx->shape[0];
  float* scratch_attn = (float*)malloc(sizeof(float) * ( (size_t)T*(Dq + Dkv + Dkv) + (size_t)T*T + (size_t)T*Dq ));
  float* scratch_moe  = (float*)malloc(sizeof(float) * (2 * d_ff)); // tmp_g/tmp_u per token (we call expert per token)
  int*   top_idx = (int*)malloc(sizeof(int) * T * 8);
  float* top_p   = (float*)malloc(sizeof(float) * T * 8);

  // In-place: x is overwritten with the final layer output
  layer_forward_f32(
    nx->data, T, d_model,
    w_norm1, 1e-6f,
    Wq,bq, Wk,bk, Wv,bv, Wo,bo, qn,kn,
    n_q, n_kv, head_dim, /*causal=*/1, /*rope_theta=*/10000.0f,
    w_norm2, 1e-6f,
    Wroute, broute,
    Wg,bg, Wu,bu, Wd,bd,
    E, /*k=*/8, d_ff,
    scratch_attn, scratch_moe, top_idx, top_p
  );

  float diff = max_abs_diff(nx->data, ny->data, T*d_model);
  printf("Max abs diff: %.6g\n", diff);
  printf("%s\n", (diff < 1e-5f) ? "PASS" : "FAIL");

  free(scratch_attn); free(scratch_moe); free(top_idx); free(top_p);
  npy_free(nx); npy_free(ny); bin_free(bf);
  free(Wg); free(bg); free(Wu); free(bu); free(Wd); free(bd);
  return (diff < 1e-5f) ? 0 : 1;
}
