// test_layer.c
// Runs a single Qwen3-MoE decoder layer forward and compares against numpy goldens.
// Usage: ./test_layer <l0_layer.bin> <x.npy> <y.npy> <rope_theta>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "io.h"
#include "utils.h"
#include "kernels.h"

static const float* must(BinFile* bf, const char* k) {
  TensorBin* t = bin_find(bf, k);
  if (!t) { fprintf(stderr, "Key not found: %s\n", k); exit(1); }
  return (const float*)t->data;
}
static const float* opt(BinFile* bf, const char* k) {
  TensorBin* t = bin_find(bf, k);
  return t ? (const float*)t->data : NULL;
}

static const float* opt_f32(BinFile* bf, const char* k) {
  TensorBin* t = bin_find(bf, k);
  return (t && t->dtype == 0) ? (const float*)t->data : NULL; // 0=f32 in your loader
}

int main(int argc, char** argv){
  if (argc < 5){
    fprintf(stderr, "Usage: %s <weights.bin> <x.npy> <y.npy> <rope_theta>\n", argv[0]);
    return 1;
  }
  const char* wpath = argv[1];
  const char* xfile = argv[2];
  const char* yfile = argv[3];
  float rope_theta = strtof(argv[4], NULL);

  fprintf(stderr, "[load] reading weights: %s\n", wpath);
  BinFile* bf = bin_load(wpath);
  if(!bf){ fprintf(stderr,"bin load fail\n"); return 1; }

  // Shapes / heads from layer0 tensors
  const float* Wq = must(bf, "model.layers.0.self_attn.q_proj.weight");
  const float* Wk = must(bf, "model.layers.0.self_attn.k_proj.weight");
  const float* Wv = must(bf, "model.layers.0.self_attn.v_proj.weight");
  const float* Wo = must(bf, "model.layers.0.self_attn.o_proj.weight");

  const float* bq = opt_f32(bf, "model.layers.0.self_attn.q_proj.bias");
  const float* bk = opt_f32(bf, "model.layers.0.self_attn.k_proj.bias");
  const float* bv = opt_f32(bf, "model.layers.0.self_attn.v_proj.bias");
  const float* bo = opt_f32(bf, "model.layers.0.self_attn.o_proj.bias");

  const float* qn = must(bf, "model.layers.0.self_attn.q_norm.weight");
  const float* kn = must(bf, "model.layers.0.self_attn.k_norm.weight");

  int d_model = bin_find(bf, "model.layers.0.self_attn.o_proj.weight")->shape[0];
  int Dq = bin_find(bf, "model.layers.0.self_attn.q_proj.weight")->shape[0];
  int Dk = bin_find(bf, "model.layers.0.self_attn.k_proj.weight")->shape[0];
  int Dv = bin_find(bf, "model.layers.0.self_attn.v_proj.weight")->shape[0];
  int head_dim = bin_find(bf, "model.layers.0.self_attn.q_norm.weight")->shape[0];
  int n_q  = Dq / head_dim;
  int n_kv = Dk / head_dim;

  const float* w1 = must(bf, "model.layers.0.input_layernorm.weight");
  const float* w2 = must(bf, "model.layers.0.post_attention_layernorm.weight");

  const float* router_w = must(bf, "model.layers.0.mlp.gate.weight");
  const float* router_b = opt_f32(bf, "model.layers.0.mlp.gate.bias");
  if (!router_b) router_b = opt_f32(bf, "model.layers.0.mlp.router.gate.bias");


  // Collect expert weights (E,d_ff inferred by scanning down_proj)
  int E = 0, d_ff = 0;
  for (int e = 0; e < 2048; ++e){
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
    snprintf(k2,sizeof(k2),"model.layers.0.mlp.experts.%d.up_proj.weight",   e);
    snprintf(k3,sizeof(k3),"model.layers.0.mlp.experts.%d.down_proj.weight", e);
    snprintf(k1b,sizeof(k1b),"model.layers.0.mlp.experts.%d.gate_proj.bias", e);
    snprintf(k2b,sizeof(k2b),"model.layers.0.mlp.experts.%d.up_proj.bias",   e);
    snprintf(k3b,sizeof(k3b),"model.layers.0.mlp.experts.%d.down_proj.bias", e);
    TensorBin* t;
    if ((t=bin_find(bf,k1)))  Wg[e] = (const float*)t->data;
    if ((t=bin_find(bf,k2)))  Wu[e] = (const float*)t->data;
    if ((t=bin_find(bf,k3)))  Wd[e] = (const float*)t->data;
    if ((t=bin_find(bf,k1b))) bg[e] = (const float*)t->data;
    if ((t=bin_find(bf,k2b))) bu[e] = (const float*)t->data;
    if ((t=bin_find(bf,k3b))) bd[e] = (const float*)t->data;

    bg[e] = opt_f32(bf, k1b);
    bu[e] = opt_f32(bf, k2b);
    bd[e] = opt_f32(bf, k3b);

  }

  NpyArray* nx = npy_load_float32(xfile);
  NpyArray* ny = npy_load_float32(yfile);
  int T = nx->shape[0];
  if (nx->shape[1]!=d_model || ny->shape[1]!=d_model || ny->shape[0]!=T){
    fprintf(stderr,"shape mismatch: T=%d d_model=%d\n", T, d_model); return 1;
  }

  // scratch sizes like model.c
  size_t attn_f  = (size_t)T*(Dq + 2*Dk) + (size_t)T*T + (size_t)T*Dq;
  size_t temps_f = 5ull * (size_t)T * d_model;
  float* scratch_attn = (float*)malloc(sizeof(float)*(attn_f+temps_f));
  float* scratch_moe  = (float*)malloc(sizeof(float)*(2ull*(size_t)T*d_ff));
  int*   top_idx      = (int*)  malloc(sizeof(int)*T*8);
  float* top_p        = (float*)malloc(sizeof(float)*T*8);

  layer_forward_f32(
    nx->data, T, d_model,
    w1, 1e-6f,
    Wq,bq, Wk,bk, Wv,bv, Wo,bo, qn,kn,
    n_q, n_kv, head_dim, /*causal=*/1, rope_theta,
    w2, 1e-6f,
    router_w, router_b,
    Wg,bg, Wu,bu, Wd,bd,
    E, /*k=*/8, d_ff,
    scratch_attn, scratch_moe, top_idx, top_p
  );

  float diff = max_abs_diff(nx->data, ny->data, T*d_model);
  printf("Max abs diff: %.6g\n%s\n", diff, (diff<1e-5f?"PASS":"FAIL"));

  free(scratch_attn); free(scratch_moe); free(top_idx); free(top_p);
  npy_free(nx); npy_free(ny); bin_free(bf);
  free(Wg); free(bg); free(Wu); free(bu); free(Wd); free(bd);
  return (diff < 1e-5f) ? 0 : 1;
}
