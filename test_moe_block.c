
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "io.h"
#include "utils.h"
#include "kernels.h"

static const char* mode_name(RouterMode m){
  return (m==ROUTER_TOPK_KONLY) ? "konly" : "full";
}
static RouterMode parse_mode(const char* s){
  if (!s) return ROUTER_TOPK_KONLY;
  if (strcmp(s,"konly")==0) return ROUTER_TOPK_KONLY;
  if (strcmp(s,"full")==0)  return ROUTER_SOFTMAX_ALL_TOPK;
  fprintf(stderr, "mode must be konly|full\n"); exit(1);
}

int main(int argc, char** argv){
  if (argc < 5){
    fprintf(stderr, "Usage: %s <weights.bin> <x.npy> <y.npy> <konly|full>\n", argv[0]);
    return 1;
  }
  const char* wpath = argv[1];
  const char* xpath = argv[2];
  const char* ypath = argv[3];
  RouterMode mode = parse_mode(argv[4]);

  BinFile* bf = bin_load(wpath);
  if(!bf){ fprintf(stderr,"bin load fail\n"); return 1; }

  TensorBin* Wroute = bin_find(bf, "model.layers.0.mlp.gate.weight");
  if (!Wroute) Wroute = bin_find(bf, "model.layers.0.mlp.router.gate.weight");
  TensorBin* Broute = bin_find(bf, "model.layers.0.mlp.gate.bias");
  if (!Broute) Broute = bin_find(bf, "model.layers.0.mlp.router.gate.bias");

  if(!Wroute){ fprintf(stderr,"router weight missing\n"); return 1; }
  int E = Wroute->shape[0];
  int d_model = Wroute->shape[1];
  const float* router_w = (const float*)Wroute->data;
  const float* router_b = (Broute && Broute->dtype==0) ? (const float*)Broute->data : NULL;

  // infer d_ff from any expert down_proj
  int d_ff = -1;
  for (int e=0; e<E; ++e){
    char key[256];
    snprintf(key,sizeof(key),"model.layers.0.mlp.experts.%d.down_proj.weight", e);
    TensorBin* Wd = bin_find(bf, key);
    if (Wd){ d_ff = Wd->shape[1]; break; }
  }
  if (d_ff<=0){ fprintf(stderr,"d_ff infer fail\n"); return 1; }

  const float** Wg = (const float**)calloc(E, sizeof(float*));
  const float** bg = (const float**)calloc(E, sizeof(float*));
  const float** Wu = (const float**)calloc(E, sizeof(float*));
  const float** bu = (const float**)calloc(E, sizeof(float*));
  const float** Wd = (const float**)calloc(E, sizeof(float*));
  const float** bd = (const float**)calloc(E, sizeof(float*));

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
    if ((t=bin_find(bf,k1b)) && t->dtype==0) bg[e]=(const float*)t->data;
    if ((t=bin_find(bf,k2b)) && t->dtype==0) bu[e]=(const float*)t->data;
    if ((t=bin_find(bf,k3b)) && t->dtype==0) bd[e]=(const float*)t->data;
  }

  NpyArray* xin = npy_load_float32(xpath);
  NpyArray* ygd = npy_load_float32(ypath);
  if(!xin||!ygd){ fprintf(stderr,"npy load fail\n"); return 1; }
  if (xin->ndim!=2 || xin->shape[1]!=d_model){ fprintf(stderr,"x shape mismatch\n"); return 1; }
  if (ygd->ndim!=2 || ygd->shape[1]!=d_model || ygd->shape[0]!=xin->shape[0]){
    fprintf(stderr,"y shape mismatch\n"); return 1;
  }
  int T = xin->shape[0];
  float* y = (float*)malloc(sizeof(float)*T*d_model);
  float* tmp_g = (float*)malloc(sizeof(float)*d_ff);
  float* tmp_u = (float*)malloc(sizeof(float)*d_ff);
  int k = 8;
  int* top_idx = (int*)malloc(sizeof(int)*T*k);
  float* top_p = (float*)malloc(sizeof(float)*T*k);

  moe_forward_f32_mode(
    xin->data, T, d_model,
    router_w, router_b,
    E, k, d_ff, mode,
    Wg, bg, Wu, bu, Wd, bd,
    y, tmp_g, tmp_u, top_idx, top_p
  );

  float mad = max_abs_diff(y, ygd->data, T*d_model);
  printf("Mode=%s Max abs diff: %.6g\n", mode_name(mode), mad);
  printf("%s\n", (mad < 1e-4f) ? "PASS" : "FAIL");

  free(y); free(tmp_g); free(tmp_u); free(top_idx); free(top_p);
  npy_free(xin); npy_free(ygd); bin_free(bf);
  free(Wg); free(bg); free(Wu); free(bu); free(Wd); free(bd);
  return (mad < 1e-4f) ? 0 : 1;
}

