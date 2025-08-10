// test_moe_block.c
// 2-space indentation; links with io.o, utils.o, kernels.o
// Usage: ./test_moe_block l0_moe.bin qwen3_L0_MOE.x.npy qwen3_L0_MOE.y.npy

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "io.h"
#include "utils.h"
#include "kernels.h"

int main(int argc, char** argv){
  if (argc < 4){
    fprintf(stderr, "Usage: %s <weights.bin> <x.npy> <y.npy>\n", argv[0]);
    return 1;
  }
  const char* wpath = argv[1];
  const char* xpath = argv[2];
  const char* ypath = argv[3];

  BinFile* bf = bin_load(wpath);
  if(!bf){ fprintf(stderr, "Failed to load %s\n", wpath); return 1; }

  // Find router weights first
  TensorBin* Wgate = NULL;
  TensorBin* bgate = NULL;
  for (int i=0;i<bf->count;i++){
    if (strstr(bf->arr[i].name, ".mlp.gate.weight")) Wgate = &bf->arr[i];
    if (strstr(bf->arr[i].name, ".mlp.gate.bias"))   bgate = &bf->arr[i];
  }
  if(!Wgate){ fprintf(stderr, "Router gate weight not found\n"); return 1; }
  int E = Wgate->shape[0];
  int d_model = Wgate->shape[1];

  // Collect expert params (sparse subset or all) by scanning names
  // Expect names like: model.layers.0.mlp.experts.<e>.{gate,up,down}_proj.{weight,bias}
  const int MAXE = 2048;
  const float* Wg_arr[MAXE] = {0}, *bg_arr[MAXE] = {0};
  const float* Wu_arr[MAXE] = {0}, *bu_arr[MAXE] = {0};
  const float* Wd_arr[MAXE] = {0}, *bd_arr[MAXE] = {0};

  int d_ff = -1;
  for (int i=0;i<bf->count;i++){
    TensorBin* t = &bf->arr[i];
    char* p = strstr(t->name, ".experts.");
    if (!p) continue;
    int e = atoi(p + strlen(".experts."));
    if (e < 0 || e >= MAXE) continue;

    if (strstr(t->name, ".gate_proj.weight")) { Wg_arr[e] = (const float*)t->data; d_ff = t->shape[0]; }
    if (strstr(t->name, ".gate_proj.bias"))   { bg_arr[e] = (const float*)t->data; }
    if (strstr(t->name, ".up_proj.weight"))   { Wu_arr[e] = (const float*)t->data; }
    if (strstr(t->name, ".up_proj.bias"))     { bu_arr[e] = (const float*)t->data; }
    if (strstr(t->name, ".down_proj.weight")) { Wd_arr[e] = (const float*)t->data; }
    if (strstr(t->name, ".down_proj.bias"))   { bd_arr[e] = (const float*)t->data; }
  }
  if (d_ff <= 0){ fprintf(stderr, "Failed to infer d_ff\n"); return 1; }

  NpyArray* xin = npy_load_float32(xpath);
  NpyArray* ygd = npy_load_float32(ypath);
  if (!xin || !ygd){ fprintf(stderr, "Failed to load npy\n"); return 1; }
  int T = xin->shape[0];
  if (xin->shape[1] != d_model){ fprintf(stderr, "x.npy d_model mismatch\n"); return 1; }
  if (ygd->shape[0] != T || ygd->shape[1] != d_model){ fprintf(stderr, "y.npy shape mismatch\n"); return 1; }

  float* y = (float*)malloc(sizeof(float)*T*d_model);
  float* tmp_g = (float*)malloc(sizeof(float)*d_ff); // per-token scratch
  float* tmp_u = (float*)malloc(sizeof(float)*d_ff);
  int topk = 8; // consistent with A3B
  int* top_idx = (int*)malloc(sizeof(int)*T*topk);
  float* top_p  = (float*)malloc(sizeof(float)*T*topk);

  moe_forward_f32(
    xin->data, T, d_model,
    (const float*)Wgate->data, bgate ? (const float*)bgate->data : NULL,
    E, topk, d_ff,
    Wg_arr, bg_arr, Wu_arr, bu_arr, Wd_arr, bd_arr,
    y, tmp_g, tmp_u, top_idx, top_p
  );

  float mad = max_abs_diff(y, ygd->data, T*d_model);
  printf("Max abs diff: %.6g\n", mad);
  printf("%s\n", (mad < 1e-4f) ? "PASS" : "FAIL");

  free(y); free(tmp_g); free(tmp_u); free(top_idx); free(top_p);
  npy_free(xin); npy_free(ygd);
  bin_free(bf);
  return (mad < 1e-4f) ? 0 : 1;
}
