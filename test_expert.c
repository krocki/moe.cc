// test_expert.c
// 2-space indentation, modular includes.
// Build: make
// Run:   ./test_expert l0_e0.bin qwen3_L0_E0.x.npy qwen3_L0_E0.y.npy

#include <stdio.h>
#include <stdlib.h>
#include "io.h"
#include "kernels.h"
#include "utils.h"

int main(int argc, char** argv){
  if (argc < 4){
    fprintf(stderr, "Usage: %s <weights.bin> <x.npy> <y_golden.npy>\n", argv[0]);
    return 1;
  }
  const char* wpath = argv[1];
  const char* xpath = argv[2];
  const char* ypath = argv[3];

  BinFile* bf = bin_load(wpath);
  if(!bf){ fprintf(stderr, "Failed to load weights\n"); return 1; }

  TensorBin* Wg = bin_find(bf, "model.layers.0.mlp.experts.0.gate_proj.weight");
  TensorBin* bg = bin_find(bf, "model.layers.0.mlp.experts.0.gate_proj.bias");
  TensorBin* Wu = bin_find(bf, "model.layers.0.mlp.experts.0.up_proj.weight");
  TensorBin* bu = bin_find(bf, "model.layers.0.mlp.experts.0.up_proj.bias");
  TensorBin* Wd = bin_find(bf, "model.layers.0.mlp.experts.0.down_proj.weight");
  TensorBin* bd = bin_find(bf, "model.layers.0.mlp.experts.0.down_proj.bias");

  if(!Wg || !Wu || !Wd){
    fprintf(stderr, "Missing one or more REQUIRED expert weights in %s\n", wpath);
    return 1;
  }

  int d_ff = Wg->shape[0];
  int d_model = Wd->shape[0];
  if (Wg->dtype!=0 || Wu->dtype!=0 || Wd->dtype!=0){
    fprintf(stderr, "Weights must be float32 in this test.\n");
    return 1;
  }

  NpyArray* xin = npy_load_float32(xpath);
  NpyArray* ygd = npy_load_float32(ypath);
  if (!xin || !ygd){ fprintf(stderr, "Failed to load .npy\n"); return 1; }
  if (xin->ndim!=2 || xin->shape[1]!=d_model){
    fprintf(stderr, "x.npy shape mismatch: expected [T,%d], got [%d,%d]\n",
            d_model, xin->shape[0], xin->shape[1]);
    return 1;
  }
  if (ygd->ndim!=2 || ygd->shape[1]!=d_model || ygd->shape[0]!=xin->shape[0]){
    fprintf(stderr, "y.npy shape mismatch\n"); return 1;
  }

  int T = xin->shape[0];
  float* y = (float*)malloc(sizeof(float)*T*d_model);
  float* tmp_g = (float*)malloc(sizeof(float)*T*d_ff);
  float* tmp_u = (float*)malloc(sizeof(float)*T*d_ff);
  if(!y||!tmp_g||!tmp_u){ fprintf(stderr, "OOM\n"); return 1; }

  expert_forward_f32(
    xin->data, T, d_model, d_ff,
    (const float*)Wg->data, bg ? (const float*)bg->data : NULL,
    (const float*)Wu->data, bu ? (const float*)bu->data : NULL,
    (const float*)Wd->data, bd ? (const float*)bd->data : NULL,
    y, tmp_g, tmp_u
  );

  int N = T*d_model;
  float mad = max_abs_diff(y, ygd->data, N);
  printf("Max abs diff: %.6g\n", mad);
  printf("%s\n", (mad < 1e-4f) ? "PASS" : "FAIL");

  free(y); free(tmp_g); free(tmp_u);
  npy_free(xin); npy_free(ygd);
  bin_free(bf);
  return (mad < 1e-4f) ? 0 : 1;
}

