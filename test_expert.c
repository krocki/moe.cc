
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "io.h"
#include "utils.h"
#include "kernels.h"

// Reuse kernels: expert_forward_f32()

int main(int argc, char** argv){
  if (argc < 4){
    fprintf(stderr, "Usage: %s <weights.bin> <x.npy> <y.npy>\n", argv[0]);
    return 1;
  }
  const char* wpath = argv[1];
  const char* xpath = argv[2];
  const char* ypath = argv[3];

  BinFile* bf = bin_load(wpath);
  if(!bf){ fprintf(stderr,"bin load fail\n"); return 1; }

  TensorBin* Wg = bin_find(bf, "model.layers.0.mlp.experts.0.gate_proj.weight");
  TensorBin* bg = bin_find(bf, "model.layers.0.mlp.experts.0.gate_proj.bias");
  TensorBin* Wu = bin_find(bf, "model.layers.0.mlp.experts.0.up_proj.weight");
  TensorBin* bu = bin_find(bf, "model.layers.0.mlp.experts.0.up_proj.bias");
  TensorBin* Wd = bin_find(bf, "model.layers.0.mlp.experts.0.down_proj.weight");
  TensorBin* bd = bin_find(bf, "model.layers.0.mlp.experts.0.down_proj.bias");

  if(!Wg||!Wu||!Wd){ fprintf(stderr,"missing expert weights\n"); return 1; }
  int d_ff = Wg->shape[0];
  int d_model = Wd->shape[0];

  NpyArray* xin = npy_load_float32(xpath);
  NpyArray* ygd = npy_load_float32(ypath);
  if(!xin||!ygd){ fprintf(stderr,"npy load fail\n"); return 1; }
  int T = xin->shape[0];
  float* y = (float*)malloc(sizeof(float)*T*d_model);
  float* tmp_g = (float*)malloc(sizeof(float)*T*d_ff);
  float* tmp_u = (float*)malloc(sizeof(float)*T*d_ff);

  expert_forward_f32(xin->data,
                     (const float*)Wg->data, bg?(const float*)bg->data:NULL,
                     (const float*)Wu->data, bu?(const float*)bu->data:NULL,
                     (const float*)Wd->data, bd?(const float*)bd->data:NULL,
                     T, d_model, d_ff, tmp_g, tmp_u, y);

  float mad = max_abs_diff(y, ygd->data, T*d_model);
  printf("Max abs diff: %.6g\n", mad);
  printf("%s\n", (mad < 1e-4f) ? "PASS" : "FAIL");

  free(y); free(tmp_g); free(tmp_u);
  npy_free(xin); npy_free(ygd);
  bin_free(bf);
  return (mad < 1e-4f) ? 0 : 1;
}

