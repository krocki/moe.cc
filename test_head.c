#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "io.h"
#include "utils.h"
#include "kernels.h"

static TensorBin* find_any(BinFile* b, const char** keys){
  for (int i=0; keys[i]; ++i){ TensorBin* t = bin_find(b, keys[i]); if (t) return t; }
  return NULL;
}

int main(int argc, char** argv){
  if (argc < 5){
    fprintf(stderr,"Usage: %s <weights.bin> <x_finalhidden.npy> <logits.npy> <eps>\n", argv[0]);
    return 1;
  }
  const char* wfile = argv[1];
  const char* xfile = argv[2];
  const char* yfile = argv[3];
  float eps = strtof(argv[4], NULL);

  fprintf(stderr, "[load] reading weights: %s\n", wfile);
  BinFile* bin = bin_load(wfile);
  if (!bin){ fprintf(stderr,"bin load fail\n"); return 1; }

  const char* final_keys[] = {
    "model.norm.weight",
    "model.final_layernorm.weight",
    NULL
  };
  TensorBin* wfin = find_any(bin, final_keys);
  if (!wfin){ fprintf(stderr,"final norm weight not found\n"); return 1; }

  TensorBin* Wemb = bin_find(bin, "model.embed_tokens.weight");
  TensorBin* Wout = bin_find(bin, "lm_head.weight");
  if (!Wemb){ fprintf(stderr,"embed not found\n"); return 1; }
  const float* headW = Wout ? (const float*)Wout->data : (const float*)Wemb->data;

  NpyArray* xin = npy_load_float32(xfile);
  NpyArray* ygd = npy_load_float32(yfile);
  int T = xin->shape[0];
  int d_model = xin->shape[1];
  int vocab = Wout ? Wout->shape[0] : Wemb->shape[0];
  if (ygd->shape[0]!=T || ygd->shape[1]!=vocab){
    fprintf(stderr,"logits shape mismatch\n"); return 1;
  }

  float* xnorm = (float*)malloc(sizeof(float)*T*d_model);
  float* logits= (float*)malloc(sizeof(float)*T*vocab);

  rmsnorm_forward_f32(xin->data, (const float*)wfin->data, T, d_model, eps, xnorm);
  matmul_f32(xnorm, headW, logits, T, vocab, d_model);

  float mad = max_abs_diff(logits, ygd->data, T*vocab);
  printf("Max abs diff: %.6g\n%s\n", mad, (mad<1e-4f?"PASS":"FAIL"));

  free(xnorm); free(logits);
  npy_free(xin); npy_free(ygd); bin_free(bin);
  return (mad<1e-4f)?0:1;
}
