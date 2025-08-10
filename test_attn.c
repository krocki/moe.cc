#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "io.h"
#include "utils.h"
#include "kernels.h"

static const float* find_or_null(BinFile* bf, const char* k) {
  TensorBin* t = bin_find(bf, k);
  return t ? (const float*)t->data : NULL;
}

static int infer_head_dim(int Dq, int Dkv){
  int cand[] = {128, 96, 80, 64, 48, 40, 32};
  for (int i=0;i<(int)(sizeof(cand)/sizeof(cand[0]));++i){
    if (Dq % cand[i] == 0 && Dkv % cand[i] == 0) return cand[i];
  }
  // fallback: gcd
  int a = Dq, b = Dkv; while (b){ int r = a % b; a = b; b = r; }
  return a ? a : Dq;
}

int main(int argc, char** argv) {
  if (argc < 5) {
    fprintf(stderr, "Usage: %s <weights.bin> <x.npy> <y.npy> <causal:0|1>\n", argv[0]);
    return 1;
  }
  const char* wfile = argv[1];
  const char* xfile = argv[2];
  const char* yfile = argv[3];
  int causal = atoi(argv[4]);

  BinFile* bf = bin_load(wfile);
  if (!bf) { fprintf(stderr, "bin load fail\n"); return 1; }

  TensorBin* Wq = bin_find(bf, "model.layers.0.self_attn.q_proj.weight");
  TensorBin* Wk = bin_find(bf, "model.layers.0.self_attn.k_proj.weight");
  TensorBin* Wv = bin_find(bf, "model.layers.0.self_attn.v_proj.weight");
  TensorBin* Wo = bin_find(bf, "model.layers.0.self_attn.o_proj.weight");
  if (!Wq || !Wk || !Wv || !Wo) { fprintf(stderr, "missing q/k/v/o weights\n"); return 1; }

  const float* bq = find_or_null(bf, "model.layers.0.self_attn.q_proj.bias");
  const float* bk = find_or_null(bf, "model.layers.0.self_attn.k_proj.bias");
  const float* bv = find_or_null(bf, "model.layers.0.self_attn.v_proj.bias");
  const float* bo = find_or_null(bf, "model.layers.0.self_attn.o_proj.bias");
  const float* qn = find_or_null(bf, "model.layers.0.self_attn.q_norm.weight");
  const float* kn = find_or_null(bf, "model.layers.0.self_attn.k_norm.weight");

  int Dq = Wq->shape[0];
  int Dkv = Wk->shape[0];
  int d_model = Wq->shape[1];
  int head_dim = infer_head_dim(Dq, Dkv);
  int n_q  = Dq  / head_dim;
  int n_kv = Dkv / head_dim;

  NpyArray* nx = npy_load_float32(xfile);
  NpyArray* ny = npy_load_float32(yfile);
  if (!nx || !ny) { fprintf(stderr, "npy load fail\n"); return 1; }
  if (nx->shape[1] != d_model || ny->shape[1] != d_model || ny->shape[0] != nx->shape[0]) {
    fprintf(stderr, "shape mismatch\n"); return 1;
  }
  int T = nx->shape[0];

  // scratch: T*(Dq + Dkv + Dkv) + T*T + T*Dq
  size_t need = (size_t)T*(Dq + Dkv + Dkv) + (size_t)T*T + (size_t)T*Dq;
  float* scratch = (float*)malloc(sizeof(float)*need);
  float* y = (float*)malloc(sizeof(float)*T*d_model);

  attn_forward_f32_gqa(
    nx->data, T, d_model,
    (const float*)Wq->data, bq,
    (const float*)Wk->data, bk,
    (const float*)Wv->data, bv,
    (const float*)Wo->data, bo,
    qn, kn,
    n_q, n_kv, head_dim, causal,
    scratch, y
  );

  double diff = max_abs_diff(y, ny->data, T*d_model);
  printf("Max abs diff: %g\n", diff);
  printf("%s\n", (diff < 1e-5) ? "PASS" : "FAIL");

  free(scratch); free(y);
  npy_free(nx); npy_free(ny); bin_free(bf);
  return (diff < 1e-5) ? 0 : 1;
}
