#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "io.h"
#include "utils.h"
#include "kernels.h"

static TensorBin* need(BinFile* b, const char* k){
  TensorBin* t = bin_find(b, k);
  if(!t){ fprintf(stderr,"Key not found: %s\n", k); exit(1); }
  return t;
}
static TensorBin* maybe(BinFile* b, const char* k){ return bin_find(b, k); }

static int infer_head_dim(int Dq, int Dkv){
  int cand[] = {128,96,80,64,48,40,32};
  for (int i=0;i<7;++i) if (Dq % cand[i] == 0 && Dkv % cand[i] == 0) return cand[i];
  int a=Dq,b=Dkv; while (b){ int r=a%b; a=b; b=r; } return a?a:Dq;
}

int main(int argc, char** argv){
  if (argc < 6){
    fprintf(stderr, "Usage: %s <weights.bin> <x.npy> <y.npy> <causal:0|1> <rope_theta>\n", argv[0]);
    return 1;
  }
  const char* wfile = argv[1];
  const char* xfile = argv[2];
  const char* yfile = argv[3];
  int causal = atoi(argv[4]);
  float rope_theta = strtof(argv[5], NULL);

  BinFile* bf = bin_load(wfile);
  if (!bf) { fprintf(stderr,"bin load fail\n"); return 1; }

  TensorBin* Wq = need(bf, "model.layers.0.self_attn.q_proj.weight");
  TensorBin* Wk = need(bf, "model.layers.0.self_attn.k_proj.weight");
  TensorBin* Wv = need(bf, "model.layers.0.self_attn.v_proj.weight");
  TensorBin* Wo = need(bf, "model.layers.0.self_attn.o_proj.weight");
  TensorBin* bq = maybe(bf, "model.layers.0.self_attn.q_proj.bias");
  TensorBin* bk = maybe(bf, "model.layers.0.self_attn.k_proj.bias");
  TensorBin* bv = maybe(bf, "model.layers.0.self_attn.v_proj.bias");
  TensorBin* bo = maybe(bf, "model.layers.0.self_attn.o_proj.bias");
  TensorBin* qn = maybe(bf, "model.layers.0.self_attn.q_norm.weight");
  TensorBin* kn = maybe(bf, "model.layers.0.self_attn.k_norm.weight");

  int Dq = Wq->shape[0], Dkv = Wk->shape[0], d_model = Wq->shape[1];
  int head_dim = infer_head_dim(Dq, Dkv);
  int n_q = Dq / head_dim, n_kv = Dkv / head_dim;
  int qn_len = qn ? qn->shape[0] : 0;
  int kn_len = kn ? kn->shape[0] : 0;

  NpyArray* nx = npy_load_float32(xfile);
  NpyArray* ny = npy_load_float32(yfile);
  if (!nx || !ny){ fprintf(stderr,"npy load fail\n"); return 1; }
  if (nx->shape[1]!=d_model || ny->shape[1]!=d_model || ny->shape[0]!=nx->shape[0]){
    fprintf(stderr,"shape mismatch\n"); return 1;
  }
  int T = nx->shape[0];

  size_t need_f = (size_t)T*(Dq + 2*Dkv) + (size_t)T*T + (size_t)T*Dq;
  float* scratch = (float*)malloc(sizeof(float)*need_f);
  float* y = (float*)malloc(sizeof(float)*T*d_model);

  attn_forward_f32_gqa(
    nx->data, T, d_model,
    (const float*)Wq->data, bq?(const float*)bq->data:NULL,
    (const float*)Wk->data, bk?(const float*)bk->data:NULL,
    (const float*)Wv->data, bv?(const float*)bv->data:NULL,
    (const float*)Wo->data, bo?(const float*)bo->data:NULL,
    qn?(const float*)qn->data:NULL, head_dim,
    kn?(const float*)kn->data:NULL, head_dim,
    n_q, n_kv, head_dim, causal, rope_theta,
    scratch, y
  );

  double diff = max_abs_diff(y, ny->data, T*d_model);
  printf("Max abs diff: %g\n", diff);
  printf("%s\n", (diff < 1e-5) ? "PASS" : "FAIL");

  free(scratch); free(y);
  npy_free(nx); npy_free(ny); bin_free(bf);
  return (diff < 1e-5) ? 0 : 1;
}
