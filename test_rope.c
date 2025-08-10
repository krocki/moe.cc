#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "io.h"
#include "utils.h"
#include "kernels.h"

// This test loads Q.npy and K.npy (pre-RoPE) and YQ.npy, YK.npy (post-RoPE goldens),
// then applies rope_apply_inplace_f32_gqa in C and compares.

static void expect_shape(NpyArray* a, int T, int D, const char* name){
  if (a->ndim != 2 || a->shape[0] != T || a->shape[1] != D){
    fprintf(stderr, "%s shape mismatch, expected [%d,%d], got [%d,%d]\n",
            name, T, D, a->shape[0], a->shape[1]);
    exit(1);
  }
}

int main(int argc, char** argv){
  if (argc < 10){
    fprintf(stderr, "Usage: %s <Q.npy> <K.npy> <YQ.npy> <YK.npy> <T> <n_q> <n_kv> <head_dim> <theta> [pos0]\n", argv[0]);
    return 1;
  }
  const char* qfile  = argv[1];
  const char* kfile  = argv[2];
  const char* yqfile = argv[3];
  const char* ykfile = argv[4];
  int T        = atoi(argv[5]);
  int n_q      = atoi(argv[6]);
  int n_kv     = atoi(argv[7]);
  int head_dim = atoi(argv[8]);
  float theta  = strtof(argv[9], NULL);
  int pos0     = (argc >= 11) ? atoi(argv[10]) : 0;

  NpyArray* Q  = npy_load_float32(qfile);
  NpyArray* K  = npy_load_float32(kfile);
  NpyArray* YQ = npy_load_float32(yqfile);
  NpyArray* YK = npy_load_float32(ykfile);

  int Dq  = n_q  * head_dim;
  int Dkv = n_kv * head_dim;
  expect_shape(Q,  T, Dq,  "Q");
  expect_shape(K,  T, Dkv, "K");
  expect_shape(YQ, T, Dq,  "YQ");
  expect_shape(YK, T, Dkv, "YK");

  float* q = (float*)malloc(sizeof(float)*T*Dq);
  float* k = (float*)malloc(sizeof(float)*T*Dkv);
  memcpy(q, Q->data,  sizeof(float)*T*Dq);
  memcpy(k, K->data,  sizeof(float)*T*Dkv);

  rope_apply_inplace_f32_gqa(q, k, T, n_q, n_kv, head_dim, pos0, theta);

  float diff_q = max_abs_diff(q,  YQ->data, T*Dq);
  float diff_k = max_abs_diff(k,  YK->data, T*Dkv);
  printf("Q diff: %.6g  K diff: %.6g\n", diff_q, diff_k);
  printf("%s\n", (diff_q < 1e-6f && diff_k < 1e-6f) ? "PASS" : "FAIL");

  free(q); free(k);
  npy_free(Q); npy_free(K); npy_free(YQ); npy_free(YK);
  return (diff_q < 1e-6f && diff_k < 1e-6f) ? 0 : 1;
}
