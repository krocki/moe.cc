#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "io.h"
#include "utils.h"

static void gather_embed(const float* Wemb, int vocab, int d_model,
                         const int32_t* ids, int T, float* x) {
  for (int t = 0; t < T; ++t) {
    int id = (int)ids[t];
    if (id < 0 || id >= vocab) { fprintf(stderr, "bad id %d\n", id); exit(1); }
    const float* row = Wemb + (size_t)id * d_model;
    memcpy(&x[(size_t)t*d_model], row, sizeof(float)*d_model);
  }
}

int main(int argc, char** argv) {
  if (argc < 4) {
    fprintf(stderr, "Usage: %s <weights.bin> <ids.npy:int32> <emb_out.npy:float32>\n", argv[0]);
    return 1;
  }
  const char* wfile = argv[1];
  const char* ids_npy = argv[2];
  const char* y_npy = argv[3];

  fprintf(stderr, "[load] reading weights: %s\n", wfile);
  BinFile* bin = bin_load(wfile);
  if (!bin) { fprintf(stderr, "bin_load failed\n"); return 1; }

  TensorBin* Wemb = bin_find(bin, "model.embed_tokens.weight");
  if (!Wemb) { fprintf(stderr, "missing model.embed_tokens.weight\n"); return 1; }
  int vocab = Wemb->shape[0];
  int d_model = Wemb->shape[1];

  NpyArrayI32* ids = npy_load_int32(ids_npy);
  NpyArray* yref = npy_load_float32(y_npy);
  int T = ids->shape[0];
  if (yref->shape[0]!=T || yref->shape[1]!=d_model){
    fprintf(stderr,"emb_out shape mismatch\n"); return 1;
  }

  float* x = (float*)malloc(sizeof(float)*T*d_model);
  gather_embed((const float*)Wemb->data, vocab, d_model, ids->data, T, x);

  float mad = max_abs_diff(x, yref->data, T*d_model);
  printf("Max abs diff: %.6g\n%s\n", mad, (mad<1e-5f ? "PASS":"FAIL"));

  free(x);
  npy_free_i32(ids);
  npy_free(yref);
  bin_free(bin);
  return (mad<1e-5f)?0:1;
}
