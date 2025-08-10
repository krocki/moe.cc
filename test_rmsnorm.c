#include <stdio.h>
#include <stdlib.h>
#include "io.h"
#include "utils.h"
#include "kernels.h"

int main(int argc, char** argv) {
  if (argc != 6) {
    fprintf(stderr, "Usage: %s <weights.bin> <x.npy> <y.npy> <weight_key> <eps>\n", argv[0]);
    return 1;
  }
  const char* wfile = argv[1];
  const char* xfile = argv[2];
  const char* yfile = argv[3];
  const char* wkey  = argv[4];
  float eps = strtof(argv[5], NULL);

  BinFile* bf = bin_load(wfile);
  TensorBin* tw = bin_find(bf, wkey);
  if (!tw) {
    fprintf(stderr, "Key not found: %s\n", wkey);
    return 1;
  }
  int d_model = tw->shape[0];

  NpyArray* nx = npy_load_float32(xfile);
  NpyArray* ny = npy_load_float32(yfile);
  int T = nx->shape[0];

  float* y = malloc(sizeof(float) * T * d_model);
  rmsnorm_forward_f32(nx->data, (float*)tw->data, T, d_model, eps, y);

  float diff = max_abs_diff(y, ny->data, T * d_model);
  printf("Max abs diff: %.6g\n", diff);
  printf("%s\n", (diff < 1e-6f) ? "PASS" : "FAIL");

  free(y);
  npy_free(nx);
  npy_free(ny);
  bin_free(bf);
  return 0;
}
