#include <stdio.h>
#include "io.h"

int main(int argc, char** argv){
  if (argc != 2){
    fprintf(stderr, "Usage: %s <weights.bin>\n", argv[0]);
    return 1;
  }
  BinFile* bf = bin_load(argv[1]);
  if(!bf){ fprintf(stderr, "Failed to load %s\n", argv[1]); return 1; }
  for (int i=0; i<bf->count; ++i){
    TensorBin* t = &bf->arr[i];
    printf("%s\n", t->name);
  }
  bin_free(bf);
  return 0;
}
