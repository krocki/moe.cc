// io.h
#ifndef IO_H
#define IO_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

typedef struct {
  char*  name;
  int    dtype;   // 0=f32, 1=f16, 2=i8, 3=i4
  int    ndim;
  int*   shape;   // length=ndim
  void*  data;    // raw pointer (malloc'ed)
  size_t nbytes;  // size of data in bytes
} TensorBin;

typedef struct {
  TensorBin* arr;
  int count;
} BinFile;

BinFile* bin_load(const char* path);
void bin_free(BinFile* bf);
TensorBin* bin_find(BinFile* bf, const char* name);

typedef struct {
  int    ndim;
  int*   shape;
  float* data;
  size_t nbytes;
} NpyArray;

NpyArray* npy_load_float32(const char* path);
void npy_free(NpyArray* a);

#endif // IO_H

