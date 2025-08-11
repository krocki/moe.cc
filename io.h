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

typedef struct {
  int    ndim;
  int*   shape;
  int32_t* data;
  size_t nbytes;
} NpyArrayI32;

NpyArrayI32* npy_load_int32(const char* path);
void npy_free_i32(NpyArrayI32* a);

void print_progress_bar(size_t done, size_t total);
void finish_progress_bar(void);

#endif // IO_H
