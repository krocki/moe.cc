#ifndef IO_H
#define IO_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>

typedef struct {
  char*  name;
  int    dtype;   // 0=f32, 1=f16, 2=i8, 3=i4
  int    ndim;
  int*   shape;   // length=ndim
  void*  data;    // raw pointer (malloc'ed)
  size_t nbytes;  // size of data in bytes
  size_t group_size; // Group size for quantized tensors (0 = rowwise or non-quantized)
} TensorBin;

typedef struct {
  TensorBin* arr;
  int count;
} BinFile;

BinFile* bin_load(const char* path);
void bin_free(BinFile* bf);
TensorBin* bin_find(BinFile* bf, const char* name);

// Streaming I/O structures and functions for memory-efficient processing
typedef struct {
  FILE* file;               // Input file handle
  size_t total_bytes;       // Total file size for progress tracking
  size_t bytes_read;        // Bytes read so far
  int file_version;         // Binary format version (1 or 2)
  uint32_t total_tensors;   // Total tensor count from header
  uint32_t tensors_read;    // Tensors processed so far
} BinStreamReader;

typedef struct {
  FILE* file;               // Output file handle
  int file_version;         // Binary format version to write (2 for group_size support)
  uint32_t tensor_count;    // Number of tensors written
} BinStreamWriter;

// Streaming reader functions
BinStreamReader* bin_stream_reader_open(const char* path);
int bin_stream_reader_next_tensor(BinStreamReader* reader, TensorBin* tensor);
void bin_stream_reader_close(BinStreamReader* reader);

// Streaming writer functions  
BinStreamWriter* bin_stream_writer_open(const char* path);
int bin_stream_writer_write_tensor(BinStreamWriter* writer, const TensorBin* tensor);
int bin_stream_writer_finalize(BinStreamWriter* writer);
void bin_stream_writer_close(BinStreamWriter* writer);

// Additional functions for saving/writing tensors
int bin_save(const BinFile* bf, const char* path);
int bin_save_single_tensor(const TensorBin* tensor, const char* path);
TensorBin* tensor_create(const char* name, int dtype, int ndim, const int* shape, const void* data);
TensorBin* tensor_create_with_group_size(const char* name, int dtype, int ndim, const int* shape, const void* data, size_t group_size);
void tensor_free_single(TensorBin* tensor);
BinFile* binfile_create(void);
int binfile_add_tensor(BinFile* bf, const TensorBin* tensor);

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

// Progress tracking functions
void progress_draw(size_t done_bytes, size_t total_bytes, size_t done_tensors, size_t total_tensors, const char* current_tensor);
void progress_done(void);
void print_progress_bar(size_t done, size_t total);
void finish_progress_bar(void);

#endif // IO_H
