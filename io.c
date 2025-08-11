
#define _POSIX_C_SOURCE 200809L
#include "io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/stat.h>
#include <unistd.h>

static uint32_t read_u32(FILE* f){
  uint32_t v;
  if (fread(&v, 4, 1, f) != 1) { perror("read_u32"); exit(1); }
  return v;
}

static void* xmalloc(size_t n){
  void* p = malloc(n);
  if(!p){ fprintf(stderr, "OOM %zu\n", n); exit(1); }
  return p;
}

static void read_exact(FILE* f, void* buf, size_t n){
  if (fread(buf, 1, n, f) != n) { perror("read_exact"); exit(1); }
}

// -------- progress bar (in-place, no env var) --------
static void progress_draw(size_t done, size_t total){
  if (!total) return;  // nothing to show (e.g., non-regular file)
  int pct = (int)((done * 100.0) / (double)total);
  if (pct < 0) pct = 0; if (pct > 100) pct = 100;
  const int width = 40;
  int fill = (pct * width) / 100;
  // CSI 2K clears the line; \r returns carriage; redraw single line
  fprintf(stderr, "\x1b[2K\rLoading tensors [");
  for (int i = 0; i < width; ++i) fputc(i < fill ? '#' : '.', stderr);
  fprintf(stderr, "] %3d%%", pct);
  fflush(stderr);
}

static void progress_done(void){
  fprintf(stderr, "\x1b[2K\rLoading tensors [########################################] 100%%\n");
  fflush(stderr);
}

// -------- .bin loader (export.py format) --------
BinFile* bin_load(const char* path){
  FILE* f = fopen(path, "rb");
  if(!f){ perror(path); return NULL; }

  // Determine total file size (for progress); if not a regular file, total=0
  struct stat st;
  size_t total_bytes = 0;
  if (fstat(fileno(f), &st) == 0 && S_ISREG(st.st_mode)) {
    total_bytes = (size_t)st.st_size;
  }

  unsigned char magic[6];
  if (fread(magic, 1, 6, f) != 6) { perror("magic"); exit(1); }
  const unsigned char ref[6] = { 'Q','W','3','W',0x00,0x01 };
  if (memcmp(magic, ref, 6) != 0){
    fprintf(stderr, "Bad magic in %s\n", path);
    exit(1);
  }

  uint32_t nt = read_u32(f);
  BinFile* bf = (BinFile*)xmalloc(sizeof(BinFile));
  bf->count = (int)nt;
  bf->arr   = (TensorBin*)xmalloc(sizeof(TensorBin)*nt);

  // initial progress snapshot (after header)
  if (total_bytes){
    size_t done = (size_t)ftell(f);
    progress_draw(done, total_bytes);
  }

  for (uint32_t i=0;i<nt;i++){
    TensorBin* t = &bf->arr[i];

    uint32_t name_len = read_u32(f);
    t->name = (char*)xmalloc(name_len+1);
    read_exact(f, t->name, name_len);
    t->name[name_len] = 0;

    t->dtype = (int)read_u32(f);   // 0=f32, 1=f16, 2=i8, 3=i4
    t->ndim  = (int)read_u32(f);

    t->shape = (int*)xmalloc(sizeof(int)*t->ndim);
    size_t elems = 1;
    for (int d=0; d<t->ndim; ++d){
      t->shape[d] = (int)read_u32(f);
      elems *= (size_t)t->shape[d];
    }

    size_t bpe;
    switch (t->dtype){
      case 0: bpe = 4; break; // f32
      case 1: bpe = 2; break; // f16
      case 2: bpe = 1; break; // i8
      case 3: bpe = 1; break; // i4 (packed across cols by exporter)
      default: fprintf(stderr, "Unknown dtype %d\n", t->dtype); exit(1);
    }

    t->nbytes = elems * bpe;
    t->data   = xmalloc(t->nbytes);
    read_exact(f, t->data, t->nbytes);

    if (total_bytes){
      size_t done = (size_t)ftell(f);
      progress_draw(done, total_bytes);
    }
  }
  fclose(f);
  if (total_bytes) progress_done();
  return bf;
}

void bin_free(BinFile* bf){
  if(!bf) return;
  for (int i=0;i<bf->count;i++){
    free(bf->arr[i].name);
    free(bf->arr[i].shape);
    free(bf->arr[i].data);
  }
  free(bf->arr);
  free(bf);
}

TensorBin* bin_find(BinFile* bf, const char* name){
  for (int i=0;i<bf->count;i++){
    if (strcmp(bf->arr[i].name, name)==0) return &bf->arr[i];
  }
  return NULL;
}

// -------- .npy (float32, LE, C-order) --------
static void read_npy_header(FILE* f, char** header_out, size_t* header_len_out){
  unsigned char magic[6];
  if (fread(magic,1,6,f)!=6){ perror("npy magic"); exit(1); }
  if (memcmp(magic, "\x93NUMPY", 6)!=0){ fprintf(stderr, "Not .npy\n"); exit(1); }
  unsigned char ver[2];
  if (fread(ver,1,2,f)!=2){ perror("npy ver"); exit(1); }
  size_t hlen=0;
  if (ver[0]==1 || ver[0]==2){
    uint16_t l; if (fread(&l,2,1,f)!=1){ perror("npy hlen"); exit(1); } hlen=l;
  } else {
    uint32_t l; if (fread(&l,4,1,f)!=1){ perror("npy hlen"); exit(1); } hlen=l;
  }
  char* header = (char*)xmalloc(hlen+1);
  if (fread(header,1,hlen,f)!=hlen){ perror("npy header"); exit(1); }
  header[hlen]=0;
  *header_out = header; *header_len_out = hlen;
}

// Replace the whole parse_shape_tuple with this
static int parse_shape_tuple(const char* header, int** shape_out){
  const char* p = strstr(header, "'shape': (");
  if(!p){ fprintf(stderr,"npy: shape missing\n"); exit(1); }
  p += strlen("'shape': (");
  const char* end = strchr(p, ')');
  if(!end){ fprintf(stderr,"npy: shape malformed\n"); exit(1); }

  // parse numbers robustly (ignore trailing commas)
  int cap = 4, cnt = 0;
  int* shape = (int*)malloc(sizeof(int)*cap);
  const char* s = p;
  while (s < end){
    // skip spaces/commas
    while (s < end && (*s==' ' || *s==',')) s++;
    if (s >= end) break;
    char* next = NULL;
    long v = strtol(s, &next, 10);
    if (next == s) break;  // nothing parsed (e.g., right before ')')
    if (cnt == cap){
      cap *= 2;
      shape = (int*)realloc(shape, sizeof(int)*cap);
    }
    shape[cnt++] = (int)v;
    s = next;
  }
  if (cnt == 0){ fprintf(stderr,"npy: empty shape tuple\n"); exit(1); }
  // shrink to fit
  shape = (int*)realloc(shape, sizeof(int)*cnt);
  *shape_out = shape;
  return cnt;
}

NpyArray* npy_load_float32(const char* path){
  FILE* f = fopen(path, "rb");
  if(!f){ perror(path); return NULL; }
  char* header=NULL; size_t hlen=0;
  read_npy_header(f,&header,&hlen);
  if (!strstr(header, "'descr': '<f4'")){ fprintf(stderr,"npy: need <f4\n"); exit(1); }
  if (!strstr(header, "'fortran_order': False")){ fprintf(stderr,"npy: need C-order\n"); exit(1); }
  int* shape=NULL; int ndim = parse_shape_tuple(header, &shape);
  free(header);
  size_t elems=1; for (int i=0;i<ndim;i++) elems *= (size_t)shape[i];
  float* data=(float*)xmalloc(elems*sizeof(float));
  if (fread(data,sizeof(float),elems,f)!=elems){ perror("npy data"); exit(1); }
  fclose(f);
  NpyArray* arr=(NpyArray*)xmalloc(sizeof(NpyArray));
  arr->ndim=ndim; arr->shape=shape; arr->data=data; arr->nbytes=elems*sizeof(float);
  return arr;
}

void npy_free(NpyArray* a){
  if(!a) return;
  free(a->shape);
  free(a->data);
  free(a);
}

NpyArrayI32* npy_load_int32(const char* path){
  FILE* f = fopen(path, "rb");
  if(!f){ perror(path); return NULL; }
  char* header=NULL; size_t hlen=0;
  read_npy_header(f,&header,&hlen);
  if (!strstr(header, "'descr': '<i4'")){ fprintf(stderr,"npy: need <i4\n"); exit(1); }
  if (!strstr(header, "'fortran_order': False")){ fprintf(stderr,"npy: need C-order\n"); exit(1); }
  int* shape=NULL; int ndim = parse_shape_tuple(header, &shape);
  free(header);
  size_t elems=1; for (int i=0;i<ndim;i++) elems *= (size_t)shape[i];
  int32_t* data=(int32_t*)malloc(elems*sizeof(int32_t));
  if (fread(data,sizeof(int32_t),elems,f)!=elems){ perror("npy data"); exit(1); }
  fclose(f);
  NpyArrayI32* arr=(NpyArrayI32*)malloc(sizeof(NpyArrayI32));
  arr->ndim=ndim; arr->shape=shape; arr->data=data; arr->nbytes=elems*sizeof(int32_t);
  return arr;
}

void npy_free_i32(NpyArrayI32* a){
  if(!a) return;
  free(a->shape);
  free(a->data);
  free(a);
}

/* ---------- progress bar (same style as test_model.c) ---------- */
void print_progress_bar(size_t done, size_t total){
  const int width = 40;
  int filled = (total == 0) ? width : (int)((done * width) / total);
  if (filled < 0) filled = 0; if (filled > width) filled = width;
  fputs("\rLoading tensors [", stderr);
  for (int i=0;i<filled;i++) fputc('#', stderr);
  for (int i=filled;i<width;i++) fputc(' ', stderr);
  fputs("] ", stderr);
  fprintf(stderr, "%3zu%%", (total==0)?100:(done*100/total));
  fflush(stderr);
}

void finish_progress_bar(void){
  fputs("\n", stderr);
}
