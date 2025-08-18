
#define _POSIX_C_SOURCE 200809L
#include "io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>
#include <math.h>

// ---- progress state / stats ----
static struct timespec g_progress_start;
static int g_progress_started = 0;
// static size_t g_progress_total_bytes = 0; // Unused variable
static size_t g_tensor_total = 0;
static size_t g_tensor_loaded = 0;

// group stats by (dtype, shape)
typedef struct {
  char* key;         // e.g., "f32 [a,b,c]"
  size_t count;
  size_t bytes;      // accumulated bytes
} GroupStat;

static GroupStat* g_groups = NULL;
static size_t g_groups_n = 0, g_groups_cap = 0;

static size_t g_dtype_counts[4] = {0,0,0,0};
static size_t g_dtype_bytes[4]  = {0,0,0,0};
static size_t g_rms_count = 0;

// util
static const char* dtype_str(int dt){
  switch(dt){
    case 0: return "f32";
    case 1: return "f16";
    case 2: return "i8";
    case 3: return "i4";
    default: return "unk";
  }
}

static double now_monotonic(){
  struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + (double)t.tv_nsec*1e-9;
}

static int strcasestr_contains(const char* hay, const char* needle){
  if(!hay || !needle) return 0;
  for(const char* p = hay; *p; ++p){
    const char* a=p; const char* b=needle;
    while(*a && *b){
      char ca = *a; if(ca>='A' && ca<='Z') ca += 'a'-'A';
      char cb = *b; if(cb>='A' && cb<='Z') cb += 'a'-'A';
      if(ca != cb) break;
      ++a; ++b;
    }
    if(!*b) return 1;
  }
  return 0;
}

static char* shape_key(int dtype, int ndim, const int* shape){
  // build "f32 [a,b,c]" string
  char buf[256];
  int off = snprintf(buf, sizeof(buf), "%s [", dtype_str(dtype));
  for (int i=0;i<ndim;i++){
    off += snprintf(buf+off, sizeof(buf)-off, "%d%s", shape[i], (i+1<ndim)?",":"]");
    if (off >= (int)sizeof(buf)-8) break;
  }
  char* s = (char*)malloc(off+1);
  memcpy(s, buf, off+1);
  return s;
}

static void add_group_stat(int dtype, int ndim, const int* shape, size_t bytes){
  char* key = shape_key(dtype, ndim, shape);
  // find existing
  for (size_t i=0;i<g_groups_n;i++){
    if (strcmp(g_groups[i].key, key)==0){
      g_groups[i].count += 1;
      g_groups[i].bytes += bytes;
      free(key);
      return;
    }
  }
  if (g_groups_n == g_groups_cap){
    g_groups_cap = g_groups_cap? g_groups_cap*2 : 16;
    g_groups = (GroupStat*)realloc(g_groups, g_groups_cap*sizeof(GroupStat));
  }
  g_groups[g_groups_n].key = key;
  g_groups[g_groups_n].count = 1;
  g_groups[g_groups_n].bytes = bytes;
  g_groups_n++;
}

// ---- progress tracking and display ----
static struct timespec g_progress_start = {0,0};
static int g_progress_multiline_started = 0;

// Progress line - Made public for streaming convert usage
void progress_draw(size_t done_bytes, size_t total_bytes, size_t loaded, size_t total_tensors, const char* name){
    if (!total_bytes) return;

    if (!g_progress_started) {
        clock_gettime(CLOCK_MONOTONIC, &g_progress_start);
        g_progress_started = 1;
    }

    double t_now = now_monotonic();
    double t_start = g_progress_start.tv_sec + g_progress_start.tv_nsec * 1e-9;
    double elapsed = t_now - t_start;
    if (elapsed <= 1e-9) elapsed = 1e-9;

    double done_gib = (double)done_bytes / (1ull<<30);
    double total_gib = (double)total_bytes / (1ull<<30);
    double speed = done_gib / elapsed;
    double rem_bytes = (done_bytes < total_bytes) ? total_bytes - done_bytes : 0;
    double eta_s = (speed > 1e-12) ? (rem_bytes / (1ull<<30)) / speed : -1;

    int pct = (int)((done_bytes * 100.0) / total_bytes);
    if (pct < 0) pct = 0; else if (pct > 100) pct = 100;

    const int width = 20;
    char bar[width+1];
    int filled = (pct * width) / 100;
    for (int i = 0; i < width; i++) bar[i] = (i < filled ? '#' : '.');
    bar[width] = '\0';

    char shown[96] = "";
    if (name && *name) {
        int maxn = 80;
        int n = strlen(name);
        if (n <= maxn) snprintf(shown, sizeof(shown), "%s", name);
        else snprintf(shown, sizeof(shown), "%.*s…", maxn-1, name);
    }

    long s = eta_s >= 0 ? (long)(eta_s + 0.5) : -1;
    long h = (s >= 0) ? (s / 3600) : 0;
    long m = (s >= 0) ? (s / 60) % 60 : 0;
    long sec = (s >= 0) ? (s % 60) : 0;
    char etabuf[16];
    if (eta_s < 0) snprintf(etabuf, sizeof(etabuf), "--:--:--");
    else snprintf(etabuf, sizeof(etabuf), "%02ld:%02ld:%02ld", h, m, sec);

    if (g_progress_multiline_started) {
        fprintf(stderr, "\x1b[1A"); // move up one line
    } else {
        g_progress_multiline_started = 1;
    }

    // Line 1
    fprintf(stderr, "\x1b[2K\rLoading tensors [%s] %3d%% %.2f GiB/s  (%.2f/%.2f GiB)  ETA %s\n",
            bar, pct, speed, done_gib, total_gib, etabuf);

    // Line 2
    fprintf(stderr, "\x1b[2K\r%8zu %s%s ", loaded,
            shown[0] ? "" : "tensor(s) ", shown);

}

void progress_done(void){
  if (!g_progress_started){ fputs("\n", stderr); return; }
  const int width = 20;
  fputs("\x1b[2K\rLoading tensors [", stderr);
  for (int i=0;i<width;i++) fputc('#', stderr);
  fputs("]  100%\n", stderr);
  fflush(stderr);
}


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
  const unsigned char ref_v1[6] = { 'Q','W','3','W',0x00,0x01 };
  const unsigned char ref_v2[6] = { 'Q','W','3','W',0x00,0x02 };
  int file_version = 1;
  if (memcmp(magic, ref_v1, 6) == 0) {
    file_version = 1;
  } else if (memcmp(magic, ref_v2, 6) == 0) {
    file_version = 2;
  } else {
    fprintf(stderr, "Bad magic in %s\n", path);
    exit(1);
  }

  uint32_t nt = read_u32(f);
  // Note: nt is the original tensor count, but file may contain more entries
  // when some tensors are quantized (each becomes 2 entries: .q8/.q4 + .scale)
  // So we'll grow the array dynamically and count actual entries loaded
  
  BinFile* bf = (BinFile*)xmalloc(sizeof(BinFile));
  bf->count = 0;  // Start with 0, increment as we load
  size_t capacity = nt * 2; // Initial capacity (assumes some quantization)
  bf->arr = (TensorBin*)xmalloc(sizeof(TensorBin)*capacity);
  
  g_tensor_total = 0; // Will be updated as we discover actual count
  g_tensor_loaded = 0;

  // initial progress snapshot (after header)
  if (total_bytes){
    size_t done = (size_t)ftell(f);
    progress_draw(done, total_bytes, g_tensor_loaded, g_tensor_total, "");
  }

  // Load tensors until EOF (dynamic count due to quantization)
  while (!feof(f)) {
    // Check if we need to grow the array
    if ((size_t)bf->count >= capacity) {
      capacity *= 2;
      bf->arr = (TensorBin*)realloc(bf->arr, sizeof(TensorBin)*capacity);
      if (!bf->arr) { fprintf(stderr, "Failed to resize tensor array\n"); exit(1); }
    }
    
    TensorBin* t = &bf->arr[bf->count];
    
    // Try to read name length - if EOF, we're done
    uint32_t name_len;
    if (fread(&name_len, sizeof(uint32_t), 1, f) != 1) break;
    
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
    
    // Read group_size for v2 files (0 for v1 files)
    if (file_version >= 2) {
      t->group_size = (size_t)read_u32(f);
    } else {
      t->group_size = 0; // Default for v1 files (rowwise or non-quantized)
    }

    // Increment count and stats
    bf->count++;
    g_tensor_loaded++;
    g_tensor_total = bf->count; // Update total as we discover entries
    if (t->dtype>=0 && t->dtype<4){ g_dtype_counts[t->dtype]++; g_dtype_bytes[t->dtype]+=t->nbytes; }
    add_group_stat(t->dtype, t->ndim, t->shape, t->nbytes);
    if (strcasestr_contains(t->name, "rms")) g_rms_count++;

    if (total_bytes){
      size_t done = (size_t)ftell(f);
      progress_draw(done, total_bytes, g_tensor_loaded, g_tensor_total, t->name);
    }
  }
  fclose(f);
  if (total_bytes) progress_done();
// timing summary
double t_end = now_monotonic();
double t_start = (double)g_progress_start.tv_sec + (double)g_progress_start.tv_nsec*1e-9;
double elapsed = (g_progress_started? (t_end - t_start) : 0.0);
double total_gib = (double)total_bytes / (double)(1ull<<30);
double speed_gib_s = (elapsed>0)? (total_gib/elapsed) : 0.0;
fprintf(stderr, "Loaded %.3f GiB in %.2f s (%.3f GiB/s)\n", total_gib, elapsed, speed_gib_s);

// tensor stats
fprintf(stderr, "Tensors loaded: %zu\n", g_tensor_loaded);
// by dtype
const char* dnames[4] = {"f32","f16","i8","i4"};
for (int d=0; d<4; ++d){
  if (g_dtype_counts[d]){
    double gib = (double)g_dtype_bytes[d] / (double)(1ull<<30);
    fprintf(stderr, "  %-3s: %zu tensors, %.3f GiB\n", dnames[d], g_dtype_counts[d], gib);
  }
}
// top groups (by (dtype,shape))
for (size_t i=0; i<g_groups_n && i<50; ++i){ // show up to 50 lines max
  double gib = (double)g_groups[i].bytes / (double)(1ull<<30);
  fprintf(stderr, "  %s : %zu tensors, %.3f GiB\n", g_groups[i].key, g_groups[i].count, gib);
}
if (g_rms_count){
  fprintf(stderr, "  rms/rmsnorm tensors: %zu\n", g_rms_count);
}
// cleanup group keys
for (size_t i=0;i<g_groups_n;i++) free(g_groups[i].key);
free(g_groups); g_groups=NULL; g_groups_n=g_groups_cap=0;
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
  const int width = 20;
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

/**
 * Write a single tensor to a file in the same format as export.py
 * Includes magic header, tensor count (1), and tensor data
 */
static void write_tensor_to_file(FILE* f, const TensorBin* tensor) {
    // Write tensor name
    uint32_t name_len = (uint32_t)strlen(tensor->name);
    fwrite(&name_len, sizeof(uint32_t), 1, f);
    fwrite(tensor->name, 1, name_len, f);
    
    // Write tensor metadata
    uint32_t dtype = (uint32_t)tensor->dtype;
    uint32_t ndim = (uint32_t)tensor->ndim;
    fwrite(&dtype, sizeof(uint32_t), 1, f);
    fwrite(&ndim, sizeof(uint32_t), 1, f);
    
    // Write shape
    for (int i = 0; i < tensor->ndim; i++) {
        uint32_t dim = (uint32_t)tensor->shape[i];
        fwrite(&dim, sizeof(uint32_t), 1, f);
    }
    
    // Write tensor data
    fwrite(tensor->data, 1, tensor->nbytes, f);
    
    // Write group_size (v2 format)
    uint32_t group_size = (uint32_t)tensor->group_size;
    fwrite(&group_size, sizeof(uint32_t), 1, f);
}

/**
 * Save a complete BinFile to disk with proper magic header
 * Returns 0 on success, -1 on error
 */
int bin_save(const BinFile* bf, const char* path) {
    if (!bf || !path) return -1;
    
    FILE* f = fopen(path, "wb");
    if (!f) {
        perror(path);
        return -1;
    }
    
    // Write magic header (v2 format with group_size support)
    const unsigned char magic[6] = { 'Q','W','3','W',0x00,0x02 };
    fwrite(magic, 1, 6, f);
    
    // Write tensor count
    uint32_t count = (uint32_t)bf->count;
    fwrite(&count, sizeof(uint32_t), 1, f);
    
    // Write all tensors
    for (int i = 0; i < bf->count; i++) {
        write_tensor_to_file(f, &bf->arr[i]);
    }
    
    fclose(f);
    return 0;
}

/**
 * Save a single tensor to disk as a standalone .bin file
 * Returns 0 on success, -1 on error
 */
int bin_save_single_tensor(const TensorBin* tensor, const char* path) {
    if (!tensor || !path) return -1;
    
    FILE* f = fopen(path, "wb");
    if (!f) {
        perror(path);
        return -1;
    }
    
    // Write magic header (v2 format with group_size support)
    const unsigned char magic[6] = { 'Q','W','3','W',0x00,0x02 };
    fwrite(magic, 1, 6, f);
    
    // Write tensor count (1)
    uint32_t count = 1;
    fwrite(&count, sizeof(uint32_t), 1, f);
    
    // Write the tensor
    write_tensor_to_file(f, tensor);
    
    fclose(f);
    return 0;
}

/**
 * Create a new tensor with the given parameters
 * Allocates memory and copies the data
 */
TensorBin* tensor_create(const char* name, int dtype, int ndim, const int* shape, const void* data) {
    if (!name || !shape || !data || ndim <= 0) return NULL;
    
    TensorBin* tensor = (TensorBin*)malloc(sizeof(TensorBin));
    if (!tensor) return NULL;
    
    // Copy name
    size_t name_len = strlen(name);
    tensor->name = (char*)malloc(name_len + 1);
    if (!tensor->name) {
        free(tensor);
        return NULL;
    }
    strcpy(tensor->name, name);
    
    // Copy shape
    tensor->shape = (int*)malloc(sizeof(int) * ndim);
    if (!tensor->shape) {
        free(tensor->name);
        free(tensor);
        return NULL;
    }
    memcpy(tensor->shape, shape, sizeof(int) * ndim);
    
    // Calculate data size based on dtype and shape
    size_t elements = 1;
    for (int i = 0; i < ndim; i++) {
        elements *= (size_t)shape[i];
    }
    
    size_t bytes_per_element;
    switch (dtype) {
        case 0: bytes_per_element = 4; break; // f32
        case 1: bytes_per_element = 2; break; // f16
        case 2: bytes_per_element = 1; break; // i8
        case 3: bytes_per_element = 1; break; // i4 (packed)
        default:
            free(tensor->shape);
            free(tensor->name);
            free(tensor);
            return NULL;
    }
    
    size_t data_size = elements * bytes_per_element;
    
    // Copy data
    tensor->data = malloc(data_size);
    if (!tensor->data) {
        free(tensor->shape);
        free(tensor->name);
        free(tensor);
        return NULL;
    }
    memcpy(tensor->data, data, data_size);
    
    tensor->dtype = dtype;
    tensor->ndim = ndim;
    tensor->nbytes = data_size;
    tensor->group_size = 0; // Default to 0 (rowwise or non-quantized)
    
    return tensor;
}

/**
 * Create a TensorBin with specified group_size metadata
 * Used for quantized tensors that need group_size information
 */
TensorBin* tensor_create_with_group_size(const char* name, int dtype, int ndim, const int* shape, const void* data, size_t group_size) {
    TensorBin* tensor = tensor_create(name, dtype, ndim, shape, data);
    if (tensor) {
        tensor->group_size = group_size;
    }
    return tensor;
}

/**
 * Free a single tensor (used when tensor is not part of a BinFile array)
 */
void tensor_free_single(TensorBin* tensor) {
    if (!tensor) return;
    
    free(tensor->name);
    free(tensor->shape);
    free(tensor->data);
    free(tensor);
}

/**
 * Create an empty BinFile structure
 */
BinFile* binfile_create(void) {
    BinFile* bf = (BinFile*)malloc(sizeof(BinFile));
    if (!bf) return NULL;
    
    bf->arr = NULL;
    bf->count = 0;
    return bf;
}

/**
 * Add a tensor to a BinFile (copies the tensor data)
 * Returns 0 on success, -1 on error
 */
int binfile_add_tensor(BinFile* bf, const TensorBin* tensor) {
    if (!bf || !tensor) return -1;
    
    // Resize array if needed
    TensorBin* new_arr = (TensorBin*)realloc(bf->arr, sizeof(TensorBin) * (bf->count + 1));
    if (!new_arr) return -1;
    
    bf->arr = new_arr;
    
    // Copy tensor data
    TensorBin* dest = &bf->arr[bf->count];
    
    // Copy name
    size_t name_len = strlen(tensor->name);
    dest->name = (char*)malloc(name_len + 1);
    if (!dest->name) return -1;
    strcpy(dest->name, tensor->name);
    
    // Copy shape
    dest->shape = (int*)malloc(sizeof(int) * tensor->ndim);
    if (!dest->shape) {
        free(dest->name);
        return -1;
    }
    memcpy(dest->shape, tensor->shape, sizeof(int) * tensor->ndim);
    
    // Copy data
    dest->data = malloc(tensor->nbytes);
    if (!dest->data) {
        free(dest->shape);
        free(dest->name);
        return -1;
    }
    memcpy(dest->data, tensor->data, tensor->nbytes);
    
    dest->dtype = tensor->dtype;
    dest->ndim = tensor->ndim;
    dest->nbytes = tensor->nbytes;
    dest->group_size = tensor->group_size;  // CRITICAL FIX: Copy group_size field
    
    bf->count++;
    return 0;
}

/**
 * STREAMING I/O FUNCTIONS FOR MEMORY-EFFICIENT TENSOR PROCESSING
 * 
 * These functions enable processing large binary files tensor-by-tensor
 * without loading the entire file into memory, significantly reducing
 * memory usage during quantization conversion.
 */

/**
 * Open a binary file for streaming tensor reading
 * 
 * @param path Path to the binary file to read
 * @return BinStreamReader* on success, NULL on failure
 * 
 * This function reads and validates the file header, determines the file version,
 * and initializes streaming state for tensor-by-tensor processing.
 */
BinStreamReader* bin_stream_reader_open(const char* path) {
  FILE* f = fopen(path, "rb");
  if (!f) {
    perror(path);
    return NULL;
  }

  // Determine total file size for progress tracking
  struct stat st;
  size_t total_bytes = 0;
  if (fstat(fileno(f), &st) == 0 && S_ISREG(st.st_mode)) {
    total_bytes = (size_t)st.st_size;
  }

  // Read and validate magic header
  unsigned char magic[6];
  if (fread(magic, 1, 6, f) != 6) {
    perror("Failed to read magic header");
    fclose(f);
    return NULL;
  }

  // Determine file version
  const unsigned char ref_v1[6] = { 'Q','W','3','W',0x00,0x01 };
  const unsigned char ref_v2[6] = { 'Q','W','3','W',0x00,0x02 };
  int file_version = 1;
  if (memcmp(magic, ref_v1, 6) == 0) {
    file_version = 1;
  } else if (memcmp(magic, ref_v2, 6) == 0) {
    file_version = 2;
  } else {
    fprintf(stderr, "Invalid magic header in %s\n", path);
    fclose(f);
    return NULL;
  }

  // Read tensor count from header
  uint32_t total_tensors = read_u32(f);

  // Allocate and initialize stream reader
  BinStreamReader* reader = (BinStreamReader*)malloc(sizeof(BinStreamReader));
  if (!reader) {
    fprintf(stderr, "Failed to allocate stream reader\n");
    fclose(f);
    return NULL;
  }

  reader->file = f;
  reader->total_bytes = total_bytes;
  reader->bytes_read = ftell(f);  // Current position after header
  reader->file_version = file_version;
  reader->total_tensors = total_tensors;
  reader->tensors_read = 0;

  return reader;
}

/**
 * Read the next tensor from a streaming binary file
 * 
 * @param reader The stream reader instance
 * @param tensor Tensor structure to populate (caller must free tensor data)
 * @return 1 if tensor read successfully, 0 if EOF reached, -1 on error
 * 
 * This function reads one tensor from the stream, allocating memory for
 * tensor data. The caller is responsible for freeing tensor->name, 
 * tensor->shape, and tensor->data when done.
 */
int bin_stream_reader_next_tensor(BinStreamReader* reader, TensorBin* tensor) {
  if (!reader || !reader->file || !tensor) {
    return -1;
  }

  FILE* f = reader->file;

  // Try to read name length - if EOF, we're done
  uint32_t name_len;
  if (fread(&name_len, sizeof(uint32_t), 1, f) != 1) {
    if (feof(f)) {
      return 0;  // EOF reached normally
    }
    return -1;  // Read error
  }

  // Read tensor name
  tensor->name = (char*)xmalloc(name_len + 1);
  read_exact(f, tensor->name, name_len);
  tensor->name[name_len] = 0;

  // Read tensor metadata
  tensor->dtype = (int)read_u32(f);   // 0=f32, 1=f16, 2=i8, 3=i4
  tensor->ndim = (int)read_u32(f);

  // Read tensor shape
  tensor->shape = (int*)xmalloc(sizeof(int) * tensor->ndim);
  size_t elems = 1;
  for (int d = 0; d < tensor->ndim; ++d) {
    tensor->shape[d] = (int)read_u32(f);
    elems *= (size_t)tensor->shape[d];
  }

  // Calculate tensor data size
  size_t bpe;
  switch (tensor->dtype) {
    case 0: bpe = 4; break; // f32
    case 1: bpe = 2; break; // f16
    case 2: bpe = 1; break; // i8
    case 3: bpe = 1; break; // i4 (packed)
    default:
      fprintf(stderr, "Unknown dtype %d\n", tensor->dtype);
      free(tensor->name);
      free(tensor->shape);
      return -1;
  }

  tensor->nbytes = elems * bpe;
  tensor->data = xmalloc(tensor->nbytes);
  read_exact(f, tensor->data, tensor->nbytes);

  // Read group_size for v2 files (0 for v1 files)
  if (reader->file_version >= 2) {
    tensor->group_size = (size_t)read_u32(f);
  } else {
    tensor->group_size = 0; // Default for v1 files
  }

  // Update streaming state
  reader->tensors_read++;
  reader->bytes_read = ftell(f);

  return 1; // Successfully read tensor
}

/**
 * Close a streaming binary file reader and free resources
 * 
 * @param reader The stream reader instance to close
 */
void bin_stream_reader_close(BinStreamReader* reader) {
  if (!reader) return;
  
  if (reader->file) {
    fclose(reader->file);
  }
  free(reader);
}

/**
 * Open a binary file for streaming tensor writing
 * 
 * @param path Path to the output binary file
 * @return BinStreamWriter* on success, NULL on failure
 * 
 * This function creates a new binary file and writes the header.
 * The tensor count will be updated when the file is finalized.
 */
BinStreamWriter* bin_stream_writer_open(const char* path) {
  FILE* f = fopen(path, "wb");
  if (!f) {
    perror(path);
    return NULL;
  }

  // Write magic header (v2 format for group_size support)
  const unsigned char magic[6] = { 'Q','W','3','W',0x00,0x02 };
  if (fwrite(magic, 1, 6, f) != 6) {
    perror("Failed to write magic header");
    fclose(f);
    return NULL;
  }

  // Write placeholder tensor count (will be updated on finalize)
  uint32_t placeholder_count = 0;
  if (fwrite(&placeholder_count, sizeof(uint32_t), 1, f) != 1) {
    perror("Failed to write tensor count placeholder");
    fclose(f);
    return NULL;
  }

  // Allocate and initialize stream writer
  BinStreamWriter* writer = (BinStreamWriter*)malloc(sizeof(BinStreamWriter));
  if (!writer) {
    fprintf(stderr, "Failed to allocate stream writer\n");
    fclose(f);
    return NULL;
  }

  writer->file = f;
  writer->file_version = 2;  // Always write v2 format for group_size support
  writer->tensor_count = 0;

  return writer;
}

/**
 * Write a tensor to the streaming output file
 * 
 * @param writer The stream writer instance
 * @param tensor The tensor to write
 * @return 0 on success, -1 on error
 */
int bin_stream_writer_write_tensor(BinStreamWriter* writer, const TensorBin* tensor) {
  if (!writer || !writer->file || !tensor) {
    return -1;
  }

  FILE* f = writer->file;

  // Write tensor name
  uint32_t name_len = (uint32_t)strlen(tensor->name);
  if (fwrite(&name_len, sizeof(uint32_t), 1, f) != 1) return -1;
  if (fwrite(tensor->name, 1, name_len, f) != name_len) return -1;

  // Write tensor metadata
  uint32_t dtype = (uint32_t)tensor->dtype;
  uint32_t ndim = (uint32_t)tensor->ndim;
  if (fwrite(&dtype, sizeof(uint32_t), 1, f) != 1) return -1;
  if (fwrite(&ndim, sizeof(uint32_t), 1, f) != 1) return -1;

  // Write tensor shape
  for (int i = 0; i < tensor->ndim; i++) {
    uint32_t dim = (uint32_t)tensor->shape[i];
    if (fwrite(&dim, sizeof(uint32_t), 1, f) != 1) return -1;
  }

  // Write tensor data
  if (fwrite(tensor->data, 1, tensor->nbytes, f) != tensor->nbytes) return -1;

  // Write group_size (v2 format)
  uint32_t group_size = (uint32_t)tensor->group_size;
  if (fwrite(&group_size, sizeof(uint32_t), 1, f) != 1) return -1;

  writer->tensor_count++;
  return 0;
}

/**
 * Finalize the streaming output file by updating the tensor count in header
 * 
 * @param writer The stream writer instance
 * @return 0 on success, -1 on error
 */
int bin_stream_writer_finalize(BinStreamWriter* writer) {
  if (!writer || !writer->file) {
    return -1;
  }

  // Save current position
  long current_pos = ftell(writer->file);
  if (current_pos == -1) {
    perror("Failed to get current file position");
    return -1;
  }

  // Seek to tensor count position (after magic header)
  if (fseek(writer->file, 6, SEEK_SET) != 0) {
    perror("Failed to seek to tensor count position");
    return -1;
  }

  // Write actual tensor count
  uint32_t count = writer->tensor_count;
  if (fwrite(&count, sizeof(uint32_t), 1, writer->file) != 1) {
    perror("Failed to write final tensor count");
    return -1;
  }

  // Return to end of file
  if (fseek(writer->file, current_pos, SEEK_SET) != 0) {
    perror("Failed to return to end of file");
    return -1;
  }

  return 0;
}

/**
 * Close a streaming binary file writer and free resources
 * 
 * @param writer The stream writer instance to close
 */
void bin_stream_writer_close(BinStreamWriter* writer) {
  if (!writer) return;
  
  if (writer->file) {
    fclose(writer->file);
  }
  free(writer);
}

