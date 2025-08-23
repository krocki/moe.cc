// test_model_trace.c
// Reference-matching forward w/ optional prefill length and hierarchical profiling.
// Build: gcc -O2 -Wall -std=c11 test_model_trace.c -o test_model_trace
//
// Usage (positional, same as before):
//   ./test_model_trace <all.bin> <outbase> <steps>
//
// Extras (optional flags):
//   --steps N         override positional steps (keeps backward compat)
//   --prompt_len P    explicitly set prompt length (P in [1 .. len(ids)])
//
// Defaults / behavior:
//   • If --prompt_len is NOT given: we match your current behavior:
//       prompt_len = max(1, IDS_len - steps)
//   • If --prompt_len IS given: we clamp to [1 .. IDS_len] and also clamp
//       steps <= IDS_len - prompt_len
//
// Profiler:
//   • Use PROF_START("name %d %d", a, b) and PROF_ENDF() pairs.
//   • Nested scopes handled; prints a table at exit.
//   • Labels are aggregated by formatted string => “unique function+args”.

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <time.h>
#include <stdint.h>
#include <inttypes.h>

#include "model.h"
#include "io.h"     // bin_load / bin_find / npy_load_* already available
#include "io_mmap.h"
#include "utils.h"  // max_abs_diff, etc.
#include "kernels.h" // matmul_adaptive for quantized weights

#ifdef DEBUG
  #define DBG(...) do { fprintf(stderr, __VA_ARGS__); } while(0)
#else
  #define DBG(...)
#endif

#ifdef BENCH
  #define TIMER_DECL struct timespec __t0, __t1
  #define TIMER_START() clock_gettime(CLOCK_MONOTONIC, &__t0)
  #define TIMER_END_MS(ms_out) do { \
      clock_gettime(CLOCK_MONOTONIC, &__t1); \
      double __ms = (__t1.tv_sec - __t0.tv_sec) * 1000.0 + \
                    (__t1.tv_nsec - __t0.tv_nsec) / 1.0e6; \
      (ms_out) = __ms; \
    } while(0)
#else
  #define TIMER_DECL
  #define TIMER_START()     do{}while(0)
  #define TIMER_END_MS(x)   do{ (void)(x); }while(0)
#endif

//==============================
// Toggle profiling on/off here
//==============================
#define PROFILING 1

//==============================
// Tiny hierarchical profiler
//==============================
#if PROFILING
  #define PROF_KEY_MAX   128
  #define PROF_MAX_STACK 256
  #define PROF_TABLE_SZ  4096  // simple power-of-two hash table

  typedef struct {
    char     key[PROF_KEY_MAX];
    uint64_t t_start_ns;
    double   child_ns;    // sum of inclusive time of direct children
  } prof_frame_t;

  typedef struct {
    uint64_t calls;
    double   inclusive_ns;
    double   exclusive_ns;
    int      used;
    char     key[PROF_KEY_MAX];
  } prof_slot_t;

  // Thread-local stack for nesting
  static __thread struct {
    prof_frame_t stack[PROF_MAX_STACK];
    int sp;
  } __prof_tls = { .sp = 0 };

  // Global aggregate table (single-threaded test path; if multi-threaded, add a mutex)
  static prof_slot_t __prof_tab[PROF_TABLE_SZ];

  static inline uint64_t prof_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
  }

  static inline uint64_t prof_hash(const char* s) {
    // FNV-1a 64-bit
    uint64_t h = 1469598103934665603ull;
    while (*s) { h ^= (unsigned char)(*s++); h *= 1099511628211ull; }
    return h;
  }

  static void prof_accumulate(const char* key, uint64_t calls, double incl_ns, double excl_ns) {
    uint64_t h = prof_hash(key);
    uint32_t mask = PROF_TABLE_SZ - 1;
    uint32_t idx = (uint32_t)(h & mask);
    for (uint32_t i = 0; i < PROF_TABLE_SZ; ++i) {
      uint32_t p = (idx + i) & mask;
      if (!__prof_tab[p].used) {
        __prof_tab[p].used = 1;
        __prof_tab[p].calls = calls;
        __prof_tab[p].inclusive_ns = incl_ns;
        __prof_tab[p].exclusive_ns = excl_ns;
        strncpy(__prof_tab[p].key, key, PROF_KEY_MAX-1);
        __prof_tab[p].key[PROF_KEY_MAX-1] = '\0';
        return;
      }
      if (__prof_tab[p].used && strncmp(__prof_tab[p].key, key, PROF_KEY_MAX) == 0) {
        __prof_tab[p].calls       += calls;
        __prof_tab[p].inclusive_ns+= incl_ns;
        __prof_tab[p].exclusive_ns+= excl_ns;
        return;
      }
    }
    // Table full – silently drop in this tiny implementation
  }
  int cmp(const void* a, const void* b){
    const prof_slot_t* x = (const prof_slot_t*)a;
    const prof_slot_t* y = (const prof_slot_t*)b;
    if (y->inclusive_ns > x->inclusive_ns) return 1;
    if (y->inclusive_ns < x->inclusive_ns) return -1;
    return 0;
  }

  static void prof_dump(void) {
    // Collect used entries
    int n = 0;
    for (int i = 0; i < PROF_TABLE_SZ; ++i) if (__prof_tab[i].used) ++n;
    if (n == 0) return;

    prof_slot_t* arr = (prof_slot_t*)malloc(sizeof(prof_slot_t) * (size_t)n);
    int j = 0;
    for (int i = 0; i < PROF_TABLE_SZ; ++i) if (__prof_tab[i].used) arr[j++] = __prof_tab[i];

    // Sort by inclusive time descending
    qsort(arr, (size_t)n, sizeof(prof_slot_t), cmp);

    printf("\n=== Profiling (inclusive/exclusive) ===\n");
    printf("%-48s %8s %12s %12s %12s\n", "Key", "Calls", "Incl(ms)", "Excl(ms)", "AvgExcl(µs)");
    for (int i = 0; i < n; ++i) {
      double incl_ms = arr[i].inclusive_ns / 1.0e6;
      double excl_ms = arr[i].exclusive_ns / 1.0e6;
      double avg_excl_us = (arr[i].calls > 0) ? (arr[i].exclusive_ns / (double)arr[i].calls) / 1.0e3 : 0.0;
      printf("%-48.48s %8" PRIu64 " %12.3f %12.3f %12.3f\n",
             arr[i].key, arr[i].calls, incl_ms, excl_ms, avg_excl_us);
    }
    printf("=======================================\n\n");
    free(arr);
  }

  static void prof_atexit(void){ prof_dump(); }

  // Install at exit once
  static void prof_init_once(void) {
    static int once = 0;
    if (!once) { atexit(prof_atexit); once = 1; }
  }

  // Macros
  #define PROF_START(fmt, ...) do { \
    prof_init_once(); \
    if (__prof_tls.sp >= PROF_MAX_STACK) break; \
    prof_frame_t* __fr = &__prof_tls.stack[__prof_tls.sp++]; \
    snprintf(__fr->key, sizeof(__fr->key), (fmt), ##__VA_ARGS__); \
    __fr->t_start_ns = prof_now_ns(); \
    __fr->child_ns = 0.0; \
  } while(0)

  #define PROF_ENDF() do { \
    if (__prof_tls.sp <= 0) break; \
    prof_frame_t __fr = __prof_tls.stack[--__prof_tls.sp]; \
    uint64_t __t_end = prof_now_ns(); \
    double __incl = (double)(__t_end - __fr.t_start_ns); \
    double __excl = __incl - __fr.child_ns; \
    prof_accumulate(__fr.key, 1, __incl, __excl); \
    if (__prof_tls.sp > 0) __prof_tls.stack[__prof_tls.sp-1].child_ns += __incl; \
  } while(0)

#else
  #define PROF_START(fmt, ...) do{}while(0)
  #define PROF_ENDF()          do{}while(0)
#endif

//==============================
// Debug logging
//==============================
#ifdef DEBUG
  #define DBG(...) do { fprintf(stderr, __VA_ARGS__); } while(0)
#else
  #define DBG(...)
#endif

//==============================
// Math kernels (profiled)
//==============================
static void matmul_f32(const float* A, const float* B, float* C,
                       int M, int N, int K) {
  PROF_START("matmul_f32 M=%d N=%d K=%d", M, N, K);
  for (int m = 0; m < M; ++m) {
    const float* a = A + m*K;
    float* c = C + m*N;
    for (int n = 0; n < N; ++n) {
      const float* b = B + n*K;
      float acc = 0.f;
      for (int k = 0; k < K; ++k) acc += a[k] * b[k];
      c[n] = acc;
    }
  }
  PROF_ENDF();
}

static void silu_f32(float* x, int n) {
  PROF_START("silu_f32 n=%d", n);
  for (int i = 0; i < n; ++i) {
    float v = x[i];
    x[i] = v / (1.0f + expf(-v));
  }
  PROF_ENDF();
}

// RMSNorm forward
// x: [T, d_model]; w: [d_model] (scale weights); eps: epsilon
// y: [T, d_model]
static void rmsnorm_forward_f32(const float* x, const float* w,
                                int T, int d_model, float eps,
                                float* y) {
  PROF_START("rmsnorm T=%d D=%d", T, d_model);
  for (int t = 0; t < T; ++t) {
    const float* xt = x + (size_t)t * d_model;
    float* yt = y + (size_t)t * d_model;
    float msq = 0.0f;
    for (int i = 0; i < d_model; ++i) msq += xt[i] * xt[i];
    msq /= (float)d_model;
    float inv = 1.0f / sqrtf(msq + eps);
    for (int i = 0; i < d_model; ++i) yt[i] = xt[i] * inv * (w ? w[i] : 1.0f);
  }
  PROF_ENDF();
}

//==============================
// Helpers
//==============================
static void softmax_rows(float* mat, int rows, int cols) {
  PROF_START("softmax rows=%d cols=%d", rows, cols);
  for (int r = 0; r < rows; ++r) {
    float* row = &mat[(size_t)r * cols];
    float maxv = row[0];
    for (int c = 1; c < cols; ++c) if (row[c] > maxv) maxv = row[c];
    float sum = 0.f;
    for (int c = 0; c < cols; ++c) { float e = expf(row[c] - maxv); row[c] = e; sum += e; }
    float inv = 1.0f / (sum + 1e-9f);
    for (int c = 0; c < cols; ++c) row[c] *= inv;
  }
  PROF_ENDF();
}

// top-k over vector x[E] -> k indices/values (descending)
static void topk_desc(const float* x, int E, int top_k, int* out_idx, float* out_val) {
  PROF_START("topk_desc E=%d k=%d", E, top_k);
  for (int i = 0; i < top_k; ++i) { out_val[i] = -INFINITY; out_idx[i] = -1; }
  for (int e = 0; e < E; ++e) {
    float v = x[e];
    int pos = -1;
    for (int i = 0; i < top_k; ++i) { if (v > out_val[i]) { pos = i; break; } }
    if (pos >= 0) {
      for (int j = top_k - 1; j > pos; --j) { out_val[j] = out_val[j-1]; out_idx[j] = out_idx[j-1]; }
      out_val[pos] = v; out_idx[pos] = e;
    }
  }
  PROF_ENDF();
}

static inline void rope_rotate_pair(float* even, float* odd, float c, float s) {
  float e = *even, o = *odd;
  *even =  e * c - o * s;
  *odd  =  o * c + e * s;
}

// Apply standard RoPE to Q and K (GQA-aware), in-place.
static void rope_apply_inplace_f32_gqa(
  float* Q, float* K,
  int T, int n_q, int n_kv, int head_dim,
  int pos0, float theta)
{
  PROF_START("rope T=%d nq=%d nkv=%d dh=%d", T, n_q, n_kv, head_dim);
  const int Dq  = n_q  * head_dim;
  const int Dkv = n_kv * head_dim;
  const int d2  = head_dim / 2;
  if ((head_dim & 1) != 0) {
    fprintf(stderr, "[rope] head_dim must be even, got %d\n", head_dim);
    exit(1);
  }
  float* inv = (float*)malloc(sizeof(float) * (size_t)d2);
  for (int i = 0; i < d2; ++i) {
    float exponent = -2.0f * (float)i / (float)head_dim;
    inv[i] = powf(theta, exponent);
  }
  for (int t = 0; t < T; ++t) {
    const float p = (float)(pos0 + t);
    for (int h = 0; h < n_q; ++h) {
      float* qh = &Q[(size_t)t*Dq + (size_t)h*head_dim];
      for (int i = 0; i < d2; ++i) {
        float ang = p * inv[i];
        float c = cosf(ang), s = sinf(ang);
        rope_rotate_pair(&qh[i], &qh[i + d2], c, s);
      }
    }
    for (int h = 0; h < n_kv; ++h) {
      float* kh = &K[(size_t)t*Dkv + (size_t)h*head_dim];
      for (int i = 0; i < d2; ++i) {
        float ang = p * inv[i];
        float c = cosf(ang), s = sinf(ang);
        rope_rotate_pair(&kh[i], &kh[i + d2], c, s);
      }
    }
  }
  free(inv);
  PROF_ENDF();
}

//==============================
// Attention + MoE (single layer)
//==============================
static void attention_forward_f32(
  const float* x, int T, int d_model,
  const float* Wq, const float* bq,
  const float* Wk, const float* bk,
  const float* Wv, const float* bv,
  const float* Wo, const float* bo,
  const float* qn, const float* kn,
  int n_q, int n_kv, int head_dim, int causal,
  const float rope_theta, const float rms_eps,
  float* scratch, float* y_out)
{
  PROF_START("attention T=%d dq=%d nkv=%d dh=%d", T, n_q*head_dim, n_kv, head_dim);
  const float scale = 1.0f / sqrtf((float)head_dim);
  const int Dq  = n_q  * head_dim;
  const int Dkv = n_kv * head_dim;

  float* Q    = scratch;            // [T, Dq]
  float* K    = Q + (size_t)T*Dq;   // [T, Dkv]
  float* V    = K + (size_t)T*Dkv;  // [T, Dkv]
  float* S    = V + (size_t)T*Dkv;  // [T, T]
  float* Hcat = S + (size_t)T*T;    // [T, Dq]

  matmul_f32(x, Wq, Q, T, Dq,  d_model);
  matmul_f32(x, Wk, K, T, Dkv, d_model);
  matmul_f32(x, Wv, V, T, Dkv, d_model);

  if (bq) for (int t=0; t<T; ++t) for (int i=0; i<Dq;  ++i) Q[(size_t)t*Dq  + i] += bq[i];
  if (bk) for (int t=0; t<T; ++t) for (int i=0; i<Dkv; ++i) K[(size_t)t*Dkv + i] += bk[i];
  if (bv) for (int t=0; t<T; ++t) for (int i=0; i<Dkv; ++i) V[(size_t)t*Dkv + i] += bv[i];

  // Q/K RMSNorm per head
  if (qn) {
    for (int t=0; t<T; ++t) {
      float* Qt = &Q[(size_t)t*Dq];
      for (int h=0; h<n_q; ++h) {
        float* v = &Qt[(size_t)h*head_dim];
        float msq=0.f; for(int d=0; d<head_dim; ++d){ float z=v[d]; msq+=z*z; }
        float inv = 1.0f / sqrtf(msq/(float)head_dim + rms_eps);
        for (int d=0; d<head_dim; ++d) v[d] = (v[d]*inv) * qn[d];
      }
    }
  }
  if (kn) {
    for (int t=0; t<T; ++t) {
      float* Kt = &K[(size_t)t*Dkv];
      for (int h=0; h<n_kv; ++h) {
        float* v = &Kt[(size_t)h*head_dim];
        float msq=0.f; for(int d=0; d<head_dim; ++d){ float z=v[d]; msq+=z*z; }
        float inv = 1.0f / sqrtf(msq/(float)head_dim + rms_eps);
        for (int d=0; d<head_dim; ++d) v[d] = (v[d]*inv) * kn[d];
      }
    }
  }

  rope_apply_inplace_f32_gqa(Q, K, T, n_q, n_kv, head_dim, /*pos0=*/0, rope_theta);

  const int group = n_q / n_kv;

  for (int h=0; h<n_q; ++h) {
    const int kvh = h / group;
    for (int tq=0; tq<T; ++tq) {
      const float* qv = &Q[(size_t)tq*Dq + (size_t)h*head_dim];
      float* Sout = &S[(size_t)tq*T];
      for (int tk=0; tk<T; ++tk) {
        const float* kv = &K[(size_t)tk*Dkv + (size_t)kvh*head_dim];
        float dot=0.f; for (int d=0; d<head_dim; ++d) dot += qv[d]*kv[d];
        Sout[tk] = dot * scale;
      }
      if (causal) for (int tk=tq+1; tk<T; ++tk) S[(size_t)tq*T + tk] = -INFINITY;
      float m=Sout[0]; for(int i=1;i<T;++i) if(Sout[i]>m) m=Sout[i];
      float ssum=0.f; for(int i=0;i<T;++i){ float e = expf(Sout[i]-m); Sout[i]=e; ssum+=e; }
      float inv=1.f/(ssum+1e-9f);
      for(int i=0;i<T;++i) Sout[i]*=inv;
    }
    for (int tq=0; tq<T; ++tq) {
      const float* Prow = &S[(size_t)tq*T];
      float* out = &Hcat[(size_t)tq*Dq + (size_t)h*head_dim];
      for (int d=0; d<head_dim; ++d) out[d]=0.f;
      for (int tk=0; tk<T; ++tk) {
        const float* vv = &V[(size_t)tk*Dkv + (size_t)kvh*head_dim];
        const float p = Prow[tk];
        for (int d=0; d<head_dim; ++d) out[d] += p * vv[d];
      }
    }
  }

  matmul_f32(Hcat, Wo, y_out, T, d_model, Dq);
  if (bo) for (int t=0; t<T; ++t) for (int i=0; i<d_model; ++i) y_out[(size_t)t*d_model + i] += bo[i];
  PROF_ENDF();
}

// One layer: norm1 -> attn -> +res -> norm2 -> MoE -> +res
static void layer_forward_f32(
  float* x, int T, int d_model,
  const QwenLayerWeights* lw,
  int n_q, int n_kv, int head_dim, int causal,
  const float rope_theta, const float rms_eps,
  int n_experts, int top_k, int d_ff,
  float* scratch_attn, float* scratch_moe, int* tmp_idx, float* tmp_val)
{
  PROF_START("layer_forward T=%d", T);

  // temp buffers (reuse scratch_attn tail as temp D vectors)
  float* x_norm1  = scratch_attn + (size_t)T*(n_q*head_dim + 2*n_kv*head_dim) + (size_t)T*T + (size_t)T*(n_q*head_dim);
  float* attn_out = x_norm1 + (size_t)T*d_model;
  float* x_after  = attn_out + (size_t)T*d_model;
  float* x_norm2  = x_after  + (size_t)T*d_model;

  rmsnorm_forward_f32(x, lw->rms1_w, T, d_model, rms_eps, x_norm1);

  attention_forward_f32(
    x_norm1, T, d_model,
    lw->Wq, lw->bq, lw->Wk, lw->bk, lw->Wv, lw->bv, lw->Wo, lw->bo,
    lw->q_norm, lw->k_norm,
    n_q, n_kv, head_dim, causal,
    rope_theta, rms_eps,
    scratch_attn, attn_out
  );

  for (int i=0;i<T*d_model;++i) x_after[i] = x[i] + attn_out[i];
  rmsnorm_forward_f32(x_after, lw->rms2_w, T, d_model, rms_eps, x_norm2);

  // router logits
  float* logits = scratch_moe;                    // [T, E]
  matmul_f32(x_norm2, lw->router_w, logits, T, n_experts, d_model);
  if (lw->router_b) {
    for (int t=0;t<T;++t) for (int e=0;e<n_experts;++e) logits[(size_t)t*n_experts + e] += lw->router_b[e];
  }

  // experts (top-k routing)
  float* moe_out = scratch_moe + (size_t)T * n_experts; // [T, d_model]
  for (int i=0;i<T*d_model;++i) moe_out[i] = 0.f;

  float* tmp_g = (float*)malloc(sizeof(float)*d_ff);
  float* tmp_u = (float*)malloc(sizeof(float)*d_ff);
  float* tmp_y = (float*)malloc(sizeof(float)*d_model);

  for (int t=0; t<T; ++t) {
    const float* log_t = &logits[(size_t)t * n_experts];
    topk_desc(log_t, n_experts, top_k, tmp_idx, tmp_val);
    // softmax over top-k
    float maxv = tmp_val[0];
    for (int i=1;i<top_k;++i) if (tmp_val[i] > maxv) maxv = tmp_val[i];
    float sum = 0.f;
    for (int i=0;i<top_k;++i){ tmp_val[i] = expf(tmp_val[i] - maxv); sum += tmp_val[i]; }
    float inv = 1.0f / (sum + 1e-9f);
    for (int i=0;i<top_k;++i) tmp_val[i] *= inv;

    for (int i=0;i<top_k;++i){
      int e = tmp_idx[i];
      float p = tmp_val[i];
      
      // Use adaptive matrix multiplication for expert weights
      if (lw->experts_quantized) {
        // Quantized expert forward pass
        matmul_adaptive(&x_norm2[(size_t)t*d_model], NULL, &lw->Wg_q[e], 
                       tmp_g, 1, d_ff, d_model);
        silu_f32(tmp_g, d_ff);
        matmul_adaptive(&x_norm2[(size_t)t*d_model], NULL, &lw->Wu_q[e],
                       tmp_u, 1, d_ff, d_model);
        for (int q=0;q<d_ff;++q) tmp_g[q] *= tmp_u[q];
        matmul_adaptive(tmp_g, NULL, &lw->Wd_q[e],
                       tmp_y, 1, d_model, d_ff);
      } else {
        // Original FP32 expert forward pass
        matmul_f32(&x_norm2[(size_t)t*d_model], lw->Wg[e], tmp_g, 1, d_ff, d_model);
        silu_f32(tmp_g, d_ff);
        matmul_f32(&x_norm2[(size_t)t*d_model], lw->Wu[e], tmp_u, 1, d_ff, d_model);
        for (int q=0;q<d_ff;++q) tmp_g[q] *= tmp_u[q];
        matmul_f32(tmp_g, lw->Wd[e], tmp_y, 1, d_model, d_ff);
      }
      
      for (int q=0;q<d_model;++q) moe_out[(size_t)t*d_model + q] += p * tmp_y[q];
    }
  }

  for (int i=0;i<T*d_model;++i) x[i] = x_after[i] + moe_out[i];

  free(tmp_g); free(tmp_u); free(tmp_y);
  PROF_ENDF();
}

//==============================
// QwenStates: All intermediate state allocations
//==============================
// Similar to llama2.c RunState, this struct contains all temporary buffers
// needed during model forward pass, allocated once and reused across steps
typedef struct {
  // Main activation buffers - store intermediate layer outputs
  float* x;           // [T, d_model] - primary activation buffer, flows through all layers
  float* x_final;     // [T, d_model] - buffer for final RMSNorm output before lm_head
  
  // Attention computation scratch space
  // Size calculation: T*(Dq + 2*Dkv) + T*T + T*Dq + 4*T*d_model
  // Where: Dq = n_q * head_dim, Dkv = n_kv * head_dim
  // Layout: [Q: T*Dq] + [K: T*Dkv] + [V: T*Dkv] + [S: T*T] + [Hcat: T*Dq] + [temps: 4*T*d_model]
  // Q/K/V: projected query/key/value matrices for all heads
  // S: attention scores matrix [seq_len, seq_len] for softmax computation  
  // Hcat: concatenated head outputs before output projection
  // temps: temporary buffers for norm1/attn_out/x_after/norm2 in layer_forward_f32
  float* scratch_attn;
  
  // MoE (Mixture of Experts) computation scratch space  
  // Size: T * n_experts + T * d_model
  // Layout: [router_logits: T*n_experts] + [moe_output: T*d_model]
  // router_logits: expert selection scores for each token
  // moe_output: accumulated weighted expert outputs
  float* scratch_moe;
  
  // Top-K selection buffers for MoE routing
  int* tmp_idx;       // [top_k] - indices of selected experts per token
  float* tmp_val;     // [top_k] - corresponding expert weights after softmax
  
  // Final output buffer
  float* out_logits;  // [vocab_size] - logits for next token prediction
} QwenStates;

// Allocate all intermediate states once - prevents repeated malloc/free in forward passes
// This matches the llama2.c pattern where RunState is allocated once in main()
static QwenStates* malloc_qwen_states(const QwenConfig* cfg, int T) {
  QwenStates* s = (QwenStates*)calloc(1, sizeof(QwenStates));
  if (!s) return NULL;
  
  const int D = cfg->d_model;           // model dimension (e.g., 4096)
  const int V = cfg->vocab_size;        // vocabulary size (e.g., 32000)
  const int Dq0  = cfg->n_q  * cfg->head_dim;  // total query dimension across all heads
  const int Dkv0 = cfg->n_kv * cfg->head_dim;  // total key/value dimension (GQA: n_kv <= n_q)
  
  // Attention scratch calculation breakdown:
  // - Q matrix: T * Dq0 floats
  // - K matrix: T * Dkv0 floats  
  // - V matrix: T * Dkv0 floats
  // - Attention scores S: T * T floats (seq_len x seq_len attention matrix)
  // - Head concatenation Hcat: T * Dq0 floats
  // - Layer temps: 4 * T * D floats (x_norm1, attn_out, x_after, x_norm2)
  size_t attn_f  = (size_t)T*(Dq0 + 2*Dkv0) + (size_t)T*T + (size_t)T*Dq0;
  size_t temps_f = 4ull * (size_t)T * D;
  
  // MoE scratch calculation:
  // - Router logits: T * n_experts floats (expert scores for each token)
  // - MoE output accumulator: T * d_model floats
  size_t moe_f = (size_t)T * cfg->n_experts + (size_t)T * D;
  
  // Allocate all buffers with zero initialization
  s->x = (float*)calloc((size_t)T * D, sizeof(float));
  s->x_final = (float*)calloc((size_t)T * D, sizeof(float));
  s->scratch_attn = (float*)calloc(attn_f + temps_f, sizeof(float));
  s->scratch_moe = (float*)calloc(moe_f, sizeof(float));
  s->tmp_idx = (int*)calloc(cfg->top_k, sizeof(int));
  s->tmp_val = (float*)calloc(cfg->top_k, sizeof(float));
  s->out_logits = (float*)calloc(V, sizeof(float));
  
  // Verify all allocations succeeded - if any fail, clean up and return NULL
  if (!s->x || !s->x_final || !s->scratch_attn || !s->scratch_moe || 
      !s->tmp_idx || !s->tmp_val || !s->out_logits) {
    free(s->x); free(s->x_final); free(s->scratch_attn); free(s->scratch_moe);
    free(s->tmp_idx); free(s->tmp_val); free(s->out_logits);
    free(s);
    return NULL;
  }
  
  return s;
}

// Free all intermediate state buffers
static void free_qwen_states(QwenStates* s) {
  if (!s) return;
  free(s->x); free(s->x_final); free(s->scratch_attn); free(s->scratch_moe);
  free(s->tmp_idx); free(s->tmp_val); free(s->out_logits);
  free(s);
}

//==============================
// Full forward (no file I/O)
//==============================
static void model_forward_f32(
  const int* ids, int T,
  const QwenConfig* cfg,
  const QwenWeights* w,
  QwenStates* s,      // pre-allocated intermediate state buffers
  int apply_softmax,
  float* out_last     // [1, vocab] - output buffer for final logits
){
  PROF_START("model_forward T=%d", T);
  const int D = cfg->d_model, V = cfg->vocab_size, L = cfg->n_layers;

  // Use pre-allocated buffers from QwenStates instead of malloc/free
  float* x = s->x;

  // embeddings
  PROF_START("embedding_lookup T=%d D=%d", T, D);
  for (int t=0; t<T; ++t) {
    int id = ids[t];
    memcpy(&x[(size_t)t*D], &w->tok_emb[(size_t)id*D], sizeof(float)*(size_t)D);
  }
  PROF_ENDF();

  // stack
  for (int l=0; l<L; ++l) {
    layer_forward_f32(
      x, T, D, &w->layers[l],
      cfg->n_q, cfg->n_kv, cfg->head_dim, cfg->causal,
      cfg->rope_theta, cfg->rms_eps,
      cfg->n_experts, cfg->top_k, cfg->d_ff,
      s->scratch_attn, s->scratch_moe, s->tmp_idx, s->tmp_val
    );
  }

  // final norm + head (only last row into out_last)
  float* x_final = s->x_final;  // use pre-allocated buffer
  rmsnorm_forward_f32(x, w->final_norm_w, T, D, cfg->rms_eps, x_final);

  const float* Wout = w->lm_head ? w->lm_head : w->tok_emb;
  matmul_f32(&x_final[(size_t)(T-1)*D], Wout, out_last, 1, V, D);

  if (apply_softmax) softmax_rows(out_last, 1, V);

  // No free() calls - buffers are reused across forward passes
  PROF_ENDF();
}

//==============================
// Weight loading (external)
//==============================
// Expect: void load_all_weights(BinFile*, QwenConfig*, QwenWeights*, int use_mmap);
// Provided in model.h with detailed memory mapping and explicit loading support.

//==============================
// Simple argument parser
//==============================
typedef struct {
  const char* allfile;
  const char* outbase;
  int   steps;            // how many generation steps to evaluate
  int   prompt_len_opt;   // 0 if not specified; else a positive value
  int   prompt_len;       // resolved prompt length
  int   use_mmap;         // 1 = use mmap (default), 0 = explicit loading
} Args;

static void usage(const char* prog){
  fprintf(stderr,
    "Usage:\n"
    "  %s <all.bin> <outbase> <steps> [OPTIONS]\n"
    "  %s <all.bin> <outbase> --steps N [OPTIONS]\n"
    "\nOptions:\n"
    "  --prompt_len P    Set explicit prompt length (default: auto-calculated)\n"
    "  --no-mmap         Load weights into memory instead of memory mapping (default: use mmap)\n"
    "\nNotes:\n"
    "  • Weights are memory-mapped by default for better performance and lower memory usage\n"
    "  • Use --no-mmap to explicitly load weights into allocated memory (shows progress)\n"
    "  • If --prompt_len is not given, defaults to IDS_len - steps (clamped to >=1)\n"
    "  • If --prompt_len is given, steps will be clamped so prompt_len + steps <= IDS_len\n",
    prog, prog);
}

static int parse_args(int argc, char** argv, Args* a){
  if (argc < 2) { usage(argv[0]); return -1; }
  memset(a, 0, sizeof(*a));
  
  // Set defaults
  a->use_mmap = 1;  // Use mmap by default

  // First pass: collect positional if present
  int pos = 0;
  for (int i=1; i<argc; ++i){
    const char* s = argv[i];
    if (s[0] == '-' && s[1] == '-') break; // flags begin
    if (pos == 0) a->allfile = s;
    else if (pos == 1) a->outbase = s;
    else if (pos == 2) a->steps = atoi(s);
    ++pos;
  }
  // Second pass: flags
  for (int i=1; i<argc; ++i){
    const char* s = argv[i];
    if (strcmp(s,"--steps")==0 && i+1<argc) { a->steps = atoi(argv[++i]); continue; }
    if (strcmp(s,"--prompt_len")==0 && i+1<argc) { a->prompt_len_opt = 1; a->prompt_len = atoi(argv[++i]); continue; }
    if (strcmp(s,"--no-mmap")==0) { a->use_mmap = 0; continue; }
  }

  if (!a->allfile || !a->outbase || a->steps <= 0) {
    usage(argv[0]);
    return -1;
  }
  return 0;
}

//#define MMAP
//==============================
// Main
//==============================
int main(int argc, char** argv) {
  Args args;
  if (parse_args(argc, argv, &args) != 0) return 1;

  // Load the model weights from the binary file
#ifdef MMAP
  BinFile* bin_file = bin_load_mmap(args.allfile);
#else
  BinFile* bin_file = bin_load(args.allfile);
#endif
  if (!bin_file) { fprintf(stderr, "Failed to load binary file\n"); return 1; }

  QwenConfig config;
  QwenWeights weights;
  printf("[loading] weights mode: %s\n", args.use_mmap ? "memory-mapped" : "explicit loading");
  load_all_weights(bin_file, &config, &weights);
  printf("[model] d_model=%d n_layers=%d head_dim=%d n_q=%d n_kv=%d vocab=%d\n",
         config.d_model, config.n_layers, config.head_dim, config.n_q, config.n_kv, config.vocab_size);

  // Load reference tokens, logits, and probs from numpy files
  char ids_path[512], logits_path[512], probs_path[512];
  snprintf(ids_path, sizeof(ids_path), "%s.ids.npy", args.outbase);
  snprintf(logits_path, sizeof(logits_path), "%s.logits.npy", args.outbase);
  snprintf(probs_path, sizeof(probs_path), "%s.probs.npy", args.outbase);

  NpyArrayI32* ref_tokens = npy_load_int32(ids_path);
  NpyArray* ref_logits = npy_load_float32(logits_path);
  NpyArray* ref_probs = npy_load_float32(probs_path);
  if (!ref_tokens || !ref_logits || !ref_probs) {
    fprintf(stderr, "Missing ids/logits/probs dumps\n"); return 1;
  }

  const int seq_len = ref_tokens->shape[0];
  const int vocab_size = config.vocab_size;

  if (ref_logits->shape[1] != vocab_size || ref_probs->shape[1] != vocab_size) {
    fprintf(stderr, "Vocab mismatch: vocab_size=%d vs dumps [%d,%d]\n",
            vocab_size, (int)ref_logits->shape[1], (int)ref_probs->shape[1]);
    return 1;
  }

  // Resolve prompt length and steps based on presence of --prompt_len
  // If not specified, assume generation starts after the first token
  int prompt_len = args.prompt_len_opt
                   ? args.prompt_len
                   : (seq_len - args.steps);

  if (prompt_len < 1) prompt_len = 1;
  if (prompt_len > seq_len) prompt_len = seq_len;
  int steps = args.steps;
  if (steps < 1) steps = 1;
  if (steps > seq_len - prompt_len) steps = seq_len - prompt_len;

  if (steps <= 0) {
    fprintf(stderr, "Nothing to generate: prompt_len=%d covers all %d tokens.\n", prompt_len, seq_len);
    return 1;
  }

  // The reference logits/probs are assumed to cover predictions starting from position 1 to seq_len-1
  // We need at least (prompt_len - 1 + steps) reference entries
  const int required_refs = prompt_len - 1 + steps;
  if (ref_logits->shape[0] < required_refs || ref_probs->shape[0] < required_refs) {
    fprintf(stderr, "steps=%d with prompt_len=%d requires %d refs, but dumps have only %d\n",
            steps, prompt_len, required_refs, (int)ref_logits->shape[0]);
    return 1;
  }

  printf("[run] seq_len=%d prompt_len=%d steps=%d\n", seq_len, prompt_len, steps);

  // Allocate QwenStates once for the maximum sequence length we'll process
  // This follows the llama2.c pattern of allocating RunState in main()
  int max_seq_len = prompt_len + steps;
  QwenStates* qwen_states = malloc_qwen_states(&config, max_seq_len);
  if (!qwen_states) {
    fprintf(stderr, "Failed to allocate QwenStates\n");
    return 1;
  }

  float* last_logits = (float*)malloc(sizeof(float) * vocab_size);

  // Run model forward for all steps at once - similar to llama2.c approach
  // where forward() processes the entire sequence and generates multiple tokens
  #ifdef BENCH
    double total_ms = 0.0;
    TIMER_DECL;
    TIMER_START();
  #endif

  // Process each step individually but reuse the same QwenStates
  for (int step = 0; step < steps; ++step) {
    // Compute the current input length: prompt + previously "generated" tokens (from reference)
    int input_len = prompt_len + step;

    DBG("[model_forward_f32] step=%d input_len=%d\n", step, input_len);

    // Run model forward on the prefix tokens[0..input_len-1], get logits for the next token
    // Now uses pre-allocated QwenStates instead of allocating buffers each time
    model_forward_f32(
      ref_tokens->data, input_len,
      &config, &weights, qwen_states,
      /*apply_softmax=*/0,
      last_logits
    );

    // The reference index: references start from predicting token 1 (LREF[0]),
    // so for predicting token (prompt_len + step), the ref index is (prompt_len + step - 1)
    int ref_index = prompt_len - 1 + step;

    // Compare logits to the corresponding reference
    double max_abs_diff_logits = max_abs_diff(last_logits, &ref_logits->data[(size_t)ref_index * vocab_size], vocab_size);

    // Softmax the model logits and compare probs to reference
    softmax_rows(last_logits, 1, vocab_size);
    double max_abs_diff_probs = max_abs_diff(last_logits, &ref_probs->data[(size_t)ref_index * vocab_size], vocab_size);

    // Compute argmax for model and reference
    int argmax_model = 0, argmax_ref = 0;
    float max_prob_model = last_logits[0];
    float max_prob_ref = ref_probs->data[(size_t)ref_index * vocab_size + 0];
    for (int i = 1; i < vocab_size; ++i) {
      if (last_logits[i] > max_prob_model) { max_prob_model = last_logits[i]; argmax_model = i; }
      float ref_val = ref_probs->data[(size_t)ref_index * vocab_size + i];
      if (ref_val > max_prob_ref) { max_prob_ref = ref_val; argmax_ref = i; }
    }

    printf("[step %d] logits MAD=%.8g  probs MAD=%.8g  argmax=%d  ref=%d\n",
           step, max_abs_diff_logits, max_abs_diff_probs, argmax_model, argmax_ref);
  }

  #ifdef BENCH
    TIMER_END_MS(total_ms);
    DBG("[total] %d steps in %.3f ms, %.3f ms/step, %.3f steps/s\n", 
        steps, total_ms, total_ms/steps, 1e3*steps/total_ms);
  #endif

  // Clean up allocated resources
  free_qwen_states(qwen_states);  // Free all intermediate state buffers
  free(last_logits);
  npy_free_i32(ref_tokens);
  npy_free(ref_logits);
  npy_free(ref_probs);

#ifdef MMAP
  bin_free_mmap(bin_file);
#else
  // Free layer expert arrays
  for (int layer = 0; layer < config.n_layers; ++layer) {
    free((void*)weights.layers[layer].Wg);
    free((void*)weights.layers[layer].Wu);
    free((void*)weights.layers[layer].Wd);
  }
  free(weights.layers);
  bin_free(bin_file);
#endif
  return 0;
}

