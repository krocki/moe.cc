// test_layer_trace.c
// Compare per-op intermediates against Python dumps.
// Usage:
//   ./test_layer_trace <l0_layer.bin> <outbase> <step> <T> <rope_theta>
//
// This matches the Raschka Qwen-3 MoE reference, including:
//   • Q/K RMSNorm per head (scale only, epsilon=1e-6, no centering)
//   • RoPE pairing by half-split: x = [x1 | x2], rot = [-x2 | x1],
//       x' = x * cos + rot * sin, with angles φ = p * theta^(-2m/d)
//   • GQA head mapping: kvh = h / group_size
//
// We only read the reference .npy dumps for *optional* compares (Q/K/V, Q_qknorm/K_qknorm,
// Q_rope/K_rope, attn_out). RoPE now uses an internal implementation (no external cos/sin).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>

#include "io.h"
#include "utils.h"
#include "kernels.h"

// -------------------------------------------------------------
// Small helper: try to load an npy; return NULL if it's missing.
// This lets us print MAD when a reference tensor exists and
// silently skip otherwise.
// -------------------------------------------------------------
static NpyArray* try_load_f32(const char* path_fmt, const char* outbase, int step, int L, const char* name){
  char p[512];
  snprintf(p, sizeof(p), path_fmt, outbase, step, L, name);
  return npy_load_float32(p); // NULL => skip compare
}

// -------------------------------------------------------------------
// RoPE (Raschka/Qwen-3 compatible): half-split pairing + on-the-fly
// cos/sin via φ = (pos0+t) * theta^(-2m/d).
//
// Layout:
//   X: [T, n_heads * head_dim], each head contiguous
// Pairing (IMPORTANT):
//   Let d = head_dim, d2 = d/2
//   x1 = X[..., 0:d2], x2 = X[..., d2:d]
//   rot = [-x2, x1]
//   X' = X * cos + rot * sin
//
// Angles:
//   inv_freq[m] = theta^(-2m/d), m=0..d2-1
//   φ(t,m) = (pos0 + t) * inv_freq[m]
//   c = cos(φ), s = sin(φ)
// Rotation (per (x1[i], x2[i])) equals:
//   new_x1 = x1*c - x2*s
//   new_x2 = x2*c + x1*s
//
// This mirrors the notebook's compute_rope_params + apply_rope.
// -------------------------------------------------------------------
static void rope_apply_inplace_f32_gqa_halfsplit(
  float* Q, float* K,
  int T, int n_q, int n_kv, int head_dim,
  int pos0, float theta)
{
  const int d = head_dim;
  if ((d & 1) != 0) {
    fprintf(stderr, "[rope] head_dim must be even, got %d\n", d);
    exit(1);
  }
  const int d2 = d >> 1;
  const size_t Dq  = (size_t)n_q  * (size_t)d;
  const size_t Dkv = (size_t)n_kv * (size_t)d;

  // precompute inv_freq[m] = theta^(-2m/d)
  float* inv_freq = (float*)malloc(sizeof(float)* (size_t)d2);
  if (!inv_freq) { fprintf(stderr,"[rope] OOM inv_freq\n"); exit(1); }
  const float inv_d = 1.0f / (float)d; // for (-2m/d)
  for (int m=0; m<d2; ++m) {
    const float exp_ = -2.0f * (float)m * inv_d;
    inv_freq[m] = powf(theta, exp_);
  }

  // rotate each token position
  for (int t=0; t<T; ++t) {
    const int p = pos0 + t;

    // --- Q heads ---
    for (int h=0; h<n_q; ++h) {
      float* base = &Q[(size_t)t*Dq + (size_t)h*d];
      float* x1 = base;       // [0..d2-1]
      float* x2 = base + d2;  // [d2..d-1]
      for (int i=0; i<d2; ++i) {
        const float ang = (float)p * inv_freq[i];
        const float c = cosf(ang), s = sinf(ang);
        const float a = x1[i], b = x2[i];
        // rot = [-b, a];  x' = x * c + rot * s
        x1[i] = a * c - b * s;
        x2[i] = b * c + a * s;
      }
    }

    // --- KV heads ---
    for (int h=0; h<n_kv; ++h) {
      float* base = &K[(size_t)t*Dkv + (size_t)h*d];
      float* x1 = base;
      float* x2 = base + d2;
      for (int i=0; i<d2; ++i) {
        const float ang = (float)p * inv_freq[i];
        const float c = cosf(ang), s = sinf(ang);
        const float a = x1[i], b = x2[i];
        x1[i] = a * c - b * s;
        x2[i] = b * c + a * s;
      }
    }
  }

  free(inv_freq);
}

// -------------------------------------------------------------
// Attention forward with optional per-op comparisons against
// Python dumps. This uses the Raschka-consistent RoPE above.
// -------------------------------------------------------------
static void attn_forward_f32_check(
  const float* x, int T, int d_model,
  const float* Wq, const float* bq,
  const float* Wk, const float* bk,
  const float* Wv, const float* bv,
  const float* Wo, const float* bo,
  const float* qn, int qn_len,
  const float* kn, int kn_len,
  int n_q, int n_kv, int head_dim, int causal, float rope_theta,
  float* scratch, float* y_out,
  const char* outbase, int step, int L
){
  const float scale = 1.0f / sqrtf((float)head_dim);
  const int   Dq    = n_q  * head_dim;
  const int   Dkv   = n_kv * head_dim;

  float* Q    = scratch;            // [T, Dq]
  float* K    = Q + T*Dq;           // [T, Dkv]
  float* V    = K + T*Dkv;          // [T, Dkv]
  float* S    = V + T*Dkv;          // [T, T] scratch row per head
  float* Hcat = S + T*T;            // [T, Dq]

  // 1) Projections (+ optional biases)
  matmul_f32(x, Wq, Q, T, Dq,  d_model);
  matmul_f32(x, Wk, K, T, Dkv, d_model);
  matmul_f32(x, Wv, V, T, Dkv, d_model);
  if (bq) for (int t=0; t<T; ++t) for (int i=0; i<Dq;  ++i) Q[t*Dq  + i] += bq[i];
  if (bk) for (int t=0; t<T; ++t) for (int i=0; i<Dkv; ++i) K[t*Dkv + i] += bk[i];
  if (bv) for (int t=0; t<T; ++t) for (int i=0; i<Dkv; ++i) V[t*Dkv + i] += bv[i];

  // Optional compares (flat layout, as dumped by verify_greedy.py)
  {
    const char* fmt = "%s.step%d.L%d.%s.npy";
    if (NpyArray* Qexp = try_load_f32(fmt, outbase, step, L, "Q_proj_flat")) {
      printf("[attn] Q mad = %.6g\n", max_abs_diff(Q, Qexp->data, (size_t)T*Dq)); npy_free(Qexp);
    }
    if (NpyArray* Kexp = try_load_f32(fmt, outbase, step, L, "K_proj_flat")) {
      printf("[attn] K mad = %.6g\n", max_abs_diff(K, Kexp->data, (size_t)T*Dkv)); npy_free(Kexp);
    }
    if (NpyArray* Vexp = try_load_f32(fmt, outbase, step, L, "V_proj_flat")) {
      printf("[attn] V mad = %.6g\n", max_abs_diff(V, Vexp->data, (size_t)T*Dkv)); npy_free(Vexp);
    }
  }

  // 2) Q/K RMSNorm (per head), epsilon=1e-6, scale-only (no centering)
  const float eps_qk = 1e-6f;
  if (qn && qn_len == head_dim) {
    for (int t = 0; t < T; ++t) {
      float* Qt = &Q[t*Dq];
      for (int h=0; h<n_q; ++h) {
        float* v = &Qt[h*head_dim];
        float msq = 0.f; for (int d=0; d<head_dim; ++d){ float z=v[d]; msq += z*z; }
        float inv = 1.0f / sqrtf(msq / (float)head_dim + eps_qk);
        for (int d=0; d<head_dim; ++d) v[d] = (v[d] * inv) * qn[d];
      }
    }
  }
  if (kn && kn_len == head_dim) {
    for (int t = 0; t < T; ++t) {
      float* Kt = &K[t*Dkv];
      for (int h=0; h<n_kv; ++h) {
        float* v = &Kt[h*head_dim];
        float msq = 0.f; for (int d=0; d<head_dim; ++d){ float z=v[d]; msq += z*z; }
        float inv = 1.0f / sqrtf(msq / (float)head_dim + eps_qk);
        for (int d=0; d<head_dim; ++d) v[d] = (v[d] * inv) * kn[d];
      }
    }
  }
  {
    const char* fmt = "%s.step%d.L%d.%s.npy";
    if (NpyArray* Qn = try_load_f32(fmt, outbase, step, L, "Q_qknorm_flat")) {
      printf("[attn] Q_qknorm mad = %.6g\n", max_abs_diff(Q, Qn->data, (size_t)T*Dq)); npy_free(Qn);
    }
    if (NpyArray* Kn = try_load_f32(fmt, outbase, step, L, "K_qknorm_flat")) {
      printf("[attn] K_qknorm mad = %.6g\n", max_abs_diff(K, Kn->data, (size_t)T*Dkv)); npy_free(Kn);
    }
  }

  // 3) RoPE (Raschka/Qwen3 half-split) — no external cos/sin
  rope_apply_inplace_f32_gqa_halfsplit(Q, K, T, n_q, n_kv, head_dim, /*pos0=*/0, rope_theta);
  {
    const char* fmt = "%s.step%d.L%d.%s.npy";
    if (NpyArray* Qr = try_load_f32(fmt, outbase, step, L, "Q_rope_flat")) {
      printf("[attn] Q_rope mad = %.6g\n", max_abs_diff(Q, Qr->data, (size_t)T*Dq)); npy_free(Qr);
    }
    if (NpyArray* Kr = try_load_f32(fmt, outbase, step, L, "K_rope_flat")) {
      printf("[attn] K_rope mad = %.6g\n", max_abs_diff(K, Kr->data, (size_t)T*Dkv)); npy_free(Kr);
    }
  }

  // 4) Scores + mask + softmax + context (GQA)
  // GQA mapping: kvh = h / group_size  (equivalent to repeat_interleave in PyTorch)
  const int group_size = n_q / n_kv;
  for (int h=0; h<n_q; ++h) {
    const int kvh = h / group_size;

    // scores for head h
    for (int tq=0; tq<T; ++tq) {
      const float* qv = &Q[tq*Dq + h*head_dim];
      float* Sout = &S[tq*T];
      for (int tk=0; tk<T; ++tk) {
        const float* kv = &K[tk*Dkv + kvh*head_dim];
        float dot = 0.f;
        for (int d=0; d<head_dim; ++d) dot += qv[d]*kv[d];
        Sout[tk] = dot * scale;
      }
      if (causal) for (int tk=tq+1; tk<T; ++tk) S[tq*T + tk] = -INFINITY;

      // stable softmax over length T
      float maxv = Sout[0];
      for (int j=1; j<T; ++j) if (Sout[j] > maxv) maxv = Sout[j];
      float sum = 0.f;
      for (int j=0; j<T; ++j){ float e = expf(Sout[j] - maxv); Sout[j] = e; sum += e; }
      const float inv = 1.0f / (sum + 1e-9f);
      for (int j=0; j<T; ++j) Sout[j] *= inv;
    }

    // context -> Hcat (concatenate all heads along last dim)
    for (int tq=0; tq<T; ++tq) {
      const float* Prow = &S[tq*T];
      float* out = &Hcat[tq*Dq + h*head_dim];
      for (int d=0; d<head_dim; ++d) out[d] = 0.f;
      for (int tk=0; tk<T; ++tk) {
        const float* vv = &V[tk*Dkv + kvh*head_dim];
        const float p   = Prow[tk];
        for (int d=0; d<head_dim; ++d) out[d] += p * vv[d];
      }
    }
  }

  // 5) Output projection (+ optional bias)
  matmul_f32(Hcat, Wo, y_out, T, d_model, Dq);
  if (bo) for (int t=0; t<T; ++t) for (int i=0; i<d_model; ++i) y_out[t*d_model + i] += bo[i];

  // Optional internal compare vs attn_out dump
  {
    char p2[512];
    snprintf(p2, sizeof(p2), "%s.step%d.L%d.attn_out.npy", outbase, step, L);
    if (NpyArray* AOexp = npy_load_float32(p2)) {
      printf("[attn] attn_out (internal) mad = %.6g\n", max_abs_diff(y_out, AOexp->data, (size_t)T*d_model));
      npy_free(AOexp);
    }
  }
}

// -------------
// Boilerplate:
//  load weights, inputs, run norm1 -> attn -> residual -> norm2 -> router -> MoE -> y
//  (unchanged from your working version except comments)
// -------------

static TensorBin* need(BinFile* b, const char* k){
  TensorBin* t = bin_find(b,k);
  if(!t){ fprintf(stderr,"missing %s\n", k); exit(1); }
  return t;
}
static TensorBin* maybe(BinFile* b, const char* k){ return bin_find(b,k); }

int main(int argc, char** argv){
  if (argc < 6){
    fprintf(stderr, "Usage: %s <l0_layer.bin> <outbase> <step> <T> <rope_theta>\n", argv[0]);
    return 1;
  }
  const char* wfile = argv[1];
  const char* outbase = argv[2];
  int step = atoi(argv[3]);
  int T = atoi(argv[4]);
  float rope_theta = strtof(argv[5], NULL);

  // weights for layer 0
  BinFile* bf = bin_load(wfile);
  if (!bf){ fprintf(stderr,"bin load fail\n"); return 1; }

  TensorBin *Wq=need(bf,"model.layers.0.self_attn.q_proj.weight"),
            *Wk=need(bf,"model.layers.0.self_attn.k_proj.weight"),
            *Wv=need(bf,"model.layers.0.self_attn.v_proj.weight"),
            *Wo=need(bf,"model.layers.0.self_attn.o_proj.weight");
  TensorBin *bq=maybe(bf,"model.layers.0.self_attn.q_proj.bias"),
            *bk=maybe(bf,"model.layers.0.self_attn.k_proj.bias"),
            *bv=maybe(bf,"model.layers.0.self_attn.v_proj.bias"),
            *bo=maybe(bf,"model.layers.0.self_attn.o_proj.bias");
  TensorBin *qn=maybe(bf,"model.layers.0.self_attn.q_norm.weight"),
            *kn=maybe(bf,"model.layers.0.self_attn.k_norm.weight");

  TensorBin *w1=maybe(bf,"model.layers.0.input_layernorm.weight");
  if(!w1) w1 = need(bf,"model.layers.0.rms_1.weight");
  TensorBin *w2=maybe(bf,"model.layers.0.post_attention_layernorm.weight");
  if(!w2) w2 = need(bf,"model.layers.0.rms_2.weight");

  TensorBin* RW = maybe(bf,"model.layers.0.mlp.gate.weight");
  if(!RW) RW = need(bf,"model.layers.0.mlp.router.gate.weight");
  TensorBin* RB = maybe(bf,"model.layers.0.mlp.gate.bias");
  if(!RB) RB = maybe(bf,"model.layers.0.mlp.router.gate.bias");

  const int d_model = Wq->shape[1];
  const int Dq = Wq->shape[0];
  const int Dkv = Wk->shape[0];
  const int head_dim = qn ? qn->shape[0] : (Dq / 32);
  const int n_q = Dq / head_dim;
  const int n_kv = Dkv / head_dim;
  const int E = RW->shape[0];

  // infer d_ff from any expert down_proj
  int d_ff = -1;
  for (int e=0;e<E;++e){
    char key[256];
    snprintf(key,sizeof(key),"model.layers.0.mlp.experts.%d.down_proj.weight", e);
    if (TensorBin* Wd = bin_find(bf,key)){ d_ff = Wd->shape[1]; break; }
  }
  if (d_ff <= 0){ fprintf(stderr,"d_ff infer fail\n"); return 1; }

  // gather expert weights (no biases in this model)
  const float** Wg = (const float**)calloc(E,sizeof(float*));
  const float** Wu = (const float**)calloc(E,sizeof(float*));
  const float** Wd = (const float**)calloc(E,sizeof(float*));
  for (int e=0;e<E;++e){
    char k1[256],k2[256],k3[256];
    snprintf(k1,sizeof(k1),"model.layers.0.mlp.experts.%d.gate_proj.weight", e);
    snprintf(k2,sizeof(k2),"model.layers.0.mlp.experts.%d.up_proj.weight",   e);
    snprintf(k3,sizeof(k3),"model.layers.0.mlp.experts.%d.down_proj.weight", e);
    if (TensorBin* t=bin_find(bf,k1)) Wg[e]=(const float*)t->data;
    if (TensorBin* t=bin_find(bf,k2)) Wu[e]=(const float*)t->data;
    if (TensorBin* t=bin_find(bf,k3)) Wd[e]=(const float*)t->data;
  }

  // load Python dumps for step
  char path[512];
  #define LOAD_F32(VAR,NAME) \
    snprintf(path,sizeof(path),"%s.step%d.L0.%s.npy", outbase, step, NAME); \
    NpyArray* VAR = npy_load_float32(path); \
    if(!VAR){ fprintf(stderr,"missing %s\n", path); return 1; }

  snprintf(path,sizeof(path),"%s.step%d.x.npy", outbase, step);
  NpyArray* X = npy_load_float32(path);
  if(!X){ fprintf(stderr,"missing %s\n", path); return 1; }
  if (X->shape[0]!=T || X->shape[1]!=d_model){ fprintf(stderr,"x shape mismatch\n"); return 1; }

  LOAD_F32(XN1, "x_norm1");
  LOAD_F32(AO,  "attn_out");
  LOAD_F32(XA,  "x_after_attn");
  LOAD_F32(XN2, "x_norm2");
  LOAD_F32(RL,  "router_logits");
  LOAD_F32(MO,  "moe_out");
  LOAD_F32(YREF,"y");

  // top-k (from Python) used only for deterministic MoE compare
  snprintf(path,sizeof(path),"%s.step%d.L0.router_topk_idx.npy", outbase, step);
  NpyArrayI32* TKIDX = npy_load_int32(path);
  if(!TKIDX){ fprintf(stderr,"missing %s\n", path); return 1; }
  snprintf(path,sizeof(path),"%s.step%d.L0.router_topk_p.npy", outbase, step);
  NpyArray* TKP = npy_load_float32(path);
  if(!TKP){ fprintf(stderr,"missing %s\n", path); return 1; }
  const int k = TKIDX->shape[1];

  // scratch
  const int Dq0 = n_q*head_dim, Dkv0 = n_kv*head_dim;
  size_t attn_f = (size_t)T*(Dq0 + 2*Dkv0) + (size_t)T*T + (size_t)T*Dq0;
  float* scratch_attn = (float*)malloc(sizeof(float)*attn_f);
  float* x_norm1 = (float*)malloc(sizeof(float)*T*d_model);
  float* attn_out= (float*)malloc(sizeof(float)*T*d_model);
  float* x_after = (float*)malloc(sizeof(float)*T*d_model);
  float* x_norm2 = (float*)malloc(sizeof(float)*T*d_model);

  // 1) norm1
  rmsnorm_forward_f32(X->data, (const float*)w1->data, T, d_model, 1e-6f, x_norm1);
  printf("[check] x_norm1 mad = %.6g\n", max_abs_diff(x_norm1, XN1->data, (size_t)T*d_model));

  // 2) attention (with internal Raschka-style RoPE)
  attn_forward_f32_check(
    x_norm1, T, d_model,
    (const float*)Wq->data, bq?(const float*)bq->data:NULL,
    (const float*)Wk->data, bk?(const float*)bk->data:NULL,
    (const float*)Wv->data, bv?(const float*)bv->data:NULL,
    (const float*)Wo->data, bo?(const float*)bo->data:NULL,
    qn?(const float*)qn->data:NULL, qn?head_dim:0,
    kn?(const float*)kn->data:NULL, kn?head_dim:0,
    n_q, n_kv, head_dim, /*causal=*/1, rope_theta,
    scratch_attn, attn_out,
    outbase, step, 0
  );
  printf("[check] attn_out mad = %.6g\n", max_abs_diff(attn_out, AO->data, (size_t)T*d_model));

  // residual
  for (int i=0;i<T*d_model;++i) x_after[i] = X->data[i] + attn_out[i];
  printf("[check] x_after_attn mad = %.6g\n", max_abs_diff(x_after, XA->data, (size_t)T*d_model));

  // 3) norm2
  rmsnorm_forward_f32(x_after, (const float*)w2->data, T, d_model, 1e-6f, x_norm2);
  printf("[check] x_norm2 mad = %.6g\n", max_abs_diff(x_norm2, XN2->data, (size_t)T*d_model));

  // 4) router logits
  float* logits = (float*)malloc(sizeof(float)* (size_t)T*E);
  matmul_f32(x_norm2, (const float*)RW->data, logits, T, E, d_model);
  if (RB && RB->dtype==0){
    const float* rb = (const float*)RB->data;
    for (int t=0;t<T;++t) for (int e=0;e<E;++e) logits[t*E+e] += rb[e];
  }
  printf("[check] router_logits mad = %.6g\n", max_abs_diff(logits, RL->data, (size_t)T*E));

  // 5) MoE using Python top-k (deterministic parity)
  float* moe_out = (float*)malloc(sizeof(float)*(size_t)T*d_model);
  for (int i=0;i<T*d_model;++i) moe_out[i] = 0.f;

  float* tmp_g = (float*)malloc(sizeof(float)*d_ff);
  float* tmp_u = (float*)malloc(sizeof(float)*d_ff);
  float* tmp_y = (float*)malloc(sizeof(float)*d_model);

  for (int t=0;t<T;++t){
    for (int j=0;j<k;++j){
      int e = TKIDX->data[t*k + j];
      float p = TKP->data[t*k + j];
      matmul_f32(&x_norm2[(size_t)t*d_model], Wg[e], tmp_g, 1, d_ff, d_model);
      silu_f32(tmp_g, d_ff);
      matmul_f32(&x_norm2[(size_t)t*d_model], Wu[e], tmp_u, 1, d_ff, d_model);
      for (int q=0;q<d_ff;++q) tmp_g[q] *= tmp_u[q];
      matmul_f32(tmp_g, Wd[e], tmp_y, 1, d_model, d_ff);
      for (int q=0;q<d_model;++q) moe_out[(size_t)t*d_model + q] += p * tmp_y[q];
    }
  }
  printf("[check] moe_out mad = %.6g\n", max_abs_diff(moe_out, MO->data, (size_t)T*d_model));

  // 6) final y
  float* y = (float*)malloc(sizeof(float)*(size_t)T*d_model);
  for (int i=0;i<T*d_model;++i) y[i] = x_after[i] + moe_out[i];
  double d7 = max_abs_diff(y, YREF->data, (size_t)T*d_model);
  printf("[check] y mad = %.6g\n", d7);

  int rc = (d7 > 2e-4) ? 1 : 0;
  printf("%s\n", rc? "FAIL":"PASS");

  // cleanup
  free(y); free(moe_out); free(tmp_g); free(tmp_u); free(tmp_y);
  free(logits); free(x_norm1); free(attn_out); free(x_after); free(x_norm2);
  free(scratch_attn);
  npy_free(X); npy_free(XN1); npy_free(AO); npy_free(XA); npy_free(XN2);
  npy_free(RL); npy_free(MO); npy_free(YREF); npy_free_i32(TKIDX); npy_free(TKP);
  free(Wg); free(Wu); free(Wd);
  bin_free(bf);
  return rc;
}
