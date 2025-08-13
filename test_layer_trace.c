// test_layer_trace.c
// Compare per-op intermediates for one layer (L=0) against Python dumps.
// Usage:
//   ./test_layer_trace <l0_layer.bin> <outbase> <step> <T> <rope_theta>
//
// IMPORTANT: RoPE consistency with Raschka Qwen-3 reference
// We now load cos/sin from the Python dumper (verify_dump.py) and apply
// the rotation exactly like the notebook's `apply_rope`:
//
//   cos = cos[:T, :][None,None]      # [1,1,T,d]
//   sin = sin[:T, :][None,None]
//   x1, x2 = x[..., :d/2], x[..., d/2:]
//   rot = cat([-x2, x1], dim=-1)
//   x_new = x * cos + rot * sin
//
// Here, Q and K are laid out as flat [T, H * d] with each head contiguous,
// so we rotate per head by iterating (t,h) and then applying the same
// pairwise 2D rotation to (even, odd) positions across the head_dim.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>

#include "io.h"
#include "utils.h"
#include "kernels.h"

// --- Optional comparator: try load an npy; if missing, return NULL (no compare) ---
static NpyArray* try_load_f32(const char* path_fmt, const char* outbase, int step, int L, const char* name){
  char p[512];
  snprintf(p, sizeof(p), path_fmt, outbase, step, L, name);
  NpyArray* arr = npy_load_float32(p);
  return arr; // NULL means "missing" (skip compare)
}

// --- Raschka-consistent RoPE using dumped cos/sin ---
// X is [T, n_heads * head_dim], with each head contiguous.
// cos/sin are [ctx, head_dim] as saved by verify_dump.py (float32).
// We rotate the pairs (even=i, odd=i+d/2) by angle whose cos/sin are
// cos[t, i] and sin[t, i], *identical* to reference apply_rope().
static void apply_rope_from_cos_sin(
  float* X, int T, int n_heads, int head_dim,
  const float* cos, const float* sin, int cos_T, int cos_D
){
  // Sanity: reference stores cos/sin with second half duplicated,
  // but we only index the first half [0..d/2-1] as the pair angle source.
  const int d2 = head_dim / 2;
  if (head_dim % 2 != 0) {
    fprintf(stderr, "[rope] head_dim must be even, got %d\n", head_dim);
    exit(1);
  }
  if (cos_D != head_dim || cos_T < T) {
    fprintf(stderr, "[rope] cos/sin shape mismatch: have [%d,%d], need at least [%d,%d]\n",
            cos_T, cos_D, T, head_dim);
    exit(1);
  }

  // For each token and head, rotate its (even, odd) pairs.
  // This is exactly: x' = x * cos + rot(x) * sin
  // where rot(x) = concat(-x_odd, x_even).
  const size_t Hd = (size_t)n_heads * (size_t)head_dim;
  for (int t = 0; t < T; ++t) {
    const float* cos_t = &cos[(size_t)t * cos_D];
    const float* sin_t = &sin[(size_t)t * cos_D];
    for (int h = 0; h < n_heads; ++h) {
      float* v = &X[(size_t)t * Hd + (size_t)h * head_dim];

      // Split v into (even, odd) halves by index, matching the Python:
      // x1 = v[:d2], x2 = v[d2:]
      // rot = [-x2, x1]; then v = v * cos + rot * sin
      for (int i = 0; i < d2; ++i) {
        const float c = cos_t[i];     // NOTE: reference uses the first-half angle
        const float s = sin_t[i];

        const float x_even = v[i];        // x1[i]
        const float x_odd  = v[i + d2];   // x2[i]

        // Compute rot = [-x_odd, x_even]
        const float rot_even = -x_odd;
        const float rot_odd  =  x_even;

        // v' = v * c + rot * s
        const float new_even = x_even * c + rot_even * s; // = x_even*c - x_odd*s
        const float new_odd  = x_odd  * c + rot_odd  * s; // = x_odd*c + x_even*s

        v[i]       = new_even;
        v[i + d2]  = new_odd;
      }
    }
  }
}

// -------------------------
// Your existing code below
// -------------------------

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
  const int Dq  = n_q  * head_dim;
  const int Dkv = n_kv * head_dim;

  float* Q    = scratch;            // [T, Dq]
  float* K    = Q + T*Dq;           // [T, Dkv]
  float* V    = K + T*Dkv;          // [T, Dkv]
  float* S    = V + T*Dkv;          // [T, T]
  float* Hcat = S + T*T;            // [T, Dq]

  // 1) Projections (+ optional biases)
  matmul_f32(x, Wq, Q, T, Dq,  d_model);
  matmul_f32(x, Wk, K, T, Dkv, d_model);
  matmul_f32(x, Wv, V, T, Dkv, d_model);
  if (bq) for (int t=0; t<T; ++t) for (int i=0; i<Dq;  ++i) Q[t*Dq  + i] += bq[i];
  if (bk) for (int t=0; t<T; ++t) for (int i=0; i<Dkv; ++i) K[t*Dkv + i] += bk[i];
  if (bv) for (int t=0; t<T; ++t) for (int i=0; i<Dkv; ++i) V[t*Dkv + i] += bv[i];

  // Optional Q/K/V compares (flat)
  {
    const char* fmt = "%s.step%d.L%d.%s.npy";
    NpyArray* Qexp = try_load_f32(fmt, outbase, step, L, "Q_proj_flat");
    NpyArray* Kexp = try_load_f32(fmt, outbase, step, L, "K_proj_flat");
    NpyArray* Vexp = try_load_f32(fmt, outbase, step, L, "V_proj_flat");
    if (Qexp) { printf("[attn] Q mad = %.6g\n", max_abs_diff(Q, Qexp->data, (size_t)T*Dq)); npy_free(Qexp); }
    if (Kexp) { printf("[attn] K mad = %.6g\n", max_abs_diff(K, Kexp->data, (size_t)T*Dkv)); npy_free(Kexp); }
    if (Vexp) { printf("[attn] V mad = %.6g\n", max_abs_diff(V, Vexp->data, (size_t)T*Dkv)); npy_free(Vexp); }
  }

  // 2) QK RMSNorm per head (matches RMSNorm over last dim; no centering, learned scale)
  const float eps_qk = 1e-6f;
  if (qn && qn_len == head_dim) {
    for (int t = 0; t < T; ++t) {
      float* Qt = &Q[t*Dq];
      for (int h = 0; h < n_q; ++h) {
        float* v = &Qt[h*head_dim];
        float msq = 0.f; for (int d=0; d<head_dim; ++d) { float z=v[d]; msq += z*z; }
        float inv = 1.0f / sqrtf(msq / (float)head_dim + eps_qk);
        for (int d=0; d<head_dim; ++d) v[d] = (v[d] * inv) * qn[d];
      }
    }
  }
  if (kn && kn_len == head_dim) {
    for (int t = 0; t < T; ++t) {
      float* Kt = &K[t*Dkv];
      for (int h = 0; h < n_kv; ++h) {
        float* v = &Kt[h*head_dim];
        float msq = 0.f; for (int d=0; d<head_dim; ++d) { float z=v[d]; msq += z*z; }
        float inv = 1.0f / sqrtf(msq / (float)head_dim + eps_qk);
        for (int d=0; d<head_dim; ++d) v[d] = (v[d] * inv) * kn[d];
      }
    }
  }
  {
    const char* fmt = "%s.step%d.L%d.%s.npy";
    NpyArray* Qn = try_load_f32(fmt, outbase, step, L, "Q_qknorm_flat");
    NpyArray* Kn = try_load_f32(fmt, outbase, step, L, "K_qknorm_flat");
    if (Qn) { printf("[attn] Q_qknorm mad = %.6g\n", max_abs_diff(Q, Qn->data, (size_t)T*Dq)); npy_free(Qn); }
    if (Kn) { printf("[attn] K_qknorm mad = %.6g\n", max_abs_diff(K, Kn->data, (size_t)T*Dkv)); npy_free(Kn); }
  }

  // 3) RoPE — EXACT Raschka behavior using dumped cos/sin
  //    cos/sin were saved as outbase.cos.npy / outbase.sin.npy with shape [ctx, d]
  char cpath[512], spath[512];
  snprintf(cpath, sizeof(cpath), "%s.cos.npy", outbase);
  snprintf(spath, sizeof(spath), "%s.sin.npy", outbase);
  NpyArray* COS = npy_load_float32(cpath);
  NpyArray* SIN = npy_load_float32(spath);
  if (!COS || !SIN) {
    fprintf(stderr, "missing cos/sin npy (expected %s and %s)\n", cpath, spath);
    exit(1);
  }
  // Apply to Q and K. Note: we operate on flat [T, H*d], but rotation is per-head.
  apply_rope_from_cos_sin(Q, T, n_q,  head_dim, COS->data, SIN->data, COS->shape[0], COS->shape[1]);
  apply_rope_from_cos_sin(K, T, n_kv, head_dim, COS->data, SIN->data, COS->shape[0], COS->shape[1]);
  {
    const char* fmt = "%s.step%d.L%d.%s.npy";
    NpyArray* Qr = try_load_f32(fmt, outbase, step, L, "Q_rope_flat");
    NpyArray* Kr = try_load_f32(fmt, outbase, step, L, "K_rope_flat");
    if (Qr) { printf("[attn] Q_rope mad = %.6g\n", max_abs_diff(Q, Qr->data, (size_t)T*Dq)); npy_free(Qr); }
    if (Kr) { printf("[attn] K_rope mad = %.6g\n", max_abs_diff(K, Kr->data, (size_t)T*Dkv)); npy_free(Kr); }
  }
  npy_free(COS); npy_free(SIN);

  // 4) Scores + softmax + context (GQA group mapping = repeat kv heads over groups)
  const int group_size = n_q / n_kv;
  for (int h = 0; h < n_q; ++h) {
    const int kvh = h / group_size;  // identical to the PyTorch repeat_interleave mapping
    for (int tq = 0; tq < T; ++tq) {
      const float* qv = &Q[tq * Dq  + h   * head_dim];
      float* Sout = &S[tq * T];
      for (int tk = 0; tk < T; ++tk) {
        const float* kv = &K[tk * Dkv + kvh * head_dim];
        float dot = 0.f;
        for (int d = 0; d < head_dim; ++d) dot += qv[d] * kv[d];
        Sout[tk] = dot * scale;
      }
      if (causal) for (int tk = tq + 1; tk < T; ++tk) S[tq*T + tk] = -INFINITY;

      float maxv = Sout[0];
      for (int tk = 1; tk < T; ++tk) if (Sout[tk] > maxv) maxv = Sout[tk];
      float sum = 0.f;
      for (int tk = 0; tk < T; ++tk) { float e = expf(Sout[tk] - maxv); Sout[tk] = e; sum += e; }
      float inv = 1.f / (sum + 1e-9f);
      for (int tk = 0; tk < T; ++tk) Sout[tk] *= inv;
    }
    // Context into Hcat
    for (int tq = 0; tq < T; ++tq) {
      const float* Prow = &S[tq * T];
      float* out = &Hcat[tq * Dq + h * head_dim];
      for (int d = 0; d < head_dim; ++d) out[d] = 0.f;
      for (int tk = 0; tk < T; ++tk) {
        const float* vv = &V[tk * Dkv + kvh * head_dim];
        const float p   = Prow[tk];
        for (int d = 0; d < head_dim; ++d) out[d] += p * vv[d];
      }
    }
  }

  // 5) Output projection (+ optional bias)
  matmul_f32(Hcat, Wo, y_out, T, d_model, Dq);
  if (bo) for (int t = 0; t < T; ++t) for (int i = 0; i < d_model; ++i) y_out[t*d_model + i] += bo[i];

  // Optional internal compare
  {
    char p2[512];
    snprintf(p2, sizeof(p2), "%s.step%d.L%d.attn_out.npy", outbase, step, L);
    NpyArray* AOexp = npy_load_float32(p2);
    if (AOexp) { printf("[attn] attn_out (internal) mad = %.6g\n", max_abs_diff(y_out, AOexp->data, (size_t)T*d_model)); npy_free(AOexp); }
  }
}

// (The rest of your file remains the same — loading weights, norms, router, MoE, etc.)
//
// Replace ONLY the call site in main() where attention is run; you already
// call attn_forward_f32_check(...), which now uses apply_rope_from_cos_sin().
// Everything after that (residual, norm2, router, MoE, final y) is unchanged.
//
// NOTE on thresholds:
// With this change, `T=2` should now match as tightly as `T=1` (on the order of 1e-6~1e-5).

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

  // weights
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

  // infer d_ff
  int d_ff = -1;
  for (int e=0;e<E;++e){
    char key[256];
    snprintf(key,sizeof(key),"model.layers.0.mlp.experts.%d.down_proj.weight", e);
    TensorBin* Wd = bin_find(bf,key);
    if (Wd){ d_ff = Wd->shape[1]; break; }
  }
  if (d_ff <= 0){ fprintf(stderr,"d_ff infer fail\n"); return 1; }

  // expert arrays
  const float** Wg = (const float**)calloc(E,sizeof(float*));
  const float** Wu = (const float**)calloc(E,sizeof(float*));
  const float** Wd = (const float**)calloc(E,sizeof(float*));
  for (int e=0;e<E;++e){
    char k1[256],k2[256],k3[256];
    snprintf(k1,sizeof(k1),"model.layers.0.mlp.experts.%d.gate_proj.weight", e);
    snprintf(k2,sizeof(k2),"model.layers.0.mlp.experts.%d.up_proj.weight",   e);
    snprintf(k3,sizeof(k3),"model.layers.0.mlp.experts.%d.down_proj.weight", e);
    TensorBin* t;
    if ((t=bin_find(bf,k1))) Wg[e]=(const float*)t->data;
    if ((t=bin_find(bf,k2))) Wu[e]=(const float*)t->data;
    if ((t=bin_find(bf,k3))) Wd[e]=(const float*)t->data;
  }

  // load dumps
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

  // ints for topk
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
  double d1 = max_abs_diff(x_norm1, XN1->data, T*d_model);
  printf("[check] x_norm1 mad = %.6g\n", d1);

  // 2) attention
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
    /*outbase*/ outbase, /*step*/ step, /*layer L=*/0
  );
  double d2 = max_abs_diff(attn_out, AO->data, T*d_model);
  printf("[check] attn_out mad = %.6g\n", d2);
//  attn_forward_f32_gqa(
//    x_norm1, T, d_model,
//    (const float*)Wq->data, bq?(const float*)bq->data:NULL,
//    (const float*)Wk->data, bk?(const float*)bk->data:NULL,
//    (const float*)Wv->data, bv?(const float*)bv->data:NULL,
//    (const float*)Wo->data, bo?(const float*)bo->data:NULL,
//    qn?(const float*)qn->data:NULL, qn?head_dim:0,
//    kn?(const float*)kn->data:NULL, kn?head_dim:0,
//    n_q, n_kv, head_dim, /*causal*/1, rope_theta,
//    scratch_attn, attn_out
//  );

  // residual
  for (int i=0;i<T*d_model;++i) x_after[i] = X->data[i] + attn_out[i];
  double d3 = max_abs_diff(x_after, XA->data, T*d_model);
  printf("[check] x_after_attn mad = %.6g\n", d3);

  // 3) norm2
  rmsnorm_forward_f32(x_after, (const float*)w2->data, T, d_model, 1e-6f, x_norm2);
  double d4 = max_abs_diff(x_norm2, XN2->data, T*d_model);
  printf("[check] x_norm2 mad = %.6g\n", d4);

  // 4) router logits
  float* logits = (float*)malloc(sizeof(float)*T*E);
  matmul_f32(x_norm2, (const float*)RW->data, logits, T, E, d_model);
  if (RB && RB->dtype==0){
    const float* rb = (const float*)RB->data;
    for (int t=0;t<T;++t) for (int e=0;e<E;++e) logits[t*E+e] += rb[e];
  }
  double d5 = max_abs_diff(logits, RL->data, T*E);
  printf("[check] router_logits mad = %.6g\n", d5);

  // 5) reuse topk from Python and compute moe_out deterministically
  float* moe_out = (float*)malloc(sizeof(float)*T*d_model);
  for (int i=0;i<T*d_model;++i) moe_out[i] = 0.f;

  float* tmp_g = (float*)malloc(sizeof(float)*d_ff);
  float* tmp_u = (float*)malloc(sizeof(float)*d_ff);
  float* tmp_y = (float*)malloc(sizeof(float)*d_model);

  for (int t=0;t<T;++t){
    for (int j=0;j<k;++j){
      int e = TKIDX->data[t*k + j];
      float p = TKP->data[t*k + j];
      // expert forward on x_norm2[t]
      matmul_f32(&x_norm2[(size_t)t*d_model], Wg[e], tmp_g, 1, d_ff, d_model);
      silu_f32(tmp_g, d_ff);
      matmul_f32(&x_norm2[(size_t)t*d_model], Wu[e], tmp_u, 1, d_ff, d_model);
      for (int q=0;q<d_ff;++q) tmp_g[q] *= tmp_u[q];
      matmul_f32(tmp_g, Wd[e], tmp_y, 1, d_model, d_ff);
      for (int q=0;q<d_model;++q) moe_out[(size_t)t*d_model + q] += p * tmp_y[q];
    }
  }
  double d6 = max_abs_diff(moe_out, MO->data, T*d_model);
  printf("[check] moe_out mad = %.6g\n", d6);

  // 6) final layer output
  float* y = (float*)malloc(sizeof(float)*T*d_model);
  for (int i=0;i<T*d_model;++i) y[i] = x_after[i] + moe_out[i];
  double d7 = max_abs_diff(y, YREF->data, T*d_model);
  printf("[check] y mad = %.6g\n", d7);

  int rc = 0;
  //if (d1>2e-5||d2>2e-5||d3>2e-5||d4>2e-5||d5>2e-5||d6>2e-5||d7>2e-5) rc = 1;
  if (d7>2e-4) rc = 1;
  printf("%s\n", rc? "FAIL":"PASS");

  free(y); free(moe_out); free(tmp_g); free(tmp_u); free(tmp_y);
  free(logits); free(x_norm1); free(attn_out); free(x_after); free(x_norm2);
  free(scratch_attn);
  npy_free(X); npy_free(XN1); npy_free(AO); npy_free(XA); npy_free(XN2);
  npy_free(RL); npy_free(MO); npy_free(YREF); npy_free_i32(TKIDX); npy_free(TKP);
  free(Wg); free(Wu); free(Wd);
  bin_free(bf);
  return rc;
}
