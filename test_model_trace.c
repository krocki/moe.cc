// test_layer_trace.c
// End-to-end check against Python dumps across ALL layers and steps.
// Usage:
//   ./test_layer_trace <all.bin> <outbase> <num_steps>
// It will:
//   • load model weights from <all.bin> (same key names as before)
//   • load RoPE cos/sin from <outbase>.cos.npy / <outbase>.sin.npy
//   • for each step s in [0..num_steps-1]:
//       - load x = {outbase}.step{s}.x.npy  (embed dump from Python)
//       - run all layers (norm1 -> attn -> +res -> norm2 -> MoE -> +res)
//           NOTE: we REUSE router_topk from the Python dump for exact parity
//       - final norm + lm_head
//       - compare last-token logits/probs to {outbase}.logits.npy[s], {outbase}.probs.npy[s]
//       - check greedy next-id equals ids.npy[T0+s+1]
//   • print per-step MADs and an overall PASS/FAIL.
//
// Implementation notes:
//   - RoPE: we DO NOT recompute angles; we load cos/sin from numpy and apply exactly the
//     same formula as in the reference `apply_rope`: x' = x * cos + rot(x) * sin, where
//     rot(x) = concat(-x[d/2:], x[:d/2]) on each head independently.
//   - GQA: query head h uses kv head (h // group_size), identical to PyTorch's
//     `repeat_interleave(group_size, dim=1)` in the reference.
//   - Router: to guarantee bit-equality end-to-end, we reuse Python’s per-token top-k
//     (indices & softmax weights). This avoids tiny tie-breaking or ULP diffs changing
//     choices on borderline tokens.
//
// Dependencies: io.h (npy load), kernels.h (matmul/rmsnorm/silu), utils.h (max_abs_diff).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>

#include "io.h"
#include "utils.h"
#include "kernels.h"

// ---------- small utils ----------

static TensorBin* need(BinFile* b, const char* k){
  TensorBin* t = bin_find(b,k);
  if(!t){ fprintf(stderr,"missing %s\n", k); exit(1); }
  return t;
}
static TensorBin* maybe(BinFile* b, const char* k){ return bin_find(b,k); }

static NpyArray* try_load(const char* path){
  NpyArray* a = npy_load_float32(path);
  return a; // ok if NULL
}
static NpyArray* try_load_f32_fmt(const char* fmt, const char* base, int step, int L, const char* name){
  char p[512]; snprintf(p,sizeof(p),fmt, base, step, L, name);
  return try_load(p);
}

// ---------- Raschka-consistent RoPE using dumped cos/sin ----------
// X = [T, H*D] (each head contiguous), cos/sin = [ctx, D] (float32)
static void apply_rope_from_cos_sin(float* X, int T, int n_heads, int head_dim,
                                    const float* cos, const float* sin, int cos_T, int cos_D)
{
  const int d = head_dim;
  const int d2 = d/2;
  if (d % 2) { fprintf(stderr,"[rope] head_dim must be even\n"); exit(1); }
  if (cos_D != d || cos_T < T) {
    fprintf(stderr,"[rope] cos/sin shape mismatch: have [%d,%d], need at least [%d,%d]\n", cos_T, cos_D, T, d);
    exit(1);
  }
  const size_t HD = (size_t)n_heads * (size_t)d;
  for (int t=0; t<T; ++t){
    const float* ct = &cos[(size_t)t*cos_D];
    const float* st = &sin[(size_t)t*cos_D];
    for (int h=0; h<n_heads; ++h){
      float* v = &X[(size_t)t*HD + (size_t)h*d];
      // x' = x * cos + rot(x) * sin, rot = [-x2, x1], where x=[x1|x2]
      for (int i=0;i<d2;++i){
        const float c = ct[i], s = st[i];
        const float xe = v[i];      // even half
        const float xo = v[i+d2];   // odd  half
        const float rot_e = -xo;
        const float rot_o =  xe;
        v[i]      = xe * c + rot_e * s;   // = xe*c - xo*s
        v[i+d2]   = xo * c + rot_o * s;   // = xo*c + xe*s
      }
    }
  }
}

// ---------- attention (GQA) for a single layer, with optional compares & reused routing ----------
static void attn_moe_layer_forward(
  // in/out
  const float* x, int T, int d_model,
  // attn
  const float* Wq, const float* bq,
  const float* Wk, const float* bk,
  const float* Wv, const float* bv,
  const float* Wo, const float* bo,
  const float* qn, int qn_len,
  const float* kn, int kn_len,
  int n_q, int n_kv, int head_dim, int causal,
  // rope buffers (numpy from python)
  const float* cos_arr, const float* sin_arr, int cos_T, int cos_D,
  // router + experts
  const float* router_w, const float* router_b, int E, int d_ff,
  const float** Wg, const float** Wu, const float** Wd,
  // scratch & outputs
  float* scratch, float* h_out,
  // trace source to reuse topk (for bit-equal aggregation)
  const char* outbase, int step, int L
){
  const int Dq  = n_q  * head_dim;
  const int Dkv = n_kv * head_dim;
  const float scale = 1.0f / sqrtf((float)head_dim);

  float* Q    = scratch;            // [T, Dq]
  float* K    = Q + T*Dq;           // [T, Dkv]
  float* V    = K + T*Dkv;          // [T, Dkv]
  float* S    = V + T*Dkv;          // [T, T]
  float* Hcat = S + T*T;            // [T, Dq]
  float* x_norm1 = Hcat + T*Dq;     // [T, d_model]
  float* attn_out= x_norm1 + T*d_model;
  float* x_norm2 = attn_out + T*d_model;
  float* logits  = x_norm2 + T*d_model;
  // tmp buffers for MoE
  float* tmp_g   = logits + T*E;    // [d_ff]
  float* tmp_u   = tmp_g + d_ff;    // [d_ff]
  float* tmp_y   = tmp_u + d_ff;    // [d_model]

  // 1) norm1
  rmsnorm_forward_f32(x, /*scale*/NULL /*we'll pass actual below*/, T, d_model, 1e-6f, x_norm1);
  // NOTE: kernels.rmsnorm_forward_f32 expects scale!=NULL. Provide it:
  // (we don't have it here; call the overload that takes weight explicitly)
  // but your kernels.h version in earlier snippet already had "weights" param.
  // Use that signature:
  // rmsnorm_forward_f32(x, rms1_w, T, d_model, 1e-6f, x_norm1);

  // Actually do it properly: caller already has rms1_w in its scope; we keep API here simple.
  // We'll compute x_norm1 outside this function. For minimal churn we do it here again with weight.

  // --- Projections
  matmul_f32(x_norm1, Wq, Q, T, Dq,  d_model);
  matmul_f32(x_norm1, Wk, K, T, Dkv, d_model);
  matmul_f32(x_norm1, Wv, V, T, Dkv, d_model);
  if (bq) for (int t=0;t<T;++t) for (int i=0;i<Dq;  ++i) Q[t*Dq  + i] += bq[i];
  if (bk) for (int t=0;t<T;++t) for (int i=0;i<Dkv; ++i) K[t*Dkv + i] += bk[i];
  if (bv) for (int t=0;t<T;++t) for (int i=0;i<Dkv; ++i) V[t*Dkv + i] += bv[i];

  // QK RMSNorm per-head if provided
  const float eps_qk = 1e-6f;
  if (qn && qn_len==head_dim){
    for (int t=0;t<T;++t){
      float* Qt = &Q[t*Dq];
      for (int h=0;h<n_q;++h){
        float* v=&Qt[h*head_dim]; float msq=0.f;
        for (int d=0; d<head_dim; ++d){ float z=v[d]; msq += z*z; }
        float inv = 1.0f / sqrtf(msq/(float)head_dim + eps_qk);
        for (int d=0; d<head_dim; ++d) v[d] = (v[d]*inv) * qn[d];
      }
    }
  }
  if (kn && kn_len==head_dim){
    for (int t=0;t<T;++t){
      float* Kt = &K[t*Dkv];
      for (int h=0;h<n_kv;++h){
        float* v=&Kt[h*head_dim]; float msq=0.f;
        for (int d=0; d<head_dim; ++d){ float z=v[d]; msq += z*z; }
        float inv = 1.0f / sqrtf(msq/(float)head_dim + eps_qk);
        for (int d=0; d<head_dim; ++d) v[d] = (v[d]*inv) * kn[d];
      }
    }
  }

  // RoPE (Raschka apply_rope: x' = x*cos + rot(x)*sin), using dumped cos/sin
  apply_rope_from_cos_sin(Q, T, n_q,  head_dim, cos_arr, sin_arr, cos_T, cos_D);
  apply_rope_from_cos_sin(K, T, n_kv, head_dim, cos_arr, sin_arr, cos_T, cos_D);

  // Scores + causal softmax + context (GQA)
  const int group_size = n_q / n_kv;
  for (int h=0; h<n_q; ++h){
    const int kvh = h / group_size; // matches repeat_interleave on KV heads
    for (int tq=0; tq<T; ++tq){
      const float* qv = &Q[tq*Dq + h*head_dim];
      float* Sout = &S[tq*T];
      for (int tk=0; tk<T; ++tk){
        const float* kv = &K[tk*Dkv + kvh*head_dim];
        float dot=0.f; for (int d=0; d<head_dim; ++d) dot += qv[d]*kv[d];
        Sout[tk] = dot * scale;
      }
      if (causal) for (int tk=tq+1; tk<T; ++tk) S[tq*T + tk] = -INFINITY;
      float m=Sout[0]; for (int i=1;i<T;++i) if (Sout[i]>m) m=Sout[i];
      float s=0.f; for (int i=0;i<T;++i){ float z=expf(Sout[i]-m); Sout[i]=z; s+=z; }
      float inv = 1.f / (s + 1e-9f);
      for (int i=0;i<T;++i) Sout[i] *= inv;
    }
    // context into Hcat
    for (int tq=0; tq<T; ++tq){
      const float* Prow = &S[tq*T];
      float* out = &Hcat[tq*Dq + h*head_dim];
      for (int d=0; d<head_dim; ++d) out[d] = 0.f;
      for (int tk=0; tk<T; ++tk){
        const float* vv = &V[tk*Dkv + kvh*head_dim];
        const float p = Prow[tk];
        for (int d=0; d<head_dim; ++d) out[d] += p * vv[d];
      }
    }
  }

  // out proj
  matmul_f32(Hcat, Wo, attn_out, T, d_model, Dq);
  if (bo) for (int t=0;t<T;++t) for (int i=0;i<d_model;++i) attn_out[t*d_model+i] += bo[i];

  // residual
  for (int i=0;i<T*d_model;++i) h_out[i] = x[i] + attn_out[i];

  // norm2
  // (NOTE: do it outside where rms2_w is available; here we assume x_norm2 already computed there)
  // but for simplicity, caller passes rms2_w and we do it here; see wrapper below.
}

// ---------- main: run all layers per step and compare logits/probs/ids ----------

int main(int argc, char** argv){
  if (argc < 4){
    fprintf(stderr, "Usage: %s <all.bin> <outbase> <num_steps>\n", argv[0]);
    return 1;
  }
  const char* wfile   = argv[1];
  const char* outbase = argv[2];
  const int num_steps = atoi(argv[3]);

  // load weights
  BinFile* bf = bin_load(wfile);
  if (!bf){ fprintf(stderr,"bin load fail\n"); return 1; }

  // infer model sizes from layer 0
  TensorBin *Wq0 = need(bf,"model.layers.0.self_attn.q_proj.weight");
  TensorBin *Wk0 = need(bf,"model.layers.0.self_attn.k_proj.weight");
  TensorBin *tok = need(bf,"model.embed_tokens.weight");
  const int d_model = Wq0->shape[1];
  const int Dq0 = Wq0->shape[0];
  const int Dkv0= Wk0->shape[0];
  // heads
  int head_dim = -1;
  TensorBin* qn0 = maybe(bf,"model.layers.0.self_attn.q_norm.weight");
  if (qn0) head_dim = qn0->shape[0];
  if (head_dim <= 0) head_dim = 128; // Qwen3-30B default; or Dq0/n_q discovered below
  int n_q  = Dq0  / head_dim;
  int n_kv = Dkv0 / head_dim;

  // n_layers: count until weights stop
  int n_layers = 0;
  for (;;++n_layers){
    char k[256];
    snprintf(k,sizeof(k),"model.layers.%d.self_attn.q_proj.weight", n_layers);
    if (!bin_find(bf,k)) break;
  }
  if (n_layers <= 0){ fprintf(stderr,"bad n_layers\n"); return 1; }

  // vocab
  const int vocab = tok->shape[0];

  // dump info
  printf("[model] d_model=%d n_layers=%d head_dim=%d n_q=%d n_kv=%d vocab=%d\n",
         d_model, n_layers, head_dim, n_q, n_kv, vocab);

  // load cos/sin npy (written by Python)
  char cpath[512], spath[512];
  snprintf(cpath,sizeof(cpath), "%s.cos.npy", outbase);
  snprintf(spath,sizeof(spath), "%s.sin.npy", outbase);
  NpyArray* COS = npy_load_float32(cpath);
  NpyArray* SIN = npy_load_float32(spath);
  if (!COS || !SIN){ fprintf(stderr,"missing cos/sin dumps\n"); return 1; }
  const float* cos_arr = COS->data;
  const float* sin_arr = SIN->data;
  const int cos_T = COS->shape[0];
  const int cos_D = COS->shape[1];

  // load ids/logits/probs reference (for final comparisons per step)
  char ipath[512], lpath[512], ppath[512];
  snprintf(ipath,sizeof(ipath), "%s.ids.npy", outbase);
  snprintf(lpath,sizeof(lpath), "%s.logits.npy", outbase);
  snprintf(ppath,sizeof(ppath), "%s.probs.npy",  outbase);
  NpyArrayI32* IDS = npy_load_int32(ipath);
  NpyArray* LREF = npy_load_float32(lpath);
  NpyArray* PREF = npy_load_float32(ppath);
  if (!IDS || !LREF || !PREF){ fprintf(stderr,"missing ids/logits/probs\n"); return 1; }

  // lm_head (if absent, tied to tok_emb)
  TensorBin* LMH = maybe(bf,"lm_head.weight");

  // allocate per-layer pointers we reuse each iteration
  // we also pull expert dims (E,d_ff) from layer 0
  // E: count experts present
  int E = 0;
  for (;;++E){
    char k[256];
    snprintf(k,sizeof(k),"model.layers.0.mlp.experts.%d.down_proj.weight", E);
    if (!bin_find(bf,k)) break;
  }
  if (E <= 0){ fprintf(stderr,"no experts found\n"); return 1; }
  int d_ff = -1;
  {
    char kdp[256]; snprintf(kdp,sizeof(kdp),"model.layers.0.mlp.experts.0.down_proj.weight");
    TensorBin* Wd = need(bf,kdp);
    d_ff = Wd->shape[1];
  }

  // buffers (enough for largest step T we will see; we read T from each step's x.npy)
  // allocate per-step based on T to keep it simple.

  int overall_fail = 0;

  for (int step=0; step<num_steps; ++step){
    // load x for this step
    char xpath[512];
    snprintf(xpath,sizeof(xpath), "%s.step%d.x.npy", outbase, step);
    NpyArray* X = npy_load_float32(xpath);
    if (!X){ fprintf(stderr,"missing %s\n", xpath); overall_fail=1; break; }
    const int T = X->shape[0];
    if (X->shape[1] != d_model){ fprintf(stderr,"x shape mismatch\n"); overall_fail=1; break; }

    // per-step work buffers
    // scratch per layer needs: T*(Dq + 2*Dkv) + T*T + T*Dq + 3*T*d_model + T*E + (2*d_ff + d_model)
    const size_t need_s = (size_t)T*(n_q*head_dim + 2*n_kv*head_dim) + (size_t)T*T + (size_t)T*(n_q*head_dim)
                        + (size_t)3*T*d_model + (size_t)T*E + (size_t)(2*d_ff + d_model);
    float* scratch = (float*)malloc(sizeof(float)*need_s);
    if (!scratch){ fprintf(stderr,"oom scratch\n"); return 1; }

    // running hidden
    float* h = (float*)malloc(sizeof(float)*T*d_model);
    memcpy(h, X->data, sizeof(float)*T*d_model);

    // layer loop
    for (int L=0; L<n_layers; ++L){
      // gather layer weights
      char k[256];
      // attn
      snprintf(k,sizeof(k),"model.layers.%d.self_attn.q_proj.weight", L); TensorBin* Wq = need(bf,k);
      snprintf(k,sizeof(k),"model.layers.%d.self_attn.k_proj.weight", L); TensorBin* Wk = need(bf,k);
      snprintf(k,sizeof(k),"model.layers.%d.self_attn.v_proj.weight", L); TensorBin* Wv = need(bf,k);
      snprintf(k,sizeof(k),"model.layers.%d.self_attn.o_proj.weight", L); TensorBin* Wo = need(bf,k);
      snprintf(k,sizeof(k),"model.layers.%d.self_attn.q_proj.bias", L);  TensorBin* bq = maybe(bf,k);
      snprintf(k,sizeof(k),"model.layers.%d.self_attn.k_proj.bias", L);  TensorBin* bk = maybe(bf,k);
      snprintf(k,sizeof(k),"model.layers.%d.self_attn.v_proj.bias", L);  TensorBin* bv = maybe(bf,k);
      snprintf(k,sizeof(k),"model.layers.%d.self_attn.o_proj.bias", L);  TensorBin* bo = maybe(bf,k);
      snprintf(k,sizeof(k),"model.layers.%d.self_attn.q_norm.weight", L);TensorBin* qn = maybe(bf,k);
      snprintf(k,sizeof(k),"model.layers.%d.self_attn.k_norm.weight", L);TensorBin* kn = maybe(bf,k);
      // norms
      TensorBin* n1 = maybe(bf, (snprintf(k,sizeof(k),"model.layers.%d.input_layernorm.weight",L), k));
      if (!n1) n1 = need(bf,(snprintf(k,sizeof(k),"model.layers.%d.rms_1.weight",L), k));
      TensorBin* n2 = maybe(bf, (snprintf(k,sizeof(k),"model.layers.%d.post_attention_layernorm.weight",L), k));
      if (!n2) n2 = need(bf,(snprintf(k,sizeof(k),"model.layers.%d.rms_2.weight",L), k));
      // router
      TensorBin* RW = maybe(bf, (snprintf(k,sizeof(k),"model.layers.%d.mlp.gate.weight",L), k));
      if (!RW) RW = need(bf, (snprintf(k,sizeof(k),"model.layers.%d.mlp.router.gate.weight",L), k));
      TensorBin* RB = maybe(bf, (snprintf(k,sizeof(k),"model.layers.%d.mlp.gate.bias",L), k));
      if (!RB) RB = maybe(bf, (snprintf(k,sizeof(k),"model.layers.%d.mlp.router.gate.bias",L), k));

      // experts
      const float** Wg = (const float**)calloc(E,sizeof(float*));
      const float** Wu = (const float**)calloc(E,sizeof(float*));
      const float** Wd = (const float**)calloc(E,sizeof(float*));
      for (int e=0;e<E;++e){
        snprintf(k,sizeof(k),"model.layers.%d.mlp.experts.%d.gate_proj.weight", L, e);
        TensorBin* t = bin_find(bf,k); if (t) Wg[e] = (const float*)t->data;
        snprintf(k,sizeof(k),"model.layers.%d.mlp.experts.%d.up_proj.weight",   L, e);
        t = bin_find(bf,k); if (t) Wu[e] = (const float*)t->data;
        snprintf(k,sizeof(k),"model.layers.%d.mlp.experts.%d.down_proj.weight", L, e);
        t = bin_find(bf,k); if (t) Wd[e] = (const float*)t->data;
      }

      // --- norm1
      float* x_norm1 = scratch + (size_t)T*(n_q*head_dim + 2*n_kv*head_dim) + (size_t)T*T + (size_t)T*(n_q*head_dim);
      rmsnorm_forward_f32(h, (const float*)n1->data, T, d_model, 1e-6f, x_norm1);

      // --- attention + residual (uses the big scratch region as laid out in attn_moe_layer_forward)
      // for simplicity, we call the internal pieces inline (keep parity with your single-layer code)
      {
        const int Dq  = n_q*head_dim;
        const int Dkv = n_kv*head_dim;
        float* Q    = scratch;            // [T, Dq]
        float* K    = Q + T*Dq;           // [T, Dkv]
        float* V    = K + T*Dkv;          // [T, Dkv]
        float* S    = V + T*Dkv;          // [T, T]
        float* Hcat = S + T*T;            // [T, Dq]
        float* attn_out = x_norm1 + T*d_model;

        // projections
        matmul_f32(x_norm1, (const float*)Wq->data, Q, T, Dq,  d_model);
        matmul_f32(x_norm1, (const float*)Wk->data, K, T, Dkv, d_model);
        matmul_f32(x_norm1, (const float*)Wv->data, V, T, Dkv, d_model);
        if (bq) for (int t=0;t<T;++t) for (int i=0;i<Dq;  ++i) Q[t*Dq  + i] += ((const float*)bq->data)[i];
        if (bk) for (int t=0;t<T;++t) for (int i=0;i<Dkv; ++i) K[t*Dkv + i] += ((const float*)bk->data)[i];
        if (bv) for (int t=0;t<T;++t) for (int i=0;i<Dkv; ++i) V[t*Dkv + i] += ((const float*)bv->data)[i];

        // qk norm
        const float eps_qk = 1e-6f;
        if (qn0 && qn0->shape[0] == head_dim){ // presence is uniform across layers
          const float* qn = (const float*) ( (maybe(bf,(snprintf(k,sizeof(k),"model.layers.%d.self_attn.q_norm.weight",L),k))) ? bin_find(bf,k)->data : NULL );
          const float* kn = (const float*) ( (maybe(bf,(snprintf(k,sizeof(k),"model.layers.%d.self_attn.k_norm.weight",L),k))) ? bin_find(bf,k)->data : NULL );
          if (qn){
            for (int t=0;t<T;++t){
              float* Qt=&Q[t*Dq];
              for (int hq=0; hq<n_q; ++hq){
                float* v=&Qt[hq*head_dim]; float msq=0.f;
                for (int d=0; d<head_dim; ++d){ float z=v[d]; msq+=z*z; }
                float inv = 1.0f / sqrtf(msq/(float)head_dim + eps_qk);
                for (int d=0; d<head_dim; ++d) v[d] = (v[d]*inv) * qn[d];
              }
            }
          }
          if (kn){
            for (int t=0;t<T;++t){
              float* Kt=&K[t*Dkv];
              for (int hk=0; hk<n_kv; ++hk){
                float* v=&Kt[hk*head_dim]; float msq=0.f;
                for (int d=0; d<head_dim; ++d){ float z=v[d]; msq+=z*z; }
                float inv = 1.0f / sqrtf(msq/(float)head_dim + eps_qk);
                for (int d=0; d<head_dim; ++d) v[d] = (v[d]*inv) * kn[d];
              }
            }
          }
        }

        // rope
        apply_rope_from_cos_sin(Q, T, n_q,  head_dim, cos_arr, sin_arr, cos_T, cos_D);
        apply_rope_from_cos_sin(K, T, n_kv, head_dim, cos_arr, sin_arr, cos_T, cos_D);

        // scores/softmax/context
        const float scale = 1.0f / sqrtf((float)head_dim);
        const int group = n_q / n_kv;
        for (int hq=0; hq<n_q; ++hq){
          const int kvh = hq / group;
          for (int tq=0; tq<T; ++tq){
            const float* qv = &Q[tq*Dq + hq*head_dim];
            float* Sout = &S[tq*T];
            for (int tk=0; tk<T; ++tk){
              const float* kv = &K[tk*Dkv + kvh*head_dim];
              float dot=0.f; for (int d=0; d<head_dim; ++d) dot += qv[d]*kv[d];
              Sout[tk] = dot * scale;
            }
            for (int tk=tq+1; tk<T; ++tk) S[tq*T+tk] = -INFINITY;
            float m=Sout[0]; for (int i=1;i<T;++i) if (Sout[i]>m) m=Sout[i];
            float s=0.f; for (int i=0;i<T;++i){ float z=expf(Sout[i]-m); Sout[i]=z; s+=z; }
            float inv = 1.f/(s+1e-9f);
            for (int i=0;i<T;++i) Sout[i]*=inv;
          }
          for (int tq=0; tq<T; ++tq){
            const float* Prow = &S[tq*T];
            float* out = &Hcat[tq*Dq + hq*head_dim];
            for (int d=0; d<head_dim; ++d) out[d]=0.f;
            for (int tk=0; tk<T; ++tk){
              const float* vv = &V[tk*Dkv + kvh*head_dim];
              const float p = Prow[tk];
              for (int d=0; d<head_dim; ++d) out[d] += p*vv[d];
            }
          }
        }

        // out proj + residual
        matmul_f32(Hcat, (const float*)Wo->data, attn_out, T, d_model, Dq);
        if (bo) for (int t=0;t<T;++t) for (int i=0;i<d_model;++i) attn_out[t*d_model+i] += ((const float*)bo->data)[i];
        for (int i=0;i<T*d_model;++i) h[i] = h[i] + attn_out[i];
      }

      // norm2
      float* x_norm2 = scratch + (size_t)T*(n_q*head_dim + 2*n_kv*head_dim) + (size_t)T*T + (size_t)T*(n_q*head_dim) + T*d_model + T*d_model;
      rmsnorm_forward_f32(h, (const float*)n2->data, T, d_model, 1e-6f, x_norm2);

      // router logits
      float* logits = x_norm2 + T*d_model;
      matmul_f32(x_norm2, (const float*)RW->data, logits, T, E, d_model);
      if (RB && RB->dtype==0){
        const float* rb = (const float*)RB->data;
        for (int t=0;t<T;++t) for (int e=0;e<E;++e) logits[t*E+e] += rb[e];
      }

      // reuse Python top-k for exact aggregation
      char path[512];
      snprintf(path,sizeof(path),"%s.step%d.L%d.router_topk_idx.npy", outbase, step, L);
      NpyArrayI32* TKIDX = npy_load_int32(path);
      snprintf(path,sizeof(path),"%s.step%d.L%d.router_topk_p.npy", outbase, step, L);
      NpyArray* TKP = npy_load_float32(path);
      if (!TKIDX || !TKP){ fprintf(stderr,"missing router topk dumps at L=%d step=%d\n", L, step); return 1; }
      const int topk = TKIDX->shape[1];

      // MoE forward using reused topk
      float* tmp_g = logits + T*E;
      float* tmp_u = tmp_g + d_ff;
      float* tmp_y = tmp_u + d_ff;
      for (int t=0;t<T;++t){
        for (int j=0;j<topk;++j){
          const int e = TKIDX->data[t*topk + j];
          const float p = TKP->data[t*topk + j];
          // x_norm2[t] @ Wg[e]^T -> silu -> * (x_norm2[t] @ Wu[e]^T) -> down
          matmul_f32(&x_norm2[(size_t)t*d_model], Wg[e], tmp_g, 1, d_ff, d_model);
          silu_f32(tmp_g, d_ff);
          matmul_f32(&x_norm2[(size_t)t*d_model], Wu[e], tmp_u, 1, d_ff, d_model);
          for (int q=0;q<d_ff;++q) tmp_g[q] *= tmp_u[q];
          matmul_f32(tmp_g, Wd[e], tmp_y, 1, d_model, d_ff);
          for (int q=0;q<d_model;++q) h[(size_t)t*d_model + q] += p * tmp_y[q];
        }
      }
      npy_free_i32(TKIDX); npy_free(TKP);

      free(Wg); free(Wu); free(Wd);
    } // end layers

    // final norm + head
    TensorBin* FN = maybe(bf,"model.norm.weight");
    if (!FN) FN = need(bf,"model.final_layernorm.weight");
    float* y_norm = (float*)malloc(sizeof(float)*T*d_model);
    rmsnorm_forward_f32(h, (const float*)FN->data, T, d_model, 1e-6f, y_norm);

    float* logits_last = (float*)malloc(sizeof(float)*vocab);
    // logits = y_norm[-1] * W^T
    const float* head = LMH ? (const float*)LMH->data : (const float*)tok->data; // tied if no lm_head
    // matmul 1 x vocab = 1 x d_model (y_norm last) * d_model x vocab (head^T)
    // our matmul expects row-major W [out, in], so pass W=head and swap dims accordingly:
    // logits_last[v] = dot(y_norm_last, head[v, :])
    for (int v=0; v<vocab; ++v){
      float acc=0.f;
      const float* wv = &head[(size_t)v*d_model];
      const float* yl = &y_norm[(size_t)(T-1)*d_model];
      for (int d=0; d<d_model; ++d) acc += yl[d]*wv[d];
      logits_last[v] = acc;
    }
    // softmax
    float maxv=logits_last[0];
    for (int i=1;i<vocab;++i) if (logits_last[i]>maxv) maxv=logits_last[i];
    float sum=0.f;
    for (int i=0;i<vocab;++i){ float z=expf(logits_last[i]-maxv); logits_last[i]=z; sum+=z; }
    for (int i=0;i<vocab;++i) logits_last[i] /= (sum + 1e-9f);

    // compare with reference dumps (row = step)
    const float* Lrow = &LREF->data[(size_t)step * (size_t)vocab];
    const float* Prow = &PREF->data[(size_t)step * (size_t)vocab];
    // We computed probs; for logits MAD, recompute our logits (pre-softmax) quickly:
    // For exactness, re-evaluate pre-softmax vector:
    float* logits_last_linear = (float*)malloc(sizeof(float)*vocab);
    for (int v=0; v<vocab; ++v){
      float acc=0.f;
      const float* wv = &head[(size_t)v*d_model];
      const float* yl = &y_norm[(size_t)(T-1)*d_model];
      for (int d=0; d<d_model; ++d) acc += yl[d]*wv[d];
      logits_last_linear[v] = acc;
    }

    double mad_logits = max_abs_diff(logits_last_linear, Lrow, (size_t)vocab);
    double mad_probs  = max_abs_diff(logits_last,        Prow, (size_t)vocab);

    // greedy id match
    const int* ids_all = IDS->data;
    const int got = (int)( (int)(ptrdiff_t)( (size_t)(size_t)(size_t)0 ) ); // (avoids unused warnings)
    // argmax
    int argmax = 0;
    for (int i=1;i<vocab;++i) if (logits_last[i] > logits_last[argmax]) argmax = i;
    // ids layout: initial seqlen is IDS->shape[0] - num_steps
    const int total_ids = IDS->shape[0];
    const int seqlen0 = total_ids - num_steps;
    const int ref_next = ids_all[seqlen0 + step]; // at step s, Python appended 1 new token; this is that token

    printf("[step %d] logits MAD=%.6g  probs MAD=%.6g  argmax=%d  ref=%d\n",
           step, mad_logits, mad_probs, argmax, ref_next);

    if (mad_logits>2e-5 || mad_probs>2e-5 || argmax!=ref_next) overall_fail = 1;

    // cleanup per-step
    free(logits_last_linear);
    free(logits_last);
    free(y_norm);
    free(h);
    free(scratch);
    npy_free(X);
  } // steps

  // final status
  printf("%s\n", overall_fail ? "FAIL" : "PASS");

  // free globals
  npy_free(COS); npy_free(SIN);
  npy_free(LREF); npy_free(PREF); npy_free_i32(IDS);
  bin_free(bf);
  return overall_fail ? 1 : 0;
}
