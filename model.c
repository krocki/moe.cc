// model.c
#include "model.h"
#include "kernels.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <time.h>

static inline void softmax_rows(float *mat, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        float *row = &mat[r * cols];
        // find max
        float maxv = row[0];
        for (int c = 1; c < cols; ++c)
            if (row[c] > maxv) maxv = row[c];
        // exponentiate & sum
        float sum = 0.0f;
        for (int c = 0; c < cols; ++c) {
            row[c] = expf(row[c] - maxv);
            sum += row[c];
        }
        // normalize
        float inv = 1.0f / (sum + 1e-9f);
        for (int c = 0; c < cols; ++c)
            row[c] *= inv;
    }
}

// helper: per-head RMSNorm used for Q/K "qk-norm"
static inline void rmsnorm_head(float* v, const float* w, int d, float eps){
  float msq=0.f; for (int i=0;i<d;++i){ float z=v[i]; msq+=z*z; }
  const float inv = 1.0f / sqrtf(msq/(float)d + eps);
  for (int i=0;i<d;++i) v[i] = (v[i]*inv) * w[i];
}

// one transformer layer forward (attn + MoE), in-place on x
void layer_forward_f32(
  float* x, int T, int D,
  const float* rms1_w, float rms1_eps,
  const float* Wq, const float* bq,
  const float* Wk, const float* bk,
  const float* Wv, const float* bv,
  const float* Wo, const float* bo,
  const float* qn, const float* kn,    // qk-norm (len=head_dim) or NULL
  int n_q, int n_kv, int head_dim, int causal, float rope_theta,
  const float* rms2_w, float rms2_eps,
  const float* router_w, const float* router_b,
  const float** Wg, const float** bg,
  const float** Wu, const float** bu,
  const float** Wd, const float** bd,
  int n_experts, int topk, int d_ff,
  // scratch
  float* scratch_attn,    // >= T*(n_q*dh + 2*n_kv*dh) + T*T + T*(n_q*dh) + 2*T*D
  float* scratch_moe,     // >= 2*T*d_ff
  int*   reuse_topk_idx,  // optional: [T, topk] (NULL -> compute topk here)
  float* reuse_topk_p     // optional: [T, topk]
){
  const int dh = head_dim;
  const int Dq = n_q*dh, Dkv = n_kv*dh;
  const float scale = 1.0f / sqrtf((float)dh);
  float* Q    = scratch_attn;                 // [T, Dq]
  float* K    = Q + T*Dq;                     // [T, Dkv]
  float* V    = K + T*Dkv;                    // [T, Dkv]
  float* S    = V + T*Dkv;                    // [T, T]
  float* Hcat = S + T*T;                      // [T, Dq]
  float* x_norm1 = Hcat + T*Dq;               // [T, D]
  float* attn_out= x_norm1 + T*D;             // [T, D]

  // norm1(x)
  rmsnorm_forward_f32(x, rms1_w, T, D, rms1_eps, x_norm1);

  // Q,K,V projections (+ optional bias)
  matmul_f32(x_norm1, Wq, Q, T, Dq,  D);
  matmul_f32(x_norm1, Wk, K, T, Dkv, D);
  matmul_f32(x_norm1, Wv, V, T, Dkv, D);
  if (bq) for (int t=0;t<T;++t) for (int i=0;i<Dq; ++i)  Q[t*Dq + i]  += bq[i];
  if (bk) for (int t=0;t<T;++t) for (int i=0;i<Dkv;++i)  K[t*Dkv+ i] += bk[i];
  if (bv) for (int t=0;t<T;++t) for (int i=0;i<Dkv;++i)  V[t*Dkv+ i] += bv[i];

  // qk-norm (per head)
  if (qn && kn){
    for (int t=0;t<T;++t){
      float* Qt=&Q[t*Dq];
      for (int h=0; h<n_q;  ++h) rmsnorm_head(&Qt[h*dh], qn, dh, 1e-6f);
      float* Kt=&K[t*Dkv];
      for (int h=0; h<n_kv; ++h) rmsnorm_head(&Kt[h*dh], kn, dh, 1e-6f);
    }
  }

  // RoPE (your drop-in that matches Raschka math)
  rope_apply_inplace_f32_gqa(Q, K, T, n_q, n_kv, dh, /*pos0=*/0, rope_theta);

  // scores + softmax + context (GQA head mapping)
  const int group = n_q / n_kv;
  for (int h=0; h<n_q; ++h){
    const int kvh = h / group;
    for (int tq=0; tq<T; ++tq){
      const float* qv = &Q[tq*Dq + h*dh];
      float* Sout = &S[tq*T];
      for (int tk=0; tk<T; ++tk){
        const float* kv = &K[tk*Dkv + kvh*dh];
        float dot=0.f; for (int d=0; d<dh; ++d) dot += qv[d]*kv[d];
        Sout[tk] = dot * scale;
      }
      if (causal) for (int tk=tq+1; tk<T; ++tk) S[tq*T + tk] = -INFINITY;
      float m=Sout[0]; for (int i=1;i<T;++i) if (Sout[i]>m) m=Sout[i];
      float s=0.f; for (int i=0;i<T;++i){ float z=expf(Sout[i]-m); Sout[i]=z; s+=z; }
      float inv = 1.f/(s + 1e-9f);
      for (int i=0;i<T;++i) Sout[i] *= inv;
    }
    for (int tq=0; tq<T; ++tq){
      const float* Prow = &S[tq*T];
      float* out = &Hcat[tq*Dq + h*dh];
      for (int d=0; d<dh; ++d) out[d]=0.f;
      for (int tk=0; tk<T; ++tk){
        const float* vv = &V[tk*Dkv + kvh*dh];
        const float p = Prow[tk];
        for (int d=0; d<dh; ++d) out[d] += p*vv[d];
      }
    }
  }

  // out proj + residual
  matmul_f32(Hcat, Wo, attn_out, T, D, Dq);
  if (bo) for (int t=0;t<T;++t) for (int i=0;i<D;++i) attn_out[t*D+i] += bo[i];
  for (int i=0;i<T*D;++i) x[i] = x[i] + attn_out[i];

  // norm2
  float* x_norm2 = attn_out + T*D;
  rmsnorm_forward_f32(x, rms2_w, T, D, rms2_eps, x_norm2);

  // router logits
  float* logits = x_norm2 + T*D;
  matmul_f32(x_norm2, router_w, logits, T, n_experts, D);
  if (router_b){ for (int t=0;t<T;++t) for (int e=0;e<n_experts;++e) logits[t*n_experts+e] += router_b[e]; }

  // top-k (per token)
  int*   top_idx = reuse_topk_idx;
  float* top_p   = reuse_topk_p;
  int owns = 0;
  if (!top_idx || !top_p){
    top_idx = (int*)malloc(sizeof(int)*T*topk);
    top_p   = (float*)malloc(sizeof(float)*T*topk);
    owns = 1;
    for (int t=0;t<T;++t){
      // find topk by partial selection (simple O(E*topk); fine for verifying)
      for (int j=0;j<topk;++j){
        int arg=-1; float best=-INFINITY;
        for (int e=0;e<n_experts;++e){
          float v = logits[t*n_experts+e];
          // ensure we don't pick same index again
          int used=0; for (int q=0;q<j;++q){ if (top_idx[t*topk+q]==e){ used=1; break; } }
          if(!used && v>best){ best=v; arg=e; }
        }
        top_idx[t*topk+j] = arg;
      }
      // softmax over the selected scores
      float m = -INFINITY;
      for (int j=0;j<topk;++j){ float v = logits[t*n_experts + top_idx[t*topk+j]]; if (v>m) m=v; }
      float s=0.f;
      for (int j=0;j<topk;++j){ float z = expf(logits[t*n_experts + top_idx[t*topk+j]] - m); top_p[t*topk+j]=z; s+=z; }
      float inv = 1.f/(s+1e-9f);
      for (int j=0;j<topk;++j) top_p[t*topk+j] *= inv;
    }
  }

  // MoE: weighted sum of expert outputs
  float* tmp_g = scratch_moe;       // [T*d_ff] chunked by token
  float* tmp_u = tmp_g + T*d_ff;    // [T*d_ff]
  // precompute gate/up for all experts? For parity & simplicity, do per token/expert:
  for (int t=0;t<T;++t){
    for (int j=0;j<topk;++j){
      int e = top_idx[t*topk+j];
      float p = top_p[t*topk+j];
      // x_norm2[t] @ Wg[e]^T -> silu -> * (x_norm2[t] @ Wu[e]^T) -> down
      float* g = tmp_g + t*d_ff;
      float* u = tmp_u + t*d_ff;
      matmul_f32(&x_norm2[t*D], Wg[e], g, 1, d_ff, D);
      silu_f32(g, d_ff);
      matmul_f32(&x_norm2[t*D], Wu[e], u, 1, d_ff, D);
      for (int q=0;q<d_ff;++q) g[q] *= u[q];
      float* y = (float*)alloca(sizeof(float)*D);
      matmul_f32(g, Wd[e], y, 1, D, d_ff);
      for (int q=0;q<D;++q) x[t*D + q] += p * y[q];
    }
  }

  if (owns){ free(top_idx); free(top_p); }
}

void model_forward_f32(
  const int* ids, int T,
  const QwenConfig* cfg,
  const QwenWeights* w,
  int apply_softmax,
  float* out_logits // [T, vocab]
){
  const int D    = cfg->d_model;
  const int V    = cfg->vocab_size;
  const int L    = cfg->n_layers;
  const int d_ff = cfg->d_ff;
  const int n_q  = cfg->n_q;
  const int n_kv = cfg->n_kv;
  const int dh   = cfg->head_dim;

  const int Dq0  = n_q  * dh;
  const int Dkv0 = n_kv * dh;

  size_t attn_f  = (size_t)T*(Dq0 + 2*Dkv0) + (size_t)T*T + (size_t)T*Dq0 + 2ull*(size_t)T*D;
  size_t moe_f   = 2ull * (size_t)T * d_ff;

  float* scratch_attn = (float*)malloc(sizeof(float)*(attn_f));
  float* scratch_moe  = (float*)malloc(sizeof(float)*(moe_f));

  // hidden buffer: embeddings
  float* x = (float*)malloc(sizeof(float)*(size_t)T*D);
  for (int t=0; t<T; ++t){
    int id = ids[t];
    memcpy(&x[(size_t)t*D], &w->tok_emb[(size_t)id*D], sizeof(float)*(size_t)D);
  }

  // decoder stack
  for (int l=0; l<L; ++l){
    const QwenLayerWeights* lw = &w->layers[l];
    layer_forward_f32(
      x, T, D,
      lw->rms1_w, 1e-6f,
      lw->Wq, lw->bq, lw->Wk, lw->bk, lw->Wv, lw->bv, lw->Wo, lw->bo,
      lw->q_norm, lw->k_norm,
      n_q, n_kv, dh, cfg->causal, cfg->rope_theta,
      lw->rms2_w, 1e-6f,
      lw->router_w, lw->router_b,
      lw->Wg, lw->bg, lw->Wu, lw->bu, lw->Wd, lw->bd,
      cfg->n_experts, cfg->top_k, d_ff,
      scratch_attn, scratch_moe,
      /*reuse topk*/ NULL, NULL
    );
  }

  // final norm + head
  float* x_final = (float*)malloc(sizeof(float)*(size_t)T*D);
  rmsnorm_forward_f32(x, w->final_norm_w, T, D, 1e-6f, x_final);

  const float* Wout = w->lm_head ? w->lm_head : w->tok_emb; // tied if lm_head==NULL
  matmul_f32(x_final, Wout, out_logits, T, V, D);
  if (apply_softmax) softmax_rows(out_logits, T, V);

  free(x_final);
  free(x);
  free(scratch_attn);
  free(scratch_moe);
}
