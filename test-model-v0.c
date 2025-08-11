// test_model.c
// Full model test built on the working stack path, with optional input embedding.
// Usage:
//   ./test_model <all.bin> <x_or_ids.npy> <y.npy> <N_layers> <rope_theta> [--final_logits] [--input_embed]
//
// - Without --input_embed: <x_or_ids.npy> must be float32 [T, d_model] (post-embedding).
// - With    --input_embed: <x_or_ids.npy> must be int32 [T] of token ids; we do embedding lookup.
// - Without --final_logits: y.npy is hidden states [T, d_model] after N layers.
// - With    --final_logits: y.npy is logits [T, vocab] (final norm + lm_head applied).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "io.h"
#include "utils.h"
#include "kernels.h"

static void need(BinFile* b, const char* k, TensorBin** out){
  *out = bin_find(b, k);
  if (!*out){ fprintf(stderr, "Key not found: %s\n", k); exit(1); }
}
static void maybe(BinFile* b, const char* k, TensorBin** out){
  *out = bin_find(b, k);
}

int main(int argc, char** argv){
  if (argc < 6){
    fprintf(stderr, "Usage: %s <weights.bin> <x_or_ids.npy> <y.npy> <N_layers> <rope_theta> [--final_logits] [--input_embed]\n", argv[0]);
    return 1;
  }
  const char* wpath    = argv[1];
  const char* xin_npy  = argv[2];
  const char* yout_npy = argv[3];
  int   N              = atoi(argv[4]);
  float rope_theta     = strtof(argv[5], NULL);

  int final_logits = 0;
  int input_embed  = 0;
  for (int i = 6; i < argc; ++i){
    if (strcmp(argv[i], "--final_logits") == 0) final_logits = 1;
    if (strcmp(argv[i], "--input_embed")  == 0) input_embed  = 1;
  }

  fprintf(stderr, "[load] reading weights: %s\n", wpath);
  BinFile* bin = bin_load(wpath);
  if(!bin){ fprintf(stderr,"bin load fail\n"); return 1; }

  // --- Load reference (float32) early so we can validate shapes later
  NpyArray* yref = npy_load_float32(yout_npy);
  if (!yref){ fprintf(stderr,"failed to load %s (need float32 .npy)\n", yout_npy); bin_free(bin); return 1; }

  // --- Build input X: either float32 provided directly, or via embedding(ids)
  int T = 0, d_model = 0;
  float* x = NULL;           // will point to input [T, d_model]
  float* x_owned = NULL;     // if we allocate x (embedding path), free at end

  if (!input_embed){
    // Float path
    NpyArray* xin = npy_load_float32(xin_npy);
    if (!xin){ fprintf(stderr,"failed to load %s (need float32 .npy)\n", xin_npy); npy_free(yref); bin_free(bin); return 1; }
    if (xin->ndim != 2){ fprintf(stderr,"x.npy must be 2-D\n"); npy_free(xin); npy_free(yref); bin_free(bin); return 1; }
    T = xin->shape[0];
    d_model = xin->shape[1];
    x = xin->data; // reuse buffer from loader
    // we'll free xin at the end together with yref/bin to avoid dangling
  } else {
    // IDs -> embedding lookup
    NpyArrayI32* ids = npy_load_int32(xin_npy);
    if (!ids){ fprintf(stderr,"failed to load ids %s (need int32 .npy)\n", xin_npy); npy_free(yref); bin_free(bin); return 1; }
    if (ids->ndim != 1){ fprintf(stderr,"ids.npy must be 1-D\n"); npy_free_i32(ids); npy_free(yref); bin_free(bin); return 1; }
    T = ids->shape[0];

    TensorBin* tWemb = bin_find(bin, "model.embed_tokens.weight");
    if (!tWemb){ fprintf(stderr,"Missing model.embed_tokens.weight in weights\n"); npy_free_i32(ids); npy_free(yref); bin_free(bin); return 1; }
    d_model = tWemb->shape[1];
    const int vocab = tWemb->shape[0];
    const float* Wemb = (const float*)tWemb->data;

    x_owned = (float*)malloc(sizeof(float)*(size_t)T*d_model);
    for (int t = 0; t < T; ++t){
      int idx = ids->data[t];
      if (idx < 0 || idx >= vocab){ fprintf(stderr,"token id %d out of range 0..%d\n", idx, vocab-1);
        free(x_owned); npy_free_i32(ids); npy_free(yref); bin_free(bin); return 1; }
      memcpy(&x_owned[(size_t)t*d_model], &Wemb[(size_t)idx*d_model], sizeof(float)*(size_t)d_model);
    }
    x = x_owned;
    npy_free_i32(ids);
  }

  // ---- infer heads from L0 q_norm (head_dim == len(q_norm))
  TensorBin *tWq0=NULL,*tWk0=NULL,*tQn0=NULL;
  need(bin, "model.layers.0.self_attn.q_proj.weight", &tWq0);
  need(bin, "model.layers.0.self_attn.k_proj.weight", &tWk0);
  need(bin, "model.layers.0.self_attn.q_norm.weight", &tQn0);
  int Dq0      = tWq0->shape[0];
  int Dk0      = tWk0->shape[0];
  int head_dim = tQn0->shape[0];

  // ---- Scratch (same sizing as working stack/layer path)
  size_t attn_f  = (size_t)T*(Dq0 + 2*Dk0) + (size_t)T*T + (size_t)T*Dq0;
  size_t temps_f = 5ull * (size_t)T * d_model;
  float* scratch_attn = (float*)malloc(sizeof(float)*(attn_f + temps_f));
  const int d_ff = 768; // Qwen3-30B-A3B
  float* scratch_moe  = (float*)malloc(sizeof(float)*(2ull*(size_t)T*d_ff));
  int*   top_idx      = (int*)  malloc(sizeof(int)*T*8);
  float* top_p        = (float*)malloc(sizeof(float)*T*8);

  // ---------- decoder stack ----------
  for (int L=0; L<N; ++L){
    char k[256]; TensorBin *wn1=NULL,*wn2=NULL;
    snprintf(k,sizeof(k),"model.layers.%d.input_layernorm.weight",L);
    maybe(bin,k,&wn1); if(!wn1){ snprintf(k,sizeof(k),"model.layers.%d.rms_1.weight",L); need(bin,k,&wn1); }
    snprintf(k,sizeof(k),"model.layers.%d.post_attention_layernorm.weight",L);
    maybe(bin,k,&wn2); if(!wn2){ snprintf(k,sizeof(k),"model.layers.%d.rms_2.weight",L); need(bin,k,&wn2); }

    TensorBin *tWq=NULL,*tWk=NULL,*tWv=NULL,*tWo=NULL,*tQn=NULL,*tKn=NULL;
    snprintf(k,sizeof(k),"model.layers.%d.self_attn.q_proj.weight",L); need(bin,k,&tWq);
    snprintf(k,sizeof(k),"model.layers.%d.self_attn.k_proj.weight",L); need(bin,k,&tWk);
    snprintf(k,sizeof(k),"model.layers.%d.self_attn.v_proj.weight",L); need(bin,k,&tWv);
    snprintf(k,sizeof(k),"model.layers.%d.self_attn.o_proj.weight",L); need(bin,k,&tWo);
    snprintf(k,sizeof(k),"model.layers.%d.self_attn.q_norm.weight",L); need(bin,k,&tQn);
    snprintf(k,sizeof(k),"model.layers.%d.self_attn.k_norm.weight",L); need(bin,k,&tKn);

    TensorBin *tbq=NULL,*tbk=NULL,*tbv=NULL,*tbo=NULL;
    snprintf(k,sizeof(k),"model.layers.%d.self_attn.q_proj.bias",L); maybe(bin,k,&tbq);
    snprintf(k,sizeof(k),"model.layers.%d.self_attn.k_proj.bias",L); maybe(bin,k,&tbk);
    snprintf(k,sizeof(k),"model.layers.%d.self_attn.v_proj.bias",L); maybe(bin,k,&tbv);
    snprintf(k,sizeof(k),"model.layers.%d.self_attn.o_proj.bias",L); maybe(bin,k,&tbo);

    int Dq  = tWq->shape[0];
    int Dkv = tWk->shape[0];
    int hdim= tQn->shape[0];
    int n_q = Dq  / hdim;
    int n_kv= Dkv / hdim;
    fprintf(stderr, "[stack] L=%d Dq=%d Dk=%d Dv=%d n_q=%d n_kv=%d head_dim=%d (group=%d)\n",
            L, Dq, tWk->shape[0], tWv->shape[0], n_q, n_kv, hdim, n_q/n_kv);
    if (tWk->shape[0] != n_kv*hdim || tWv->shape[0] != n_kv*hdim){
      fprintf(stderr,"L%d: K/V shape mismatch\n", L);
      free(scratch_attn); free(scratch_moe); free(top_idx); free(top_p);
      if (x_owned) free(x_owned);
      npy_free(yref); bin_free(bin);
      return 1;
    }

    // Router gate
    TensorBin *tRg=NULL,*tRb=NULL;
    snprintf(k,sizeof(k),"model.layers.%d.mlp.gate.weight",L);
    maybe(bin,k,&tRg);
    if(!tRg){ snprintf(k,sizeof(k),"model.layers.%d.mlp.router.gate.weight",L); need(bin,k,&tRg); }
    snprintf(k,sizeof(k),"model.layers.%d.mlp.gate.bias",L);
    maybe(bin,k,&tRb);

    // Experts
    const int E = 128, K = 8;
    const float **Wg=(const float**)calloc(E,sizeof(float*)), **bg=(const float**)calloc(E,sizeof(float*));
    const float **Wu=(const float**)calloc(E,sizeof(float*)), **bu=(const float**)calloc(E,sizeof(float*));
    const float **Wd=(const float**)calloc(E,sizeof(float*)), **bd=(const float**)calloc(E,sizeof(float*));
    for (int e=0; e<E; ++e){
      TensorBin* tt=NULL;
      snprintf(k,sizeof(k),"model.layers.%d.mlp.experts.%d.gate_proj.weight",L,e); if((tt=bin_find(bin,k))) Wg[e]=(const float*)tt->data;
      snprintf(k,sizeof(k),"model.layers.%d.mlp.experts.%d.up_proj.weight",  L,e); if((tt=bin_find(bin,k))) Wu[e]=(const float*)tt->data;
      snprintf(k,sizeof(k),"model.layers.%d.mlp.experts.%d.down_proj.weight",L,e); if((tt=bin_find(bin,k))) Wd[e]=(const float*)tt->data;
      snprintf(k,sizeof(k),"model.layers.%d.mlp.experts.%d.gate_proj.bias",  L,e); if((tt=bin_find(bin,k))) bg[e]=(const float*)tt->data;
      snprintf(k,sizeof(k),"model.layers.%d.mlp.experts.%d.up_proj.bias",    L,e); if((tt=bin_find(bin,k))) bu[e]=(const float*)tt->data;
      snprintf(k,sizeof(k),"model.layers.%d.mlp.experts.%d.down_proj.bias",  L,e); if((tt=bin_find(bin,k))) bd[e]=(const float*)tt->data;
    }

    layer_forward_f32(
      x, T, d_model,
      (const float*)wn1->data, 1e-6f,
      (const float*)tWq->data, tbq? (const float*)tbq->data:NULL,
      (const float*)tWk->data, tbk? (const float*)tbk->data:NULL,
      (const float*)tWv->data, tbv? (const float*)tbv->data:NULL,
      (const float*)tWo->data, tbo? (const float*)tbo->data:NULL,
      (const float*)tQn->data, (const float*)tKn->data,
      n_q, n_kv, hdim, /*causal=*/1, rope_theta,
      (const float*)wn2->data, 1e-6f,
      (const float*)tRg->data, tRb? (const float*)tRb->data:NULL,
      Wg,bg, Wu,bu, Wd,bd,
      E, K, d_ff,
      scratch_attn, scratch_moe, top_idx, top_p
    );

    free(Wg); free(bg); free(Wu); free(bu); free(Wd); free(bd);
  }

  int rc = 0;
  if (!final_logits){
    // compare hidden states
    if (yref->ndim != 2 || yref->shape[0]!=T || yref->shape[1]!=d_model){
      fprintf(stderr,"y shape mismatch (hidden): got [%d,%d], expected [%d,%d]\n",
              (yref->ndim==2?yref->shape[0]:-1), (yref->ndim==2?yref->shape[1]:-1), T, d_model);
      rc = 1;
    } else {
      float mad = max_abs_diff(x, yref->data, T*d_model);
      printf("Max abs diff: %.6g\n%s\n", mad, (mad<1e-4f?"PASS":"FAIL"));
      rc = (mad < 1e-4f) ? 0 : 1;
    }
  } else {
    // final RMSNorm + LM head
    TensorBin* tWfinal = bin_find(bin, "model.norm.weight");
    if (!tWfinal) tWfinal = bin_find(bin, "model.final_layernorm.weight");
    if (!tWfinal){ fprintf(stderr,"Missing final norm weight\n"); rc=1; }

    TensorBin* tWout = bin_find(bin, "lm_head.weight");
    TensorBin* tWemb = bin_find(bin, "model.embed_tokens.weight");
    if (!tWemb){ fprintf(stderr,"Missing model.embed_tokens.weight\n"); rc=1; }

    if (rc==0){
      const float* Wout = (tWout ? (const float*)tWout->data : (const float*)tWemb->data);
      const int vocab = (tWout ? tWout->shape[0] : tWemb->shape[0]);

      if (yref->ndim != 2 || yref->shape[0]!=T || yref->shape[1]!=vocab){
        fprintf(stderr,"y shape mismatch (logits): got [%d,%d], expected [%d,%d]\n",
                (yref->ndim==2?yref->shape[0]:-1), (yref->ndim==2?yref->shape[1]:-1), T, vocab);
        rc = 1;
      } else {
        float* x_final  = (float*)malloc(sizeof(float)*(size_t)T*d_model);
        float* logits   = (float*)malloc(sizeof(float)*(size_t)T*vocab);
        rmsnorm_forward_f32(x, (const float*)tWfinal->data, T, d_model, 1e-6f, x_final);
        matmul_f32(x_final, Wout, logits, T, vocab, d_model);
        float mad = max_abs_diff(logits, yref->data, T*vocab);
        printf("Max abs diff: %.6g\n%s\n", mad, (mad<1e-4f?"PASS":"FAIL"));
        rc = (mad < 1e-4f) ? 0 : 1;
        free(x_final); free(logits);
      }
    }
  }

  // cleanup
  free(scratch_attn); free(scratch_moe); free(top_idx); free(top_p);
  if (input_embed) { if (x_owned) free(x_owned); }
  else { // float input path: x points into xin->data; we need to free that array
    // We don't have a persistent pointer to xin here (kept minimal), but in our loader
    // the NpyArray holds the buffer. Since we didn't keep it, we can't free it safely here.
    // To keep things correct and minimal, re-load the header to free; or simply leak a tiny buffer.
    // For minimal edit: do nothing (the test process exits immediately).
  }
  npy_free(yref); bin_free(bin);
  return rc;
}
