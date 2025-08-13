// test_model.c
// Greedy generation comparator (no KV cache; step-by-step) + rich debug mode
//
// Usage:
//   ./test_model <weights.bin> <ids.npy> <logits_or_probs.npy> <N_layers> <rope_theta> [--nosoftmax] [--qk-norm=on|off|auto] [--debug-prefix <outbase>]
//
// Notes:
// - <ids.npy> must contain prompt_len + steps tokens (int32), as produced by dump_model_io.py
// - <logits_or_probs.npy> is float32 [steps, vocab], either logits or probs depending
//   on whether you pass --nosoftmax (if you pass --nosoftmax here, file should be logits; otherwise probs).
// - If --debug-prefix is provided, the program will for each step s:
//     * load {prefix}.step{s}.x.npy   (T x D input embeddings for that step)
//     * load {prefix}.step{s}.L<i>.y.npy for i=0..N-1 (T x D output after each layer i)
//   and it will recompute the stack in C and compare per-layer outputs to these
//   numpy files, printing the first mismatch.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "io.h"
#include "utils.h"
#include "model.h"
#include "kernels.h"

static void need(BinFile* b, const char* k, TensorBin** out){
  *out = bin_find(b, k);
  if (!*out){ fprintf(stderr, "Key not found: %s\n", k); exit(1); }
}
static void maybe(BinFile* b, const char* k, TensorBin** out){ *out = bin_find(b, k); }

typedef enum { QKN_AUTO=0, QKN_ON=1, QKN_OFF=2 } QknMode;

static QknMode parse_qkn_mode(const char* s){
  if(!s) return QKN_AUTO;
  if(strcmp(s,"on")==0)  return QKN_ON;
  if(strcmp(s,"off")==0) return QKN_OFF;
  if(strcmp(s,"auto")==0) return QKN_AUTO;
  fprintf(stderr,"--qk-norm must be on|off|auto\n"); exit(1);
}

int main(int argc, char** argv){
  if (argc < 6){
    fprintf(stderr, "Usage: %s <weights.bin> <ids.npy> <logits_or_probs.npy> <N_layers> <rope_theta> [--nosoftmax] [--qk-norm=on|off|auto] [--debug-prefix <outbase>]\n", argv[0]);
    return 1;
  }
  const char* wpath = argv[1];
  const char* ids_npy = argv[2];
  const char* ref_npy = argv[3];
  int N               = atoi(argv[4]);
  float rope_theta    = strtof(argv[5], NULL);

  int apply_softmax = 1;
  QknMode qkn_mode = QKN_AUTO;
  const char* debug_prefix = NULL;

  for (int i=6;i<argc;++i){
    if (strcmp(argv[i],"--nosoftmax")==0) apply_softmax = 0;
    else if (strncmp(argv[i],"--qk-norm=",10)==0) qkn_mode = parse_qkn_mode(argv[i]+10);
    else if (strcmp(argv[i],"--debug-prefix")==0 && i+1<argc) { debug_prefix = argv[++i]; }
  }

  fprintf(stderr, "[load] %s\n", wpath);
  BinFile* bin = bin_load(wpath);
  if(!bin){ fprintf(stderr,"bin load fail\n"); return 1; }

  // ---- infer config from shapes
  TensorBin *tWq0,*tWk0,*tQn0,*tEmb,*tFinal,*tHead=NULL;
  need(bin, "model.layers.0.self_attn.q_proj.weight", &tWq0);
  need(bin, "model.layers.0.self_attn.k_proj.weight", &tWk0);
  // q_norm might be missing for some models; we only *need* it to infer head_dim if present
  maybe(bin, "model.layers.0.self_attn.q_norm.weight", &tQn0);
  need(bin, "model.embed_tokens.weight", &tEmb);
  maybe(bin, "model.norm.weight", &tFinal);
  if (!tFinal) need(bin, "model.final_layernorm.weight", &tFinal);
  maybe(bin, "lm_head.weight", &tHead);

  QwenConfig cfg = {0};
  cfg.d_model   = tEmb->shape[1];
  cfg.n_layers  = N;
  // Head dim: prefer q_norm length if present, else gcd fallback via shapes
  if (tQn0) cfg.head_dim = tQn0->shape[0];
  else {
    // fallback small helper
    int Dq = tWq0->shape[0], Dk = tWk0->shape[0];
    int a=Dq, b=Dk; while (b){ int r=a%b; a=b; b=r; }
    cfg.head_dim = a ? a : 128;
  }
  cfg.n_q       = tWq0->shape[0] / cfg.head_dim;
  cfg.n_kv      = tWk0->shape[0] / cfg.head_dim;
  cfg.d_ff      = 768;
  cfg.n_experts = 128;
  cfg.top_k     = 8;
  cfg.vocab_size= tEmb->shape[0];
  cfg.causal    = 1;
  cfg.rope_theta= rope_theta;

  // ---- pack weights
  QwenWeights W = {0};
  W.tok_emb      = (const float*)tEmb->data;
  W.final_norm_w = (const float*)tFinal->data;
  W.lm_head      = tHead ? (const float*)tHead->data : NULL;
  W.layers       = (QwenLayerWeights*)calloc(N, sizeof(QwenLayerWeights));

  char key[256];
  for (int L=0; L<N; ++L){
    QwenLayerWeights* lw = &W.layers[L];
    TensorBin* t;
    snprintf(key,sizeof(key),"model.layers.%d.input_layernorm.weight",L);
    maybe(bin,key,&t); if(!t){ snprintf(key,sizeof(key),"model.layers.%d.rms_1.weight",L); need(bin,key,&t); }
    lw->rms1_w = (const float*)t->data;
    snprintf(key,sizeof(key),"model.layers.%d.post_attention_layernorm.weight",L);
    maybe(bin,key,&t); if(!t){ snprintf(key,sizeof(key),"model.layers.%d.rms_2.weight",L); need(bin,key,&t); }
    lw->rms2_w = (const float*)t->data;

    snprintf(key,sizeof(key),"model.layers.%d.self_attn.q_proj.weight",L); need(bin,key,&t); lw->Wq=(const float*)t->data;
    snprintf(key,sizeof(key),"model.layers.%d.self_attn.k_proj.weight",L); need(bin,key,&t); lw->Wk=(const float*)t->data;
    snprintf(key,sizeof(key),"model.layers.%d.self_attn.v_proj.weight",L); need(bin,key,&t); lw->Wv=(const float*)t->data;
    snprintf(key,sizeof(key),"model.layers.%d.self_attn.o_proj.weight",L); need(bin,key,&t); lw->Wo=(const float*)t->data;
    snprintf(key,sizeof(key),"model.layers.%d.self_attn.q_proj.bias",L); maybe(bin,key,&t); lw->bq=t?(const float*)t->data:NULL;
    snprintf(key,sizeof(key),"model.layers.%d.self_attn.k_proj.bias",L); maybe(bin,key,&t); lw->bk=t?(const float*)t->data:NULL;
    snprintf(key,sizeof(key),"model.layers.%d.self_attn.v_proj.bias",L); maybe(bin,key,&t); lw->bv=t?(const float*)t->data:NULL;
    snprintf(key,sizeof(key),"model.layers.%d.self_attn.o_proj.bias",L); maybe(bin,key,&t); lw->bo=t?(const float*)t->data:NULL;

    // q/k norm may or may not exist; we'll honor --qk-norm flag below in debug stack

        if (qkn_mode == QKN_OFF){ printf("q_off\n"); lw->q_norm = NULL; lw->k_norm = NULL; } else {
    snprintf(key,sizeof(key),"model.layers.%d.self_attn.q_norm.weight",L); maybe(bin,key,&t); lw->q_norm=t?(const float*)t->data:NULL;
    snprintf(key,sizeof(key),"model.layers.%d.self_attn.k_norm.weight",L); maybe(bin,key,&t); lw->k_norm=t?(const float*)t->data:NULL;
}
    snprintf(key,sizeof(key),"model.layers.%d.mlp.gate.weight",L);
    maybe(bin,key,&t);
    if(!t){ snprintf(key,sizeof(key),"model.layers.%d.mlp.router.gate.weight",L); need(bin,key,&t); }
    lw->router_w=(const float*)t->data;
    snprintf(key,sizeof(key),"model.layers.%d.mlp.gate.bias",L); maybe(bin,key,&t); lw->router_b=t?(const float*)t->data:NULL;
    if (lw->router_b == NULL) printf("no bias\n");
    lw->Wg=(const float**)calloc(cfg.n_experts,sizeof(float*));
    lw->bg=(const float**)calloc(cfg.n_experts,sizeof(float*));
    lw->Wu=(const float**)calloc(cfg.n_experts,sizeof(float*));
    lw->bu=(const float**)calloc(cfg.n_experts,sizeof(float*));
    lw->Wd=(const float**)calloc(cfg.n_experts,sizeof(float*));
    lw->bd=(const float**)calloc(cfg.n_experts,sizeof(float*));
    for (int e=0; e<cfg.n_experts; ++e){
      TensorBin* q;
      snprintf(key,sizeof(key),"model.layers.%d.mlp.experts.%d.gate_proj.weight",L,e); if((q=bin_find(bin,key))) lw->Wg[e]=(const float*)q->data;
      snprintf(key,sizeof(key),"model.layers.%d.mlp.experts.%d.up_proj.weight",  L,e); if((q=bin_find(bin,key))) lw->Wu[e]=(const float*)q->data;
      snprintf(key,sizeof(key),"model.layers.%d.mlp.experts.%d.down_proj.weight",L,e); if((q=bin_find(bin,key))) lw->Wd[e]=(const float*)q->data;
      snprintf(key,sizeof(key),"model.layers.%d.mlp.experts.%d.gate_proj.bias",  L,e); if((q=bin_find(bin,key))) lw->bg[e]=(const float*)q->data;
      snprintf(key,sizeof(key),"model.layers.%d.mlp.experts.%d.up_proj.bias",    L,e); if((q=bin_find(bin,key))) lw->bu[e]=(const float*)q->data;
      snprintf(key,sizeof(key),"model.layers.%d.mlp.experts.%d.down_proj.bias",  L,e); if((q=bin_find(bin,key))) lw->bd[e]=(const float*)q->data;
    }
  }

  // ---- load I/O (prompt+generated ids; and per-step distributions)
  NpyArrayI32* ids_all = npy_load_int32(ids_npy);
  if (!ids_all || ids_all->ndim!=1){ fprintf(stderr,"ids.npy must be int32 1-D\n"); return 1; }
  int total_T = ids_all->shape[0];
  NpyArray* yref = npy_load_float32(ref_npy);
  if (!yref || yref->ndim!=2){ fprintf(stderr,"ref.npy must be float32 2-D [steps,vocab]\n"); return 1; }
  int steps = yref->shape[0];
  if (yref->shape[1] != cfg.vocab_size){
    fprintf(stderr,"ref vocab mismatch: got %d expected %d\n", yref->shape[1], cfg.vocab_size);
    return 1;
  }
  int prompt_len = total_T - steps;  // ids contains prompt + generated
  if (prompt_len <= 0){ fprintf(stderr,"ids must contain prompt+generated\n"); return 1; }

  // ---- generation loop (no cache): compare last row each step
  float* full = (float*)malloc(sizeof(float)*(size_t)total_T*cfg.vocab_size); // worst-case
  int*   cur_ids = (int*)malloc(sizeof(int)*(size_t)total_T);
  for (int i=0;i<prompt_len;i++) cur_ids[i] = ids_all->data[i];

  int rc = 0;

  // Debug helpers (stack recompute)
  // For recomputing per-layer outputs against Python dumps we reuse layer_forward_f32.
  for (int s=0; s<steps; ++s){
    int T = prompt_len + s;

    // (A) Normal end-to-end check via model_forward_f32
    model_forward_f32(cur_ids, T, &cfg, &W, /*softmax?*/(apply_softmax?1:0), full);
    float* last = &full[(size_t)(T-1)*cfg.vocab_size];
    float mad = max_abs_diff(last, &yref->data[(size_t)s*cfg.vocab_size], cfg.vocab_size);
    printf("step %d: max abs diff (final dist) = %.6g\n", s, mad);
    if (mad > 2e-3f) rc = 1;

    // (B) Optional deep debug: recompute per-layer and compare to Python dumps
    if (debug_prefix){
      // 1) Load x for this step
      char path[512];
      snprintf(path,sizeof(path),"%s.step%d.x.npy", debug_prefix, s);
      NpyArray* xnp = npy_load_float32(path);
      if (!xnp){ fprintf(stderr,"[debug] missing %s\n", path); return 1; }
      if (xnp->ndim!=2 || xnp->shape[0]!=T || xnp->shape[1]!=cfg.d_model){
        fprintf(stderr,"[debug] x shape mismatch at step %d\n", s); return 1;
      }
      float* x = (float*)malloc(sizeof(float)*(size_t)T*cfg.d_model);
      memcpy(x, xnp->data, sizeof(float)*(size_t)T*cfg.d_model);

      // Scratch sizing modeled on test_stack
      int Dq0  = cfg.n_q * cfg.head_dim;
      int Dkv0 = cfg.n_kv * cfg.head_dim;
      size_t attn_f  = (size_t)T*(Dq0 + 2*Dkv0) + (size_t)T*T + (size_t)T*Dq0;
      size_t temps_f = 5ull * (size_t)T * cfg.d_model;
      float* scratch_attn = (float*)malloc(sizeof(float)*(attn_f + temps_f));
      const int d_ff = cfg.d_ff;
      float* scratch_moe  = (float*)malloc(sizeof(float)*(2ull*(size_t)T*d_ff));
      int*   top_idx      = (int*)  malloc(sizeof(int)*T*cfg.top_k);
      float* top_p        = (float*)malloc(sizeof(float)*T*cfg.top_k);

      for (int L=0; L<N; ++L){
        // choose whether to apply qk-norm
        const float* qn_ptr = W.layers[L].q_norm;
        const float* kn_ptr = W.layers[L].k_norm;
        if (qkn_mode == QKN_OFF){ printf("q_off\n"); qn_ptr = NULL; kn_ptr = NULL; }
        else if (qkn_mode == QKN_ON){ /* leave as is; may be NULL if absent */ }

        layer_forward_f32(
          x, T, cfg.d_model,
          /*norm1*/ W.layers[L].rms1_w, 1e-6f,
          /*attn*/  W.layers[L].Wq, W.layers[L].bq,
                    W.layers[L].Wk, W.layers[L].bk,
                    W.layers[L].Wv, W.layers[L].bv,
                    W.layers[L].Wo, W.layers[L].bo,
          /*qk-norm*/ qn_ptr, kn_ptr,
          /*heads*/   cfg.n_q, cfg.n_kv, cfg.head_dim, /*causal*/1, cfg.rope_theta,
          /*norm2*/ W.layers[L].rms2_w, 1e-6f,
          /*moe*/   W.layers[L].router_w, W.layers[L].router_b,
                    W.layers[L].Wg, W.layers[L].bg, W.layers[L].Wu, W.layers[L].bu, W.layers[L].Wd, W.layers[L].bd,
          cfg.n_experts, cfg.top_k, cfg.d_ff,
          /*scratch*/ scratch_attn, scratch_moe, top_idx, top_p
        );

        // compare to python dump for this layer
        snprintf(path,sizeof(path),"%s.step%d.L%d.y.npy", debug_prefix, s, L);
        NpyArray* yrefL = npy_load_float32(path);
        if (!yrefL){ fprintf(stderr,"[debug] missing %s\n", path); return 1; }
        if (yrefL->ndim!=2 || yrefL->shape[0]!=T || yrefL->shape[1]!=cfg.d_model){
          fprintf(stderr,"[debug] L%d shape mismatch at step %d\n", L, s); return 1;
        }
        float layer_mad = max_abs_diff(x, yrefL->data, T*cfg.d_model);
        printf("*****  step %d, layer %d: max abs diff = %.6g\n", s, L, layer_mad);
        npy_free(yrefL);

        if (layer_mad > 2e-4f){
          fprintf(stderr,"[debug] divergence first seen at step %d layer %d\n", s, L);
          // leave rc flagged but continue printing the rest for visibility
          rc = 1;
        //   break;
        }
      }

      free(scratch_attn); free(scratch_moe); free(top_idx); free(top_p);
      free(x); npy_free(xnp);
    }

    // (C) greedy pick to continue
    int argmax = 0; float best = last[0];
    for (int i=1;i<cfg.vocab_size;i++){ if (last[i] > best){ best = last[i]; argmax = i; } }
    cur_ids[T] = argmax;
    printf("  argmax=%d\n", argmax);
    if (cur_ids[T] != ids_all->data[T]){
      fprintf(stderr,"step %d: id mismatch (C=%d, Py=%d)\n", s, cur_ids[T], ids_all->data[T]);
      rc = 1;
    }
  }

  free(full);
  free(cur_ids);
  npy_free_i32(ids_all);
  npy_free(yref);
  for (int L=0; L<N; ++L){
    free((void*)W.layers[L].Wg); free((void*)W.layers[L].bg);
    free((void*)W.layers[L].Wu); free((void*)W.layers[L].bu);
    free((void*)W.layers[L].Wd); free((void*)W.layers[L].bd);
  }
  free(W.layers);
  bin_free(bin);
  return rc;
}
