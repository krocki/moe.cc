// test_model.c
// ids -> full model (N layers) -> logits; compares with Py goldens
// Usage: ./test_model <weights.bin> <ids.npy> <logits.npy> <N_layers> <rope_theta> [--nosoftmax]
// Exits 0 on success.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "io.h"
#include "utils.h"
#include "model.h"

static void need(BinFile* b, const char* k, TensorBin** out){
  *out = bin_find(b, k);
  if (!*out){ fprintf(stderr, "Key not found: %s\n", k); exit(1); }
}
static void maybe(BinFile* b, const char* k, TensorBin** out){ *out = bin_find(b, k); }

int main(int argc, char** argv){
  if (argc < 6){
    fprintf(stderr, "Usage: %s <weights.bin> <ids.npy> <logits.npy> <N_layers> <rope_theta> [--nosoftmax]\n", argv[0]);
    return 1;
  }
  const char* wpath = argv[1];
  const char* ids_npy = argv[2];
  const char* yref_npy= argv[3];
  int N               = atoi(argv[4]);
  float rope_theta    = strtof(argv[5], NULL);
  int apply_softmax   = 1;
  for (int i=6;i<argc;++i) if (strcmp(argv[i],"--nosoftmax")==0) apply_softmax = 0;

  fprintf(stderr, "[load] %s\n", wpath);
  BinFile* bin = bin_load(wpath);
  if(!bin){ fprintf(stderr,"bin load fail\n"); return 1; }

  // ---- config from shapes (L0)
  TensorBin *tWq0,*tWk0,*tQn0,*tEmb,*tFinal,*tHead=NULL;
  need(bin, "model.layers.0.self_attn.q_proj.weight", &tWq0);
  need(bin, "model.layers.0.self_attn.k_proj.weight", &tWk0);
  need(bin, "model.layers.0.self_attn.q_norm.weight", &tQn0);
  need(bin, "model.embed_tokens.weight", &tEmb);
  maybe(bin, "model.norm.weight", &tFinal);
  if (!tFinal) need(bin, "model.final_layernorm.weight", &tFinal);
  maybe(bin, "lm_head.weight", &tHead);

  QwenConfig cfg = {0};
  cfg.d_model   = tEmb->shape[1];
  cfg.n_layers  = N;
  cfg.head_dim  = tQn0->shape[0];
  cfg.n_q       = tWq0->shape[0] / cfg.head_dim;
  cfg.n_kv      = tWk0->shape[0] / cfg.head_dim;
  cfg.d_ff      = 768;         // Qwen3-30B-A3B (known)
  cfg.n_experts = 128;
  cfg.top_k     = 8;
  cfg.vocab_size= tEmb->shape[0];
  cfg.causal    = 1;
  cfg.rope_theta= rope_theta;

  // ---- allocate and fill weights struct
  QwenWeights W = {0};
  W.tok_emb      = (const float*)tEmb->data;
  W.final_norm_w = (const float*)tFinal->data;
  W.lm_head      = tHead ? (const float*)tHead->data : NULL;
  W.layers       = (QwenLayerWeights*)calloc(N, sizeof(QwenLayerWeights));

  char key[256];
  for (int L=0; L<N; ++L){
    QwenLayerWeights* lw = &W.layers[L];

    // norms
    TensorBin* t;
    snprintf(key,sizeof(key),"model.layers.%d.input_layernorm.weight",L);
    maybe(bin,key,&t); if(!t){ snprintf(key,sizeof(key),"model.layers.%d.rms_1.weight",L); need(bin,key,&t); }
    lw->rms1_w = (const float*)t->data;

    snprintf(key,sizeof(key),"model.layers.%d.post_attention_layernorm.weight",L);
    maybe(bin,key,&t); if(!t){ snprintf(key,sizeof(key),"model.layers.%d.rms_2.weight",L); need(bin,key,&t); }
    lw->rms2_w = (const float*)t->data;

    // attn
    snprintf(key,sizeof(key),"model.layers.%d.self_attn.q_proj.weight",L); need(bin,key,&t); lw->Wq=(const float*)t->data;
    snprintf(key,sizeof(key),"model.layers.%d.self_attn.k_proj.weight",L); need(bin,key,&t); lw->Wk=(const float*)t->data;
    snprintf(key,sizeof(key),"model.layers.%d.self_attn.v_proj.weight",L); need(bin,key,&t); lw->Wv=(const float*)t->data;
    snprintf(key,sizeof(key),"model.layers.%d.self_attn.o_proj.weight",L); need(bin,key,&t); lw->Wo=(const float*)t->data;

    snprintf(key,sizeof(key),"model.layers.%d.self_attn.q_proj.bias",L); maybe(bin,key,&t); lw->bq=t?(const float*)t->data:NULL;
    snprintf(key,sizeof(key),"model.layers.%d.self_attn.k_proj.bias",L); maybe(bin,key,&t); lw->bk=t?(const float*)t->data:NULL;
    snprintf(key,sizeof(key),"model.layers.%d.self_attn.v_proj.bias",L); maybe(bin,key,&t); lw->bv=t?(const float*)t->data:NULL;
    snprintf(key,sizeof(key),"model.layers.%d.self_attn.o_proj.bias",L); maybe(bin,key,&t); lw->bo=t?(const float*)t->data:NULL;

    snprintf(key,sizeof(key),"model.layers.%d.self_attn.q_norm.weight",L); need(bin,key,&t); lw->q_norm=(const float*)t->data;
    snprintf(key,sizeof(key),"model.layers.%d.self_attn.k_norm.weight",L); need(bin,key,&t); lw->k_norm=(const float*)t->data;

    // router
    snprintf(key,sizeof(key),"model.layers.%d.mlp.gate.weight",L);
    maybe(bin,key,&t);
    if(!t){ snprintf(key,sizeof(key),"model.layers.%d.mlp.router.gate.weight",L); need(bin,key,&t); }
    lw->router_w=(const float*)t->data;
    snprintf(key,sizeof(key),"model.layers.%d.mlp.gate.bias",L); maybe(bin,key,&t); lw->router_b=t?(const float*)t->data:NULL;

    // experts
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

  // ---- load I/O
  NpyArrayI32* ids = npy_load_int32(ids_npy);
  if (!ids || ids->ndim!=1){ fprintf(stderr,"ids.npy must be int32 1-D\n"); return 1; }
  int T = ids->shape[0];

  NpyArray* yref = npy_load_float32(yref_npy);
  if (!yref || yref->ndim!=2){ fprintf(stderr,"logits.npy must be float32 2-D\n"); return 1; }
  if (yref->shape[0]!=T || yref->shape[1]!=cfg.vocab_size){
    fprintf(stderr,"logits shape mismatch: got [%d,%d] expected [%d,%d]\n",
      yref->shape[0], yref->shape[1], T, cfg.vocab_size);
    return 1;
  }

  // ---- run
  float* logits = (float*)malloc(sizeof(float)*(size_t)T*cfg.vocab_size);
  model_forward_f32(ids->data, T, &cfg, &W, /*softmax?*/apply_softmax, logits);

  // ---- compare
  float mad = max_abs_diff(logits, yref->data, T*cfg.vocab_size);
  printf("Max abs diff: %.6g\n%s\n", mad, (mad < 2e-4f ? "PASS" : "FAIL"));

  // cleanup
  free(logits);
  npy_free_i32(ids);
  npy_free(yref);
  for (int L=0; L<N; ++L){
    free((void*)W.layers[L].Wg); free((void*)W.layers[L].bg);
    free((void*)W.layers[L].Wu); free((void*)W.layers[L].bu);
    free((void*)W.layers[L].Wd); free((void*)W.layers[L].bd);
  }
  free(W.layers);
  bin_free(bin);
  return (mad < 2e-4f) ? 0 : 1;
}
