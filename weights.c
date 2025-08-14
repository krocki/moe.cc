// weights.c
#include "model.h"
#include "io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static TensorBin* need(BinFile* b, const char* k){
  TensorBin* t = bin_find(b,k);
  if(!t){ fprintf(stderr,"missing %s\n", k); exit(1); }
  return t;
}
static TensorBin* maybe(BinFile* b, const char* k){ return bin_find(b,k); }

static int count_layers(BinFile* bf){
  int L=0; char k[256];
  for(;;++L){
    snprintf(k,sizeof(k),"model.layers.%d.self_attn.q_proj.weight", L);
    if(!bin_find(bf,k)) break;
  }
  return L;
}
static int count_experts(BinFile* bf, int L){
  int E=0; char k[256];
  for(;;++E){
    snprintf(k,sizeof(k),"model.layers.%d.mlp.experts.%d.down_proj.weight", L, E);
    if(!bin_find(bf,k)) break;
  }
  return E;
}

// Fills cfg + w (allocates w->layers and expert pointer tables).
// Caller owns cfg/w lifetimes; call qwen_free_weights(&w) at the end.
void qwen_load_from_allbin(const char* allbin, QwenConfig* cfg, QwenWeights* w){
  memset(w, 0, sizeof(*w));
  BinFile* bf = bin_load(allbin);
  if (!bf){ fprintf(stderr,"bin load fail\n"); exit(1); }

  TensorBin* tok = need(bf,"model.embed_tokens.weight");
  const int V = tok->shape[0];
  const int D = tok->shape[1];

  // layer 0 probes for dims
  TensorBin* Wq0 = need(bf,"model.layers.0.self_attn.q_proj.weight");
  TensorBin* Wk0 = need(bf,"model.layers.0.self_attn.k_proj.weight");
  int d_model = D;
  int head_dim;
  TensorBin* qn0 = maybe(bf,"model.layers.0.self_attn.q_norm.weight");
  if (qn0) head_dim = qn0->shape[0];
  else     head_dim = 128; // Qwen3 default if not present
  int n_q  = Wq0->shape[0]  / head_dim;
  int n_kv = Wk0->shape[0]  / head_dim;

  int n_layers = count_layers(bf);
  int n_experts = count_experts(bf, 0);
  int d_ff = need(bf,"model.layers.0.mlp.experts.0.down_proj.weight")->shape[1];

  // router hyperparams
  // gate weights exist => MoE; Raschka config uses top_k=8
  int top_k = 8;

  cfg->d_model = d_model;
  cfg->n_layers= n_layers;
  cfg->head_dim= head_dim;
  cfg->n_q     = n_q;
  cfg->n_kv    = n_kv;
  cfg->d_ff    = d_ff;
  cfg->n_experts = n_experts;
  cfg->top_k   = top_k;
  cfg->vocab_size = V;
  cfg->causal  = 1;
  cfg->rope_theta = 10000000.0f; // set from CLI if you want

  // top-level pointers
  w->tok_emb = (const float*)tok->data;
  TensorBin* FN = maybe(bf,"model.norm.weight");
  if (!FN) FN = need(bf,"model.final_layernorm.weight");
  w->final_norm_w = (const float*)FN->data;

  TensorBin* LMH = maybe(bf,"lm_head.weight");
  w->lm_head = LMH ? (const float*)LMH->data : NULL;

  // allocate layers
  w->layers = (QwenLayerWeights*)calloc(n_layers, sizeof(QwenLayerWeights));
  for (int L=0; L<n_layers; ++L){
    QwenLayerWeights* lw = &w->layers[L];
    char k[256];
    // attn weights/biases
    snprintf(k,sizeof(k),"model.layers.%d.self_attn.q_proj.weight", L); lw->Wq = (const float*)need(bf,k)->data;
    snprintf(k,sizeof(k),"model.layers.%d.self_attn.k_proj.weight", L); lw->Wk = (const float*)need(bf,k)->data;
    snprintf(k,sizeof(k),"model.layers.%d.self_attn.v_proj.weight", L); lw->Wv = (const float*)need(bf,k)->data;
    snprintf(k,sizeof(k),"model.layers.%d.self_attn.o_proj.weight", L); lw->Wo = (const float*)need(bf,k)->data;

    snprintf(k,sizeof(k),"model.layers.%d.self_attn.q_proj.bias", L);  { TensorBin* t=maybe(bf,k); lw->bq = t?(const float*)t->data:NULL; }
    snprintf(k,sizeof(k),"model.layers.%d.self_attn.k_proj.bias", L);  { TensorBin* t=maybe(bf,k); lw->bk = t?(const float*)t->data:NULL; }
    snprintf(k,sizeof(k),"model.layers.%d.self_attn.v_proj.bias", L);  { TensorBin* t=maybe(bf,k); lw->bv = t?(const float*)t->data:NULL; }
    snprintf(k,sizeof(k),"model.layers.%d.self_attn.o_proj.bias", L);  { TensorBin* t=maybe(bf,k); lw->bo = t?(const float*)t->data:NULL; }

    snprintf(k,sizeof(k),"model.layers.%d.self_attn.q_norm.weight", L);{ TensorBin* t=maybe(bf,k); lw->q_norm = t?(const float*)t->data:NULL; }
    snprintf(k,sizeof(k),"model.layers.%d.self_attn.k_norm.weight", L);{ TensorBin* t=maybe(bf,k); lw->k_norm = t?(const float*)t->data:NULL; }

    // norms
    snprintf(k,sizeof(k),"model.layers.%d.input_layernorm.weight", L);
    TensorBin* n1 = maybe(bf,k);
    if (!n1){ snprintf(k,sizeof(k),"model.layers.%d.rms_1.weight", L); n1 = need(bf,k); }
    lw->rms1_w = (const float*)n1->data;

    snprintf(k,sizeof(k),"model.layers.%d.post_attention_layernorm.weight", L);
    TensorBin* n2 = maybe(bf,k);
    if (!n2){ snprintf(k,sizeof(k),"model.layers.%d.rms_2.weight", L); n2 = need(bf,k); }
    lw->rms2_w = (const float*)n2->data;

    // router
    snprintf(k,sizeof(k),"model.layers.%d.mlp.gate.weight", L);
    TensorBin* rw = maybe(bf,k);
    if (!rw){ snprintf(k,sizeof(k),"model.layers.%d.mlp.router.gate.weight", L); rw = need(bf,k); }
    lw->router_w = (const float*)rw->data;

    snprintf(k,sizeof(k),"model.layers.%d.mlp.gate.bias", L);
    TensorBin* rb = maybe(bf,k);
    if (!rb){ snprintf(k,sizeof(k),"model.layers.%d.mlp.router.gate.bias", L); rb = maybe(bf,k); }
    lw->router_b = rb ? (const float*)rb->data : NULL;

    // experts (weights only; biases are optional and typically absent in Qwen3-A3B)
    lw->Wg = (const float**)calloc(n_experts, sizeof(float*));
    lw->Wu = (const float**)calloc(n_experts, sizeof(float*));
    lw->Wd = (const float**)calloc(n_experts, sizeof(float*));
    lw->bg = lw->bu = lw->bd = NULL; // not used (no biases)

    for (int e=0; e<n_experts; ++e){
      snprintf(k,sizeof(k),"model.layers.%d.mlp.experts.%d.gate_proj.weight", L, e);
      lw->Wg[e] = (const float*)need(bf,k)->data;
      snprintf(k,sizeof(k),"model.layers.%d.mlp.experts.%d.up_proj.weight", L, e);
      lw->Wu[e] = (const float*)need(bf,k)->data;
      snprintf(k,sizeof(k),"model.layers.%d.mlp.experts.%d.down_proj.weight", L, e);
      lw->Wd[e] = (const float*)need(bf,k)->data;
    }
  }

  // stash the BinFile pointer to free later if you want; or just leak safely for a short-run tool
  // Here we keep it simple:
  // bin_free(bf);  // DON'T free if you keep raw pointers into its memory mapped region.
  // If bin_load copies into heap and you can free safely, uncomment the line above.
}

void qwen_free_weights(const QwenConfig* cfg, QwenWeights* w){
  if (!w || !cfg) return;
  for (int L=0; L<cfg->n_layers; ++L){
    QwenLayerWeights* lw = &w->layers[L];
    free((void*)lw->Wg);
    free((void*)lw->Wu);
    free((void*)lw->Wd);
  }
  free(w->layers);
}
