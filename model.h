#ifndef MODEL_H
#define MODEL_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include "io.h"

typedef struct {
  // model sizes
  int d_model;
  int n_layers;
  int head_dim;      // per Q head
  int n_q;           // Q heads per layer (assumed constant across layers)
  int n_kv;          // KV heads per layer
  int d_ff;          // MoE FFN hidden size per expert
  int n_experts;     // total experts per layer
  int top_k;         // top-k routing
  int vocab_size;
  int causal;        // 0/1
  float rope_theta;
  float rms_eps;
} QwenConfig;

// Quantized weight structure for expert matrices
typedef struct {
  const int8_t* q;      // quantized weights
  const float* s;       // scales (per row)
  int rows, cols;       // dimensions
  int dtype;            // 2=q8, 3=q4 (matches TensorBin dtypes)
} QuantizedWeight;
#define QUANTIZED_WEIGHT_DEFINED

typedef struct {
  // attention (always FP32)
  const float* Wq; const float* bq;
  const float* Wk; const float* bk;
  const float* Wv; const float* bv;
  const float* Wo; const float* bo;
  const float* q_norm; // length = head_dim
  const float* k_norm; // length = head_dim
  // norms (always FP32)
  const float* rms1_w;
  const float* rms2_w;
  // router (always FP32)
  const float* router_w; // [E, d_model]
  const float* router_b; // [E] or NULL
  // experts: can be FP32 or quantized
  // FP32 pointers (NULL if quantized)
  const float** Wg; const float** bg; // [E][d_ff, d_model], [E][d_ff]
  const float** Wu; const float** bu; // [E][d_ff, d_model], [E][d_ff]
  const float** Wd; const float** bd; // [E][d_model, d_ff], [E][d_model]
  // Quantized versions (valid if corresponding FP32 pointer is NULL)
  QuantizedWeight* Wg_q; // [E] array of quantized gate weights
  QuantizedWeight* Wu_q; // [E] array of quantized up weights  
  QuantizedWeight* Wd_q; // [E] array of quantized down weights
  int experts_quantized; // 0=fp32, 1=quantized
} QwenLayerWeights;

typedef struct {
  // token embedding (required)
  const float* tok_emb;     // [vocab, d_model]
  // final norm (required; Qwen: "model.norm.weight"/"model.final_layernorm.weight")
  const float* final_norm_w; // [d_model]
  // output head (optional, falls back to tok_emb if NULL)
  const float* lm_head;     // [vocab, d_model] or NULL (tied)
  // layers
  QwenLayerWeights* layers; // [n_layers]
} QwenWeights;

// Full forward: ids -> embed -> N layers -> final norm -> logits (optionally softmax)

// -----------------------------
// Weight loading (one-time)
// -----------------------------
static TensorBin* need(BinFile* b, const char* k){
  TensorBin* t = bin_find(b,k);
  if(!t){ fprintf(stderr,"missing %s\n", k); exit(1); }
  return t;
}
static TensorBin* maybe(BinFile* b, const char* k){ return bin_find(b,k); }

// Helper function to find a tensor, prioritizing the original name
static TensorBin* need_adaptive(BinFile* b, const char* k) {
  TensorBin* t = bin_find(b, k);
  if (t) return t;
  
  // If original tensor not found, it might be quantized
  // This happens when we're loading a quantized model but still looking for original names
  char q8_name[512], q4_name[512];
  snprintf(q8_name, sizeof(q8_name), "%s.q8", k);
  snprintf(q4_name, sizeof(q4_name), "%s.q4", k);
  
  t = bin_find(b, q8_name);
  if (t) return t;  // Found Q8 version
  
  t = bin_find(b, q4_name);
  if (t) return t;  // Found Q4 version
  
  // Still not found - this is an error
  fprintf(stderr, "missing %s (tried fp32, q8, q4)\n", k);
  exit(1);
}

// Fill QwenWeights/QwenConfig from all.bin (Qwen/Qwen3-30B-A3B layout).
// Only strings appear here (one-time), never inside the forward loops.
static void load_all_weights(BinFile* bf, QwenConfig* cfg, QwenWeights* w) {
  // Infer sizes (consistent with your verified setup)
  // d_model, vocab, head_dim, n_q, n_kv, n_layers, d_ff, n_experts, top_k
  TensorBin* Wq0 = need(bf,"model.layers.0.self_attn.q_proj.weight");
  TensorBin* Wk0 = need(bf,"model.layers.0.self_attn.k_proj.weight");
  TensorBin* emb = need(bf,"model.embed_tokens.weight");
  TensorBin* norm = maybe(bf,"model.norm.weight");
  if (!norm) norm = need(bf,"model.final_layernorm.weight");

  cfg->d_model   = Wq0->shape[1];
  cfg->head_dim  = maybe(bf,"model.layers.0.self_attn.q_norm.weight")
                   ? maybe(bf,"model.layers.0.self_attn.q_norm.weight")->shape[0]
                   : (Wq0->shape[0] / 32);
  cfg->n_q       = Wq0->shape[0] / cfg->head_dim;
  cfg->n_kv      = Wk0->shape[0] / cfg->head_dim;
  cfg->vocab_size= emb->shape[0];
  cfg->n_layers  = 0;
  // count layers by probing L until miss
  for (;;) {
    char key[256];
    snprintf(key,sizeof(key),"model.layers.%d.self_attn.q_proj.weight", cfg->n_layers);
    if (!bin_find(bf,key)) break;
    cfg->n_layers++;
  }

  // MoE sizes
  // infer n_experts and d_ff from first layer's experts
  int E = 0, d_ff = -1;
  for (;;) {
    char k_down[256];
    snprintf(k_down,sizeof(k_down),"model.layers.0.mlp.experts.%d.down_proj.weight", E);
    TensorBin* t = bin_find(bf,k_down);
    
    // If FP32 version not found, try quantized versions
    if (!t) {
      snprintf(k_down,sizeof(k_down),"model.layers.0.mlp.experts.%d.down_proj.weight.q8", E);
      t = bin_find(bf,k_down);
    }
    if (!t) {
      snprintf(k_down,sizeof(k_down),"model.layers.0.mlp.experts.%d.down_proj.weight.q4", E);
      t = bin_find(bf,k_down);
    }
    
    if (!t) break;
    d_ff = t->shape[1];
    // For Q4, shape[1] is packed (2 values per byte), so actual cols is shape[1] * 2
    if (strstr(k_down, ".q4")) d_ff *= 2;
    E++;
  }
  if (E==0 || d_ff<=0) { fprintf(stderr,"infer MoE sizes failed\n"); exit(1); }
  cfg->n_experts = E;
  cfg->d_ff      = d_ff;
  // infer top_k from router weight vs Python dump (use common default 8)
  cfg->top_k     = 8;
  cfg->causal    = 1;
  cfg->rope_theta= 10000000.f;
  cfg->rms_eps   = 1e-6f;

  w->tok_emb      = (const float*)emb->data;
  w->final_norm_w = (const float*)norm->data;
  // head: use lm_head if present, else tie
  TensorBin* head = maybe(bf,"lm_head.weight");
  w->lm_head = head ? (const float*)head->data : NULL;

  // allocate layers
  w->layers = (QwenLayerWeights*)calloc((size_t)cfg->n_layers, sizeof(QwenLayerWeights));

  for (int L=0; L<cfg->n_layers; ++L) {
    QwenLayerWeights* lw = &w->layers[L];
    char k[256];

    // attention weights/biases
    snprintf(k,sizeof(k),"model.layers.%d.self_attn.q_proj.weight",L);
    lw->Wq = (const float*)need(bf,k)->data;
    snprintf(k,sizeof(k),"model.layers.%d.self_attn.k_proj.weight",L);
    lw->Wk = (const float*)need(bf,k)->data;
    snprintf(k,sizeof(k),"model.layers.%d.self_attn.v_proj.weight",L);
    lw->Wv = (const float*)need(bf,k)->data;
    snprintf(k,sizeof(k),"model.layers.%d.self_attn.o_proj.weight",L);
    lw->Wo = (const float*)need(bf,k)->data;

    snprintf(k,sizeof(k),"model.layers.%d.self_attn.q_proj.bias",L);
    lw->bq = (maybe(bf,k) && maybe(bf,k)->dtype==0) ? (const float*)maybe(bf,k)->data : NULL;
    snprintf(k,sizeof(k),"model.layers.%d.self_attn.k_proj.bias",L);
    lw->bk = (maybe(bf,k) && maybe(bf,k)->dtype==0) ? (const float*)maybe(bf,k)->data : NULL;
    snprintf(k,sizeof(k),"model.layers.%d.self_attn.v_proj.bias",L);
    lw->bv = (maybe(bf,k) && maybe(bf,k)->dtype==0) ? (const float*)maybe(bf,k)->data : NULL;
    snprintf(k,sizeof(k),"model.layers.%d.self_attn.o_proj.bias",L);
    lw->bo = (maybe(bf,k) && maybe(bf,k)->dtype==0) ? (const float*)maybe(bf,k)->data : NULL;

    // qk norm (auto-present for Qwen3 A3B)
    TensorBin* qn = maybe(bf, (snprintf(k,sizeof(k),"model.layers.%d.self_attn.q_norm.weight",L), k));
    TensorBin* kn = maybe(bf, (snprintf(k,sizeof(k),"model.layers.%d.self_attn.k_norm.weight",L), k));
    lw->q_norm = qn ? (const float*)qn->data : NULL;
    lw->k_norm = kn ? (const float*)kn->data : NULL;

    // norms
    TensorBin* n1 = maybe(bf, (snprintf(k,sizeof(k),"model.layers.%d.input_layernorm.weight",L), k));
    if (!n1) { snprintf(k, sizeof(k), "model.layers.%d.rms_1.weight", L); n1 = need(bf, k); }
    lw->rms1_w = (const float*)n1->data;

    TensorBin* n2 = maybe(bf, (snprintf(k,sizeof(k),"model.layers.%d.post_attention_layernorm.weight",L), k));
    if (!n2) { snprintf(k, sizeof(k), "model.layers.%d.rms_2.weight", L); n2 = need(bf, k); }
    lw->rms2_w = (const float*)n2->data;

    // router
    TensorBin* RW = maybe(bf, (snprintf(k,sizeof(k),"model.layers.%d.mlp.gate.weight",L), k));
    if (!RW) { snprintf(k, sizeof(k), "model.layers.%d.mlp.router.gate.weight", L); RW = need(bf, k); }
    lw->router_w = (const float*)RW->data;

    TensorBin* RB = maybe(bf, (snprintf(k,sizeof(k),"model.layers.%d.mlp.gate.bias",L), k));
    lw->router_b = (RB && RB->dtype==0) ? (const float*)RB->data : NULL;

    // Check if experts are quantized by looking for .q8/.q4 suffixes
    snprintf(k,sizeof(k),"model.layers.%d.mlp.experts.0.gate_proj.weight.q8",L);
    TensorBin* test_q8 = maybe(bf, k);
    snprintf(k,sizeof(k),"model.layers.%d.mlp.experts.0.gate_proj.weight.q4",L);
    TensorBin* test_q4 = maybe(bf, k);
    
    lw->experts_quantized = (test_q8 || test_q4) ? 1 : 0;
    
    if (lw->experts_quantized) {
      // Allocate quantized weight structures
      lw->Wg_q = (QuantizedWeight*)calloc((size_t)cfg->n_experts, sizeof(QuantizedWeight));
      lw->Wu_q = (QuantizedWeight*)calloc((size_t)cfg->n_experts, sizeof(QuantizedWeight));
      lw->Wd_q = (QuantizedWeight*)calloc((size_t)cfg->n_experts, sizeof(QuantizedWeight));
      lw->Wg = lw->Wu = lw->Wd = NULL; // Mark FP32 as unused
      
      // Load quantized expert weights
      for (int e=0; e<cfg->n_experts; ++e){
        // Gate projection
        if (test_q8) {
          snprintf(k,sizeof(k),"model.layers.%d.mlp.experts.%d.gate_proj.weight.q8",L,e);
          TensorBin* q_data = need(bf,k);
          snprintf(k,sizeof(k),"model.layers.%d.mlp.experts.%d.gate_proj.weight.scale",L,e);
          TensorBin* s_data = need(bf,k);
          lw->Wg_q[e].q = (const int8_t*)q_data->data;
          lw->Wg_q[e].s = (const float*)s_data->data;
          lw->Wg_q[e].rows = q_data->shape[0];
          lw->Wg_q[e].cols = q_data->shape[1];
          lw->Wg_q[e].dtype = 2; // q8
        } else {
          snprintf(k,sizeof(k),"model.layers.%d.mlp.experts.%d.gate_proj.weight.q4",L,e);
          TensorBin* q_data = need(bf,k);
          snprintf(k,sizeof(k),"model.layers.%d.mlp.experts.%d.gate_proj.weight.scale",L,e);
          TensorBin* s_data = need(bf,k);
          lw->Wg_q[e].q = (const int8_t*)q_data->data;
          lw->Wg_q[e].s = (const float*)s_data->data;
          lw->Wg_q[e].rows = q_data->shape[0];
          lw->Wg_q[e].cols = q_data->shape[1] * 2; // q4 packs 2 values per byte
          lw->Wg_q[e].dtype = 3; // q4
        }
        
        // Up projection  
        if (test_q8) {
          snprintf(k,sizeof(k),"model.layers.%d.mlp.experts.%d.up_proj.weight.q8",L,e);
          TensorBin* q_data = need(bf,k);
          snprintf(k,sizeof(k),"model.layers.%d.mlp.experts.%d.up_proj.weight.scale",L,e);
          TensorBin* s_data = need(bf,k);
          lw->Wu_q[e].q = (const int8_t*)q_data->data;
          lw->Wu_q[e].s = (const float*)s_data->data;
          lw->Wu_q[e].rows = q_data->shape[0];
          lw->Wu_q[e].cols = q_data->shape[1];
          lw->Wu_q[e].dtype = 2;
        } else {
          snprintf(k,sizeof(k),"model.layers.%d.mlp.experts.%d.up_proj.weight.q4",L,e);
          TensorBin* q_data = need(bf,k);
          snprintf(k,sizeof(k),"model.layers.%d.mlp.experts.%d.up_proj.weight.scale",L,e);
          TensorBin* s_data = need(bf,k);
          lw->Wu_q[e].q = (const int8_t*)q_data->data;
          lw->Wu_q[e].s = (const float*)s_data->data;
          lw->Wu_q[e].rows = q_data->shape[0];
          lw->Wu_q[e].cols = q_data->shape[1] * 2;
          lw->Wu_q[e].dtype = 3;
        }
        
        // Down projection
        if (test_q8) {
          snprintf(k,sizeof(k),"model.layers.%d.mlp.experts.%d.down_proj.weight.q8",L,e);
          TensorBin* q_data = need(bf,k);
          snprintf(k,sizeof(k),"model.layers.%d.mlp.experts.%d.down_proj.weight.scale",L,e);
          TensorBin* s_data = need(bf,k);
          lw->Wd_q[e].q = (const int8_t*)q_data->data;
          lw->Wd_q[e].s = (const float*)s_data->data;
          lw->Wd_q[e].rows = q_data->shape[0];
          lw->Wd_q[e].cols = q_data->shape[1];
          lw->Wd_q[e].dtype = 2;
        } else {
          snprintf(k,sizeof(k),"model.layers.%d.mlp.experts.%d.down_proj.weight.q4",L,e);
          TensorBin* q_data = need(bf,k);
          snprintf(k,sizeof(k),"model.layers.%d.mlp.experts.%d.down_proj.weight.scale",L,e);
          TensorBin* s_data = need(bf,k);
          lw->Wd_q[e].q = (const int8_t*)q_data->data;
          lw->Wd_q[e].s = (const float*)s_data->data;
          lw->Wd_q[e].rows = q_data->shape[0];
          lw->Wd_q[e].cols = q_data->shape[1] * 2;
          lw->Wd_q[e].dtype = 3;
        }
      }
    } else {
      // Load FP32 expert weights (original behavior)
      lw->Wg = (const float**)calloc((size_t)cfg->n_experts, sizeof(float*));
      lw->Wu = (const float**)calloc((size_t)cfg->n_experts, sizeof(float*));
      lw->Wd = (const float**)calloc((size_t)cfg->n_experts, sizeof(float*));
      lw->Wg_q = lw->Wu_q = lw->Wd_q = NULL; // Mark quantized as unused
      
      for (int e=0;e<cfg->n_experts;++e){
        snprintf(k,sizeof(k),"model.layers.%d.mlp.experts.%d.gate_proj.weight",L,e);
        lw->Wg[e] = (const float*)need(bf,k)->data;
        snprintf(k,sizeof(k),"model.layers.%d.mlp.experts.%d.up_proj.weight",L,e);
        lw->Wu[e] = (const float*)need(bf,k)->data;
        snprintf(k,sizeof(k),"model.layers.%d.mlp.experts.%d.down_proj.weight",L,e);
        lw->Wd[e] = (const float*)need(bf,k)->data;
      }
    }
  }
}
#endif // MODEL_H
