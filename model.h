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
  const float* s;       // scales (per row or per group)
  const float* zp;      // zero points (Q4 only, NULL for Q8)
  int rows, cols;       // dimensions
  int dtype;            // 2=q8, 3=q4 (matches Tensor dtypes)
  size_t group_size;    // group size for quantization (0 = rowwise)
} QuantizedWeight;
#define QUANTIZED_WEIGHT_DEFINED

typedef struct {
  // attention: can be FP32 or quantized
  // FP32 pointers (NULL if quantized)
  const float* Wq; const float* bq;
  const float* Wk; const float* bk;
  const float* Wv; const float* bv;
  const float* Wo; const float* bo;
  const float* q_norm; // length = head_dim
  const float* k_norm; // length = head_dim
  // Quantized attention weights (NULL if FP32)
  QuantizedWeight* Wq_q; // quantized query projection
  QuantizedWeight* Wk_q; // quantized key projection  
  QuantizedWeight* Wv_q; // quantized value projection
  QuantizedWeight* Wo_q; // quantized output projection
  bool attention_quantized; // flag: true if attention is quantized
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
  // token embedding (required) - always FP32 for initial lookup
  const float* tok_emb;     // [vocab, d_model]
  // final norm (required; Qwen: "model.norm.weight"/"model.final_layernorm.weight")
  const float* final_norm_w; // [d_model]
  // output head (optional, falls back to tok_emb if NULL)
  const float* lm_head;     // [vocab, d_model] or NULL (tied) - FP32 version
  QuantizedWeight* lm_head_q; // quantized lm_head (Q8 only) - NULL if not quantized
  bool lm_head_quantized;   // flag: true if lm_head is quantized to Q8
  // layers
  QwenLayerWeights* layers; // [n_layers]
} QwenWeights;

// Full forward: ids -> embed -> N layers -> final norm -> logits (optionally softmax)

// -----------------------------
// Weight loading (one-time)
// -----------------------------
static Tensor* need(BinFile* b, const char* k){
  Tensor* t = bin_find(b,k);
  if(!t){ fprintf(stderr,"missing %s\n", k); exit(1); }
  return t;
}
static Tensor* maybe(BinFile* b, const char* k){ return bin_find(b,k); }

// Note: need_adaptive function was removed as it was unused
// If adaptive tensor loading is needed in the future, implement a version
// that properly handles the new group_size metadata

/**
 * Load complete Qwen3-30B-A3B model weights from binary file
 * 
 * This function initializes QwenConfig and QwenWeights structures from a binary
 * file containing the complete model. It handles both FP32 and quantized weights,
 * automatically detecting quantization format and populating appropriate structures.
 * 
 * Used by: test_model_trace.c for model inference testing
 * 
 * @param bf: BinFile containing all model tensors
 * @param cfg: Output QwenConfig structure to populate
 * @param w: Output QwenWeights structure to populate  
 */
static void load_all_weights(BinFile* bf, QwenConfig* cfg, QwenWeights* w) {
  // Infer sizes (consistent with your verified setup)
  // d_model, vocab, head_dim, n_q, n_kv, n_layers, d_ff, n_experts, top_k
  
  // Try to find q_proj tensor (FP32 first, then Q8, then Q4)
  Tensor* Wq0 = maybe(bf,"model.layers.0.self_attn.q_proj.weight");
  bool is_q4_attention = false;
  if (!Wq0) {
    // Try Q8 quantized version
    Tensor* Wq0_q8 = maybe(bf,"model.layers.0.self_attn.q_proj.weight.q8");
    if (Wq0_q8) {
      Wq0 = Wq0_q8; // Use the quantized tensor for shape inference
    } else {
      // Try Q4 quantized version
      Tensor* Wq0_q4 = maybe(bf,"model.layers.0.self_attn.q_proj.weight.q4");
      if (Wq0_q4) {
        Wq0 = Wq0_q4; // Use the Q4 tensor for shape inference
        is_q4_attention = true; // Mark that we're using Q4 attention
      } else {
        Wq0 = need(bf,"model.layers.0.self_attn.q_proj.weight"); // This will fail with error message
      }
    }
  }
  
  // Try to find k_proj tensor (FP32 first, then Q8, then Q4) 
  Tensor* Wk0 = maybe(bf,"model.layers.0.self_attn.k_proj.weight");
  if (!Wk0) {
    // Try Q8 quantized version
    Tensor* Wk0_q8 = maybe(bf,"model.layers.0.self_attn.k_proj.weight.q8");
    if (Wk0_q8) {
      Wk0 = Wk0_q8; // Use the quantized tensor for shape inference
    } else {
      // Try Q4 quantized version
      Tensor* Wk0_q4 = maybe(bf,"model.layers.0.self_attn.k_proj.weight.q4");
      if (Wk0_q4) {
        Wk0 = Wk0_q4; // Use the Q4 tensor for shape inference  
      } else {
        Wk0 = need(bf,"model.layers.0.self_attn.k_proj.weight"); // This will fail with error message
      }
    }
  }
  
  Tensor* emb = need(bf,"model.embed_tokens.weight");
  Tensor* norm = maybe(bf,"model.norm.weight");
  if (!norm) norm = need(bf,"model.final_layernorm.weight");

  cfg->d_model   = Wq0->shape[1];
  // For Q4 attention, shape[1] is packed (2 values per byte), so multiply by 2
  if (is_q4_attention) cfg->d_model *= 2;
  cfg->head_dim  = maybe(bf,"model.layers.0.self_attn.q_norm.weight")
                   ? maybe(bf,"model.layers.0.self_attn.q_norm.weight")->shape[0]
                   : (Wq0->shape[0] / 32);
  cfg->n_q       = Wq0->shape[0] / cfg->head_dim;
  cfg->n_kv      = Wk0->shape[0] / cfg->head_dim;
  cfg->vocab_size= emb->shape[0];
  cfg->n_layers  = 0;
  // count layers by probing L until miss (check both FP32 and quantized)
  for (;;) {
    char key[256];
    // Try FP32 first
    snprintf(key,sizeof(key),"model.layers.%d.self_attn.q_proj.weight", cfg->n_layers);
    Tensor* layer_tensor = bin_find(bf,key);
    
    // If FP32 not found, try quantized versions
    if (!layer_tensor) {
      snprintf(key,sizeof(key),"model.layers.%d.self_attn.q_proj.weight.q8", cfg->n_layers);
      layer_tensor = bin_find(bf,key);
    }
    if (!layer_tensor) {
      snprintf(key,sizeof(key),"model.layers.%d.self_attn.q_proj.weight.q4", cfg->n_layers);  
      layer_tensor = bin_find(bf,key);
    }
    
    if (!layer_tensor) break;
    cfg->n_layers++;
  }

  // MoE sizes
  // infer n_experts and d_ff from first layer's experts
  int E = 0, d_ff = -1;
  for (;;) {
    char k_down[256];
    snprintf(k_down,sizeof(k_down),"model.layers.0.mlp.experts.%d.down_proj.weight", E);
    Tensor* t = bin_find(bf,k_down);
    
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
  Tensor* head = maybe(bf,"lm_head.weight");
  w->lm_head = head ? (const float*)head->data : NULL;
  w->lm_head_q = NULL;  // Initialize quantized version as NULL
  w->lm_head_quantized = false;  // Initialize as not quantized

  // allocate layers
  w->layers = (QwenLayerWeights*)calloc((size_t)cfg->n_layers, sizeof(QwenLayerWeights));

  for (int L=0; L<cfg->n_layers; ++L) {
    QwenLayerWeights* lw = &w->layers[L];
    char k[256];

    // attention weights/biases - check for quantized versions first
    // Initialize quantized attention pointers to NULL
    lw->Wq_q = lw->Wk_q = lw->Wv_q = lw->Wo_q = NULL;
    lw->attention_quantized = false;
    
    // Check if quantized attention weights exist (Q8 first, then Q4)
    snprintf(k, sizeof(k), "model.layers.%d.self_attn.q_proj.weight.q8", L);
    Tensor* Wq_q8 = bin_find(bf, k);
    if (!Wq_q8) {
      snprintf(k, sizeof(k), "model.layers.%d.self_attn.q_proj.weight.q4", L);
      Wq_q8 = bin_find(bf, k);
    }
    
    if (Wq_q8) {
      bool is_q4 = (strstr(k, ".q4") != NULL);
      // Load quantized attention weights (Q8 or Q4)
      const char* suffix = is_q4 ? ".q4" : ".q8";
      int dtype = is_q4 ? 1 : 2;  // Q4=1, Q8=2
      
      // Wq
      snprintf(k, sizeof(k), "model.layers.%d.self_attn.q_proj.weight%s", L, suffix);
      Tensor* Wq_data = need(bf, k);
      snprintf(k, sizeof(k), "model.layers.%d.self_attn.q_proj.weight.scale", L);
      Tensor* Wq_scales = need(bf, k);
      lw->Wq_q = (QuantizedWeight*)malloc(sizeof(QuantizedWeight));
      lw->Wq_q->q = (const int8_t*)Wq_data->data;
      lw->Wq_q->s = (const float*)Wq_scales->data;
      lw->Wq_q->rows = Wq_data->shape[0];
      lw->Wq_q->cols = Wq_data->shape[1];
      lw->Wq_q->dtype = dtype;
      lw->Wq_q->group_size = Wq_scales->group_size;
      
      // Load zero points for Q4, NULL for Q8
      if (is_q4) {
        snprintf(k, sizeof(k), "model.layers.%d.self_attn.q_proj.weight.zero_point", L);
        Tensor* Wq_zp = need(bf, k);
        lw->Wq_q->zp = (const float*)Wq_zp->data;
      } else {
        lw->Wq_q->zp = NULL;
      }
      
      // Wk
      snprintf(k, sizeof(k), "model.layers.%d.self_attn.k_proj.weight%s", L, suffix);
      Tensor* Wk_data = need(bf, k);
      snprintf(k, sizeof(k), "model.layers.%d.self_attn.k_proj.weight.scale", L);
      Tensor* Wk_scales = need(bf, k);
      lw->Wk_q = (QuantizedWeight*)malloc(sizeof(QuantizedWeight));
      lw->Wk_q->q = (const int8_t*)Wk_data->data;
      lw->Wk_q->s = (const float*)Wk_scales->data;
      lw->Wk_q->rows = Wk_data->shape[0];
      lw->Wk_q->cols = Wk_data->shape[1];
      lw->Wk_q->dtype = dtype;
      lw->Wk_q->group_size = Wk_scales->group_size;
      
      if (is_q4) {
        snprintf(k, sizeof(k), "model.layers.%d.self_attn.k_proj.weight.zero_point", L);
        Tensor* Wk_zp = need(bf, k);
        lw->Wk_q->zp = (const float*)Wk_zp->data;
      } else {
        lw->Wk_q->zp = NULL;
      }
      
      // Wv
      snprintf(k, sizeof(k), "model.layers.%d.self_attn.v_proj.weight%s", L, suffix);
      Tensor* Wv_data = need(bf, k);
      snprintf(k, sizeof(k), "model.layers.%d.self_attn.v_proj.weight.scale", L);
      Tensor* Wv_scales = need(bf, k);
      lw->Wv_q = (QuantizedWeight*)malloc(sizeof(QuantizedWeight));
      lw->Wv_q->q = (const int8_t*)Wv_data->data;
      lw->Wv_q->s = (const float*)Wv_scales->data;
      lw->Wv_q->rows = Wv_data->shape[0];
      lw->Wv_q->cols = Wv_data->shape[1];
      lw->Wv_q->dtype = dtype;
      lw->Wv_q->group_size = Wv_scales->group_size;
      
      if (is_q4) {
        snprintf(k, sizeof(k), "model.layers.%d.self_attn.v_proj.weight.zero_point", L);
        Tensor* Wv_zp = need(bf, k);
        lw->Wv_q->zp = (const float*)Wv_zp->data;
      } else {
        lw->Wv_q->zp = NULL;
      }
      
      // Wo
      snprintf(k, sizeof(k), "model.layers.%d.self_attn.o_proj.weight%s", L, suffix);
      Tensor* Wo_data = need(bf, k);
      snprintf(k, sizeof(k), "model.layers.%d.self_attn.o_proj.weight.scale", L);
      Tensor* Wo_scales = need(bf, k);
      lw->Wo_q = (QuantizedWeight*)malloc(sizeof(QuantizedWeight));
      lw->Wo_q->q = (const int8_t*)Wo_data->data;
      lw->Wo_q->s = (const float*)Wo_scales->data;
      lw->Wo_q->rows = Wo_data->shape[0];
      lw->Wo_q->cols = Wo_data->shape[1];
      lw->Wo_q->dtype = dtype;
      lw->Wo_q->group_size = Wo_scales->group_size;
      
      if (is_q4) {
        snprintf(k, sizeof(k), "model.layers.%d.self_attn.o_proj.weight.zero_point", L);
        Tensor* Wo_zp = need(bf, k);
        lw->Wo_q->zp = (const float*)Wo_zp->data;
      } else {
        lw->Wo_q->zp = NULL;
      }
      
      // Set flag and skip loading FP32 versions
      lw->attention_quantized = true;
      lw->Wq = lw->Wk = lw->Wv = lw->Wo = NULL;
    } else {
      // Load FP32 attention weights (fallback)
      snprintf(k,sizeof(k),"model.layers.%d.self_attn.q_proj.weight",L);
      lw->Wq = (const float*)need(bf,k)->data;
      snprintf(k,sizeof(k),"model.layers.%d.self_attn.k_proj.weight",L);
      lw->Wk = (const float*)need(bf,k)->data;
      snprintf(k,sizeof(k),"model.layers.%d.self_attn.v_proj.weight",L);
      lw->Wv = (const float*)need(bf,k)->data;
      snprintf(k,sizeof(k),"model.layers.%d.self_attn.o_proj.weight",L);
      lw->Wo = (const float*)need(bf,k)->data;
    }

    snprintf(k,sizeof(k),"model.layers.%d.self_attn.q_proj.bias",L);
    lw->bq = (maybe(bf,k) && maybe(bf,k)->dtype==0) ? (const float*)maybe(bf,k)->data : NULL;
    snprintf(k,sizeof(k),"model.layers.%d.self_attn.k_proj.bias",L);
    lw->bk = (maybe(bf,k) && maybe(bf,k)->dtype==0) ? (const float*)maybe(bf,k)->data : NULL;
    snprintf(k,sizeof(k),"model.layers.%d.self_attn.v_proj.bias",L);
    lw->bv = (maybe(bf,k) && maybe(bf,k)->dtype==0) ? (const float*)maybe(bf,k)->data : NULL;
    snprintf(k,sizeof(k),"model.layers.%d.self_attn.o_proj.bias",L);
    lw->bo = (maybe(bf,k) && maybe(bf,k)->dtype==0) ? (const float*)maybe(bf,k)->data : NULL;

    // qk norm (auto-present for Qwen3 A3B)
    Tensor* qn = maybe(bf, (snprintf(k,sizeof(k),"model.layers.%d.self_attn.q_norm.weight",L), k));
    Tensor* kn = maybe(bf, (snprintf(k,sizeof(k),"model.layers.%d.self_attn.k_norm.weight",L), k));
    lw->q_norm = qn ? (const float*)qn->data : NULL;
    lw->k_norm = kn ? (const float*)kn->data : NULL;

    // norms
    Tensor* n1 = maybe(bf, (snprintf(k,sizeof(k),"model.layers.%d.input_layernorm.weight",L), k));
    if (!n1) { snprintf(k, sizeof(k), "model.layers.%d.rms_1.weight", L); n1 = need(bf, k); }
    lw->rms1_w = (const float*)n1->data;

    Tensor* n2 = maybe(bf, (snprintf(k,sizeof(k),"model.layers.%d.post_attention_layernorm.weight",L), k));
    if (!n2) { snprintf(k, sizeof(k), "model.layers.%d.rms_2.weight", L); n2 = need(bf, k); }
    lw->rms2_w = (const float*)n2->data;

    // router
    Tensor* RW = maybe(bf, (snprintf(k,sizeof(k),"model.layers.%d.mlp.gate.weight",L), k));
    if (!RW) { snprintf(k, sizeof(k), "model.layers.%d.mlp.router.gate.weight", L); RW = need(bf, k); }
    lw->router_w = (const float*)RW->data;

    Tensor* RB = maybe(bf, (snprintf(k,sizeof(k),"model.layers.%d.mlp.gate.bias",L), k));
    lw->router_b = (RB && RB->dtype==0) ? (const float*)RB->data : NULL;

    // Check if experts are quantized by looking for .q8/.q4 suffixes
    snprintf(k,sizeof(k),"model.layers.%d.mlp.experts.0.gate_proj.weight.q8",L);
    Tensor* test_q8 = maybe(bf, k);
    snprintf(k,sizeof(k),"model.layers.%d.mlp.experts.0.gate_proj.weight.q4",L);
    Tensor* test_q4 = maybe(bf, k);
    
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
          Tensor* q_data = need(bf,k);
          snprintf(k,sizeof(k),"model.layers.%d.mlp.experts.%d.gate_proj.weight.scale",L,e);
          Tensor* s_data = need(bf,k);
          lw->Wg_q[e].q = (const int8_t*)q_data->data;
          lw->Wg_q[e].s = (const float*)s_data->data;
          lw->Wg_q[e].zp = NULL; // Q8 doesn't use zero points
          lw->Wg_q[e].rows = q_data->shape[0];
          lw->Wg_q[e].cols = q_data->shape[1];
          lw->Wg_q[e].dtype = 2; // q8
          lw->Wg_q[e].group_size = s_data->group_size;
        } else {
          snprintf(k,sizeof(k),"model.layers.%d.mlp.experts.%d.gate_proj.weight.q4",L,e);
          Tensor* q_data = need(bf,k);
          snprintf(k,sizeof(k),"model.layers.%d.mlp.experts.%d.gate_proj.weight.scale",L,e);
          Tensor* s_data = need(bf,k);
          snprintf(k,sizeof(k),"model.layers.%d.mlp.experts.%d.gate_proj.weight.zero_point",L,e);
          Tensor* zp_data = bin_find(bf,k); // zero_point tensor may not exist in old models
          lw->Wg_q[e].q = (const int8_t*)q_data->data;
          lw->Wg_q[e].s = (const float*)s_data->data;
          lw->Wg_q[e].zp = zp_data ? (const float*)zp_data->data : NULL;
          lw->Wg_q[e].rows = q_data->shape[0];
          lw->Wg_q[e].cols = q_data->shape[1] * 2; // q4 packs 2 values per byte
          lw->Wg_q[e].dtype = 3; // q4
          lw->Wg_q[e].group_size = s_data->group_size;
        }
        
        // Up projection  
        if (test_q8) {
          snprintf(k,sizeof(k),"model.layers.%d.mlp.experts.%d.up_proj.weight.q8",L,e);
          Tensor* q_data = need(bf,k);
          snprintf(k,sizeof(k),"model.layers.%d.mlp.experts.%d.up_proj.weight.scale",L,e);
          Tensor* s_data = need(bf,k);
          lw->Wu_q[e].q = (const int8_t*)q_data->data;
          lw->Wu_q[e].s = (const float*)s_data->data;
          lw->Wu_q[e].zp = NULL; // Q8 doesn't use zero points
          lw->Wu_q[e].rows = q_data->shape[0];
          lw->Wu_q[e].cols = q_data->shape[1];
          lw->Wu_q[e].dtype = 2;
          lw->Wu_q[e].group_size = s_data->group_size;
        } else {
          snprintf(k,sizeof(k),"model.layers.%d.mlp.experts.%d.up_proj.weight.q4",L,e);
          Tensor* q_data = need(bf,k);
          snprintf(k,sizeof(k),"model.layers.%d.mlp.experts.%d.up_proj.weight.scale",L,e);
          Tensor* s_data = need(bf,k);
          snprintf(k,sizeof(k),"model.layers.%d.mlp.experts.%d.up_proj.weight.zero_point",L,e);
          Tensor* zp_data = bin_find(bf,k); // zero_point tensor may not exist in old models
          lw->Wu_q[e].q = (const int8_t*)q_data->data;
          lw->Wu_q[e].s = (const float*)s_data->data;
          lw->Wu_q[e].zp = zp_data ? (const float*)zp_data->data : NULL;
          lw->Wu_q[e].rows = q_data->shape[0];
          lw->Wu_q[e].cols = q_data->shape[1] * 2;
          lw->Wu_q[e].dtype = 3;
          lw->Wu_q[e].group_size = s_data->group_size;
        }
        
        // Down projection
        if (test_q8) {
          snprintf(k,sizeof(k),"model.layers.%d.mlp.experts.%d.down_proj.weight.q8",L,e);
          Tensor* q_data = need(bf,k);
          snprintf(k,sizeof(k),"model.layers.%d.mlp.experts.%d.down_proj.weight.scale",L,e);
          Tensor* s_data = need(bf,k);
          lw->Wd_q[e].q = (const int8_t*)q_data->data;
          lw->Wd_q[e].s = (const float*)s_data->data;
          lw->Wd_q[e].zp = NULL; // Q8 doesn't use zero points
          lw->Wd_q[e].rows = q_data->shape[0];
          lw->Wd_q[e].cols = q_data->shape[1];
          lw->Wd_q[e].dtype = 2;
          lw->Wd_q[e].group_size = s_data->group_size;
        } else {
          snprintf(k,sizeof(k),"model.layers.%d.mlp.experts.%d.down_proj.weight.q4",L,e);
          Tensor* q_data = need(bf,k);
          snprintf(k,sizeof(k),"model.layers.%d.mlp.experts.%d.down_proj.weight.scale",L,e);
          Tensor* s_data = need(bf,k);
          snprintf(k,sizeof(k),"model.layers.%d.mlp.experts.%d.down_proj.weight.zero_point",L,e);
          Tensor* zp_data = bin_find(bf,k); // zero_point tensor may not exist in old models
          lw->Wd_q[e].q = (const int8_t*)q_data->data;
          lw->Wd_q[e].s = (const float*)s_data->data;
          lw->Wd_q[e].zp = zp_data ? (const float*)zp_data->data : NULL;
          lw->Wd_q[e].rows = q_data->shape[0];
          lw->Wd_q[e].cols = q_data->shape[1] * 2;
          lw->Wd_q[e].dtype = 3;
          lw->Wd_q[e].group_size = s_data->group_size;
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
  
  // Load quantized lm_head if it exists (always Q8, never Q4)
  Tensor* lm_head_q8 = bin_find(bf, "lm_head.weight.q8");
  if (lm_head_q8) {
    // Load quantized Q8 lm_head
    Tensor* lm_head_scales = need(bf, "lm_head.weight.scale");
    w->lm_head_q = (QuantizedWeight*)malloc(sizeof(QuantizedWeight));
    w->lm_head_q->q = (const int8_t*)lm_head_q8->data;
    w->lm_head_q->s = (const float*)lm_head_scales->data;
    w->lm_head_q->zp = NULL;  // Q8 doesn't use zero points
    w->lm_head_q->rows = lm_head_q8->shape[0];
    w->lm_head_q->cols = lm_head_q8->shape[1];
    w->lm_head_q->dtype = 2;  // Q8
    w->lm_head_q->group_size = lm_head_scales->group_size;
    w->lm_head_quantized = true;
    w->lm_head = NULL;  // Clear FP32 version to save memory
  }
}
#endif // MODEL_H
