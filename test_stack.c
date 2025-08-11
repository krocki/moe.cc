// test_stack.c
// Runs the first N Qwen3-MoE decoder layers in C and compares to Py goldens.
// Usage: ./test_stack <all.bin> <x.npy> <y.npy> <N_layers> <rope_theta>

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
    fprintf(stderr, "Usage: %s <weights.bin> <x_in.npy> <y_out.npy> <N_layers> <rope_theta>\n", argv[0]);
    return 1;
  }
  const char* wpath    = argv[1];
  const char* xin_npy  = argv[2];
  const char* yout_npy = argv[3];
  int N                = atoi(argv[4]);
  float rope_theta     = strtof(argv[5], NULL);

  fprintf(stderr, "[load] reading weights: %s\n", wpath);
  BinFile* bin = bin_load(wpath);
  if(!bin){ fprintf(stderr,"bin load fail\n"); return 1; }

  NpyArray* xin  = npy_load_float32(xin_npy);
  NpyArray* yref = npy_load_float32(yout_npy);
  int T = xin->shape[0];
  int d_model = xin->shape[1];
  if (yref->shape[0]!=T || yref->shape[1]!=d_model){ fprintf(stderr,"y shape mismatch\n"); return 1; }

  // infer heads from L0 q_norm
  TensorBin *tWq0=NULL,*tWk0=NULL,*tQn0=NULL;
  need(bin, "model.layers.0.self_attn.q_proj.weight", &tWq0);
  need(bin, "model.layers.0.self_attn.k_proj.weight", &tWk0);
  need(bin, "model.layers.0.self_attn.q_norm.weight", &tQn0);

  int Dq0      = tWq0->shape[0];
  int Dk0      = tWk0->shape[0];
  int head_dim = tQn0->shape[0];       // Qwen3: q_norm length == head_dim
  int n_q0     = Dq0 / head_dim;
  int n_kv0    = Dk0 / head_dim;

  // scratch sizing: match model.c / test_layer.c
  size_t attn_f  = (size_t)T*(Dq0 + 2*Dk0) + (size_t)T*T + (size_t)T*Dq0;
  size_t temps_f = 5ull * (size_t)T * d_model;
  float* scratch_attn = (float*)malloc(sizeof(float)*(attn_f + temps_f));
  // d_ff won’t change across layers for this model; 768 for 30B-A3B
  const int d_ff = 768;
  float* scratch_moe  = (float*)malloc(sizeof(float)*(2ull*(size_t)T*d_ff));
  int*   top_idx      = (int*)  malloc(sizeof(int)*T*8);
  float* top_p        = (float*)malloc(sizeof(float)*T*8);

  float* x = xin->data;

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
    int Dkv = tWk->shape[0]; // rows of K/V
    int hdim= tQn->shape[0]; // head_dim for this layer
    int n_q = Dq  / hdim;
    int n_kv= Dkv / hdim;
    fprintf(stderr, "[stack] L=%d Dq=%d Dk=%d Dv=%d n_q=%d n_kv=%d head_dim=%d (group=%d)\n",
            L, Dq, tWk->shape[0], tWv->shape[0], n_q, n_kv, hdim, n_q/n_kv);
    if (tWk->shape[0] != n_kv*hdim || tWv->shape[0] != n_kv*hdim){
      fprintf(stderr,"L%d: K/V shape mismatch\n", L); return 1;
    }

    // Router gate (required)
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

    // In-place: x is overwritten each layer
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

  float mad = max_abs_diff(x, yref->data, T*d_model);
  printf("Max abs diff: %.6g\n%s\n", mad, (mad<1e-4f?"PASS":"FAIL"));

  free(scratch_attn); free(scratch_moe); free(top_idx); free(top_p);
  npy_free(xin); npy_free(yref); bin_free(bin);
  return (mad<1e-5f)?0:1;
}
