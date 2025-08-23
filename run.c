/**
 * Qwen3-30B-A3B Mixture of Experts Language Model Runner
 *
 * This is a clean, standalone implementation for running Qwen3-30B-A3B inference
 * following the llama2.c pattern. It supports FP32 weights only
 * with memory-mapped or explicit loading.
 *
 * Model Architecture (from config.json):
 * - hidden_size (d_model): 2048 - Main model dimension
 * - num_attention_heads: 32 - Query heads for attention
 * - num_key_value_heads: 4 - Key/Value heads (GQA - Grouped Query Attention)
 * - head_dim: 128 - Dimension per attention head (2048/32 = 64, but config says 128)
 * - num_hidden_layers: 48 - Number of transformer layers
 * - vocab_size: 151936 - Vocabulary size
 * - num_experts: 128 - Total experts per MoE layer
 * - num_experts_per_tok: 8 - Top-k experts selected per token (top_k)
 * - moe_intermediate_size: 768 - Hidden dimension inside each expert (d_ff)
 * - rope_theta: 1000000.0 - RoPE (Rotary Position Embedding) base frequency
 * - rms_norm_eps: 1e-06 - RMS normalization epsilon
 *
 * Usage:
 *   ./run <model.bin> <steps> [outbase]
 *
 *   model.bin: Binary model file (FP32 only)
 *   steps: Number of inference steps to run
 *   outbase: Optional - base path for reference files (.ids.npy, .logits.npy, .probs.npy)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <getopt.h>

#include "model.h"
#include "io.h"
#include "io_mmap.h"
#include "utils.h"
#include "kernels.h"
#include "debug_utils.h"
#include "tokenizer.h"

// ----------------------------------------------------------------------------
// QwenStates: Pre-allocated intermediate computation buffers
// This struct contains all temporary buffers needed for model inference,
// allocated once in main() and reused across all forward passes
// ----------------------------------------------------------------------------

typedef struct {
  // Main activation buffer - stores the current token embeddings as they
  // flow through the model. Shape: [seq_len, d_model] = [T, 2048]
  float* x;

  // Post-normalization buffer - stores activations after final RMS normalization
  // Shape: [seq_len, d_model] = [T, 2048]
  float* x_final;

  // Attention computation scratch space - used for storing intermediate
  // attention results during multi-head attention computation
  // Size: max(query_proj, key_proj, value_proj, output_proj)
  // Shape: [seq_len, n_q * head_dim] = [T, 32 * 128] = [T, 4096]
  float* scratch_attn;

  // MoE (Mixture of Experts) scratch space - used for expert computations
  // Must accommodate the largest expert computation: gate_proj, up_proj outputs
  // Shape: [seq_len, max_expert_dim] = [T, 768] (moe_intermediate_size)
  float* scratch_moe;

  // Top-k expert indices - stores which experts are selected for each token
  // Shape: [seq_len, top_k] = [T, 8] (num_experts_per_tok)
  int* expert_indices;

  // Top-k expert weights - stores the routing weights for selected experts
  // Shape: [seq_len, top_k] = [T, 8]
  float* expert_weights;

  // Output logits buffer - final model outputs before softmax
  // Shape: [vocab_size] = [151936] - only for current token
  float* logits;

  // K/V Cache for efficient autoregressive generation
  // Caches key and value matrices across all layers and positions
  // Shape: [n_layers, max_seq_len, n_kv * head_dim]
  float* k_cache;  // Key cache
  float* v_cache;  // Value cache
  
  // Maximum sequence length used for cache allocation
  int max_seq_len;

} QwenStates;

// ----------------------------------------------------------------------------
// Memory allocation and deallocation for QwenStates
// ----------------------------------------------------------------------------

/**
 * Allocate all intermediate computation buffers for Qwen model inference
 *
 * @param s: QwenStates struct to initialize
 * @param cfg: Model configuration containing dimensions
 * @param max_seq_len: Maximum sequence length (T) - determines buffer sizes
 */
void malloc_qwen_states(QwenStates* s, QwenConfig* cfg, int max_seq_len) {
  int d_model = cfg->d_model;           // 2048 - main model dimension
  int n_q = cfg->n_q;                   // 32 - number of query heads
  int n_kv = cfg->n_kv;                 // 4 - number of key/value heads
  int head_dim = cfg->head_dim;         // 128 - dimension per head
  int d_ff = cfg->d_ff;                 // 768 - expert hidden dimension
  int top_k = cfg->top_k;               // 8 - experts per token
  int vocab_size = cfg->vocab_size;     // 151936 - vocabulary size

  // Main activation buffer: [max_seq_len, d_model]
  s->x = (float*)calloc(max_seq_len * d_model, sizeof(float));

  // Final normalization buffer: [max_seq_len, d_model]
  s->x_final = (float*)calloc(max_seq_len * d_model, sizeof(float));

  // Attention scratch: Q[T,Dq] + K[T,Dkv] + V[T,Dkv] + S[T,T] + Hcat[T,Dq] + temp buffers
  // For reference: T*Dq + T*Dkv + T*Dkv + T*T + T*Dq + additional temp space
  int total_q_dim = n_q * head_dim;      // 32 * 128 = 4096
  int total_kv_dim = n_kv * head_dim;    // 4 * 128 = 512
  size_t attn_scratch_size = max_seq_len * (2*total_q_dim + 2*total_kv_dim + max_seq_len + 4*d_model) + max_seq_len * d_model;  // +x_orig buffer
  s->scratch_attn = (float*)calloc(attn_scratch_size, sizeof(float));

  // MoE scratch: Need space for expert computations, output buffer, and residual buffers
  // Layout: [max_seq_len * d_model] output + [2 * d_ff + d_model] expert workspace + [max_seq_len * d_model] x_before_moe
  size_t moe_output_size = max_seq_len * d_model;  // Output buffer
  size_t expert_work_size = 2 * d_ff + d_model;  // gate_out + up_out + expert_out
  size_t x_before_moe_size = max_seq_len * d_model;  // Buffer for MoE residual connection
  s->scratch_moe = (float*)calloc(moe_output_size + expert_work_size + x_before_moe_size, sizeof(float));

  // Expert selection buffers: [max_seq_len, top_k]
  s->expert_indices = (int*)calloc(max_seq_len * top_k, sizeof(int));
  s->expert_weights = (float*)calloc(max_seq_len * top_k, sizeof(float));

  // Output logits: [vocab_size] - only current token
  s->logits = (float*)calloc(vocab_size, sizeof(float));

  // K/V Cache: [n_layers, max_seq_len, n_kv * head_dim]
  int n_layers = cfg->n_layers;         // 48 - number of transformer layers
  size_t cache_size = n_layers * max_seq_len * total_kv_dim;
  s->k_cache = (float*)calloc(cache_size, sizeof(float));
  s->v_cache = (float*)calloc(cache_size, sizeof(float));
  
  // Store max_seq_len for cache indexing
  s->max_seq_len = max_seq_len;
}

/**
 * Free all allocated QwenStates buffers
 */
void free_qwen_states(QwenStates* s) {
  free(s->x);
  free(s->x_final);
  free(s->scratch_attn);
  free(s->scratch_moe);
  free(s->expert_indices);
  free(s->expert_weights);
  free(s->logits);
  free(s->k_cache);
  free(s->v_cache);
}

// ----------------------------------------------------------------------------
// Basic matrix multiplication operations
// ----------------------------------------------------------------------------

/**
 * Basic FP32 matrix multiplication: C = A × B^T
 * A: [M, K], B: [N, K] (stored row-major), C: [M, N]
 * This matches the exact implementation from test_model_trace.c
 *
 * @param A: Input matrix [M, K]
 * @param B: Weight matrix [N, K]
 * @param C: Output matrix [M, N]
 * @param M: Number of rows in A and C
 * @param N: Number of rows in B and columns in C
 * @param K: Number of columns in A and B
 */
void matmul(const float* A, const float* B, float* C, int M, int N, int K) {
  for (int m = 0; m < M; ++m) {
    const float* a = A + m*K;
    float* c = C + m*N;
    for (int n = 0; n < N; ++n) {
      const float* b = B + n*K;
      float acc = 0.f;
      for (int k = 0; k < K; ++k) acc += a[k] * b[k];
      c[n] = acc;
    }
  }
}

/**
 * Matrix multiplication with transposed second matrix: C = A × B^T
 * A: [M, K], B: [N, K] -> conceptually B^T: [K, N], C: [M, N]
 * Used for output projection with tied embeddings
 */
void matmul_transposed(const float* A, const float* B, float* C, int M, int N, int K) {
  for (int m = 0; m < M; ++m) {
    const float* a = A + m*K;
    float* c = C + m*N;
    for (int n = 0; n < N; ++n) {
      const float* b = B + n*K;
      float acc = 0.f;
      for (int k = 0; k < K; ++k) acc += a[k] * b[k];
      c[n] = acc;
    }
  }
}


// ----------------------------------------------------------------------------
// Core neural network operations
// ----------------------------------------------------------------------------

/**
 * RMS (Root Mean Square) Normalization
 * Normalizes input using RMS instead of LayerNorm (no mean centering)
 * Formula: x_norm = x / sqrt(mean(x²) + eps) * weight
 *
 * @param out: Output buffer [size]
 * @param x: Input activations [size]
 * @param weight: Learned scale parameters [size]
 * @param size: Dimension to normalize (typically d_model=2048)
 * @param eps: Small constant for numerical stability (1e-6)
 */
void rmsnorm(float* out, const float* x, const float* weight, int T, int d_model, float eps) {
  for (int t = 0; t < T; t++) {
    const float* xt = x + (size_t)t * d_model;
    float* yt = out + (size_t)t * d_model;

    float ss = 0.0f;
    for (int i = 0; i < d_model; i++) {
      ss += xt[i] * xt[i];
    }
    ss = 1.0f / sqrtf(ss / d_model + eps);

    for (int i = 0; i < d_model; i++) {
      yt[i] = xt[i] * ss * (weight ? weight[i] : 1.0f);
    }
  }
}

/**
 * SiLU (Sigmoid Linear Unit) activation function
 * Formula: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
 * Used in the expert feed-forward networks
 *
 * @param x: Input/output buffer to apply activation in-place [size]
 * @param size: Number of elements
 */
void silu(float* x, int size) {
  for (int i = 0; i < size; i++) {
    x[i] = x[i] / (1.0f + expf(-x[i]));
  }
}

/**
 * Softmax function for probability normalization
 * Converts logits to probabilities: p_i = exp(x_i - max) / sum(exp(x_j - max))
 * Uses max subtraction for numerical stability
 *
 * @param x: Input logits, output probabilities [size]
 * @param size: Number of elements (typically vocab_size=151936)
 */
void softmax(float* x, int size) {
  // Find maximum for numerical stability
  float max_val = x[0];
  for (int i = 1; i < size; i++) {
    if (x[i] > max_val) max_val = x[i];
  }

  // Compute exp and sum
  float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }

  // Normalize to probabilities
  for (int i = 0; i < size; i++) {
    x[i] /= sum;
  }
}

/**
 * Top-k selection for expert routing
 * Selects the k experts with highest routing weights for each token
 *
 * @param values: Input router logits [num_experts=128], modified in-place to weights
 * @param indices: Output expert indices [top_k=8]
 * @param k: Number of experts to select (8)
 * @param n: Total number of experts (128)
 */
void topk(float* values, int* indices, int k, int n) {
  // Simple selection sort for top-k (efficient for small k=8)
  for (int i = 0; i < k; i++) {
    int max_idx = i;
    for (int j = i + 1; j < n; j++) {
      if (values[j] > values[max_idx]) {
        max_idx = j;
      }
    }
    // Swap values and track index
    float temp_val = values[i];
    values[i] = values[max_idx];
    values[max_idx] = temp_val;
    indices[i] = max_idx;
  }

  // Softmax normalization over top-k values (like reference)
  // Find max for numerical stability
  float maxv = values[0];
  for (int i = 1; i < k; i++) {
    if (values[i] > maxv) maxv = values[i];
  }

  // Compute exp(val - max) and sum
  float sum = 0.0f;
  for (int i = 0; i < k; i++) {
    values[i] = expf(values[i] - maxv);
    sum += values[i];
  }

  // Normalize
  float inv = 1.0f / (sum + 1e-9f);
  for (int i = 0; i < k; i++) {
    values[i] *= inv;
  }
}

// ----------------------------------------------------------------------------
// RoPE: Rotary Position Embedding
// Applies rotary embeddings to query and key vectors for position encoding
// ----------------------------------------------------------------------------

/**
 * Apply RoPE (Rotary Position Embedding) to query or key vectors
 * RoPE encodes position by rotating vector pairs in 2D planes
 *
 * @param q: Query or key vectors [head_dim=128] - modified in-place
 * @param pos: Position index of current token
 * @param head_dim: Dimension per attention head (128)
 * @param theta: Base frequency (1000000.0 from config)
 */
void rope(float* q, int pos, int head_dim, float theta) {
  for (int i = 0; i < head_dim; i += 2) {
    float freq = 1.0f / powf(theta, (float)i / (float)head_dim);
    float val = pos * freq;
    float fcr = cosf(val);  // cos component
    float fci = sinf(val);  // sin component

    // Rotate the pair (q[i], q[i+1])
    float q0 = q[i];
    float q1 = q[i + 1];
    q[i] = q0 * fcr - q1 * fci;
    q[i + 1] = q0 * fci + q1 * fcr;
  }
}

// Helper function for rope pair rotation
static inline void rope_rotate_pair(float* even, float* odd, float c, float s) {
  float e = *even, o = *odd;
  *even = e * c - o * s;
  *odd  = o * c + e * s;
}

// Apply RoPE to Q and K tensors with GQA awareness
void rope_apply_inplace_gqa(float* Q, float* K, int T, int n_q, int n_kv, int head_dim, int pos0, float theta) {
  const int Dq = n_q * head_dim;
  const int Dkv = n_kv * head_dim;
  const int d2 = head_dim / 2;

  if ((head_dim & 1) != 0) {
    fprintf(stderr, "[rope] head_dim must be even, got %d\n", head_dim);
    exit(1);
  }

  // Precompute inverse frequencies
  float* inv = (float*)malloc(sizeof(float) * (size_t)d2);
  for (int i = 0; i < d2; i++) {
    float exponent = -2.0f * (float)i / (float)head_dim;
    inv[i] = powf(theta, exponent);
  }

  // Apply RoPE to each token
  for (int t = 0; t < T; t++) {
    const float p = (float)(pos0 + t);

    // Apply to query heads
    for (int h = 0; h < n_q; h++) {
      float* qh = &Q[(size_t)t * Dq + (size_t)h * head_dim];
      for (int i = 0; i < d2; i++) {
        float ang = p * inv[i];
        float c = cosf(ang), s = sinf(ang);
        rope_rotate_pair(&qh[i], &qh[i + d2], c, s);
      }
    }

    // Apply to key heads
    for (int h = 0; h < n_kv; h++) {
      float* kh = &K[(size_t)t * Dkv + (size_t)h * head_dim];
      for (int i = 0; i < d2; i++) {
        float ang = p * inv[i];
        float c = cosf(ang), s = sinf(ang);
        rope_rotate_pair(&kh[i], &kh[i + d2], c, s);
      }
    }
  }

  free(inv);
}

// ----------------------------------------------------------------------------
// Multi-Head Attention with GQA (Grouped Query Attention)
// ----------------------------------------------------------------------------

/**
 * Multi-head attention layer with Grouped Query Attention (GQA)
 *
 * GQA reduces memory by sharing key/value heads across multiple query heads:
 * - Query heads (n_q): 32 heads, each head_dim=128 -> total 4096 dims
 * - Key/Value heads (n_kv): 4 heads, each head_dim=128 -> total 512 dims
 * - Each KV head is shared by n_q/n_kv = 32/4 = 8 query heads
 *
 * @param out: Output after attention [batch_size, d_model=2048]
 * @param x: Input activations [batch_size, d_model=2048]
 * @param layer_weights: Attention weight matrices and norms
 * @param cfg: Model configuration
 * @param s: Pre-allocated scratch buffers (sized for max_seq_len)
 * @param batch_size: Number of tokens being processed (1=autoregressive, N=prefill)
 * @param pos: Current position in sequence (for RoPE and caching)
 */
void attention(float* out, float* x, QwenLayerWeights* layer_weights,
    QwenConfig* cfg, QwenStates* s, int batch_size, int pos, int layer_idx) {

  int d_model = cfg->d_model;      // 2048
  int n_q = cfg->n_q;              // 32 - query heads
  int n_kv = cfg->n_kv;            // 4 - key/value heads
  int head_dim = cfg->head_dim;    // 128
  int total_q_dim = n_q * head_dim;    // 32 * 128 = 4096
  int total_kv_dim = n_kv * head_dim;  // 4 * 128 = 512
  float scale = 1.0f / sqrtf((float)head_dim);

  // Debug: check input x at start of attention

  // K/V Cache optimization: compute cache pointers for this layer  
  size_t cache_layer_offset = (size_t)layer_idx * s->max_seq_len * total_kv_dim;
  float* layer_k_cache = s->k_cache + cache_layer_offset;
  float* layer_v_cache = s->v_cache + cache_layer_offset;

  // Scratch buffer layout: Q [batch, Dq] + K [batch, Dkv] + V [batch, Dkv] + S [batch, context] + Hcat [batch, Dq]
  float* Q    = s->scratch_attn;                                         // [batch_size, total_q_dim]
  int context_len = pos + batch_size;  // Total context length including current tokens
  float* K_temp = Q + (size_t)batch_size * total_q_dim;                 // [batch_size, total_kv_dim] (temp for prefill)
  float* V_temp = K_temp + (size_t)batch_size * total_kv_dim;           // [batch_size, total_kv_dim] (temp for prefill)
  float* S    = V_temp + (size_t)batch_size * total_kv_dim;             // [batch_size, context_len]
  float* Hcat = S + (size_t)batch_size * context_len;                   // [batch_size, total_q_dim]

  // Project input to Q (always needed)
  matmul(x, (float*)layer_weights->Wq, Q, batch_size, total_q_dim, d_model);
  
  // Add Q bias if present
  if (layer_weights->bq) {
    for (int t = 0; t < batch_size; t++) {
      for (int i = 0; i < total_q_dim; i++) {
        Q[(size_t)t * total_q_dim + i] += ((float*)layer_weights->bq)[i];
      }
    }
  }

  // Declare K,V pointers for attention computation
  float* K;
  float* V;

  // For K, V: only compute new positions, use cache for previous ones
  if (batch_size == 1) {
    // Autoregressive mode: compute K,V for current token only
    float* current_k = K_temp;  // Use temp buffer for current computation
    float* current_v = V_temp;
    
    matmul(x, (float*)layer_weights->Wk, current_k, 1, total_kv_dim, d_model);
    matmul(x, (float*)layer_weights->Wv, current_v, 1, total_kv_dim, d_model);
    
    // Add biases to current K,V if present
    if (layer_weights->bk) {
      for (int i = 0; i < total_kv_dim; i++) {
        current_k[i] += ((float*)layer_weights->bk)[i];
      }
    }
    if (layer_weights->bv) {
      for (int i = 0; i < total_kv_dim; i++) {
        current_v[i] += ((float*)layer_weights->bv)[i];
      }
    }
    
    // Apply K normalization to current K if present
    if (layer_weights->k_norm) {
      for (int h = 0; h < n_kv; h++) {
        float* v = &current_k[(size_t)h * head_dim];
        float msq = 0.f;
        for (int d = 0; d < head_dim; d++) {
          float z = v[d];
          msq += z * z;
        }
        float inv = 1.0f / sqrtf(msq / (float)head_dim + cfg->rms_eps);
        for (int d = 0; d < head_dim; d++) {
          v[d] = (v[d] * inv) * ((float*)layer_weights->k_norm)[d];
        }
      }
    }
    
    // Store current K,V in cache after processing
    float* cache_k_pos = layer_k_cache + (size_t)pos * total_kv_dim;
    float* cache_v_pos = layer_v_cache + (size_t)pos * total_kv_dim;
    memcpy(cache_k_pos, current_k, total_kv_dim * sizeof(float));
    memcpy(cache_v_pos, current_v, total_kv_dim * sizeof(float));
    
    // Set K,V pointers to use cache directly (eliminating copies!)
    K = layer_k_cache;  // Points to start of cached K for this layer
    V = layer_v_cache;  // Points to start of cached V for this layer
  } else {
    // Prefill mode: compute K,V for entire batch, populate cache
    K = K_temp;  // Use temporary buffers
    V = V_temp;
    
    matmul(x, (float*)layer_weights->Wk, K, batch_size, total_kv_dim, d_model);
    matmul(x, (float*)layer_weights->Wv, V, batch_size, total_kv_dim, d_model);
    
    // Add biases if present
    if (layer_weights->bk) {
      for (int t = 0; t < batch_size; t++) {
        for (int i = 0; i < total_kv_dim; i++) {
          K[(size_t)t * total_kv_dim + i] += ((float*)layer_weights->bk)[i];
        }
      }
    }
    if (layer_weights->bv) {
      for (int t = 0; t < batch_size; t++) {
        for (int i = 0; i < total_kv_dim; i++) {
          V[(size_t)t * total_kv_dim + i] += ((float*)layer_weights->bv)[i];
        }
      }
    }
    
    // Apply K normalization if present
    if (layer_weights->k_norm) {
      for (int t = 0; t < batch_size; t++) {
        float* Kt = &K[(size_t)t * total_kv_dim];
        for (int h = 0; h < n_kv; h++) {
          float* v = &Kt[(size_t)h * head_dim];
          float msq = 0.f;
          for (int d = 0; d < head_dim; d++) {
            float z = v[d];
            msq += z * z;
          }
          float inv = 1.0f / sqrtf(msq / (float)head_dim + cfg->rms_eps);
          for (int d = 0; d < head_dim; d++) {
            v[d] = (v[d] * inv) * ((float*)layer_weights->k_norm)[d];
          }
        }
      }
    }
    
    // Copy K,V to cache for future use
    memcpy(layer_k_cache + (size_t)pos * total_kv_dim, K, batch_size * total_kv_dim * sizeof(float));
    memcpy(layer_v_cache + (size_t)pos * total_kv_dim, V, batch_size * total_kv_dim * sizeof(float));
  }

  // Debug: check Q, K, V after matmuls

  // Q RMSNorm per head (Qwen3 specific) 
  if (layer_weights->q_norm) {
    for (int t = 0; t < batch_size; t++) {
      float* Qt = &Q[(size_t)t * total_q_dim];
      for (int h = 0; h < n_q; h++) {
        float* v = &Qt[(size_t)h * head_dim];
        float msq = 0.f;
        for (int d = 0; d < head_dim; d++) {
          float z = v[d];
          msq += z * z;
        }
        float inv = 1.0f / sqrtf(msq / (float)head_dim + cfg->rms_eps);
        for (int d = 0; d < head_dim; d++) {
          v[d] = (v[d] * inv) * ((float*)layer_weights->q_norm)[d];
        }
      }
    }
  }

  // Debug: check Q, K after normalization

  // Apply RoPE to Q and K
  if (batch_size == 1) {
    // Autoregressive mode: Apply RoPE to current Q and stored K at current position
    float* current_k = layer_k_cache + (size_t)pos * total_kv_dim;
    rope_apply_inplace_gqa(Q, current_k, 1, n_q, n_kv, head_dim, pos, cfg->rope_theta);
  } else {
    // Prefill mode: Apply RoPE to full batch
    rope_apply_inplace_gqa(Q, K, batch_size, n_q, n_kv, head_dim, pos, cfg->rope_theta);
  }

  // Grouped Query Attention computation
  int group_size = n_q / n_kv;  // 32/4 = 8 query heads per KV head

  for (int h = 0; h < n_q; h++) {
    int kv_head = h / group_size;

    for (int tq = 0; tq < batch_size; tq++) {
      const float* q_vec = &Q[(size_t)tq * total_q_dim + (size_t)h * head_dim];
      float* S_row = &S[(size_t)tq * context_len];

      // Compute attention scores for all key positions in context
      for (int tk = 0; tk < context_len; tk++) {
        const float* k_vec = &K[(size_t)tk * total_kv_dim + (size_t)kv_head * head_dim];
        float dot = 0.f;
        for (int d = 0; d < head_dim; d++) {
          dot += q_vec[d] * k_vec[d];
        }
        S_row[tk] = dot * scale;
      }

      // Apply causal masking (no looking at future tokens)
      int current_pos = pos + tq;  // Absolute position of this query token
      for (int tk = current_pos + 1; tk < context_len; tk++) {
        S[(size_t)tq * context_len + tk] = -INFINITY;
      }

      // Softmax over the row
      float max_val = S_row[0];
      for (int i = 1; i < context_len; i++) {
        if (S_row[i] > max_val) max_val = S_row[i];
      }
      float sum = 0.f;
      for (int i = 0; i < context_len; i++) {
        float e = expf(S_row[i] - max_val);
        S_row[i] = e;
        sum += e;
      }
      float inv = 1.f / (sum + 1e-9f);
      for (int i = 0; i < context_len; i++) {
        S_row[i] *= inv;
      }
    }

    // Apply attention weights to values
    for (int tq = 0; tq < batch_size; tq++) {
      const float* P_row = &S[(size_t)tq * context_len];
      float* head_out = &Hcat[(size_t)tq * total_q_dim + (size_t)h * head_dim];

      // Initialize output for this head
      for (int d = 0; d < head_dim; d++) head_out[d] = 0.f;

      // Accumulate weighted values over full context
      for (int tk = 0; tk < context_len; tk++) {
        const float* v_vec = &V[(size_t)tk * total_kv_dim + (size_t)kv_head * head_dim];
        const float p = P_row[tk];
        for (int d = 0; d < head_dim; d++) {
          head_out[d] += p * v_vec[d];
        }
      }
    }
  }

  // Debug: check Hcat before final matmul

  // Output projection: concatenated heads -> d_model
  matmul(Hcat, (float*)layer_weights->Wo, out, batch_size, d_model, total_q_dim);

  // Add output bias if present
  if (layer_weights->bo) {
    for (int t = 0; t < batch_size; t++) {
      for (int i = 0; i < d_model; i++) {
        out[(size_t)t * d_model + i] += ((float*)layer_weights->bo)[i];
      }
    }
  }

  // Debug: check input x at end of attention
}

// ----------------------------------------------------------------------------
// Mixture of Experts (MoE) Layer
// ----------------------------------------------------------------------------

/**
 * Mixture of Experts feed-forward layer
 * Routes each token to top-k=8 experts out of 128 total experts
 * Each expert is a 2-layer MLP: gate_proj, up_proj -> SiLU -> down_proj
 *
 * @param out: Output activations [batch_size, d_model=2048]
 * @param x: Input activations [batch_size, d_model=2048]
 * @param layer_weights: MoE weights (router + experts)
 * @param cfg: Model configuration
 * @param s: Pre-allocated scratch buffers (sized for max_seq_len)
 * @param batch_size: Number of tokens being processed
 */
void moe_layer(float* out, float* x, QwenLayerWeights* layer_weights,
    QwenConfig* cfg, QwenStates* s, int batch_size, int layer_idx) {

  int d_model = cfg->d_model;        // 2048
  int d_ff = cfg->d_ff;              // 768 - expert hidden dimension
  int n_experts = cfg->n_experts;    // 128 - total experts
  int top_k = cfg->top_k;            // 8 - experts per token

  // Buffer layout: [batch_size * d_model] output + expert workspace
  size_t moe_output_size = batch_size * d_model;
  float* expert_workspace = s->scratch_moe + moe_output_size;

  // Clear output buffer
  memset(out, 0, batch_size * d_model * sizeof(float));

  // Process each token in batch
  for (int t = 0; t < batch_size; t++) {
    float* token_x = x + t * d_model;           // Input for this token [d_model]
    float* token_out = out + t * d_model;       // Output for this token [d_model]

    // Router: compute routing weights for all experts
    // router_w: [n_experts, d_model] = [128, 2048]
    float router_logits[128];  // Stack allocation for small array
    matmul(token_x, layer_weights->router_w, router_logits, 1, n_experts, d_model);

    // Add router bias if present
    if (layer_weights->router_b) {
      for (int e = 0; e < n_experts; e++) {
        router_logits[e] += layer_weights->router_b[e];
      }
    }

    // Debug: print router logits for layer 0, token 0

    // Select top-k experts and get normalized weights
    int* selected_experts = s->expert_indices + t * top_k;      // [top_k]
    float* expert_weights = s->expert_weights + t * top_k;      // [top_k]

    // Apply top-k selection directly on router_logits, then copy selected weights
    topk(router_logits, selected_experts, top_k, n_experts);
    for (int k = 0; k < top_k; k++) {
      expert_weights[k] = router_logits[k];
    }

    // Debug: print selected experts and weights for layer 0, token 0

    // Process each selected expert
    for (int k = 0; k < top_k; k++) {
      int expert_id = selected_experts[k];
      float expert_weight = expert_weights[k];

      // Expert computation: 2-layer MLP with SiLU activation
      // gate_proj: [d_model] -> [d_ff], up_proj: [d_model] -> [d_ff]
      // down_proj: [d_ff] -> [d_model]

      float* gate_out = expert_workspace;                     // [d_ff]
      float* up_out = expert_workspace + d_ff;             // [d_ff]
      float* expert_out = expert_workspace + 2 * d_ff;     // [d_model]

      // FP32 expert weights
      const float* Wg = layer_weights->Wg[expert_id];  // [d_ff, d_model]
      const float* Wu = layer_weights->Wu[expert_id];  // [d_ff, d_model]
      const float* Wd = layer_weights->Wd[expert_id];  // [d_model, d_ff]

      // Gate projection: x [d_model] * Wg^T [d_model, d_ff] -> [d_ff]
      matmul(token_x, Wg, gate_out, 1, d_ff, d_model);

      // Up projection: x [d_model] * Wu^T [d_model, d_ff] -> [d_ff]
      matmul(token_x, Wu, up_out, 1, d_ff, d_model);

      // Apply SiLU activation: gate_out = silu(gate_out) * up_out
      silu(gate_out, d_ff);
      for (int i = 0; i < d_ff; i++) {
        gate_out[i] *= up_out[i];
      }

      // Down projection: gate_out [d_ff] * Wd^T [d_ff, d_model] -> [d_model]
      matmul(gate_out, Wd, expert_out, 1, d_model, d_ff);

      // Debug: print first expert output for layer 0

      // Add weighted expert output to final result
      for (int i = 0; i < d_model; i++) {
        token_out[i] += expert_weight * expert_out[i];
      }
    }
  }
}

// ----------------------------------------------------------------------------
// Complete Transformer Layer
// ----------------------------------------------------------------------------

/**
 * Single transformer layer: RMSNorm -> Attention -> Add -> RMSNorm -> MoE -> Add
 * Implements the standard transformer architecture with pre-normalization
 *
 * @param out: Output activations [batch_size, d_model=2048]
 * @param x: Input activations [batch_size, d_model=2048]
 * @param layer_weights: All weights for this layer (attention + MoE)
 * @param cfg: Model configuration
 * @param s: Pre-allocated scratch buffers (sized for max_seq_len)
 * @param batch_size: Number of tokens being processed
 * @param pos: Current position (for RoPE)
 */
void transformer_layer(float* out, float* x, QwenLayerWeights* layer_weights,
    QwenConfig* cfg, QwenStates* s, int batch_size, int pos, int layer_idx) {

  int d_model = cfg->d_model;  // 2048

  // Store original input for residual connections (need separate buffer!)
  // Note: We can't just use float* x_orig = x because x might point to s->x
  // which gets overwritten by rmsnorm
  // Use end of attention buffer (already allocated with extra space)
  float* x_orig = s->scratch_attn + (s->max_seq_len * (2*cfg->n_q*cfg->head_dim + 2*cfg->n_kv*cfg->head_dim + s->max_seq_len + 4*d_model));
  memcpy(x_orig, x, batch_size * d_model * sizeof(float));


  // Pre-attention RMS normalization
  // x_norm = RMSNorm(x) using input_layernorm weights
  rmsnorm(s->x, x, (float*)layer_weights->rms1_w, batch_size, d_model, cfg->rms_eps);

  // Debug: save pre-attention norm

  // Multi-head attention with residual connection
  // attn_out = Attention(x_norm)
  attention(s->scratch_attn, s->x, layer_weights, cfg, s, batch_size, pos, layer_idx);

  // Debug: check x before residual add

  // Residual connection: x = x_orig + attn_out
  // Note: x_orig is the input to this layer (embedding for layer 0)
  for (int i = 0; i < batch_size * d_model; i++) {
    x[i] = x_orig[i] + s->scratch_attn[i];
  }

  // Debug: save attention output

  // Store input to MoE block for residual connection (before normalization)
  // Need separate buffer since x gets overwritten by rmsnorm
  // Place after MoE output + expert workspace to avoid conflicts
  size_t expert_work_size = 2 * cfg->d_ff + cfg->d_model;  // gate_out + up_out + expert_out
  float* x_before_moe = s->scratch_moe + s->max_seq_len * d_model + expert_work_size;  // After expert workspace
  memcpy(x_before_moe, x, batch_size * d_model * sizeof(float));


  // Pre-MoE RMS normalization
  // x_norm = RMSNorm(x) using post_attention_layernorm weights
  rmsnorm(s->x, x, (float*)layer_weights->rms2_w, batch_size, d_model, cfg->rms_eps);

  // Debug: save pre-MoE norm

  // Mixture of Experts with residual connection
  // moe_out = MoE(x_norm)
  moe_layer(s->scratch_moe, s->x, layer_weights, cfg, s, batch_size, layer_idx);

  // Debug: check MoE output before residual

  // Final residual connection: out = x_before_moe + moe_out
  for (int i = 0; i < batch_size * d_model; i++) {
    out[i] = x_before_moe[i] + s->scratch_moe[i];
  }

  // Debug: save MoE output and final result
}

// ----------------------------------------------------------------------------
// Full Model Forward Pass
// ----------------------------------------------------------------------------

/**
 * Complete Qwen3-30B-A3B forward pass
 * Input tokens -> Embeddings -> 48 Transformer Layers -> Final Norm -> Output Logits
 *
 * @param logits: Output logits [vocab_size=151936] for next token prediction
 * @param tokens: Input token sequence [batch_size]
 * @param weights: All model weights (embeddings + layers + output)
 * @param cfg: Model configuration
 * @param s: Pre-allocated computation buffers (sized for max_seq_len)
 * @param batch_size: Number of tokens in input sequence
 * @param pos: Starting position (for RoPE and caching)
 */
void model_forward(float* logits, int* tokens, QwenWeights* weights,
    QwenConfig* cfg, QwenStates* s, int batch_size, int pos) {

  int d_model = cfg->d_model;           // 2048
  int n_layers = cfg->n_layers;         // 48
  int vocab_size = cfg->vocab_size;     // 151936

  // Token embedding lookup
  // For each token, copy corresponding embedding vector from tok_emb table
  // tok_emb: [vocab_size, d_model] = [151936, 2048]
  for (int t = 0; t < batch_size; t++) {
    int token = tokens[t];
    float* embedding_vec = (float*)weights->tok_emb + token * d_model;
    float* dest = s->x + t * d_model;
    memcpy(dest, embedding_vec, d_model * sizeof(float));
  }


  // Pass through all transformer layers
  // Each layer processes: [batch_size, d_model] -> [batch_size, d_model]
  float* layer_input = s->x;
  float* layer_output = s->x_final;

  for (int layer = 0; layer < n_layers; layer++) {
    transformer_layer(layer_output, layer_input, &weights->layers[layer],
        cfg, s, batch_size, pos, layer);

    // Debug: print layer output for comparison

    // Swap buffers for next layer (avoids copying)
    float* temp = layer_input;
    layer_input = layer_output;
    layer_output = temp;
  }

  // Final RMS normalization
  // final_norm_w: [d_model] = [2048]
  float* final_x = layer_input;  // Result from last layer
  rmsnorm(s->x_final, final_x, (float*)weights->final_norm_w, batch_size, d_model, cfg->rms_eps);

  // Output projection to vocabulary logits
  // Use last token's hidden state for next token prediction
  float* last_hidden = s->x_final + (batch_size - 1) * d_model;  // [d_model]

  if (weights->lm_head) {
    // Separate output head: lm_head [vocab_size, d_model] = [151936, 2048]
    matmul(last_hidden, weights->lm_head, logits, 1, vocab_size, d_model);
  } else {
    // Tied weights: reuse token embedding matrix (transposed)
    // tok_emb^T: [d_model, vocab_size] conceptually
    matmul_transposed(logits, last_hidden, (float*)weights->tok_emb,
        1, vocab_size, d_model);
  }
}

// ----------------------------------------------------------------------------
// Main inference loop and utilities
// ----------------------------------------------------------------------------

/**
 * Find the token with maximum probability (greedy sampling)
 *
 * @param logits: Model output logits [vocab_size]
 * @param vocab_size: Size of vocabulary (151936)
 * @return: Token ID with highest probability
 */
int argmax(float* logits, int vocab_size) {
  int max_idx = 0;
  float max_val = logits[0];
  for (int i = 1; i < vocab_size; i++) {
    if (logits[i] > max_val) {
      max_val = logits[i];
      max_idx = i;
    }
  }
  return max_idx;
}

/**
 * Sample token using multinomial distribution from probabilities
 *
 * @param probabilities: Probability distribution [vocab_size]
 * @param vocab_size: Size of vocabulary
 * @return: Sampled token ID
 */
int sample_multinomial(float* probabilities, int vocab_size) {
  float r = (float)rand() / RAND_MAX;
  float cumulative_prob = 0.0f;
  
  for (int i = 0; i < vocab_size; i++) {
    cumulative_prob += probabilities[i];
    if (r < cumulative_prob) {
      return i;
    }
  }
  
  return vocab_size - 1; // fallback to last token
}

/**
 * Apply temperature scaling to logits
 *
 * @param logits: Model logits [vocab_size]
 * @param vocab_size: Size of vocabulary
 * @param temperature: Temperature parameter (1.0 = no change, <1.0 = sharper, >1.0 = smoother)
 */
void apply_temperature(float* logits, int vocab_size, float temperature) {
  if (temperature == 1.0f) return;
  
  for (int i = 0; i < vocab_size; i++) {
    logits[i] /= temperature;
  }
}

/**
 * Convert logits to probabilities using softmax for sampling
 *
 * @param logits: Input logits [vocab_size]
 * @param probabilities: Output probabilities [vocab_size]
 * @param vocab_size: Size of vocabulary
 */
void softmax_sampling(float* logits, float* probabilities, int vocab_size) {
  // Find max for numerical stability
  float max_logit = logits[0];
  for (int i = 1; i < vocab_size; i++) {
    if (logits[i] > max_logit) {
      max_logit = logits[i];
    }
  }
  
  // Compute exp and sum
  float sum = 0.0f;
  for (int i = 0; i < vocab_size; i++) {
    probabilities[i] = expf(logits[i] - max_logit);
    sum += probabilities[i];
  }
  
  // Normalize
  for (int i = 0; i < vocab_size; i++) {
    probabilities[i] /= sum;
  }
}

/**
 * Probability-index pair for efficient sorting
 */
typedef struct {
  float prob;
  int idx;
} ProbIdx;

/**
 * Quicksort comparison function for ProbIdx (descending order)
 */
int compare_prob_desc(const void* a, const void* b) {
  const ProbIdx* pa = (const ProbIdx*)a;
  const ProbIdx* pb = (const ProbIdx*)b;
  if (pa->prob > pb->prob) return -1;
  if (pa->prob < pb->prob) return 1;
  return 0;
}

/**
 * Fast top-p (nucleus) sampling using quicksort
 *
 * @param logits: Model logits [vocab_size]
 * @param vocab_size: Size of vocabulary
 * @param temperature: Temperature for scaling
 * @param top_p: Cumulative probability threshold
 * @return: Sampled token ID
 */
int sample_topp(float* logits, int vocab_size, float temperature, float top_p) {
  // Apply temperature scaling
  apply_temperature(logits, vocab_size, temperature);
  
  // Convert to probabilities and create prob-index pairs
  ProbIdx* prob_idx = (ProbIdx*)malloc(vocab_size * sizeof(ProbIdx));
  
  // Find max for numerical stability
  float max_logit = logits[0];
  for (int i = 1; i < vocab_size; i++) {
    if (logits[i] > max_logit) {
      max_logit = logits[i];
    }
  }
  
  // Compute probabilities
  float sum = 0.0f;
  for (int i = 0; i < vocab_size; i++) {
    prob_idx[i].prob = expf(logits[i] - max_logit);
    prob_idx[i].idx = i;
    sum += prob_idx[i].prob;
  }
  
  // Normalize probabilities
  for (int i = 0; i < vocab_size; i++) {
    prob_idx[i].prob /= sum;
  }
  
  // Sort by probability (descending) using quicksort
  qsort(prob_idx, vocab_size, sizeof(ProbIdx), compare_prob_desc);
  
  // Find cutoff point for top-p
  float cumulative_prob = 0.0f;
  int cutoff = vocab_size;
  
  for (int i = 0; i < vocab_size; i++) {
    cumulative_prob += prob_idx[i].prob;
    if (cumulative_prob >= top_p) {
      cutoff = i + 1;
      break;
    }
  }
  
  // Sample from the top-p subset (already normalized)
  float r = (float)rand() / RAND_MAX * cumulative_prob;
  cumulative_prob = 0.0f;
  int selected_token = prob_idx[cutoff - 1].idx; // fallback
  
  for (int i = 0; i < cutoff; i++) {
    cumulative_prob += prob_idx[i].prob;
    if (r < cumulative_prob) {
      selected_token = prob_idx[i].idx;
      break;
    }
  }
  
  free(prob_idx);
  
  return selected_token;
}

/**
 * Load reference data for testing (optional)
 * Loads .ids.npy, .logits.npy, .probs.npy files for comparison
 */
int load_reference_data(const char* outbase, NpyArrayI32** ref_tokens,
    NpyArray** ref_logits, NpyArray** ref_probs) {
  char path[512];

  // Load reference token IDs
  snprintf(path, sizeof(path), "%s.ids.npy", outbase);
  *ref_tokens = npy_load_int32(path);
  if (!*ref_tokens) return 1;

  // Load reference logits
  snprintf(path, sizeof(path), "%s.logits.npy", outbase);
  *ref_logits = npy_load_float32(path);
  if (!*ref_logits) return 1;

  // Load reference probabilities
  snprintf(path, sizeof(path), "%s.probs.npy", outbase);
  *ref_probs = npy_load_float32(path);
  if (!*ref_probs) return 1;

  return 0;  // Success
}

// ----------------------------------------------------------------------------
// Main function
// ----------------------------------------------------------------------------

void usage(char* argv0) {
  printf("Qwen3-30B-A3B Inference Engine\n\n");
  printf("Usage: %s [options]\n", argv0);
  printf("       %s --model <model.bin> --tokenizer <tokenizer.bin> --prompt \"text\" --steps N\n", argv0);
  printf("       %s <model.bin> <steps> [outbase]  # Legacy mode\n", argv0);
  printf("\n");
  printf("Options:\n");
  printf("  -m, --model <file>      Binary model file (FP32 only)\n");
  printf("  -t, --tokenizer <file>  Tokenizer binary file\n");
  printf("  -p, --prompt <text>     Input prompt to process\n");
  printf("  -s, --steps <N>         Number of generation steps\n");
  printf("  -S, --sample            Use probabilistic sampling instead of greedy argmax\n");
  printf("  -T, --temperature <T>   Temperature for sampling (default: 1.0)\n");
  printf("  -P, --top-p <P>         Top-p threshold for nucleus sampling (default: 0.9)\n");
  printf("  -N, --no-timing         Disable timing statistics display\n");
  printf("  -h, --help              Show this help message\n");
  printf("\n");
  printf("Legacy Arguments (deprecated):\n");
  printf("  model.bin  - Binary model file (FP32 only)\n");
  printf("  steps      - Number of inference steps to run\n");
  printf("  outbase    - Optional: base path for reference files (.ids.npy, .logits.npy, .probs.npy)\n");
  printf("\n");
  printf("Examples:\n");
  printf("  # Basic generation:\n");
  printf("  %s --model all_fp32.bin --tokenizer qwen3_tokenizer.bin --prompt \"Once upon a\" --steps 5\n", argv0);
  printf("  %s -m all_fp32.bin -t qwen3_tokenizer.bin -p \"Hello world\" -s 10\n", argv0);
  printf("\n");
  printf("  # With sampling:\n");
  printf("  %s -m all_fp32.bin -t qwen3_tokenizer.bin -p \"Tell me a story\" -s 20 --sample\n", argv0);
  printf("  %s -m all_fp32.bin -t qwen3_tokenizer.bin -p \"Write code\" -s 10 -S -T 0.7 -P 0.95\n", argv0);
  printf("\n");
  printf("  # Legacy interface:\n");
  printf("  %s all_fp32.bin 10                    # Generate 10 tokens with placeholder\n", argv0);
  printf("  %s all_fp32.bin 5 q3_trace           # Run 5 steps, compare with q3_trace.*.npy\n", argv0);
}

int main(int argc, char** argv) {
  // Initialize debug system
  debug_init("debug_activations");

  // Command-line arguments
  char* model_path = NULL;
  char* tokenizer_path = NULL;
  char* prompt = NULL;
  int steps = 0;
  char* outbase = NULL;  // Legacy mode only
  int use_tokenizer = 0;
  int use_sampling = 0;
  float temperature = 1.0f;
  float top_p = 0.95f;
  int show_timing = 1;  // Show timing by default

  // Check if this is legacy mode (old interface: ./run model.bin steps [outbase])
  if (argc >= 3 && argv[1][0] != '-') {
    // Legacy mode
    model_path = argv[1];
    steps = atoi(argv[2]);
    outbase = (argc >= 4) ? argv[3] : NULL;
    
    if (steps <= 0) {
      fprintf(stderr, "Error: steps must be positive\n");
      return 1;
    }
    
    printf("Running in legacy mode (no tokenizer)\n");
    printf("Loading model: %s\n", model_path);
    printf("Running %d inference steps\n", steps);
    if (outbase) {
      printf("Comparing with reference: %s.*.npy\n", outbase);
    }
  } else {
    // New command-line interface with getopt
    static struct option long_options[] = {
      {"model",       required_argument, 0, 'm'},
      {"tokenizer",   required_argument, 0, 't'},
      {"prompt",      required_argument, 0, 'p'},
      {"steps",       required_argument, 0, 's'},
      {"sample",      no_argument,       0, 'S'},
      {"temperature", required_argument, 0, 'T'},
      {"top-p",       required_argument, 0, 'P'},
      {"no-timing",   no_argument,       0, 'N'},
      {"help",        no_argument,       0, 'h'},
      {0, 0, 0, 0}
    };

    int c;
    int option_index = 0;
    
    while ((c = getopt_long(argc, argv, "m:t:p:s:ST:P:Nh", long_options, &option_index)) != -1) {
      switch (c) {
        case 'm':
          model_path = optarg;
          break;
        case 't':
          tokenizer_path = optarg;
          break;
        case 'p':
          prompt = optarg;
          break;
        case 's':
          steps = atoi(optarg);
          break;
        case 'S':
          use_sampling = 1;
          break;
        case 'T':
          temperature = atof(optarg);
          break;
        case 'P':
          top_p = atof(optarg);
          break;
        case 'N':
          show_timing = 0;
          break;
        case 'h':
          usage(argv[0]);
          return 0;
        case '?':
          usage(argv[0]);
          return 1;
        default:
          usage(argv[0]);
          return 1;
      }
    }

    // Validate required arguments for new interface
    if (!model_path || !tokenizer_path || !prompt || steps <= 0) {
      fprintf(stderr, "Error: Missing required arguments\n");
      usage(argv[0]);
      return 1;
    }
    
    use_tokenizer = 1;
    printf("Loading model: %s\n", model_path);
    printf("Loading tokenizer: %s\n", tokenizer_path);
    printf("Prompt: \"%s\"\n", prompt);
    printf("Generation steps: %d\n", steps);
  }

  // Load model weights - use mmap for efficiency
  struct timeval start_time, end_time;
  gettimeofday(&start_time, NULL);

  BinFile* bin_file = bin_load_mmap(model_path);
  if (!bin_file) {
    fprintf(stderr, "Failed to load model file: %s\n", model_path);
    return 1;
  }

  // Initialize model configuration and weights
  QwenConfig config;
  QwenWeights weights;
  load_all_weights(bin_file, &config, &weights);

  gettimeofday(&end_time, NULL);
  double load_time = (end_time.tv_sec - start_time.tv_sec) +
    (end_time.tv_usec - start_time.tv_usec) / 1000000.0;

  printf("Model loaded in %.2f seconds\n", load_time);
  printf("Model: d_model=%d layers=%d heads=%d/%d vocab=%d experts=%d/%d\n",
      config.d_model, config.n_layers, config.n_q, config.n_kv,
      config.vocab_size, config.n_experts, config.top_k);

  // Load tokenizer if using new interface
  Tokenizer* tokenizer = NULL;
  if (use_tokenizer) {
    tokenizer = tokenizer_create();
    if (!tokenizer) {
      fprintf(stderr, "Failed to create tokenizer\n");
      return 1;
    }
    
    if (tokenizer_load(tokenizer, tokenizer_path) != 0) {
      fprintf(stderr, "Failed to load tokenizer from: %s\n", tokenizer_path);
      tokenizer_free(tokenizer);
      return 1;
    }
    
    printf("Tokenizer loaded: vocabulary size %d\n", tokenizer_vocab_size(tokenizer));
    
    int bos_id, eos_id, pad_id;
    tokenizer_get_special_tokens(tokenizer, &bos_id, &eos_id, &pad_id);
    printf("Special tokens: BOS=%d, EOS=%d, PAD=%d\n", bos_id, eos_id, pad_id);
  }

  // Load reference data if provided
  NpyArrayI32* ref_tokens = NULL;
  NpyArray* ref_logits = NULL;
  NpyArray* ref_probs = NULL;
  int has_reference = 0;

  if (outbase) {
    if (load_reference_data(outbase, &ref_tokens, &ref_logits, &ref_probs) == 0) {
      has_reference = 1;
      printf("Reference data loaded: %d tokens\n", ref_tokens->shape[0]);
    } else {
      printf("Warning: Could not load reference data from %s\n", outbase);
    }
  }

  // Set up input tokens
  int* input_tokens;
  int prompt_len = 0;  // Number of prompt tokens
  int max_seq_len;

  if (has_reference) {
    // Legacy mode with reference data
    input_tokens = ref_tokens->data;
    prompt_len = ref_tokens->shape[0] - 1;  // Exclude the last token for generation
    max_seq_len = ref_tokens->shape[0];
  } else if (use_tokenizer) {
    // New interface: tokenize the prompt
    TokenizerResult encode_result;
    if (tokenizer_encode(tokenizer, prompt, &encode_result) != 0) {
      fprintf(stderr, "Failed to tokenize prompt: \"%s\"\n", prompt);
      tokenizer_free(tokenizer);
      return 1;
    }
    
    prompt_len = encode_result.count;
    max_seq_len = prompt_len + steps;
    input_tokens = (int*)calloc(max_seq_len, sizeof(int));
    
    // Copy prompt tokens
    memcpy(input_tokens, encode_result.token_ids, prompt_len * sizeof(int));
    
    printf("\nTokenized prompt (%d tokens): [", prompt_len);
    for (int i = 0; i < prompt_len; i++) {
      printf("%d", encode_result.token_ids[i]);
      if (i < prompt_len - 1) printf(", ");
    }
    printf("]\n");
    
    // Show decoded tokens for verification
    TokenizerResult decode_result;
    if (tokenizer_decode(tokenizer, encode_result.token_ids, encode_result.count, &decode_result) == 0) {
      printf("Token breakdown:\n");
      for (int i = 0; i < decode_result.count; i++) {
        printf("  [%d] -> \"%s\"\n", encode_result.token_ids[i], decode_result.token_strings[i]);
      }
      tokenizer_result_free(&decode_result);
    }
    printf("\n");
    
    tokenizer_result_free(&encode_result);
  } else {
    // Legacy mode without reference data
    prompt_len = 1;  // Just BOS token
    max_seq_len = steps + 1;
    input_tokens = (int*)calloc(max_seq_len, sizeof(int));
    input_tokens[0] = 148350;  // BOS token from config
    for (int i = 1; i < max_seq_len; i++) {
      input_tokens[i] = 1000 + i;  // Placeholder tokens
    }
  }

  // Allocate computation buffers
  QwenStates qwen_states;
  malloc_qwen_states(&qwen_states, &config, max_seq_len);

  // Initialize random seed for sampling
  srand((unsigned int)time(NULL));

  // Run inference  
  struct timeval prompt_start, prompt_end, gen_start, gen_end, step_start, step_end;
  gettimeofday(&prompt_start, NULL);

  printf("\nRunning inference...\n");
  
  // Process prompt tokens first (if any) to populate K/V cache
  if (use_tokenizer && prompt_len > 1) {
    printf("Processing prompt tokens (%d tokens)...\n", prompt_len - 1);
    for (int i = 0; i < prompt_len - 1; i++) {
      int* current_token = &input_tokens[i];
      model_forward(qwen_states.logits, current_token, &weights, &config,
          &qwen_states, 1, i);
      if (show_timing) {
        printf("  Processed token %d: %d\n", i, input_tokens[i]);
      }
    }
    printf("Prompt processing complete. Starting generation...\n\n");
  }
  
  gettimeofday(&prompt_end, NULL);
  gettimeofday(&gen_start, NULL);
  
  // Generate new tokens
  int generation_start = use_tokenizer ? prompt_len - 1 : 0;
  for (int step = 0; step < steps; step++) {
    gettimeofday(&step_start, NULL);
    int current_pos = generation_start + step;

    // Forward pass: predict next token
    int* current_token = &input_tokens[current_pos];
    model_forward(qwen_states.logits, current_token, &weights, &config,
        &qwen_states, 1, current_pos);

    // Select next token (sampling or argmax)
    int predicted_token;
    if (use_sampling) {
      // Create a copy of logits for sampling (to avoid modifying original)
      float* logits_copy = (float*)malloc(config.vocab_size * sizeof(float));
      memcpy(logits_copy, qwen_states.logits, config.vocab_size * sizeof(float));
      predicted_token = sample_topp(logits_copy, config.vocab_size, temperature, top_p);
      free(logits_copy);
    } else {
      predicted_token = argmax(qwen_states.logits, config.vocab_size);
    }

    // Store predicted token for next iteration
    if (current_pos + 1 < max_seq_len) {
      input_tokens[current_pos + 1] = predicted_token;
    }

    gettimeofday(&step_end, NULL);
    double step_time = (step_end.tv_sec - step_start.tv_sec) * 1000.0 +
                      (step_end.tv_usec - step_start.tv_usec) / 1000.0;

    // Decode and display the generated token
    if (use_tokenizer) {
      TokenizerResult decode_result;
      if (tokenizer_decode(tokenizer, &predicted_token, 1, &decode_result) == 0) {
        if (show_timing) {
          printf("Step %d: token=%d -> \"%s\" (%.2f ms)\n", step, predicted_token, 
                 decode_result.token_strings[0], step_time);
        } else {
          printf("Step %d: token=%d -> \"%s\"\n", step, predicted_token, decode_result.token_strings[0]);
        }
        tokenizer_result_free(&decode_result);
      } else {
        if (show_timing) {
          printf("Step %d: token=%d (decode failed, %.2f ms)\n", step, predicted_token, step_time);
        } else {
          printf("Step %d: token=%d (decode failed)\n", step, predicted_token);
        }
      }
    } else {
      if (show_timing) {
        printf("Step %d: predicted_token=%d (%.2f ms)\n", step, predicted_token, step_time);
      } else {
        printf("Step %d: predicted_token=%d\n", step, predicted_token);
      }
    }

    // Compare with reference if available
    if (has_reference && step < ref_tokens->shape[0] - 1) {
      int expected_token = input_tokens[step + 1];  // Next token in sequence

      // Compare logits
      float* expected_logits = ref_logits->data + step * config.vocab_size;
      float logit_error = 0.0f;
      for (int i = 0; i < config.vocab_size; i++) {
        float diff = qwen_states.logits[i] - expected_logits[i];
        logit_error += diff * diff;
      }
      logit_error = sqrtf(logit_error / config.vocab_size);  // RMS error

      // Compare probabilities
      softmax(qwen_states.logits, config.vocab_size);  // Convert to probs
      float* expected_probs = ref_probs->data + step * config.vocab_size;
      float prob_error = 0.0f;
      for (int i = 0; i < config.vocab_size; i++) {
        float diff = qwen_states.logits[i] - expected_probs[i];
        prob_error += fabsf(diff);
      }
      prob_error /= config.vocab_size;  // Mean absolute error

      printf(" expected=%d logit_rmse=%.6f prob_mae=%.6f",
          expected_token, logit_error, prob_error);
    }

    printf("\n");
  }

  gettimeofday(&gen_end, NULL);
  gettimeofday(&end_time, NULL);
  
  double total_time = (end_time.tv_sec - start_time.tv_sec) +
    (end_time.tv_usec - start_time.tv_usec) / 1000000.0;
  double prompt_time = (prompt_end.tv_sec - prompt_start.tv_sec) +
    (prompt_end.tv_usec - prompt_start.tv_usec) / 1000000.0;
  double gen_time = (gen_end.tv_sec - gen_start.tv_sec) +
    (gen_end.tv_usec - gen_start.tv_usec) / 1000000.0;

  if (show_timing) {
    printf("\n=== PERFORMANCE STATISTICS ===\n");
    if (use_tokenizer && prompt_len > 1) {
      double prompt_tokens = prompt_len - 1;
      printf("Prompt processing: %.2f seconds (%.1f tok/s)\n", 
             prompt_time, prompt_tokens / prompt_time);
    }
    printf("Generation: %.2f seconds (%.1f tok/s, %.2f ms/tok)\n", 
           gen_time, steps / gen_time, gen_time * 1000.0 / steps);
    printf("Total inference: %.2f seconds\n", total_time);
  } else {
    printf("\nInference completed in %.2f seconds (%.2f ms/step)\n",
           total_time, total_time * 1000.0 / steps);
  }
  
  // Generate final output for tokenizer mode
  if (use_tokenizer) {
    int output_len = prompt_len + steps;
    TokenizerResult final_decode;
    if (tokenizer_decode(tokenizer, input_tokens, output_len, &final_decode) == 0) {
      printf("\n============================================================\n");
      printf("COMPLETE GENERATED TEXT:\n");
      printf("============================================================\n");
      for (int i = 0; i < final_decode.count; i++) {
        printf("%s", final_decode.token_strings[i]);
      }
      printf("\n");
      printf("============================================================\n");
      
      tokenizer_result_free(&final_decode);
    }
  }

  // Cleanup
  if (tokenizer) {
    tokenizer_free(tokenizer);
  }
  if (!has_reference) {
    free(input_tokens);
  }
  if (ref_tokens) npy_free_i32(ref_tokens);
  if (ref_logits) npy_free(ref_logits);
  if (ref_probs) npy_free(ref_probs);

  free_qwen_states(&qwen_states);
  bin_free_mmap(bin_file);

  return 0;
}
