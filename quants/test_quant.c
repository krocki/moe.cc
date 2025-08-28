/**
 * Quantization Test Driver
 * OMP_NUM_THREADS=16 OMP_WAIT_POLICY=active OMP_PROC_BIND=spread ./test_quant
 * Comprehensive testing suite for quantization functions including:
 * - Performance benchmarks vs FP32 matrix multiplication
 * - Accuracy measurements for quantize/dequantize roundtrip
 * - Multi-threading performance analysis
 * - Various matrix sizes and configurations
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>
#include <assert.h>
#include "quant.h"
#include "tensor.h"

// Test configuration
#define Q8_GROUP_SIZE 128
#define Q4_GROUP_SIZE 128
#define MIN_THREADS 16
#define MAX_THREADS 16
#define WARMUP_ITERATIONS 50000
#define BENCHMARK_ITERATIONS 1000
#define ACCURACY_TOLERANCE_Q8 0.01f
#define ACCURACY_TOLERANCE_Q4 0.1f

// Matrix size configurations to test
typedef struct {
  int M, N, K;
  const char* name;
} MatrixSize;

static const MatrixSize MATRIX_SIZES[] = {
  // Key transpose comparisons - Original vs Transposed weight layouts
  {1,  128,  2048, "MoE_Router_T1"},     // [1,2048] × [128,2048]^T → [1,128] - Router
  {1,  768,  2048, "MoE_Gate_T1"},       // [1,2048] × [768,2048]^T → [1,768] - Expert gate
  {1, 2048,   768, "MoE_Down_T1"},       // [1,768] × [2048,768]^T → [1,2048] - Expert down  
  {1, 4096,  2048, "Att_Wq_T1"},        // [1,2048] × [4096,2048]^T → [1,4096] - Query proj
  {1, 2048,  4096, "Att_Wq_T1_Trans"},   // [1,2048] × [2048,4096] → [1,4096] - Query with transposed weights  
  {1,  512,  2048, "Att_Wk_T1"},        // [1,2048] × [512,2048]^T → [1,512] - Key proj
  {1, 2048,   512, "Att_Wk_T1_Trans"},   // [1,2048] × [2048,512] → [1,512] - Key with transposed weights
  {8, 4096,  2048, "Att_Wq_T8"},        // [8,2048] × [4096,2048]^T → [8,4096] - Query batch
  {8, 2048,  4096, "Att_Wq_T8_Trans"},  // [8,2048] × [2048,4096] → [8,4096] - Query batch transposed
  
};

static const int NUM_MATRIX_SIZES = sizeof(MATRIX_SIZES) / sizeof(MATRIX_SIZES[0]);

// Timing utilities
static double get_time_ms() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

// Random data generation
static void fill_random_f32(float* data, size_t n) {
  for (size_t i = 0; i < n; i++) {
    data[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 4.0f;  // Range [-2, 2]
  }
}

// Simple FP32 matrix multiplication reference
static void matmul_f32_f32_f32(const float* A, const float* B, float* C, 
                               int M, int N, int K) {
  #pragma omp parallel for collapse(2)
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      float sum = 0.0f;
      for (int k = 0; k < K; k++) {
        sum += A[m * K + k] * B[n * K + k];  // B is transposed
      }
      C[m * N + n] = sum;
    }
  }
}

// Calculate Mean Absolute Error
static float calculate_mae(const float* a, const float* b, size_t n) {
  double sum = 0.0;
  for (size_t i = 0; i < n; i++) {
    sum += fabs(a[i] - b[i]);
  }
  return (float)(sum / n);
}

// Calculate Mean Squared Error
static float calculate_mse(const float* a, const float* b, size_t n) {
  double sum = 0.0;
  for (size_t i = 0; i < n; i++) {
    double diff = a[i] - b[i];
    sum += diff * diff;
  }
  return (float)(sum / n);
}

// Calculate Cosine Similarity
static float calculate_cosine_similarity(const float* a, const float* b, size_t n) {
  double dot_product = 0.0, norm_a = 0.0, norm_b = 0.0;
  
  for (size_t i = 0; i < n; i++) {
    dot_product += a[i] * b[i];
    norm_a += a[i] * a[i];
    norm_b += b[i] * b[i];
  }
  
  if (norm_a > 0 && norm_b > 0) {
    return (float)(dot_product / (sqrt(norm_a) * sqrt(norm_b)));
  }
  return 0.0f;
}

// Test Q8 quantization accuracy
static void test_q8_accuracy(int size) {
  printf("\n=== Q8 Quantization Accuracy Test (size: %d, gs: %d) ===\n", size, Q8_GROUP_SIZE);
  
  float* orig = (float*)malloc(size * sizeof(float));
  int8_t* q = (int8_t*)malloc(size * sizeof(int8_t));
  float* scales = (float*)malloc((size / Q8_GROUP_SIZE) * sizeof(float));
  float* restored = (float*)malloc(size * sizeof(float));
  
  fill_random_f32(orig, size);
  
  // Quantize
  quantize_q8(orig, q, scales, size, Q8_GROUP_SIZE);
  
  // Dequantize
  dequantize_q8(q, scales, restored, size, Q8_GROUP_SIZE);
  
  // Calculate errors
  float mae = calculate_mae(orig, restored, size);
  float mse = calculate_mse(orig, restored, size);
  
  printf("MAE: %.6f, MSE: %.6f\n", mae, mse);
  printf("Q8 Accuracy: %s (MAE < %.3f)\n", 
         mae < ACCURACY_TOLERANCE_Q8 ? "PASS" : "FAIL", ACCURACY_TOLERANCE_Q8);
  
  free(orig);
  free(q);
  free(scales);
  free(restored);
}

// Test Q4 quantization accuracy with both group sizes
static void test_q4_accuracy(int size, int group_size) {
  printf("\n=== Q4 Quantization Accuracy Test (size: %d, gs: %d) ===\n", size, group_size);
  
  if (size % group_size != 0) {
    printf("SKIP: Size %d not divisible by group size %d\n", size, group_size);
    return;
  }
  
  float* orig = (float*)malloc(size * sizeof(float));
  uint8_t* q = (uint8_t*)malloc((size + 1) / 2);  // Packed 4-bit
  float* scales = (float*)malloc((size / group_size) * sizeof(float));
  float* zero_points = (float*)malloc((size / group_size) * sizeof(float));
  float* restored = (float*)malloc(size * sizeof(float));
  
  if (!orig || !q || !scales || !zero_points || !restored) {
    printf("Memory allocation failed\n");
    goto cleanup;
  }
  
  fill_random_f32(orig, size);
  
  // Quantize
  quantize_q4(orig, 1, size, group_size, scales, zero_points, q);
  
  // Dequantize
  dequantize_q4(q, scales, zero_points, restored, 1, size, group_size);
  
  // Calculate errors
  float mae = calculate_mae(orig, restored, size);
  float mse = calculate_mse(orig, restored, size);
  float cosine = calculate_cosine_similarity(orig, restored, size);
  
  printf("MAE: %.6f, MSE: %.6f, Cosine: %.6f\n", mae, mse, cosine);
  printf("Q4 Accuracy: %s (MAE < %.3f)\n", 
         mae < ACCURACY_TOLERANCE_Q4 ? "PASS" : "FAIL", ACCURACY_TOLERANCE_Q4);

cleanup:
  free(orig);
  free(q);
  free(scales);
  free(zero_points);
  free(restored);
}

// Multi-threaded benchmark testing all methods across different thread counts
static void benchmark_threading(const MatrixSize* size) {
  printf("\n=== Multi-threaded Matrix Multiplication: %s %dx%dx%d ===\n", 
         size->name, size->M, size->N, size->K);
  
  int M = size->M, N = size->N, K = size->K;
  
  // Check if dimensions work with group size 128
  if (K % Q8_GROUP_SIZE != 0) {
    printf("SKIP: K dimension (%d) not divisible by %d\n", K, Q8_GROUP_SIZE);
    return;
  }
  
  // Generate thread counts between MIN_THREADS and MAX_THREADS
  int thread_counts[8];  // Max possible: 1, 2, 4, 8, 16, 32, 64, 128
  int num_thread_counts = 0;
  for (int t = MIN_THREADS; t <= MAX_THREADS; t *= 2) {
    if (t >= MIN_THREADS && t <= MAX_THREADS) {
      thread_counts[num_thread_counts++] = t;
    }
  }
  // If MIN_THREADS is not a power of 2, add it
  if (MIN_THREADS > 1 && (MIN_THREADS & (MIN_THREADS - 1)) != 0) {
    // MIN_THREADS is not a power of 2, add it at the beginning
    for (int i = num_thread_counts; i > 0; i--) {
      thread_counts[i] = thread_counts[i-1];
    }
    thread_counts[0] = MIN_THREADS;
    num_thread_counts++;
  }
  const char* method_names[] = {"FP32", "Q8xQ8", "Q8xQ4", "F32xQ8", "F32xQ4"};
  const int num_methods = 5;
  
  // Results storage: [method][thread_count] - declare early to avoid VLA issues
  double times[num_methods][num_thread_counts];
  
  // Allocate matrices (shared across all thread tests)
  float* A_f32 = (float*)malloc(M * K * sizeof(float));
  float* B_f32 = (float*)malloc(N * K * sizeof(float));
  float* C_f32 = (float*)malloc(M * N * sizeof(float));
  float* C_q8q8 = (float*)malloc(M * N * sizeof(float));
  float* C_q8q4 = (float*)malloc(M * N * sizeof(float));
  float* C_f32q8 = (float*)malloc(M * N * sizeof(float));
  float* C_f32q4 = (float*)malloc(M * N * sizeof(float));
  
  // Scratch spaces for F32×Q matmuls
  int8_t* A_scratch = (int8_t*)malloc(M * K * sizeof(int8_t));
  float* A_scales = (float*)malloc(M * (K / Q8_GROUP_SIZE) * sizeof(float));

  if (!A_f32 || !B_f32 || !C_f32 || !C_q8q8 || !C_q8q4 || 
      !C_f32q8 || !C_f32q4 || !A_scratch || !A_scales) {
    printf("Memory allocation failed\n");
    goto cleanup;
  }
  
  // Initialize data
  fill_random_f32(A_f32, M * K);
  fill_random_f32(B_f32, N * K);
  
  // Create tensors for proper quantization (like convert.c)
  int A_shape[2] = {M, K};
  int B_shape[2] = {N, K};
  
  Tensor* A_tensor = tensor_create("A", 0, 2, A_shape, A_f32);
  Tensor* B_tensor = tensor_create("B", 0, 2, B_shape, B_f32);
  
  if (!A_tensor || !B_tensor) {
    printf("Tensor creation failed\n");
    goto cleanup;
  }
  
  // Quantize tensors (exclude from timing) - use group size 128 for both Q8 and Q4
  QuantizedTensor* A_q8 = quantize_tensor(A_tensor, QUANT_Q8, Q8_GROUP_SIZE);
  QuantizedTensor* B_q8 = quantize_tensor(B_tensor, QUANT_Q8, Q8_GROUP_SIZE);  
  QuantizedTensor* B_q4 = quantize_tensor(B_tensor, QUANT_Q4, Q4_GROUP_SIZE);
  
  if (!A_q8 || !B_q8 || !B_q4) {
    printf("Quantization failed\n");
    goto cleanup_tensors;
  }
  
  // Test each thread count
  for (int t = 0; t < num_thread_counts; t++) {
    int threads = thread_counts[t];
    omp_set_num_threads(threads);
    
    // Warmup
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {  // Very reduced warmup for multi-threading test
      matmul_f32_f32_f32(A_f32, B_f32, C_f32, M, N, K);
      matmul_q8_q8_f32((const int8_t*)A_q8->q_data, A_q8->scales, 
                       (const int8_t*)B_q8->q_data, B_q8->scales, 
                       C_q8q8, M, N, K, Q8_GROUP_SIZE);
      matmul_q8_q4_f32((const int8_t*)A_q8->q_data, A_q8->scales,
                       (const uint8_t*)B_q4->q_data, B_q4->scales, B_q4->zero_points,
                       C_q8q4, M, N, K, Q4_GROUP_SIZE);
      matmul_f32_q8_f32(A_f32, (const int8_t*)B_q8->q_data, B_q8->scales,
                        C_f32q8, M, N, K, Q8_GROUP_SIZE, A_scratch, A_scales);
      matmul_f32_q4_f32_with_zeros(A_f32, (const uint8_t*)B_q4->q_data, B_q4->scales, B_q4->zero_points,
                        C_f32q4, M, N, K, Q4_GROUP_SIZE, A_scratch, A_scales);
    }
    
    // Benchmark each method
    double start;
    
    // FP32 baseline
    start = get_time_ms();
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
      matmul_f32_f32_f32(A_f32, B_f32, C_f32, M, N, K);
    }
    times[0][t] = (get_time_ms() - start) / BENCHMARK_ITERATIONS;
    
    // Q8×Q8
    start = get_time_ms();
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
      matmul_q8_q8_f32((const int8_t*)A_q8->q_data, A_q8->scales, 
                       (const int8_t*)B_q8->q_data, B_q8->scales, 
                       C_q8q8, M, N, K, Q8_GROUP_SIZE);
    }
    times[1][t] = (get_time_ms() - start) / BENCHMARK_ITERATIONS;
    
    // Q8×Q4
    start = get_time_ms();
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
      matmul_q8_q4_f32((const int8_t*)A_q8->q_data, A_q8->scales,
                       (const uint8_t*)B_q4->q_data, B_q4->scales, B_q4->zero_points,
                       C_q8q4, M, N, K, Q4_GROUP_SIZE);
    }
    times[2][t] = (get_time_ms() - start) / BENCHMARK_ITERATIONS;
    
    // F32×Q8
    start = get_time_ms();
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
      matmul_f32_q8_f32(A_f32, (const int8_t*)B_q8->q_data, B_q8->scales,
                        C_f32q8, M, N, K, Q8_GROUP_SIZE, A_scratch, A_scales);
    }
    times[3][t] = (get_time_ms() - start) / BENCHMARK_ITERATIONS;
    
    // F32×Q4
    start = get_time_ms();
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
      matmul_f32_q4_f32_with_zeros(A_f32, (const uint8_t*)B_q4->q_data, B_q4->scales, B_q4->zero_points,
                        C_f32q4, M, N, K, Q4_GROUP_SIZE, A_scratch, A_scales);
    }
    times[4][t] = (get_time_ms() - start) / BENCHMARK_ITERATIONS;
  }
  
  // Calculate accuracy once for each method (using single-threaded results)
  omp_set_num_threads(1);
  float* results[] = {C_f32, C_q8q8, C_q8q4, C_f32q8, C_f32q4};
  
  // Run reference calculation for accuracy
  matmul_f32_f32_f32(A_f32, B_f32, C_f32, M, N, K);
  matmul_q8_q8_f32((const int8_t*)A_q8->q_data, A_q8->scales, 
                   (const int8_t*)B_q8->q_data, B_q8->scales, 
                   C_q8q8, M, N, K, Q8_GROUP_SIZE);
  matmul_q8_q4_f32((const int8_t*)A_q8->q_data, A_q8->scales,
                   (const uint8_t*)B_q4->q_data, B_q4->scales, B_q4->zero_points,
                   C_q8q4, M, N, K, Q4_GROUP_SIZE);
  matmul_f32_q8_f32(A_f32, (const int8_t*)B_q8->q_data, B_q8->scales,
                    C_f32q8, M, N, K, Q8_GROUP_SIZE, A_scratch, A_scales);
  matmul_f32_q4_f32_with_zeros(A_f32, (const uint8_t*)B_q4->q_data, B_q4->scales, B_q4->zero_points,
                    C_f32q4, M, N, K, Q4_GROUP_SIZE, A_scratch, A_scales);
  
  // Print results table with method+thread combinations as rows
  printf("┌──────────┬───────┬─────────────┬─────────────┬─────────────┬─────────────┬─────────────┬─────────────┐\n");
  printf("│ Method   │threads│ Time (ms)   │ Speedup     │ Thread Scal │ GFLOPS      │ MAE         │ Cosine Sim  │\n");
  printf("├──────────┼───────┼─────────────┼─────────────┼─────────────┼─────────────┼─────────────┼─────────────┤\n");
  
  for (int t = 0; t < num_thread_counts; t++) {
    for (int m = 0; m < num_methods; m++) {
      float mae, cosine_sim;
      if (m == 0) {
        mae = 0.0f; cosine_sim = 1.0f; // FP32 baseline
      } else {
        mae = calculate_mae(C_f32, results[m], M * N);
        cosine_sim = calculate_cosine_similarity(C_f32, results[m], M * N);
      }
      
      double gflops = (2.0 * M * N * K) / (times[m][t] * 1e6);
      double speedup = times[0][t] / times[m][t];  // Speedup vs FP32 at same thread count
      double thread_speedup = times[m][0] / times[m][t];  // Thread scaling for same method
      
      printf("│ %-8s │  %2d   │ %11.2f │ %10.2fx │ %10.2fx │ %11.2f │ %11.6f │ %11.6f │\n",
             method_names[m], thread_counts[t], times[m][t], speedup, thread_speedup, gflops, mae, cosine_sim);
    }
  }
  
  printf("└──────────┴───────┴─────────────┴─────────────┴─────────────┴─────────────┴─────────────┴─────────────┘\n");

cleanup_tensors:
  if (A_tensor) tensor_free_single(A_tensor);
  if (B_tensor) tensor_free_single(B_tensor);
  if (A_q8) quantized_tensor_free(A_q8);
  if (B_q8) quantized_tensor_free(B_q8);
  if (B_q4) quantized_tensor_free(B_q4);

cleanup:
  free(A_f32); free(B_f32); free(C_f32); free(C_q8q8); free(C_q8q4);
  free(C_f32q8); free(C_f32q4);
  free(A_scratch); free(A_scales);
}

// Original single-threaded benchmark (keep for compatibility)
static void benchmark(const MatrixSize* size, int num_threads) {
  printf("\n=== Comprehensive Matrix Multiplication: %s %dx%dx%d (threads: %d) ===\n", 
         size->name, size->M, size->N, size->K, num_threads);
  
  omp_set_num_threads(num_threads);
  
  int M = size->M, N = size->N, K = size->K;
  
  // Check if dimensions work with group size 128
  if (K % Q8_GROUP_SIZE != 0) {
    printf("SKIP: K dimension (%d) not divisible by %d\n", K, Q8_GROUP_SIZE);
    return;
  }
  
  // Allocate matrices
  float* A_f32 = (float*)malloc(M * K * sizeof(float));
  float* B_f32 = (float*)malloc(N * K * sizeof(float));
  float* C_f32 = (float*)malloc(M * N * sizeof(float));
  float* C_q8q8 = (float*)malloc(M * N * sizeof(float));
  float* C_q8q4 = (float*)malloc(M * N * sizeof(float));
  float* C_f32q8 = (float*)malloc(M * N * sizeof(float));
  float* C_f32q4 = (float*)malloc(M * N * sizeof(float));
  
  // Scratch spaces for F32×Q matmuls
  int8_t* A_scratch = (int8_t*)malloc(M * K * sizeof(int8_t));
  float* A_scales = (float*)malloc(M * (K / Q8_GROUP_SIZE) * sizeof(float));
  
  if (!A_f32 || !B_f32 || !C_f32 || !C_q8q8 || !C_q8q4 || 
      !C_f32q8 || !C_f32q4 || !A_scratch || !A_scales) {
    printf("Memory allocation failed\n");
    goto cleanup;
  }
  
  // Initialize data
  fill_random_f32(A_f32, M * K);
  fill_random_f32(B_f32, N * K);
  
  // Create tensors for proper quantization (like convert.c)
  int A_shape[2] = {M, K};
  int B_shape[2] = {N, K};
  
  Tensor* A_tensor = tensor_create("A", 0, 2, A_shape, A_f32);
  Tensor* B_tensor = tensor_create("B", 0, 2, B_shape, B_f32);
  
  if (!A_tensor || !B_tensor) {
    printf("Tensor creation failed\n");
    goto cleanup;
  }
  
  // Quantize tensors (exclude from timing) - use group size 128 for both Q8 and Q4
  QuantizedTensor* A_q8 = quantize_tensor(A_tensor, QUANT_Q8, Q8_GROUP_SIZE);
  QuantizedTensor* B_q8 = quantize_tensor(B_tensor, QUANT_Q8, Q8_GROUP_SIZE);  
  QuantizedTensor* B_q4 = quantize_tensor(B_tensor, QUANT_Q4, Q4_GROUP_SIZE);
  
  if (!A_q8 || !B_q8 || !B_q4) {
    printf("Quantization failed\n");
    goto cleanup_tensors;
  }
  
  // FP32 baseline
  matmul_f32_f32_f32(A_f32, B_f32, C_f32, M, N, K);
  
  // Warmup all methods  
  for (int i = 0; i < WARMUP_ITERATIONS; i++) {
    matmul_f32_f32_f32(A_f32, B_f32, C_f32, M, N, K);
    matmul_q8_q8_f32((const int8_t*)A_q8->q_data, A_q8->scales, 
                     (const int8_t*)B_q8->q_data, B_q8->scales, 
                     C_q8q8, M, N, K, Q8_GROUP_SIZE);
    matmul_q8_q4_f32((const int8_t*)A_q8->q_data, A_q8->scales,
                     (const uint8_t*)B_q4->q_data, B_q4->scales, B_q4->zero_points,
                     C_q8q4, M, N, K, Q4_GROUP_SIZE);
    matmul_f32_q8_f32(A_f32, (const int8_t*)B_q8->q_data, B_q8->scales,
                      C_f32q8, M, N, K, Q8_GROUP_SIZE, A_scratch, A_scales);
    matmul_f32_q4_f32_with_zeros(A_f32, (const uint8_t*)B_q4->q_data, B_q4->scales, B_q4->zero_points,
                      C_f32q4, M, N, K, Q4_GROUP_SIZE, A_scratch, A_scales);
  }
  
  // Benchmark all methods  
  double times[5];
  const char* method_names[] = {"FP32", "Q8xQ8", "Q8xQ4", "F32xQ8", "F32xQ4"};
  double start;
  
  // FP32 baseline
  start = get_time_ms();
  for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
    matmul_f32_f32_f32(A_f32, B_f32, C_f32, M, N, K);
  }
  times[0] = (get_time_ms() - start) / BENCHMARK_ITERATIONS;
  
  // Q8×Q8 (direct)
  start = get_time_ms();
  for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
    matmul_q8_q8_f32((const int8_t*)A_q8->q_data, A_q8->scales, 
                     (const int8_t*)B_q8->q_data, B_q8->scales, 
                     C_q8q8, M, N, K, Q8_GROUP_SIZE);
  }
  times[1] = (get_time_ms() - start) / BENCHMARK_ITERATIONS;
  
  // Q8×Q4 (direct, gs=128 only)
  start = get_time_ms();
  for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
    matmul_q8_q4_f32((const int8_t*)A_q8->q_data, A_q8->scales,
                     (const uint8_t*)B_q4->q_data, B_q4->scales, B_q4->zero_points,
                     C_q8q4, M, N, K, Q4_GROUP_SIZE);
  }
  times[2] = (get_time_ms() - start) / BENCHMARK_ITERATIONS;
  
  // F32×Q8 (dynamic quantization)
  start = get_time_ms();
  for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
    matmul_f32_q8_f32(A_f32, (const int8_t*)B_q8->q_data, B_q8->scales,
                      C_f32q8, M, N, K, Q8_GROUP_SIZE, A_scratch, A_scales);
  }
  times[3] = (get_time_ms() - start) / BENCHMARK_ITERATIONS;
  
  // F32×Q4 (dynamic quantization, gs=128, with zero points)
  start = get_time_ms();
  for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
    matmul_f32_q4_f32_with_zeros(A_f32, (const uint8_t*)B_q4->q_data, B_q4->scales, B_q4->zero_points,
                      C_f32q4, M, N, K, Q4_GROUP_SIZE, A_scratch, A_scales);
  }
  times[4] = (get_time_ms() - start) / BENCHMARK_ITERATIONS;
  
  // Calculate accuracy metrics for each method
  float* results[] = {C_f32, C_q8q8, C_q8q4, C_f32q8, C_f32q4};
  
  printf("┌────────────────┬─────────────┬─────────────┬─────────────┬─────────────┬─────────────┐\n");
  printf("│ Method         │ Time (ms)   │ Speedup     │ GFLOPS      │ MAE         │ Cosine Sim  │\n");
  printf("├────────────────┼─────────────┼─────────────┼─────────────┼─────────────┼─────────────┤\n");
  
  for (int i = 0; i < 5; i++) {
    float mae, cosine_sim;
    if (i == 0) {
      mae = 0.0f; cosine_sim = 1.0f; // baseline
    } else {
      mae = calculate_mae(C_f32, results[i], M * N);
      cosine_sim = calculate_cosine_similarity(C_f32, results[i], M * N);
    }
    
    double gflops = (2.0 * M * N * K) / (times[i] * 1e6);
    double speedup = times[0] / times[i];
    
    printf("│ %-14s │ %11.2f │ %10.2fx │ %11.2f │ %11.6f │ %11.6f │\n",
           method_names[i], times[i], speedup, gflops, mae, cosine_sim);
  }
  printf("└────────────────┴─────────────┴─────────────┴─────────────┴─────────────┴─────────────┘\n");
  
  // Memory usage
  size_t fp32_mem = (M * K + N * K + M * N) * sizeof(float);
  size_t q8_mem = (M * K + N * K) * sizeof(int8_t) + (M + N) * (K / 128) * sizeof(float) + M * N * sizeof(float);
  size_t q4_mem = M * K * sizeof(float) + N * (K / 2) + 2 * N * (K / 128) * sizeof(float) + M * N * sizeof(float);
  
  printf("Memory: FP32=%.1fMB, Q8×Q8=%.1fMB (%.1fx), Q8×Q4=%.1fMB (%.1fx)\n",
         fp32_mem / 1024.0 / 1024.0,
         q8_mem / 1024.0 / 1024.0, (float)q8_mem / fp32_mem,
         q4_mem / 1024.0 / 1024.0, (float)q4_mem / fp32_mem);

cleanup_tensors:
  if (A_tensor) tensor_free_single(A_tensor);
  if (B_tensor) tensor_free_single(B_tensor);
  if (A_q8) quantized_tensor_free(A_q8);
  if (B_q8) quantized_tensor_free(B_q8);
  if (B_q4) quantized_tensor_free(B_q4);

cleanup:
  free(A_f32); free(B_f32); free(C_f32); free(C_q8q8); free(C_q8q4);
  free(C_f32q8); free(C_f32q4);
  free(A_scratch); free(A_scales);
}


int main(int argc, char* argv[]) {
  printf("Quantization Library Test Driver\n");
  printf("================================\n");
  printf("Compiled with OpenMP: %s\n", _OPENMP ? "Yes" : "No");
  printf("Available threads: %d\n", omp_get_max_threads());
  printf("Q8 Group Size: %d, Q4 Group Size: 32/128\n", Q8_GROUP_SIZE);
  
  // Skip accuracy tests for faster transpose optimization analysis
  printf("Skipping accuracy tests - focusing on transpose optimization...\n");
  
  
  int default_threads = omp_get_max_threads();
  
  // Skip separate transpose tests - now integrated into threaded benchmark
  printf("Transpose optimization integrated into threaded benchmark for direct comparison...\n");
  
  // Test multi-threaded benchmark on all matrix sizes
  printf("\n================================================================================\n");
  printf("MULTI-THREADED PERFORMANCE SCALING TEST\n");
  printf("================================================================================\n");
  for (int i = 0; i < NUM_MATRIX_SIZES; i++) {
    if ((MATRIX_SIZES[i].M == 1024 && MATRIX_SIZES[i].N == 1024 && MATRIX_SIZES[i].K == 1024) ||
        (MATRIX_SIZES[i].M == 2048 && MATRIX_SIZES[i].N == 2048 && MATRIX_SIZES[i].K == 2048)) {
      continue; // Skip large sizes
    }
    if (MATRIX_SIZES[i].N > 50000) continue;  // Skip LM_Head for threading test too
    
    benchmark_threading(&MATRIX_SIZES[i]);
  }
  
  return 0;
}
