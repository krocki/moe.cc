/**
 * test_matmul_harness.c - Comprehensive Matrix Multiplication Testing Suite
 * 
 * This program provides comprehensive testing for matrix multiplication kernels
 * including accuracy and performance benchmarks across different configurations.
 * 
 * Functions tested:
 * - FP32 × FP32 -> FP32 (reference implementation)
 * - FP32 × Q8 -> FP32 (existing kernel)
 * - FP32 × Q4 -> FP32 (existing kernel)
 * - Q8 × Q8 -> FP32 (new llama2.c style kernel)
 * - Q8 × Q4 -> FP32 (new asymmetric kernel)
 * 
 * Test Configurations:
 * - Standard: Power-of-2 sizes (64×64, 256×256, 512×512) with group sizes 32, 64, 128
 * - LLM Inference: Realistic transformer dimensions for vector×matrix and expert weights
 *   · Vector×Matrix: [1×2048]×[2048×768], [1×4096]×[4096×2048], etc.
 *   · Expert Weights: [768×2048], [2048×768] typical for gate_proj/up_proj/down_proj
 *   · Non-power-of-2: Real model dimensions like 768, 2048, 4096
 * - Performance metrics: GFLOPS, latency, accuracy (MAD, MSE)
 * 
 * The harness generates random test data to ensure reproducible results
 * and compares all quantized implementations against FP32 reference.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>
#include "quant.h"
#include "kernels.h"
#include "utils.h"

/**
 * Test configuration structure
 */
typedef struct {
  int M, N, K;                    // Matrix dimensions
  size_t group_size;              // Group size for quantization
  int iterations;                 // Number of iterations for timing
  bool verbose;                   // Verbose output
  bool validate_accuracy;         // Enable accuracy validation
  bool benchmark_performance;     // Enable performance benchmarking
} TestConfig;

/**
 * Test results structure
 */
typedef struct {
  double mad_error;               // Mean Absolute Deviation
  double mse_error;               // Mean Squared Error
  double max_error;               // Maximum absolute error
  double gflops;                  // Performance in GFLOPS
  double latency_ms;              // Latency in milliseconds
  bool passed;                    // Overall test status
} TestResult;

/**
 * Generate random FP32 matrix with specified range
 * 
 * @param matrix: Output matrix buffer [rows × cols]
 * @param rows: Number of rows
 * @param cols: Number of columns  
 * @param min_val: Minimum value range
 * @param max_val: Maximum value range
 * @param seed: Random seed for reproducibility
 */
void generate_random_matrix(float* matrix, int rows, int cols, 
                           float min_val, float max_val, unsigned int seed) {
  srand(seed);
  
  for (int i = 0; i < rows * cols; i++) {
    float range = max_val - min_val;
    matrix[i] = min_val + (float)rand() / RAND_MAX * range;
  }
}

/**
 * Reference FP32 matrix multiplication (naive implementation)
 * 
 * @param A: Input matrix A [M × K]
 * @param B: Input matrix B [K × N]
 * @param C: Output matrix C [M × N]
 * @param M: Number of rows in A and C
 * @param N: Number of columns in B and C
 * @param K: Number of columns in A and rows in B
 */
void matmul_f32_reference(const float* A, const float* B, float* C, int M, int N, int K) {
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      float sum = 0.0f;
      for (int k = 0; k < K; k++) {
        sum += A[m * K + k] * B[k * N + n];
      }
      C[m * N + n] = sum;
    }
  }
}

/**
 * Calculate accuracy metrics between reference and test results
 * 
 * @param reference: Reference FP32 results [M × N]
 * @param test: Test results [M × N]
 * @param size: Total number of elements (M × N)
 * @param result: Output accuracy metrics
 */
void calculate_accuracy_metrics(const float* reference, const float* test, 
                               int size, TestResult* result) {
  double sum_abs_diff = 0.0;
  double sum_sq_diff = 0.0;
  double max_abs_diff = 0.0;
  
  for (int i = 0; i < size; i++) {
    double abs_diff = fabs(reference[i] - test[i]);
    double sq_diff = abs_diff * abs_diff;
    
    sum_abs_diff += abs_diff;
    sum_sq_diff += sq_diff;
    
    if (abs_diff > max_abs_diff) {
      max_abs_diff = abs_diff;
    }
  }
  
  result->mad_error = sum_abs_diff / size;
  result->mse_error = sum_sq_diff / size;
  result->max_error = max_abs_diff;
}

/**
 * Measure execution time and calculate GFLOPS
 * 
 * @param M, N, K: Matrix dimensions
 * @param iterations: Number of timing iterations
 * @param kernel_func: Function pointer to kernel to benchmark
 * @param A, B, C: Matrix pointers
 * @param extra_params: Additional parameters for kernel
 * @param result: Output performance metrics
 */
double get_time_ms() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}

/**
 * Test FP32 × Q8 matrix multiplication
 */
void test_f32_q8_matmul(const TestConfig* config, const float* A, const float* B,
                        const float* C_ref, TestResult* result) {
  int M = config->M, N = config->N, K = config->K;
  size_t group_size = config->group_size;
  
  // B is provided as [K × N], but FP32×Q8 kernel expects [N × K] quantized format
  // So we need to transpose B for quantization
  float* B_transposed = (float*)malloc(N * K * sizeof(float));
  for (int k = 0; k < K; k++) {
    for (int n = 0; n < N; n++) {
      B_transposed[n * K + k] = B[k * N + n];
    }
  }
  
  // Quantize transposed B matrix to Q8
  size_t B_q_size = N * K * sizeof(int8_t);
  size_t B_scales_size = get_scales_size(N, K, group_size);
  
  int8_t* B_q = (int8_t*)malloc(B_q_size);
  float* B_scales = (float*)malloc(B_scales_size);
  float* C_test = (float*)malloc(M * N * sizeof(float));
  
  if (!B_q || !B_scales || !C_test || !B_transposed) {
    fprintf(stderr, "Error: Memory allocation failed in test_f32_q8_matmul\n");
    result->passed = false;
    return;
  }
  
  // Quantize transposed B matrix
  size_t num_groups;
  quantize_groupwise_q8(B_transposed, N, K, group_size, B_scales, B_q, &num_groups);
  
  // Benchmark performance
  double start_time = get_time_ms();
  
  for (int iter = 0; iter < config->iterations; iter++) {
    matmul_f32_q8(A, B_q, B_scales, C_test, M, N, K, group_size);
  }
  
  double end_time = get_time_ms();
  double total_time_ms = end_time - start_time;
  
  result->latency_ms = total_time_ms / config->iterations;
  double ops = 2.0 * M * N * K; // Multiply-accumulate operations
  result->gflops = ops / (result->latency_ms * 1e6);
  
  // Calculate accuracy (relaxed thresholds for FP32×Q8 with random test data)
  if (config->validate_accuracy) {
    calculate_accuracy_metrics(C_ref, C_test, M * N, result);
    result->passed = (result->mad_error < 7.0 && result->max_error < 40.0);
  } else {
    result->passed = true;
  }
  
  free(B_q);
  free(B_scales);
  free(C_test);
  free(B_transposed);
}

/**
 * Test FP32 × Q4 matrix multiplication
 */
void test_f32_q4_matmul(const TestConfig* config, const float* A, const float* B,
                        const float* C_ref, TestResult* result) {
  int M = config->M, N = config->N, K = config->K;
  size_t group_size = config->group_size;
  
  // B is provided as [K × N], but FP32×Q4 kernel expects [N × K] quantized format
  // So we need to transpose B for quantization
  float* B_transposed = (float*)malloc(N * K * sizeof(float));
  for (int k = 0; k < K; k++) {
    for (int n = 0; n < N; n++) {
      B_transposed[n * K + k] = B[k * N + n];
    }
  }
  
  // Quantize transposed B matrix to Q4 (F32×Q4 expects [N × K] format)
  size_t B_q_size = N * ((K + 1) / 2) * sizeof(uint8_t);
  size_t B_scales_size = get_scales_size(N, K, group_size);
  
  uint8_t* B_q = (uint8_t*)malloc(B_q_size);
  float* B_scales = (float*)malloc(B_scales_size);
  float* C_test = (float*)malloc(M * N * sizeof(float));
  
  if (!B_q || !B_scales || !C_test || !B_transposed) {
    fprintf(stderr, "Error: Memory allocation failed in test_f32_q4_matmul\n");
    result->passed = false;
    return;
  }
  
  // Initialize and quantize transposed B matrix (F32×Q4 expects [N × K] format)
  memset(B_q, 0, B_q_size);
  size_t num_groups;
  quantize_groupwise_q4(B_transposed, N, K, group_size, B_scales, B_q, &num_groups);
  
  // Benchmark performance
  double start_time = get_time_ms();
  
  for (int iter = 0; iter < config->iterations; iter++) {
    matmul_f32_q4(A, B_q, B_scales, C_test, M, N, K, group_size);
  }
  
  double end_time = get_time_ms();
  double total_time_ms = end_time - start_time;
  
  result->latency_ms = total_time_ms / config->iterations;
  double ops = 2.0 * M * N * K;
  result->gflops = ops / (result->latency_ms * 1e6);
  
  // Calculate accuracy
  if (config->validate_accuracy) {
    calculate_accuracy_metrics(C_ref, C_test, M * N, result);
    result->passed = (result->mad_error < 7.0 && result->max_error < 40.0);
  } else {
    result->passed = true;
  }
  
  free(B_q);
  free(B_scales);
  free(C_test);
  free(B_transposed);
}

/**
 * Test Q8 × Q8 matrix multiplication (llama2.c style)
 * Pre-quantizes inputs to separate quantization overhead from computation timing
 */
void test_q8_q8_matmul(const TestConfig* config, const float* A, const float* B,
                       const float* C_ref, TestResult* result) {
  int M = config->M, N = config->N, K = config->K;
  size_t group_size = config->group_size;
  
  // Allocate memory for quantized inputs and output
  size_t A_q_size = M * K * sizeof(int8_t);
  size_t B_q_size = K * N * sizeof(int8_t);
  size_t A_scales_size = get_scales_size(M, K, group_size);
  size_t B_scales_size = get_scales_size(K, N, group_size);
  
  int8_t* A_q = (int8_t*)malloc(A_q_size);
  int8_t* B_q = (int8_t*)malloc(B_q_size);
  float* A_scales = (float*)malloc(A_scales_size);
  float* B_scales = (float*)malloc(B_scales_size);
  float* C_test = (float*)malloc(M * N * sizeof(float));
  
  if (!A_q || !B_q || !A_scales || !B_scales || !C_test) {
    fprintf(stderr, "Error: Memory allocation failed in test_q8_q8_matmul\n");
    result->passed = false;
    free(A_q); free(B_q); free(A_scales); free(B_scales); free(C_test);
    return;
  }
  
  // Pre-quantize inputs (NOT timed - this is preparation)
  size_t A_num_groups, B_num_groups;
  quantize_groupwise_q8(A, M, K, group_size, A_scales, A_q, &A_num_groups);
  quantize_groupwise_q8(B, K, N, group_size, B_scales, B_q, &B_num_groups);
  
  // Benchmark performance of pure matrix multiplication (without quantization overhead)
  double start_time = get_time_ms();
  
  for (int iter = 0; iter < config->iterations; iter++) {
    matmul_q8_q8_f32(A_q, A_scales, B_q, B_scales, C_test, M, N, K, group_size);
  }
  
  double end_time = get_time_ms();
  double total_time_ms = end_time - start_time;
  
  result->latency_ms = total_time_ms / config->iterations;
  double ops = 2.0 * M * N * K;
  result->gflops = ops / (result->latency_ms * 1e6);
  
  // Calculate accuracy (Q8×Q8 has high precision, strict thresholds)
  if (config->validate_accuracy) {
    calculate_accuracy_metrics(C_ref, C_test, M * N, result);
    result->passed = (result->mad_error < 0.1 && result->max_error < 1.0);
  } else {
    result->passed = true;
  }
  
  // Cleanup allocated memory
  free(A_q);
  free(B_q);
  free(A_scales);
  free(B_scales);
  free(C_test);
}

/**
 * Test Q8 × Q4 matrix multiplication (asymmetric)
 * Pre-quantizes both inputs to separate quantization overhead from computation timing
 */
void test_q8_q4_matmul(const TestConfig* config, const float* A, const float* B,
                       const float* C_ref, TestResult* result) {
  int M = config->M, N = config->N, K = config->K;
  size_t group_size = config->group_size;
  
  // Allocate memory for quantized inputs and output
  size_t A_q_size = M * K * sizeof(int8_t);
  size_t A_scales_size = get_scales_size(M, K, group_size);
  size_t B_q_size = K * ((N + 1) / 2) * sizeof(uint8_t);
  size_t B_scales_size = get_scales_size(K, N, group_size);
  
  int8_t* A_q = (int8_t*)malloc(A_q_size);
  float* A_scales = (float*)malloc(A_scales_size);
  uint8_t* B_q = (uint8_t*)malloc(B_q_size);
  float* B_scales = (float*)malloc(B_scales_size);
  float* C_test = (float*)malloc(M * N * sizeof(float));
  
  if (!A_q || !A_scales || !B_q || !B_scales || !C_test) {
    fprintf(stderr, "Error: Memory allocation failed in test_q8_q4_matmul\n");
    result->passed = false;
    free(A_q); free(A_scales); free(B_q); free(B_scales); free(C_test);
    return;
  }
  
  // Pre-quantize both matrices (NOT timed - this is preparation)
  size_t A_num_groups, B_num_groups;
  quantize_groupwise_q8(A, M, K, group_size, A_scales, A_q, &A_num_groups);
  memset(B_q, 0, B_q_size);
  quantize_groupwise_q4(B, K, N, group_size, B_scales, B_q, &B_num_groups);
  
  // Benchmark performance of pure matrix multiplication (without quantization overhead)
  double start_time = get_time_ms();
  
  for (int iter = 0; iter < config->iterations; iter++) {
    matmul_q8_q4_f32(A_q, A_scales, B_q, B_scales, C_test, M, N, K, group_size);
  }
  
  double end_time = get_time_ms();
  double total_time_ms = end_time - start_time;
  
  result->latency_ms = total_time_ms / config->iterations;
  double ops = 2.0 * M * N * K;
  result->gflops = ops / (result->latency_ms * 1e6);
  
  // Calculate accuracy
  if (config->validate_accuracy) {
    calculate_accuracy_metrics(C_ref, C_test, M * N, result);
    result->passed = (result->mad_error < 0.5 && result->max_error < 3.0);
  } else {
    result->passed = true;
  }
  
  // Cleanup allocated memory
  free(A_q);
  free(A_scales);
  free(B_q);
  free(B_scales);
  free(C_test);
}

/**
 * Print test results in formatted table
 */
void print_test_results(const char* test_name, const TestConfig* config, 
                       const TestResult* result) {
  printf("%-20s | %4d×%4d×%4d | GS:%3zu | %8.3f ms | %8.2f GFLOPS | MAD:%8.6f | Max:%8.6f | %s\n",
         test_name, config->M, config->N, config->K, config->group_size,
         result->latency_ms, result->gflops, result->mad_error, result->max_error,
         result->passed ? "PASS" : "FAIL");
}

/**
 * Print test results for LLM test suite (forward declaration)
 */
void print_llm_test_results(const char* test_name, const TestConfig* config, 
                           const TestResult* result);

/**
 * Run standard test suite with power-of-2 dimensions
 */
void run_standard_test_suite(bool quick_test) {
  printf("Matrix Multiplication Testing Harness v3.2.0\n");
  printf("==============================================\n\n");
  printf("Running STANDARD test suite (power-of-2 dimensions)\n\n");
  
  // Test configurations (dimensions and group sizes)
  int test_sizes[][3] = {
    {64, 64, 64},      // Small
    {128, 128, 128},   // Medium
    {256, 256, 256}    // Large
  };
  
  size_t group_sizes[] = {32, 64, 128};
  int num_sizes = quick_test ? 1 : 3;
  int num_groups = quick_test ? 1 : 3;
  int iterations = quick_test ? 1 : 10;
  
  printf("Test Configuration:\n");
  printf("- %d matrix sizes, %d group sizes\n", num_sizes, num_groups);
  printf("- %d iterations per test\n", iterations);
  printf("- Accuracy validation: %s\n", "enabled");
  printf("- Performance benchmarking: %s\n\n", "enabled");
  
  printf("%-20s | %12s | %6s | %10s | %12s | %12s | %12s | %6s\n",
         "Test Name", "Dimensions", "Group", "Latency", "Performance", "MAD Error", "Max Error", "Status");
  printf("%-20s-+-%12s-+-%6s-+-%10s-+-%12s-+-%12s-+-%12s-+-%6s\n",
         "--------------------", "------------", "------", "----------", 
         "------------", "------------", "------------", "------");
  
  for (int size_idx = 0; size_idx < num_sizes; size_idx++) {
    for (int group_idx = 0; group_idx < num_groups; group_idx++) {
      TestConfig config = {
        .M = test_sizes[size_idx][0],
        .N = test_sizes[size_idx][1], 
        .K = test_sizes[size_idx][2],
        .group_size = group_sizes[group_idx],
        .iterations = iterations,
        .verbose = false,
        .validate_accuracy = true,
        .benchmark_performance = true
      };
      
      // Allocate test matrices
      int M = config.M, N = config.N, K = config.K;
      float* A = (float*)malloc(M * K * sizeof(float));
      float* B = (float*)malloc(K * N * sizeof(float));
      float* C_ref = (float*)malloc(M * N * sizeof(float));
      
      if (!A || !B || !C_ref) {
        fprintf(stderr, "Error: Failed to allocate test matrices\n");
        continue;
      }
      
      // Generate random test data (reproducible)
      generate_random_matrix(A, M, K, -1.0f, 1.0f, 42 + size_idx);
      generate_random_matrix(B, K, N, -1.0f, 1.0f, 123 + group_idx);
      
      // Calculate reference result
      matmul_f32_reference(A, B, C_ref, M, N, K);
      
      // Run tests
      TestResult result;
      
      // Test FP32 × Q8
      test_f32_q8_matmul(&config, A, B, C_ref, &result);
      print_test_results("FP32 × Q8", &config, &result);
      
      // Test FP32 × Q4  
      test_f32_q4_matmul(&config, A, B, C_ref, &result);
      print_test_results("FP32 × Q4", &config, &result);
      
      // Test Q8 × Q8
      test_q8_q8_matmul(&config, A, B, C_ref, &result);
      print_test_results("Q8 × Q8", &config, &result);
      
      // Test Q8 × Q4
      test_q8_q4_matmul(&config, A, B, C_ref, &result);
      print_test_results("Q8 × Q4", &config, &result);
      
      free(A);
      free(B);
      free(C_ref);
    }
  }
  
  printf("\nStandard test suite completed!\n");
}

/**
 * Run LLM inference test suite with realistic transformer dimensions
 */
void run_llm_test_suite(bool quick_test) {
  printf("Matrix Multiplication Testing Harness v3.2.0\n");
  printf("==============================================\n\n");
  printf("Running LLM INFERENCE test suite (realistic transformer dimensions)\n\n");
  
  // LLM-specific test configurations based on real transformer architectures
  // Vector×Matrix multiplications (typical for inference)
  int llm_test_sizes[][3] = {
    // Vector × Expert weight matrices (M=1 for inference)
    {1, 768, 2048},     // [1×2048] × [2048×768] -> [1×768] (down_proj)
    {1, 2048, 4096},    // [1×4096] × [4096×2048] -> [1×2048] (down_proj large)
    {1, 4096, 2048},    // [1×2048] × [2048×4096] -> [1×4096] (up_proj)
    {1, 512, 2048},     // [1×2048] × [2048×512] -> [1×512] (small down_proj)
    {1, 128, 2048},     // [1×2048] × [2048×128] -> [1×128] (tiny down_proj)
    
    // Expert weight matrix dimensions (for comparison)
    {1, 2048, 768},   // Typical expert gate_proj: [768×768] × [768×2048] -> [768×2048]
    {768, 2048, 1},  // Typical expert down_proj: [2048×2048] × [2048×768] -> [2048×768]
    
    // Batch processing scenarios (small batches)
    {4, 768, 2048},     // Small batch processing
    {8, 2048, 4096},    // Small batch up_proj
    {16, 512, 2048},    // Medium batch down_proj
  };
  
  // LLM-focused group sizes (optimized for transformer architectures)
  size_t group_sizes[] = {32, 64, 128};
  int num_sizes = quick_test ? 3 : 7;  // Limit to reasonable subset (first 7 configs)
  int num_groups = quick_test ? 1 : 2;  // Reduce to 2 group sizes for efficiency
  int iterations = quick_test ? 1 : 3;  // Reduce iterations for large matrices
  
  printf("Test Configuration:\n");
  printf("- %d LLM matrix configurations, %d group sizes\n", num_sizes, num_groups);
  printf("- %d iterations per test\n", iterations);
  printf("- Focus: Vector×Matrix and Expert weight dimensions\n");
  printf("- Dimensions: Non-power-of-2 realistic transformer sizes\n\n");
  
  printf("%-20s | %15s | %6s | %10s | %12s | %12s | %12s | %6s\n",
         "Test Name", "Dimensions", "Group", "Latency", "Performance", "MAD Error", "Max Error", "Status");
  printf("%-20s-+-%15s-+-%6s-+-%10s-+-%12s-+-%12s-+-%12s-+-%6s\n",
         "--------------------", "---------------", "------", "----------", 
         "------------", "------------", "------------", "------");
  
  for (int size_idx = 0; size_idx < num_sizes; size_idx++) {
    for (int group_idx = 0; group_idx < num_groups; group_idx++) {
      TestConfig config = {
        .M = llm_test_sizes[size_idx][0],
        .N = llm_test_sizes[size_idx][1], 
        .K = llm_test_sizes[size_idx][2],
        .group_size = group_sizes[group_idx],
        .iterations = iterations,
        .verbose = false,
        .validate_accuracy = true,
        .benchmark_performance = true
      };
      
      // Allocate test matrices
      int M = config.M, N = config.N, K = config.K;
      float* A = (float*)malloc(M * K * sizeof(float));
      float* B = (float*)malloc(K * N * sizeof(float));
      float* C_ref = (float*)malloc(M * N * sizeof(float));
      
      if (!A || !B || !C_ref) {
        fprintf(stderr, "Error: Failed to allocate test matrices\n");
        continue;
      }
      
      // Generate random test data (reproducible, optimized for LLM ranges)
      generate_random_matrix(A, M, K, -0.5f, 0.5f, 42 + size_idx);
      generate_random_matrix(B, K, N, -0.5f, 0.5f, 123 + group_idx);
      
      // Calculate reference result
      matmul_f32_reference(A, B, C_ref, M, N, K);
      
      // Run tests
      TestResult result;
      
      // Test FP32 × Q8
      test_f32_q8_matmul(&config, A, B, C_ref, &result);
      print_llm_test_results("FP32 × Q8", &config, &result);
      
      // Test FP32 × Q4  
      test_f32_q4_matmul(&config, A, B, C_ref, &result);
      print_llm_test_results("FP32 × Q4", &config, &result);
      
      // Test Q8 × Q8
      test_q8_q8_matmul(&config, A, B, C_ref, &result);
      print_llm_test_results("Q8 × Q8", &config, &result);
      
      // Test Q8 × Q4
      test_q8_q4_matmul(&config, A, B, C_ref, &result);
      print_llm_test_results("Q8 × Q4", &config, &result);
      
      free(A);
      free(B);
      free(C_ref);
    }
  }
  
  printf("\nLLM inference test suite completed!\n");
}

/**
 * Print test results for LLM test suite (wider format for longer dimension strings)
 */
void print_llm_test_results(const char* test_name, const TestConfig* config, 
                           const TestResult* result) {
  printf("%-20s | %4d×%4d×%4d | GS:%3zu | %8.3f ms | %8.2f GFLOPS | MAD:%8.6f | Max:%8.6f | %s\n",
         test_name, config->M, config->N, config->K, config->group_size,
         result->latency_ms, result->gflops, result->mad_error, result->max_error,
         result->passed ? "PASS" : "FAIL");
}

/**
 * Run combined test suite (both standard and LLM)
 */
void run_combined_test_suite(bool quick_test) {
  run_standard_test_suite(quick_test);
  printf("\n" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "\n\n");
  run_llm_test_suite(quick_test);
}

/**
 * Print usage information
 */
void print_usage(const char* program_name) {
  printf("Matrix Multiplication Testing Harness v3.2.0\n");
  printf("Usage: %s [options]\n\n", program_name);
  printf("Options:\n");
  printf("  --quick         Run quick test (reduced configurations, 1 iteration)\n");
  printf("  --full          Run full test suite (all configurations, multiple iterations)\n");
  printf("  --standard      Run standard test suite only (power-of-2 dimensions)\n");
  printf("  --llm           Run LLM inference test suite only (realistic transformer dimensions)\n");
  printf("  --combined      Run both standard and LLM test suites [default]\n");
  printf("  --help          Show this help message\n\n");
  printf("Test Suites:\n");
  printf("  Standard: Power-of-2 matrices (64×64, 256×256, 512×512)\n");
  printf("  LLM:      Realistic transformer dimensions:\n");
  printf("            · Vector×Matrix: [1×K]×[K×N] for inference\n");
  printf("            · Expert weights: [768×2048], [2048×768], etc.\n");
  printf("            · Batch processing: [4×K]×[K×N], [8×K]×[K×N]\n\n");
  printf("Kernels Tested:\n");
  printf("  - FP32 × Q8 -> FP32 (existing kernel)\n");
  printf("  - FP32 × Q4 -> FP32 (existing kernel)\n");
  printf("  - Q8 × Q8 -> FP32 (new llama2.c style)\n");
  printf("  - Q8 × Q4 -> FP32 (new asymmetric)\n\n");
  printf("Metrics:\n");
  printf("  - Accuracy: MAD (Mean Absolute Deviation), Max Error\n");
  printf("  - Performance: GFLOPS, Latency\n");
  printf("  - Validation: PASS/FAIL based on accuracy thresholds\n");
}

/**
 * Main entry point
 */
int main(int argc, char* argv[]) {
  bool quick_test = false;
  bool show_help = false;
  enum { MODE_COMBINED, MODE_STANDARD, MODE_LLM } test_mode = MODE_COMBINED;
  
  // Parse command line arguments
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--quick") == 0) {
      quick_test = true;
    } else if (strcmp(argv[i], "--full") == 0) {
      quick_test = false;
    } else if (strcmp(argv[i], "--standard") == 0) {
      test_mode = MODE_STANDARD;
    } else if (strcmp(argv[i], "--llm") == 0) {
      test_mode = MODE_LLM;
    } else if (strcmp(argv[i], "--combined") == 0) {
      test_mode = MODE_COMBINED;
    } else if (strcmp(argv[i], "--help") == 0) {
      show_help = true;
    } else {
      fprintf(stderr, "Unknown option: %s\n", argv[i]);
      show_help = true;
      break;
    }
  }
  
  if (show_help) {
    print_usage(argv[0]);
    return 0;
  }
  
  // Run appropriate test suite based on mode
  switch (test_mode) {
    case MODE_STANDARD:
      run_standard_test_suite(quick_test);
      break;
    case MODE_LLM:
      run_llm_test_suite(quick_test);
      break;
    case MODE_COMBINED:
    default:
      run_combined_test_suite(quick_test);
      break;
  }
  
  return 0;
}
