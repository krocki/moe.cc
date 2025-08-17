/**
 * test_convert_integration.c - Integration Test for Convert Program
 * 
 * This test creates actual tensor files and runs the convert program
 * to verify end-to-end functionality and compatibility with export.py output.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>
#include <math.h>
#include "io.h"
#include "quant.h"

static int test_count = 0;
static int test_passed = 0;

#define TEST_ASSERT(condition, message) do { \
  test_count++; \
  if (condition) { \
    test_passed++; \
    printf("PASS: %s\n", message); \
  } else { \
    printf("FAIL: %s\n", message); \
  } \
} while(0)

/**
 * Create a test tensor with known data for in-memory use
 */
static TensorBin* create_test_tensor(const char* name, int rows, int cols, float base_value) {
  int shape[2] = {rows, cols};
  size_t total_elements = rows * cols;
  
  float* data = malloc(total_elements * sizeof(float));
  if (!data) return NULL;
  
  // Fill with test pattern
  for (int i = 0; i < total_elements; i++) {
    data[i] = base_value + (float)(i % 100) * 0.1f;
  }
  
  TensorBin* tensor = tensor_create(name, 0, 2, shape, data);
  free(data);
  return tensor;
}

/**
 * Create a test tensor file with known data
 */
static int create_test_file(const char* filename, const char* tensor_name, 
                           int rows, int cols, float base_value) {
  // Create test data
  int shape[2] = {rows, cols};
  size_t total_elements = rows * cols;
  
  float* data = malloc(total_elements * sizeof(float));
  if (!data) return -1;
  
  // Fill with test pattern
  for (int i = 0; i < total_elements; i++) {
    data[i] = base_value + (float)(i % 100) * 0.1f;
  }
  
  // Create tensor and save
  TensorBin* tensor = tensor_create(tensor_name, 0, 2, shape, data);
  free(data);
  
  if (!tensor) return -1;
  
  int result = bin_save_single_tensor(tensor, filename);
  tensor_free_single(tensor);
  
  return result;
}

/**
 * Run convert program and return exit status
 */
static int run_convert(const char* input, const char* output, const char* quant) {
  char cmd[1024];
  snprintf(cmd, sizeof(cmd), "./convert --input %s --output %s --quant %s", 
           input, output, quant);
  
  return system(cmd);
}

/**
 * Compare tensor counts between input and output files
 */
static void test_tensor_counts(const char* input_file, const char* output_file, 
                              const char* quant_type, int expected_ratio) {
  BinFile* input_bf = bin_load(input_file);
  BinFile* output_bf = bin_load(output_file);
  
  if (!input_bf || !output_bf) {
    printf("FAIL: Could not load files for tensor count test\n");
    test_count++;
    return;
  }
  
  printf("Input tensors: %d, Output tensors: %d\n", input_bf->count, output_bf->count);
  
  // For expert tensors with quantization, we expect 2 output tensors per quantized input
  // Non-expert tensors remain unchanged (1:1 ratio)
  
  char message[256];
  snprintf(message, sizeof(message), "Tensor count validation for %s quantization", quant_type);
  
  // Basic validation - output should have >= input count
  TEST_ASSERT(output_bf->count >= input_bf->count, message);
  
  bin_free(input_bf);
  bin_free(output_bf);
}

/**
 * Verify that quantized tensors have correct naming
 */
static void test_quantized_tensor_names(const char* output_file, const char* quant_type) {
  BinFile* output_bf = bin_load(output_file);
  if (!output_bf) {
    printf("FAIL: Could not load output file for name test\n");
    test_count++;
    return;
  }
  
  bool found_scale = false;
  bool found_quant = false;
  
  for (int i = 0; i < output_bf->count; i++) {
    const char* name = output_bf->arr[i].name;
    if (strstr(name, ".scale")) found_scale = true;
    if (strcmp(quant_type, "q8") == 0 && strstr(name, ".q8")) found_quant = true;
    if (strcmp(quant_type, "q4") == 0 && strstr(name, ".q4")) found_quant = true;
  }
  
  char message[256];
  snprintf(message, sizeof(message), "Found .scale tensors in %s output", quant_type);
  TEST_ASSERT(found_scale, message);
  
  snprintf(message, sizeof(message), "Found .%s tensors in %s output", quant_type, quant_type);
  TEST_ASSERT(found_quant, message);
  
  bin_free(output_bf);
}

/**
 * Main integration test
 */
int main(void) {
  printf("Convert Program Integration Test\n");
  printf("==============================\n");
  
  // Clean up any existing test files
  unlink("/tmp/test_input.bin");
  unlink("/tmp/test_expert.bin");
  unlink("/tmp/test_mixed.bin");
  unlink("/tmp/test_output_q8.bin");
  unlink("/tmp/test_output_q4.bin");
  unlink("/tmp/test_output_none.bin");
  
  // Test 1: Create test files
  printf("\n1. Creating test files...\n");
  
  int result1 = create_test_file("/tmp/test_expert.bin", 
                                "model.layers.0.mlp.experts.0.gate_proj.weight", 
                                768, 2048, 100.0f);
  TEST_ASSERT(result1 == 0, "Created expert tensor test file");
  
  // Test 2: Create mixed model file
  printf("\n2. Creating mixed model file...\n");
  
  BinFile* mixed_bf = binfile_create();
  if (mixed_bf) {
    // Add expert tensor
    TensorBin* expert = create_test_tensor("model.layers.0.mlp.experts.0.gate_proj.weight", 
                                          100, 200, 50.0f);
    if (expert) {
      binfile_add_tensor(mixed_bf, expert);
      tensor_free_single(expert);
    }
    
    // Add non-expert tensor  
    TensorBin* attention = create_test_tensor("model.layers.0.self_attn.q_proj.weight",
                                             150, 100, 25.0f);
    if (attention) {
      binfile_add_tensor(mixed_bf, attention);
      tensor_free_single(attention);
    }
    
    int mixed_result = bin_save(mixed_bf, "/tmp/test_mixed.bin");
    TEST_ASSERT(mixed_result == 0, "Created mixed model test file");
    bin_free(mixed_bf);
  }
  
  // Test 3: Test Q8 conversion
  printf("\n3. Testing Q8 conversion...\n");
  
  int q8_result = run_convert("/tmp/test_mixed.bin", "/tmp/test_output_q8.bin", "q8");
  TEST_ASSERT(q8_result == 0, "Q8 conversion completed successfully");
  
  if (q8_result == 0) {
    test_tensor_counts("/tmp/test_mixed.bin", "/tmp/test_output_q8.bin", "q8", 2);
    test_quantized_tensor_names("/tmp/test_output_q8.bin", "q8");
  }
  
  // Test 4: Test Q4 conversion
  printf("\n4. Testing Q4 conversion...\n");
  
  int q4_result = run_convert("/tmp/test_mixed.bin", "/tmp/test_output_q4.bin", "q4");
  TEST_ASSERT(q4_result == 0, "Q4 conversion completed successfully");
  
  if (q4_result == 0) {
    test_tensor_counts("/tmp/test_mixed.bin", "/tmp/test_output_q4.bin", "q4", 2);
    test_quantized_tensor_names("/tmp/test_output_q4.bin", "q4");
  }
  
  // Test 5: Test no quantization
  printf("\n5. Testing no quantization...\n");
  
  int none_result = run_convert("/tmp/test_mixed.bin", "/tmp/test_output_none.bin", "none");
  TEST_ASSERT(none_result == 0, "No quantization conversion completed successfully");
  
  if (none_result == 0) {
    test_tensor_counts("/tmp/test_mixed.bin", "/tmp/test_output_none.bin", "none", 1);
  }
  
  // Test 6: Test help option
  printf("\n6. Testing help option...\n");
  
  int help_result = system("./convert --help");
  TEST_ASSERT(help_result == 0, "Help option works correctly");
  
  // Test 7: Test error cases
  printf("\n7. Testing error cases...\n");
  
  int error1 = system("./convert --input /nonexistent.bin --output /tmp/out.bin --quant q8");
  TEST_ASSERT(error1 != 0, "Convert handles non-existent input file");
  
  int error2 = system("./convert --input /tmp/test_mixed.bin --quant invalid --output /tmp/out.bin");
  TEST_ASSERT(error2 != 0, "Convert handles invalid quantization type");
  
  // Summary
  printf("\n==================================================\n");
  printf("Integration Test Summary\n");
  printf("Tests passed: %d/%d\n", test_passed, test_count);
  printf("Success rate: %.1f%%\n", test_count > 0 ? (100.0 * test_passed / test_count) : 0.0);
  
  // Cleanup
  unlink("/tmp/test_input.bin");
  unlink("/tmp/test_expert.bin");
  unlink("/tmp/test_mixed.bin");
  unlink("/tmp/test_output_q8.bin");
  unlink("/tmp/test_output_q4.bin");
  unlink("/tmp/test_output_none.bin");
  
  return (test_passed == test_count) ? 0 : 1;
}