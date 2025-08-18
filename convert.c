/**
 * convert.c - Tensor Quantization Conversion Tool
 * 
 * This program converts FP32 tensors to quantized formats (Q8 or Q4) without
 * requiring the Python export pipeline. It can process:
 * 1. Individual tensor files (e.g., layer.weight.bin)
 * 2. Complete model files (e.g., all.bin with all tensors)
 * 
 * Usage examples:
 *   ./convert --input all.bin --quant q8 --output all_q8.bin
 *   ./convert --input tensor.bin --quant q4 --output tensor_q4.bin
 *   ./convert --help
 * 
 * The output format matches export.py conventions:
 * - Q8: Creates .scale (f32) and .q8 (i8) tensor pairs
 * - Q4: Creates .scale (f32) and .q4 (u8 packed) tensor pairs
 * - Only expert weight matrices are quantized (gate_proj, up_proj, down_proj)
 */

#define CONVERT_VERSION "2.1.0"  // Version for debugging and compatibility tracking - Added streaming I/O

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <sys/stat.h>
#include <errno.h>
#include "io.h"
#include "quant.h"

/**
 * Program configuration structure
 */
typedef struct {
  char* input_path;      // Input .bin file path
  char* output_path;     // Output .bin file path  
  QuantType quant_type;  // Quantization type (Q8, Q4, or NONE)
  size_t group_size;     // Group size for quantization (0 = rowwise)
  bool verbose;          // Enable verbose output
  bool help;             // Show help and exit
  bool version;          // Show version and exit
} ConvertConfig;

/**
 * Print usage information
 */
static void print_usage(const char* program_name) {
  printf("Tensor Quantization Conversion Tool v%s\n", CONVERT_VERSION);
  printf("Converts FP32 tensors to quantized formats (Q8/Q4) in C\n\n");
  printf("Usage: %s --input INPUT.bin --quant QUANT_TYPE --output OUTPUT.bin\n\n", program_name);
  printf("Required Arguments:\n");
  printf("  --input PATH    Input .bin file (FP32 tensors)\n");
  printf("  --quant TYPE    Quantization type: q8, q4, or none\n");
  printf("  --output PATH   Output .bin file (quantized tensors)\n\n");
  printf("Optional Arguments:\n");
  printf("  --group-size N  Group size for quantization (default: 0 = rowwise)\n");
  printf("  --verbose       Enable verbose output\n");
  printf("  --version       Show version information\n");
  printf("  --help          Show this help message\n\n");
  printf("Examples:\n");
  printf("  # Convert complete model to Q8\n");
  printf("  %s --input all.bin --quant q8 --output all_q8.bin\n\n", program_name);
  printf("  # Convert complete model to Q4\n");
  printf("  %s --input all.bin --quant q4 --output all_q4.bin\n\n", program_name);
  printf("  # Convert single tensor\n");
  printf("  %s --input layer.weight.bin --quant q8 --output layer.weight_q8.bin\n\n", program_name);
  printf("  # Convert with group quantization\n");
  printf("  %s --input all.bin --quant q4 --group-size 128 --output all_q4_g128.bin\n\n", program_name);
  printf("Notes:\n");
  printf("  - Only expert weight matrices are quantized (gate_proj, up_proj, down_proj)\n");
  printf("  - Other tensors (attention, norms, embeddings) remain FP32\n");
  printf("  - Output format matches export.py conventions\n");
  printf("  - Quantized tensors create .scale + .q8/.q4 pairs\n");
}

/**
 * Parse command line arguments
 */
static int parse_arguments(int argc, char* argv[], ConvertConfig* config) {
  // Initialize config with defaults
  memset(config, 0, sizeof(ConvertConfig));
  config->quant_type = QUANT_NONE;
  config->group_size = 0; // Default to rowwise quantization
  
  // Define long options
  static struct option long_options[] = {
    {"input",      required_argument, 0, 'i'},
    {"output",     required_argument, 0, 'o'},
    {"quant",      required_argument, 0, 'q'},
    {"group-size", required_argument, 0, 'g'},
    {"verbose",    no_argument,       0, 'v'},
    {"version",    no_argument,       0, 'V'},
    {"help",       no_argument,       0, 'h'},
    {0, 0, 0, 0}
  };
  
  int option_index = 0;
  int c;
  
  while ((c = getopt_long(argc, argv, "i:o:q:g:vVh", long_options, &option_index)) != -1) {
    switch (c) {
      case 'i':
        config->input_path = strdup(optarg);
        break;
      case 'o':
        config->output_path = strdup(optarg);
        break;
      case 'q':
        if (strcmp(optarg, "q8") == 0) {
          config->quant_type = QUANT_Q8;
        } else if (strcmp(optarg, "q4") == 0) {
          config->quant_type = QUANT_Q4;
        } else if (strcmp(optarg, "none") == 0) {
          config->quant_type = QUANT_NONE;
        } else {
          fprintf(stderr, "Error: Invalid quantization type '%s'. Use q8, q4, or none.\n", optarg);
          return -1;
        }
        break;
      case 'g':
        {
          char* endptr;
          long group_size = strtol(optarg, &endptr, 10);
          if (*endptr != '\0' || group_size < 0) {
            fprintf(stderr, "Error: Invalid group size '%s'. Must be a non-negative integer.\n", optarg);
            return -1;
          }
          config->group_size = (size_t)group_size;
        }
        break;
      case 'v':
        config->verbose = true;
        break;
      case 'V':
        config->version = true;
        return 0;
      case 'h':
        config->help = true;
        return 0;
      case '?':
        // getopt_long already printed an error message
        return -1;
      default:
        fprintf(stderr, "Error: Unknown option\n");
        return -1;
    }
  }
  
  // Validate required arguments
  if (!config->help) {
    if (!config->input_path) {
      fprintf(stderr, "Error: --input is required\n");
      return -1;
    }
    if (!config->output_path) {
      fprintf(stderr, "Error: --output is required\n");
      return -1;
    }
    if (config->quant_type == QUANT_NONE) {
      fprintf(stderr, "Error: --quant is required (q8, q4, or none)\n");
      return -1;
    }
  }
  
  return 0;
}

/**
 * Free configuration memory
 */
static void free_config(ConvertConfig* config) {
  free(config->input_path);
  free(config->output_path);
}

/**
 * Check if a file exists and is readable
 */
static bool file_exists(const char* path) {
  struct stat st;
  return (stat(path, &st) == 0) && S_ISREG(st.st_mode);
}


/**
 * Convert a single tensor by applying quantization if appropriate (streaming version)
 * Writes output tensors directly to stream writer instead of accumulating in memory
 * Returns the number of tensors written to output (1 for unquantized, 2 for quantized)
 */
static int convert_single_tensor_streaming(const TensorBin* input_tensor, BinStreamWriter* writer, 
                                          QuantType quant_type, size_t group_size, bool verbose) {
  if (!input_tensor || !writer) return 0;
  
  const char* tensor_name = input_tensor->name;
  bool should_quantize = (quant_type != QUANT_NONE) && 
                        should_quantize_tensor(tensor_name) && 
                        (input_tensor->ndim == 2) && 
                        (input_tensor->dtype == 0); // Only quantize f32 2D tensors
  
  if (verbose) {
    printf("Processing tensor: %s [%s] ", tensor_name, 
           should_quantize ? "quantize" : "keep_fp32");
    for (int i = 0; i < input_tensor->ndim; i++) {
      printf("%s%d", (i == 0) ? "[" : ",", input_tensor->shape[i]);
    }
    printf("]");
    if (should_quantize) {
      printf(" group_size=%zu", group_size);
    }
    printf("\n");
  }
  
  if (!should_quantize) {
    // Write tensor as-is (no quantization)
    if (bin_stream_writer_write_tensor(writer, input_tensor) != 0) {
      fprintf(stderr, "Error: Failed to write tensor %s to output stream\n", tensor_name);
      return 0;
    }
    return 1;
  }
  
  // Quantize the tensor
  QuantizedTensor* qt = quantize_tensor(input_tensor, quant_type, group_size);
  if (!qt) {
    fprintf(stderr, "Error: Failed to quantize tensor %s\n", tensor_name);
    return 0;
  }
  
  // Create scale tensor name and data
  char scale_name[512];
  snprintf(scale_name, sizeof(scale_name), "%s.scale", tensor_name);
  
  // Create scale tensor (1D array of per-row or per-group scales)
  int scale_shape[1] = { (int)qt->num_groups };
  TensorBin* scale_tensor = tensor_create_with_group_size(scale_name, 0, 1, scale_shape, qt->scales, group_size);
  if (!scale_tensor) {
    fprintf(stderr, "Error: Failed to create scale tensor for %s\n", tensor_name);
    quantized_tensor_free(qt);
    return 0;
  }
  
  // Create quantized data tensor name and parameters
  char quant_name[512];
  int quant_dtype;
  int quant_shape[2];
  
  if (quant_type == QUANT_Q8) {
    snprintf(quant_name, sizeof(quant_name), "%s.q8", tensor_name);
    quant_dtype = 2; // i8
    quant_shape[0] = input_tensor->shape[0]; // rows
    quant_shape[1] = input_tensor->shape[1]; // cols
  } else { // QUANT_Q4
    snprintf(quant_name, sizeof(quant_name), "%s.q4", tensor_name);
    quant_dtype = 3; // i4 (packed)
    quant_shape[0] = input_tensor->shape[0]; // rows
    quant_shape[1] = (input_tensor->shape[1] + 1) / 2; // packed cols
  }
  
  // Create quantized tensor
  TensorBin* quant_tensor = tensor_create_with_group_size(quant_name, quant_dtype, 2, quant_shape, qt->q_data, group_size);
  if (!quant_tensor) {
    fprintf(stderr, "Error: Failed to create quantized tensor for %s\n", tensor_name);
    tensor_free_single(scale_tensor);
    quantized_tensor_free(qt);
    return 0;
  }
  
  // Write both tensors to output stream
  int written_count = 0;
  if (bin_stream_writer_write_tensor(writer, scale_tensor) == 0) written_count++;
  if (bin_stream_writer_write_tensor(writer, quant_tensor) == 0) written_count++;
  
  // Cleanup
  tensor_free_single(scale_tensor);
  tensor_free_single(quant_tensor);
  quantized_tensor_free(qt);
  
  if (written_count != 2) {
    fprintf(stderr, "Error: Failed to write quantized tensors for %s\n", tensor_name);
    return 0;
  }
  
  return 2; // Wrote scale + quantized tensors
}

/**
 * Show progress during streaming conversion
 * Reuses existing progress drawing logic from io.c with streaming reader state
 */
static void show_conversion_progress(BinStreamReader* reader, const char* tensor_name, 
                                   uint32_t tensors_processed) {
  if (!reader || reader->total_bytes == 0) return;
  
  size_t done_bytes = reader->bytes_read;
  size_t total_bytes = reader->total_bytes;
  
  // Use the existing progress_draw function from io.c
  progress_draw(done_bytes, total_bytes, (size_t)tensors_processed, (size_t)reader->total_tensors, tensor_name);
}

/**
 * Main conversion function using streaming I/O for memory efficiency
 * 
 * This function processes tensors one-by-one without loading the entire
 * input file into memory, significantly reducing peak memory usage.
 */
static int convert_tensors(const ConvertConfig* config) {
  // Validate input file exists
  if (!file_exists(config->input_path)) {
    fprintf(stderr, "Error: Input file does not exist: %s\n", config->input_path);
    return 1;
  }
  
  if (config->verbose) {
    printf("Tensor Quantization Conversion Tool v%s (Streaming Mode)\n", CONVERT_VERSION);
    printf("Processing tensors from: %s\n", config->input_path);
  }
  
  // Open input file for streaming reading
  BinStreamReader* reader = bin_stream_reader_open(config->input_path);
  if (!reader) {
    fprintf(stderr, "Error: Failed to open input file for streaming: %s\n", config->input_path);
    return 1;
  }
  
  // Open output file for streaming writing
  BinStreamWriter* writer = bin_stream_writer_open(config->output_path);
  if (!writer) {
    fprintf(stderr, "Error: Failed to open output file for streaming: %s\n", config->output_path);
    bin_stream_reader_close(reader);
    return 1;
  }
  
  if (config->verbose) {
    printf("Estimated input tensors: %u\n", reader->total_tensors);
    const char* quant_str = (config->quant_type == QUANT_Q8) ? "Q8" : 
                           (config->quant_type == QUANT_Q4) ? "Q4" : "NONE";
    printf("Converting with quantization: %s", quant_str);
    if (config->group_size > 0) {
      printf(" (group_size=%zu)", config->group_size);
    }
    printf("\n");
  }
  
  // Process tensors one by one in streaming fashion
  TensorBin tensor;
  int total_output_tensors = 0;
  int quantized_tensors = 0;
  int result;
  
  // Show initial progress
  if (reader->total_bytes > 0) {
    show_conversion_progress(reader, "", 0);
  }
  
  while ((result = bin_stream_reader_next_tensor(reader, &tensor)) == 1) {
    // Convert and write tensor using streaming approach
    int written = convert_single_tensor_streaming(&tensor, writer, config->quant_type, config->group_size, config->verbose);
    
    if (written == 0) {
      fprintf(stderr, "Error: Failed to convert tensor %s\n", tensor.name);
      // Cleanup allocated tensor data
      free(tensor.name);
      free(tensor.shape);
      free(tensor.data);
      bin_stream_writer_close(writer);
      bin_stream_reader_close(reader);
      return 1;
    }
    
    total_output_tensors += written;
    if (written == 2) quantized_tensors++; // Quantized tensors create 2 outputs
    
    // Update progress display
    if (reader->total_bytes > 0) {
      show_conversion_progress(reader, tensor.name, reader->tensors_read);
    }
    
    // Free tensor data (allocated by streaming reader)
    free(tensor.name);
    free(tensor.shape);
    free(tensor.data);
  }
  
  if (result == -1) {
    fprintf(stderr, "Error: Failed to read tensor from input stream\n");
    bin_stream_writer_close(writer);
    bin_stream_reader_close(reader);
    return 1;
  }
  
  // Complete progress display
  if (reader->total_bytes > 0) {
    progress_done();
  }
  
  if (config->verbose) {
    printf("Conversion complete:\n");
    printf("  Input tensors processed: %u\n", reader->tensors_read);
    printf("  Output tensors written: %d\n", total_output_tensors);
    printf("  Quantized tensors: %d\n", quantized_tensors);
    printf("Finalizing output file: %s\n", config->output_path);
  }
  
  // Finalize output file (updates tensor count in header)
  if (bin_stream_writer_finalize(writer) != 0) {
    fprintf(stderr, "Error: Failed to finalize output file: %s\n", config->output_path);
    bin_stream_writer_close(writer);
    bin_stream_reader_close(reader);
    return 1;
  }
  
  // Cleanup
  bin_stream_writer_close(writer);
  bin_stream_reader_close(reader);
  
  printf("Successfully quantized %u tensors to %s\n", 
         quantized_tensors, config->output_path);
  
  return 0;
}

/**
 * Main entry point
 */
int main(int argc, char* argv[]) {
  ConvertConfig config;
  
  // Parse command line arguments
  int parse_result = parse_arguments(argc, argv, &config);
  if (parse_result != 0) {
    if (parse_result == -1) {
      print_usage(argv[0]);
      return 1;
    }
    return 0; // Help was shown
  }
  
  // Show version if requested
  if (config.version) {
    printf("convert v%s\n", CONVERT_VERSION);
    free_config(&config);
    return 0;
  }
  
  // Show help if requested
  if (config.help) {
    print_usage(argv[0]);
    free_config(&config);
    return 0;
  }
  
  // Perform conversion
  int result = convert_tensors(&config);
  
  // Cleanup and exit
  free_config(&config);
  return result;
}
