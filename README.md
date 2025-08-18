# Qwen3 MoE (C) — Group-wise Quantized MoE Inference Engine

A C implementation for Qwen3-30B-A3B MoE inference with group-wise quantized expert weights:

## Key Features
- **Group-wise Quantization Only**: 8-bit (Q8) and 4-bit (Q4) quantization with configurable group sizes (32, 64, 128, etc.)
- **Mixed Precision**: FP32 for attention/norms, quantized weights for experts  
- **Memory Efficient**: Up to 4x memory reduction for expert weights (75% of model size)
- **Fast Inference**: Optimized group-wise quantized matrix multiplication kernels
- **Superior Accuracy**: Group-wise quantization provides better accuracy than legacy rowwise methods
- **Comprehensive Testing**: Built-in matrix multiplication test harness for accuracy and performance validation
- **llama2.c Style Kernels**: Q8×Q8 and Q8×Q4 matrix multiplication for flexible quantization strategies

## Architecture Support
- **Model**: Qwen3-30B-A3B (48 layers, 256 experts per layer)
- **Expert Types**: gate_proj, up_proj, down_proj weight matrices
- **Quantization**: Group-wise Q8/Q4 with configurable group sizes (32, 64, 128, 256)
- **Precision**: FP32 for attention, router, norms; Q8/Q4 for expert weights
- **Matrix Kernels**: FP32×Q8, FP32×Q4, Q8×Q8, Q8×Q4 matrix multiplication

## Setup

```bash
conda create -n moe python=3.12
conda activate moe
conda install pytorch torchvision torchaudio cpuonly -c pytorch
conda install transformers accelerate
pip install tqdm  # Optional: for progress bars
```



---

## Quick Start

### 1. Build the C Inference Engine

```bash
make clean
make
```

This builds:
- `test_model_trace` - Main inference engine
- `convert` - Tensor quantization tool v3.0.0 (requires --group-size)
- `list_bin` - Tensor file inspector
- `test_matmul_harness` - Matrix multiplication testing and benchmarking suite
- Test programs and utilities

### 2. Export Model Weights

#### FP32 Baseline (Full Precision)
```bash
python3 export.py \
  --model Qwen/Qwen3-30B-A3B \
  --all \
  --quant none \
  --outdir qwen3-30b-a3_f32

python3 merge_dir.py all_fp32.bin qwen3-30b-a3_f32
```

#### Q8 Quantized Expert Weights (Recommended)
```bash
python3 export.py \
  --model Qwen/Qwen3-30B-A3B \
  --all \
  --quant q8 \
  --outdir qwen3-30b-a3_q8

python3 merge_dir.py all_q8.bin qwen3-30b-a3_q8
```

#### Q4 Quantized Expert Weights (Maximum Compression)  
```bash
python3 export.py \
  --model Qwen/Qwen3-30B-A3B \
  --all \
  --quant q4 \
  --outdir qwen3-30b-a3_q4

python3 merge_dir.py all_q4.bin qwen3-30b-a3_q4
```

#### Alternative: Direct C Quantization (Recommended)

For faster quantization without Python dependencies using group-wise quantization:

```bash
# Export FP32 model once
python3 export.py --model Qwen/Qwen3-30B-A3B --all --quant none --outdir qwen3-30b-a3_f32
python3 merge_dir.py all.bin qwen3-30b-a3_f32

# Convert to Q8 with group-size 128 (recommended for accuracy)
./convert --input all.bin --quant q8 --group-size 128 --output all_q8_g128.bin

# Convert to Q8 with group-size 64 (balanced accuracy/compression)
./convert --input all.bin --quant q8 --group-size 64 --output all_q8_g64.bin

# Convert to Q4 with group-size 32 (maximum compression)
./convert --input all.bin --quant q4 --group-size 32 --output all_q4_g32.bin

# Convert to Q4 with group-size 64 (better accuracy)
./convert --input all.bin --quant q4 --group-size 64 --output all_q4_g64.bin
```

**Note**: The `--group-size` parameter is now required in v3.0.0. Rowwise quantization has been removed for better accuracy.

### 3. Generate Reference Traces (Optional)
```bash
python3 verify_greedy.py \
  --model Qwen/Qwen3-30B-A3B \
  --layers 48 \
  --seqlen 2 \
  --steps 4 \
  --qk-norm auto \
  --rope-theta 10000000 \
  --outbase q3_trace \
  --first-id 151644 \
  --force-f32
```

### 4. Run Inference Tests

#### Test FP32 Baseline
```bash
./test_model_trace all_fp32.bin q3_trace --steps 3 --prompt_len 1
```

#### Test Q8 Quantized Model  
```bash
./test_model_trace all_q8.bin q3_trace --steps 3 --prompt_len 1
```

#### Test Q4 Quantized Model
```bash
./test_model_trace all_q4.bin q3_trace --steps 3 --prompt_len 1
```

### 5. Test Matrix Multiplication Kernels

The comprehensive testing harness v3.1.0 validates accuracy and performance of all matrix multiplication kernels with both standard and realistic LLM dimensions:

#### Quick Test (Reduced Configurations)
```bash
./test_matmul_harness --quick
```

#### Standard Test Suite (Power-of-2 Dimensions)
```bash
./test_matmul_harness --standard
```

#### LLM Inference Test Suite (Realistic Transformer Dimensions)
```bash
./test_matmul_harness --llm
```

#### Combined Test Suite (Standard + LLM) [Default]
```bash
./test_matmul_harness --combined
./test_matmul_harness --full  # Same as --combined
```

#### Test Configurations

**Standard Test Suite:**
- **Matrix Sizes**: Power-of-2 dimensions (64×64, 256×256, 512×512) for algorithm validation
- **Group Sizes**: 32, 64, 128 for quantization granularity testing

**LLM Inference Test Suite:**
- **Vector×Matrix**: `[1×K]×[K×N]` for single-token inference scenarios:
  - `[1×2048]×[2048×768]` → `[1×768]` (down_proj)
  - `[1×4096]×[4096×2048]` → `[1×2048]` (large down_proj)
  - `[1×2048]×[2048×4096]` → `[1×4096]` (up_proj)
  - `[1×2048]×[2048×512]` → `[1×512]` (small projections)
  - `[1×2048]×[2048×128]` → `[1×128]` (tiny projections)

- **Expert Weight Matrices**: Typical transformer expert dimensions:
  - `[768×768]×[768×2048]` → `[768×2048]` (gate_proj)
  - `[2048×2048]×[2048×768]` → `[2048×768]` (down_proj)

- **Batch Processing**: Small batch scenarios (4×, 8×, 16× batch sizes)

#### Test Results (llama2.c Style Pre-Quantized Kernels)
- **FP32×Q8 Kernel**: MAD error 0.007-0.016, ~1.85-1.90 GFLOPS
- **FP32×Q4 Kernel**: MAD error 0.125-0.300, ~1.85-1.90 GFLOPS
- **Q8×Q8 Kernel**: MAD error 0.010-0.023, ~1.59-1.79 GFLOPS (**64% faster** with pre-quantized inputs)
- **Q8×Q4 Kernel**: MAD error 0.125-0.300, ~1.61-1.78 GFLOPS (optimized with pre-quantized inputs)
- **Performance Improvement**: Q8×Q8 kernels now competitive with FP32×Q8/Q4 performance
- **Vector×Matrix Performance**: Excellent accuracy for inference scenarios (MAD < 0.02 for Q8×Q8)
- **Large Matrix Performance**: Consistent ~1.6-1.9 GFLOPS across realistic transformer dimensions
- **Non-power-of-2 Validation**: All realistic LLM dimensions (768, 2048, 4096) pass accuracy tests
- **Benchmark Accuracy**: Pure matrix multiplication timing (quantization overhead excluded)

## Tensor Quantization in C

The `convert` program provides direct tensor quantization in C, eliminating the need for the Python export pipeline for quantization.

### Convert Program Usage (v3.0.0)

```bash
# Convert complete model to Q8 with group-size 128 (recommended)
./convert --input all.bin --quant q8 --group-size 128 --output all_q8_g128.bin --verbose

# Convert complete model to Q8 with group-size 64 (balanced)
./convert --input all.bin --quant q8 --group-size 64 --output all_q8_g64.bin --verbose

# Convert complete model to Q4 with group-size 32 (maximum compression)
./convert --input all.bin --quant q4 --group-size 32 --output all_q4_g32.bin --verbose

# Convert complete model to Q4 with group-size 64 (better accuracy)
./convert --input all.bin --quant q4 --group-size 64 --output all_q4_g64.bin --verbose

# Convert single tensor file
./convert --input tensor.bin --quant q8 --group-size 128 --output tensor_q8.bin --verbose

# Show version information
./convert --version

# Show help
./convert --help
```

### Features

- **Group-wise Only**: Exclusive support for group-wise quantization (rowwise removed in v3.0.0)
- **Required Group Size**: `--group-size` parameter is now mandatory for all quantization operations
- **Selective Quantization**: Only expert weight matrices are quantized (gate_proj, up_proj, down_proj)
- **Superior Accuracy**: Group-wise quantization provides better accuracy than legacy rowwise methods
- **Multiple Formats**: Supports Q8 (8-bit) and Q4 (4-bit) with configurable group sizes
- **Configurable Groups**: Support for group sizes 32, 64, 128, 256 for optimal accuracy/compression trade-off
- **Fast Processing**: Direct C implementation for efficient conversion with streaming I/O
- **Version Tracking**: Built-in version information for debugging and compatibility
- **Verbose Mode**: Optional detailed progress reporting with group-size information

### File Compatibility

The convert program maintains full compatibility with the existing pipeline:

- **Input**: Standard .bin files (from export.py or convert)
- **Output**: Compatible with test_model_trace and existing inference code
- **Format**: Follows export.py conventions (.scale + .q8/.q4 tensor pairs)

### Testing

```bash
# Run all tests
make test

# Run unit tests only
make test-unit

# Run integration tests only  
make test-integration
```

### Diagnostic Tools

The repository includes diagnostic tools for debugging quantization issues:

```bash
# Inspect group-size metadata in quantized files
./debug_group_size all_q8_g128.bin

# Test group-size I/O functionality  
./test_group_size_io
```

### Performance Comparison

| Method | Time | Memory | File Size | Features |
|--------|------|--------|-----------|----------|
| export.py + merge | ~15min | ~60GB | 33GB | Python dependencies |
| ./convert | ~3min | ~35GB | 33GB | C-only, group-size support |

## Validation and Testing Commands

### Complete Validation Suite

To validate the v3.0.0 implementation with group-wise quantization:

```bash
# 1. Build all components
make clean && make

# 2. Test convert tool functionality
./convert --help
./convert --version

# 3. Test matrix multiplication kernels (quick)
./test_matmul_harness --quick

# 4. Full comprehensive kernel testing
./test_matmul_harness --full

# 5. Test convert tool with small tensor (if available)
# ./convert --input small_tensor.bin --quant q8 --group-size 64 --output test_q8.bin --verbose

# 6. Verify group-size requirement enforcement
./convert --input test.bin --quant q8 --output test_fail.bin 2>&1 | grep "group-size is required"
```

### Expected Results

- **Matrix Multiplication Tests**:
  - Q8×Q8: MAD error 0.011-0.033, Max error <0.2, Performance ~1.7 GFLOPS
  - Q8×Q4: MAD error 0.145-0.425, Max error <2.4, Performance ~1.8 GFLOPS
  - All tests should show "PASS" status

- **Convert Tool**:
  - Version should show "3.0.0"
  - Help should mention required `--group-size` parameter
  - Attempting conversion without `--group-size` should fail with clear error

- **Error Handling**:
  - All components should reject rowwise quantization attempts
  - Clear error messages for missing group_size parameter

### Performance Benchmarks

Expected performance on typical hardware:

| Matrix Size | Q8×Q8 (GFLOPS) | Q8×Q4 (GFLOPS) | Latency (ms) |
|-------------|----------------|----------------|--------------|
| 64×64×64    | ~1.7           | ~1.8           | <2           |
| 256×256×256 | ~1.8           | ~1.8           | <20          |
| 512×512×512 | ~1.7           | ~1.7           | <160         |

## Contributing

This implementation provides a foundation for quantized MoE inference. Areas for improvement:

1. **Parallelization**: Add threading/distributed inference using Tensor Parallelism: use pthreads/openmp/MPI etc
2. **Vectorization**: Add SIMD optimizations for quantized kernels
3. **Advanced Quantization**: Additional group sizes, mixed precision schemes
4. **Optimization**: Enhanced kernels, parallelization, profiling
---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{qwen3_moe_quantized,
  title={MoE Inference in C/C++},
  author={Kamil Rocki},
  year={2024},
  url={https://github.com/krocki-repo/moe.cc}
}
```
