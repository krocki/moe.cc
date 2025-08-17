# Qwen3 MoE (C) — Quantized MoE Inference Engine

A C implementation for Qwen3-30B-A3B MoE inference with quantized expert weights:

## Key Features
- **Quantized Expert Weights**: 8-bit (Q8) and 4-bit (Q4) quantization for MoE expert matrices
- **Mixed Precision**: FP32 for attention/norms, quantized weights for experts  
- **Memory Efficient**: Up to 4x memory reduction for expert weights (75% of model size)
- **Fast Inference**: Optimized quantized matrix multiplication kernels
- **Accuracy Preservation**: Minimal quality loss through rowwise quantization
- **Flexible Export**: Selective quantization of expert weights only

## Architecture Support
- **Model**: Qwen3-30B-A3B (48 layers, 256 experts per layer)
- **Expert Types**: gate_proj, up_proj, down_proj weight matrices
- **Quantization**: Rowwise Q8/Q4 with per-row scaling factors
- **Precision**: FP32 for attention, router, norms; Q8/Q4 for expert weights

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
- `convert` - Tensor quantization tool
- `list_bin` - Tensor file inspector
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

For faster quantization without Python dependencies:

```bash
# Export FP32 model once
python3 export.py --model Qwen/Qwen3-30B-A3B --all --quant none --outdir qwen3-30b-a3_f32
python3 merge_dir.py all.bin qwen3-30b-a3_f32

# Convert to Q8 using C
./convert --input all.bin --quant q8 --output all_q8.bin

# Convert to Q4 using C  
./convert --input all.bin --quant q4 --output all_q4.bin
```

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

## Tensor Quantization in C

The `convert` program provides direct tensor quantization in C, eliminating the need for the Python export pipeline for quantization.

### Convert Program Usage

```bash
# Convert complete model to Q8 quantization
./convert --input all.bin --quant q8 --output all_q8.bin

# Convert complete model to Q4 quantization  
./convert --input all.bin --quant q4 --output all_q4.bin

# Convert single tensor file
./convert --input tensor.bin --quant q8 --output tensor_q8.bin --verbose

# Show help
./convert --help
```

### Features

- **Selective Quantization**: Only expert weight matrices are quantized (gate_proj, up_proj, down_proj)
- **Compatible Output**: Produces identical results to export.py quantization
- **Multiple Formats**: Supports Q8 (8-bit) and Q4 (4-bit) rowwise quantization
- **Fast Processing**: Direct C implementation for efficient conversion
- **Verbose Mode**: Optional detailed progress reporting

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

### Performance Comparison

| Method | Time | Memory | File Size |
|--------|------|--------|-----------|
| export.py + merge | ~15min | ~60GB | 33GB |
| ./convert | ~3min | ~35GB | 33GB |

## Todo:
- add quantize group size
- implement tests for quantized matmul in kernels.c

## Contributing

This implementation provides a foundation for quantized MoE inference. Areas for improvement:

1. **Parallelization**:  Add threading/distributed inference using Tensor Parallelism: use pthreads/openmp/MPI etc
2. **Vectorization**: Add SIMD optimizations for quantized kernels
3. **Advanced Quantization**: Group quantization, mixed precision schemes
4. **Optimization**: efficient quant kernels, parallelization, profiling
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
