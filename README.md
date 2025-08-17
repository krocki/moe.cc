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

---

## Quantization Details

### Supported Quantization Schemes

| Scheme | Bits | Memory | Accuracy | Speed |
|--------|------|---------|----------|-------|
| FP32   | 32   | 100%    | Baseline | 1.0x  |
| Q8     | 8    | 25%     | 99.9%    | 1.2x  |
| Q4     | 4    | 12.5%   | 99.5%    | 1.5x  |

### What Gets Quantized

The quantization is **selective** - only MoE expert weight matrices are quantized:

**Quantized (Q8/Q4):**
- `model.layers.*.mlp.experts.*.gate_proj.weight` - Gate projection weights
- `model.layers.*.mlp.experts.*.up_proj.weight` - Up projection weights  
- `model.layers.*.mlp.experts.*.down_proj.weight` - Down projection weights

**Remains FP32:**
- Token embeddings
- Attention weights (Q, K, V, O projections)
- Layer normalization weights
- Router weights
- Output head weights

### Quantization Algorithm

**Q8 Quantization (Recommended):**
```
For each weight matrix row i:
  scale[i] = max(abs(row[i])) / 127.0
  quantized[i] = round(row[i] / scale[i]).clamp(-127, 127)
```

**Q4 Quantization (Maximum Compression):**
```  
For each weight matrix row i:
  scale[i] = max(abs(row[i])) / 7.0
  quantized[i] = round(row[i] / scale[i]).clamp(-8, 7) + 8
  packed[i] = pack_two_q4_per_byte(quantized[i])
```

### Binary Format

The quantized tensors are stored with modified names:
- `original_name.q8` or `original_name.q4` - Quantized weight data
- `original_name.scale` - Per-row FP32 scaling factors

Example: `model.layers.0.mlp.experts.3.gate_proj.weight` becomes:
- `model.layers.0.mlp.experts.3.gate_proj.weight.q8`
- `model.layers.0.mlp.experts.3.gate_proj.weight.scale`

---

## Performance Benchmarks

### Memory Usage (Qwen3-30B-A3B)

| Component | FP32 (GB) | Q8 (GB) | Q4 (GB) | Reduction |
|-----------|-----------|---------|---------|-----------|
| Experts   | 84.0      | 21.0    | 10.5    | 4.0x / 8.0x |
| Other     | 30.0      | 30.0    | 30.0    | 1.0x |
| **Total** | **114.0** | **51.0**| **40.5**| **2.2x / 2.8x** |

### Inference Speed (tokens/second)

Results will vary based on hardware. Quantized models are typically:
- **Q8**: 20-30% faster than FP32 due to reduced memory bandwidth
- **Q4**: 40-50% faster than FP32 due to increased arithmetic intensity

### Accuracy Validation

Expected Mean Absolute Differences vs FP32 baseline:
- **Q8**: logits MAD < 1e-3, identical argmax 99.9% of the time  
- **Q4**: logits MAD < 1e-2, identical argmax 99.5% of the time

---

## Advanced Usage

### Export Specific Layers/Experts

Export only layer 0 experts 0-7 with Q8 quantization:
```bash
python3 export.py \
  --model Qwen/Qwen3-30B-A3B \
  --layer 0 \
  --part mlp \
  --experts 0,1,2,3,4,5,6,7 \
  --quant q8 \
  --outdir layer0_experts_q8
```

### Testing Quantization Quality

Use the provided test script to validate quantization:
```bash
./test_quantization.sh
```

This script will:
1. Export small test sets with different quantization levels
2. Generate reference traces
3. Compare outputs between FP32/Q8/Q4 models
4. Report accuracy metrics and performance differences

### Custom Quantization

To implement different quantization schemes:

1. **Add quantization function** in `export.py`:
   ```python
   def _custom_quant(w):
       # Your quantization logic here
       return scales, quantized_weights
   ```

2. **Add dtype enum** in `_save_tensor()` (e.g., dtype=4 for Q16)

3. **Implement matmul kernel** in `kernels.c`:
   ```c
   static void matmul_f32_custom(const float* A, const custom_t* B_q, 
                                 const float* B_s, float* C, int M, int N, int K)
   ```

4. **Update adaptive dispatcher** in `matmul_adaptive()`

### Debugging and Profiling

Enable detailed profiling and debug output:
```bash
make clean
make CFLAGS="-O2 -Wall -DDEBUG -DBENCH -DPROFILING=1"
```

The profiler will output detailed timing information for each operation.

---

## Implementation Notes

### Quantized Matrix Multiplication

The core quantized matmul algorithms are implemented in `kernels.c`:

**Q8 Algorithm:**
```c
// Compute: C = A (fp32) × B_quantized (q8)
for each output C[m,n]:
    int32_accumulator = 0
    for k in K:
        a_scaled = A[m,k] * 127.0f  // Scale to Q8 range
        int32_accumulator += a_scaled * B_q[n,k]  
    C[m,n] = (float(int32_accumulator) / 127.0f) * B_scale[n]
```

**Q4 Algorithm:**
```c  
// Compute: C = A (fp32) × B_quantized (q4_packed)  
for each output C[m,n]:
    int32_accumulator = 0
    for k in 0,2,4,...:
        // Unpack two Q4 values from one byte
        q4_pair = B_q[n,k/2]
        val0 = (q4_pair & 0xF) - 8        // First Q4: [0,15] -> [-8,7]
        val1 = ((q4_pair >> 4) & 0xF) - 8 // Second Q4: [0,15] -> [-8,7]
        
        a0_scaled = A[m,k] * 7.0f     // Scale to Q4 range
        a1_scaled = A[m,k+1] * 7.0f
        int32_accumulator += a0_scaled * val0 + a1_scaled * val1
    C[m,n] = (float(int32_accumulator) / 7.0f) * B_scale[n]
```

### File Structure

```
.
├── README.md               # This documentation
├── Makefile               # Build system
├── export.py              # Weight export with quantization  
├── merge_dir.py           # Merge exported weights into binary
├── test_quantization.sh   # Quantization validation script
├── model.h                # Model structures and weight loading
├── kernels.h              # Kernel function declarations
├── kernels.c              # Quantized matmul implementations
├── io.h/.c                # Binary I/O for weights and traces
├── test_model_trace.c     # Main inference test program
└── utils.h/.c             # Utility functions
```

---

## Troubleshooting

### Common Issues

**"missing tensor" errors:**
- Ensure you're using `--all` flag when exporting the full model
- Check that the model name is correct: `Qwen/Qwen3-30B-A3B`

**Compilation errors:**
- Make sure you have a recent GCC version (GCC 9+)
- Try `make clean` before rebuilding

**High memory usage:**
- Use Q8 or Q4 quantization: `--quant q8` or `--quant q4`
- Export only specific layers with `--layer N` for testing

**Accuracy issues:**
- Q8 should match FP32 very closely (MAD < 1e-3)
- Q4 may show small differences (MAD < 1e-2) but should preserve generation quality
- Use `test_quantization.sh` to validate your setup

### Performance Tuning

**For maximum speed:**
- Use Q4 quantization (`--quant q4`)  
- Compile with `-O3 -march=native`
- Consider using OpenMP parallelization

**For maximum accuracy:**  
- Use Q8 quantization (`--quant q8`)
- Keep critical layers in FP32 if needed

**For minimum memory:**
- Use Q4 quantization with selective layer export
- Only export the layers you need for testing

---

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
