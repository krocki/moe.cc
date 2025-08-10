# Qwen3 MoE (C) — Minimal Expert, MoE Block, and RMSNorm

A tiny, readable C testbed for Qwen3-30B-A3B MoE parts:
- experts (SwiGLU MLP)
- router (two modes)
- MoE block (router + experts)
- RMSNorm

### Setup

```
conda create -n moe python=3.12
conda activate moe
conda install pytorch torchvision torchaudio cpuonly -c pytorch
conda install transformers
conda install accelerate
```



## Files

- `io.h`, `io.c` — simple readers for:
  - `.bin` (from `export.py`) with named tensors
  - `.npy` (float32 C-order) for I/O goldens
- `utils.h`, `utils.c` — helpers (`max_abs_diff`)
- `kernels.h`, `kernels.c` — kernels:
  - `matmul_f32`, `silu_f32`
  - `expert_forward_f32` (SwiGLU: `down(silu(gate(x)) * up(x))`)
  - Router:
    - `ROUTER_TOPK_KONLY`: top‑k logits → softmax over **k**
    - `ROUTER_SOFTMAX_ALL_TOPK`: softmax over **all E** → top‑k by prob
  - `moe_forward_f32_mode(...)` — full MoE forward with a routing mode
  - `rmsnorm_forward_f32` — RMSNorm
- `test_expert.c` — validates a single expert
- `test_moe_block.c` — validates MoE block with selectable routing
- `test_rmsnorm.c` — validates RMSNorm
- `export.py` — exports weights from HF into a `.bin` (router + selected experts or “all”)
- `dump_expert_io.py` — creates goldens for a single expert
- `dump_moe_io.py` — creates goldens for a full MoE block (union-of-topk experts list)
- `dump_rmsnorm_io.py` — creates goldens for RMSNorm

> **Tip:** Compile with `-DDEBUG -DBENCH` for verbose prints and timings.

---

## 0) Build

```bash
make clean
make CFLAGS="-O2 -Wall -DDEBUG -DBENCH"
```

This builds:
- `test_expert`
- `test_moe_block`

---

## 1) Test **one expert**

1) Dump goldens for layer 0, expert 0:
```bash
python3 dump_expert_io.py --model Qwen/Qwen3-30B-A3B \
  --layer 0 --expert 0 --seqlen 1 --seed 123 \
  --outbase qwen3_L0_E0
```

2) Export only that expert’s weights:
```bash
python3 export.py --model Qwen/Qwen3-30B-A3B \
  --out l0_e0.bin --layer 0 --part mlp --experts 0 --quant none
```

3) Run the C test:
```bash
./test_expert l0_e0.bin qwen3_L0_E0.x.npy qwen3_L0_E0.y.npy
# Expect: Max abs diff ~1e-6 → PASS
```

---

## 2) Test **MoE block** — **konly** routing
*(top‑k on logits, softmax over the selected k)*

1) Dump MoE goldens (`konly`) and a list of used experts:
```bash
python3 dump_moe_io.py --model Qwen/Qwen3-30B-A3B \
  --layer 0 --seqlen 1 --seed 123 --topk 8 \
  --route konly --outbase qwen3_L0_MOE
```

2) Export router + the **union‑of‑topk** experts:
```bash
# If your shell needs commas instead of spaces:
# python3 export.py ... --experts $(paste -sd, qwen3_L0_MOE.experts.txt)
python3 export.py --model Qwen/Qwen3-30B-A3B \
  --out l0_moe.bin --layer 0 --part mlp \
  --experts $(cat qwen3_L0_MOE.experts.txt) --quant none
```

3) Run the C test (`konly`):
```bash
./test_moe_block l0_moe.bin qwen3_L0_MOE.x.npy qwen3_L0_MOE.y.npy konly
# Expect: Max abs diff ~1e-6 → PASS
```

> To export **all experts** instead: `--experts all` and use the same test command.

---

## 3) Test **MoE block** — **full** routing
*(softmax over all E experts, then pick top‑k by probability)*

1) Dump MoE goldens (`full`) and corresponding expert list:
```bash
python3 dump_moe_io.py --model Qwen/Qwen3-30B-A3B \
  --layer 0 --seqlen 1 --seed 123 --topk 8 \
  --route full --outbase qwen3_L0_MOE_full
```

2) Export router + union‑of‑topk (or `--experts all`):
```bash
python3 export.py --model Qwen/Qwen3-30B-A3B \
  --out l0_moe_full.bin --layer 0 --part mlp \
  --experts $(cat qwen3_L0_MOE_full.experts.txt) --quant none
```

3) Run the C test in `full` mode:
```bash
./test_moe_block l0_moe_full.bin qwen3_L0_MOE_full.x.npy qwen3_L0_MOE_full.y.npy full
# Expect: Max abs diff ~1e-6 → PASS
```

## 3) Test **RMSNorm**

1) Generate goldens (choose input or post):
```
python3 export.py --model Qwen/Qwen3-30B-A3B --out l0_norms.bin --layer 0 --part norms --quant none
```

Check names:

```
./list_bin l0_norms.bin | head -50

model.layers.0.input_layernorm.weight
model.layers.0.post_attention_layernorm.weight
```
2) Export norms for that layer:

choose input/output
```
python3 dump_rmsnorm_io.py --model Qwen/Qwen3-30B-A3B \
  --layer 0 --which input --eps 1e-6 --seqlen 4 --seed 123 \
  --outbase qwen3_L0_RMS_in
```

3) Run test

```
./test_rmsnorm l0_norms.bin qwen3_L0_RMS_in.x.npy qwen3_L0_RMS_in.y.npy "model.layers.0.input_layernorm.weight" 1e-6
# Expect ~1e-7 → PASS
```

### 4. Test attention (GQA) block

> Note: This path is RoPE‑free, so keep `--seqlen 1` to match the dumper.

1) Dump attention I/O for a given layer (example: layer 0, causal mask on):
```bash
python3 dump_attn_io.py --model Qwen/Qwen3-30B-A3B \
  --layer 0 --seqlen 1 --seed 123 --causal 1 \
  --outbase qwen3_L0_ATTN
```

2) Export the attention weights
```
python3 export.py --model Qwen/Qwen3-30B-A3B \
  --out l0_attn.bin --layer 0 --part attn --quant none
```

3) Run the C test for GQA attention (last arg is causal flag: 1=causal, 0=non‑causal):

```
./test_attn l0_attn.bin qwen3_L0_ATTN.x.npy qwen3_L0_ATTN.y.npy 1
```
Example output:
```
[attn/gqa] T=1 d_model=2048 n_q=32 n_kv=4 head_dim=128 causal=1
[matmul] A[1,2048] * W^T[4096,2048] -> Y[1,4096]
...
Max abs diff: 6.4e-06
PASS
```

---

## Debug & Bench

- `-DDEBUG`:
  - matmul shapes & completion
  - router mode + per‑token top‑k (indices & probabilities)
  - per‑token expert choices and weights
- `-DBENCH`:
  - ms timings for matmuls and routing

---

## Notes
- Names embedded in `.bin` match Hugging Face keys (e.g. `model.layers.0.mlp.experts.7.up_proj.weight`).  
- Biases are optional — handled as zeros if missing.  
- Router tensors are always included for `--part mlp` exports.

