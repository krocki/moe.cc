# Qwen3 MoE (C) — Minimal Expert & MoE Block Tests

Everything is split into small, readable pieces (2‑space indentation). Makefile rules use tabs.

### Setup

```
conda create -n moe python=3.12
conda activate moe
conda install pytorch torchvision torchaudio cpuonly -c pytorch
conda install transformers
conda install accelerate
```


## Files
- `io.h`, `io.c` — tiny `.bin` (export.py) and `.npy` readers
- `utils.h`, `utils.c` — helpers (`max_abs_diff`)
- `kernels.h`, `kernels.c` — matmul, SiLU, expert forward, router, MoE forward (both routing modes)
- `test_expert.c` — validates a single expert forward
- `test_moe_block.c` — validates the MoE block with selectable routing mode (`konly` / `full`)
- `export.py` — exports router + experts into a single `.bin` (accepts `--experts all` or IDs list)
- `dump_expert_io.py` — creates `x.npy`/`y.npy` for a single expert
- `dump_moe_io.py` — creates `x.npy`/`y.npy` for the full MoE block; emits `experts.txt` and suggested export commands

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

