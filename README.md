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



---

## 0) Build

```bash
make clean
make CFLAGS="-O2 -Wall -DDEBUG -DBENCH"
```

Export & merge weights
```
python3 export.py --model Qwen/Qwen3-30B-A3B --all --quant none --outdir qwen3-30b-a3_f32
python3 merge_dir.py all.bin qwen3-30b-a3_f32
```

Run test:
```
./test_model_trace all.bin q3_trace --steps 2 --prompt_len 1
```
