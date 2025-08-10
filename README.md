### Setup

```
conda create -n moe python=3.12
conda activate moe
conda install pytorch torchvision torchaudio cpuonly -c pytorch
conda install transformers
conda install accelerate
```

```
make
```

### Expert test
```
python3 export.py --model Qwen/Qwen3-30B-A3B --out l0_e0.bin --layer 0 --part mlp --experts 0 --quant none
python3 dump_expert_io.py --model Qwen/Qwen3-30B-A3B --layer 0 --expert 0 --seqlen 1 --seed 123 --outbase qwen3_L0_E0

./test_expert l0_e0.bin qwen3_L0_E0.x.npy qwen3_L0_E0.y.npy
```

### MoE block test

```
python dump_moe_io.py --model Qwen/Qwen3-30B-A3B --layer 0 --seqlen 1 --seed 123 --topk 8 --outbase qwen3_L0_MOE
python3 export.py --model Qwen/Qwen3-30B-A3B --out l0_moe.bin --layer 0 --part mlp --experts $(paste -sd, qwen3_L0_MOE.experts.txt) --quant none

./test_moe_block l0_moe.bin qwen3_L0_MOE.x.npy qwen3_L0_MOE.y.npy
```
