#!/usr/bin/env python3
import argparse, numpy as np, torch
from transformers import AutoModelForCausalLM

def rmsnorm(x, w, eps):
    msq = (x * x).mean(dim=-1, keepdim=True)
    inv = torch.rsqrt(msq + eps)
    return x * inv * w

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--which", choices=["input", "post"], default="input")
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--seqlen", type=int, default=4)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--outbase", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, device_map="cpu")
    sd = model.state_dict()

    if args.which == "input":
        key = f"model.layers.{args.layer}.input_layernorm.weight"
    else:
        key = f"model.layers.{args.layer}.post_attention_layernorm.weight"

    w = sd[key].float()
    d_model = w.shape[0]
    x = torch.randn(args.seqlen, d_model, dtype=torch.float32)
    y = rmsnorm(x, w, args.eps)

    np.save(args.outbase + ".x.npy", x.numpy())
    np.save(args.outbase + ".y.npy", y.numpy())
    print(f"Weight key: {key}")
    print(f"Run export.py to dump norms:\n"
          f"python export.py --model {args.model} --out l{args.layer}_norms.bin --layer {args.layer} --part norms --quant none")
    print(f"Then test:\n"
          f"./test_rmsnorm l{args.layer}_norms.bin {args.outbase}.x.npy {args.outbase}.y.npy \"{key}\" {args.eps}")

if __name__ == "__main__":
    main()
