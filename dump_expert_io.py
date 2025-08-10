#!/usr/bin/env python3
"""
dump_expert_io.py
-----------------
Generate small, reproducible test I/O for a single expert in Qwen3 MoE.
Saves:
  - <outbase>.x.npy : input activations [T, d_model] (float32)
  - <outbase>.y.npy : expert output      [T, d_model] computed via PyTorch
  - <outbase>.manifest.json : shapes and key names

This script pulls only the expert tensors from the state_dict and evaluates:
  y = down( silu(gate(x)) * up(x) )

It also tolerates models without expert biases (common in Qwen3) by using zeros.
"""

import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoConfig

def find_weight_keys(sd, L, E):
  """
  Return the three weight keys for the chosen layer/expert.
  Biases might be absent in some checkpoints (then we'll synthesize zeros).
  """
  prefixes = [
    f"model.layers.{L}.mlp.experts.{E}.",
    f"model.layers.{L}.mlp.experts[{E}].",
  ]
  weight_triplets = [
    ("gate_proj.weight", "up_proj.weight", "down_proj.weight"),
  ]
  for pref in prefixes:
    for (gw, uw, dw) in weight_triplets:
      k = (pref + gw, pref + uw, pref + dw)
      if all(tk in sd for tk in k):
        return k
  raise KeyError(f"Couldn't find expert WEIGHT keys for layer={L}, expert={E}. "
                 f"Sample keys: {list(sd.keys())[:30]}")

def find_bias_key(sd, base):
  """
  Try to find a bias next to a given weight key by replacing '.weight' -> '.bias'.
  Return tensor if found, else None.
  """
  kb = base.replace(".weight", ".bias")
  return sd.get(kb, None)

@torch.no_grad()
def expert_forward_torch(x, Wg, bg, Wu, bu, Wd, bd):
  g = F.silu(x @ Wg.T + (bg if bg is not None else 0.0))
  u = x @ Wu.T + (bu if bu is not None else 0.0)
  h = g * u
  y = h @ Wd.T + (bd if bd is not None else 0.0)
  return y

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", required=True)
  ap.add_argument("--layer", type=int, required=True)
  ap.add_argument("--expert", type=int, required=True)
  ap.add_argument("--seqlen", type=int, default=4)
  ap.add_argument("--seed", type=int, default=123)
  ap.add_argument("--outbase", required=True)
  args = ap.parse_args()

  torch.manual_seed(args.seed)
  np.random.seed(args.seed)

  # Load model on CPU
  model = AutoModelForCausalLM.from_pretrained(
    args.model, torch_dtype=torch.float32, device_map="cpu"
  )
  cfg = AutoConfig.from_pretrained(args.model)
  sd = model.state_dict()

  # Locate expert weights (bias may be missing)
  gw, uw, dw = find_weight_keys(sd, args.layer, args.expert)
  Wg = sd[gw].float()
  Wu = sd[uw].float()
  Wd = sd[dw].float()

  bg_t = find_bias_key(sd, gw)
  bu_t = find_bias_key(sd, uw)
  bd_t = find_bias_key(sd, dw)

  # Shapes
  d_model = Wd.shape[0]
  d_ff = Wd.shape[1]
  T = args.seqlen

  # Build biases as tensors or None
  bg = bg_t.float() if bg_t is not None else None
  bu = bu_t.float() if bu_t is not None else None
  bd = bd_t.float() if bd_t is not None else None

  # Input
  x = torch.randn(T, d_model, dtype=torch.float32)

  # Golden
  y = expert_forward_torch(x, Wg, bg, Wu, bu, Wd, bd)

  # Save .npy
  np.save(args.outbase + ".x.npy", x.cpu().numpy().astype(np.float32))
  np.save(args.outbase + ".y.npy", y.cpu().numpy().astype(np.float32))

  manifest = {
    "d_model": int(d_model),
    "d_ff": int(d_ff),
    "T": int(T),
    "layer": args.layer,
    "expert": args.expert,
    "weights": {
      "gate_proj.weight": gw,
      "up_proj.weight": uw,
      "down_proj.weight": dw
    },
    "bias_present": {
      "gate_proj.bias": bg_t is not None,
      "up_proj.bias": bu_t is not None,
      "down_proj.bias": bd_t is not None
    }
  }
  with open(args.outbase + ".manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

  print(f"Wrote {args.outbase}.x.npy and {args.outbase}.y.npy")
  print(f"d_model={d_model} d_ff={d_ff} T={T}")
  if bg is None or bu is None or bd is None:
    print("Note: one or more expert biases were absent; zeros were used.")

if __name__ == "__main__":
  main()

