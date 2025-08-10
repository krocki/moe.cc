
#!/usr/bin/env python3
import argparse, json, numpy as np, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoConfig

def find_weight_keys(sd, L, E):
  prefixes = [
    f"model.layers.{L}.mlp.experts.{E}.",
    f"model.layers.{L}.mlp.experts[{E}].",
  ]
  for pref in prefixes:
    gw = pref + "gate_proj.weight"
    uw = pref + "up_proj.weight"
    dw = pref + "down_proj.weight"
    if gw in sd and uw in sd and dw in sd:
      return gw, uw, dw, pref+"gate_proj.bias", pref+"up_proj.bias", pref+"down_proj.bias"
  raise KeyError("expert weight keys not found")

@torch.no_grad()
def expert_forward_torch(x, Wg,bg, Wu,bu, Wd,bd):
  import torch.nn.functional as F
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

  torch.manual_seed(args.seed); np.random.seed(args.seed)
  model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, device_map="cpu")
  sd = model.state_dict()

  gw, uw, dw, gb, ub, db = find_weight_keys(sd, args.layer, args.expert)
  Wg, Wu, Wd = sd[gw].float(), sd[uw].float(), sd[dw].float()
  bg = sd[gb].float() if gb in sd else None
  bu = sd[ub].float() if ub in sd else None
  bd = sd[db].float() if db in sd else None

  d_model = Wd.shape[0]
  x = torch.randn(args.seqlen, d_model, dtype=torch.float32)
  y = expert_forward_torch(x, Wg,bg, Wu,bu, Wd,bd)
  np.save(args.outbase + ".x.npy", x.cpu().numpy().astype(np.float32))
  np.save(args.outbase + ".y.npy", y.cpu().numpy().astype(np.float32))

if __name__ == "__main__":
  main()

