#!/usr/bin/env python3
# dump_moe_io.py (2-space indent)
import argparse, json, numpy as np, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoConfig

@torch.no_grad()
def collect_router_and_experts(sd, L):
  # Router: prefer mlp.gate.*, fallback to mlp.router.gate.*
  rk_w = f"model.layers.{L}.mlp.gate.weight"
  rk_b = f"model.layers.{L}.mlp.gate.bias"
  if rk_w not in sd:
    rk_w = f"model.layers.{L}.mlp.router.gate.weight"
    rk_b = f"model.layers.{L}.mlp.router.gate.bias"
  router_w = sd[rk_w].float()
  router_b = sd.get(rk_b, None)
  # Experts: scan keys
  experts = {}
  for k,t in sd.items():
    if k.startswith(f"model.layers.{L}.mlp.experts.") and k.endswith(".down_proj.weight"):
      e = int(k.split(".experts.")[1].split(".")[0])
      experts[e] = 1
  return rk_w, rk_b, sorted(experts.keys()), router_w, router_b

def expert_forward_torch(x, Wg,bg, Wu,bu, Wd,bd):
  g = F.silu(x @ Wg.T + (bg if bg is not None else 0.0))
  u = x @ Wu.T + (bu if bu is not None else 0.0)
  h = g * u
  y = h @ Wd.T + (bd if bd is not None else 0.0)
  return y

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", required=True)
  ap.add_argument("--layer", type=int, required=True)
  ap.add_argument("--seqlen", type=int, default=4)
  ap.add_argument("--seed", type=int, default=123)
  ap.add_argument("--topk", type=int, default=8)
  ap.add_argument("--outbase", required=True)
  args = ap.parse_args()

  torch.manual_seed(args.seed); np.random.seed(args.seed)

  model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, device_map="cpu")
  sd = model.state_dict()

  rk_w, rk_b, all_expert_ids, router_w, router_b_t = collect_router_and_experts(sd, args.layer)
  router_b = router_b_t.float() if router_b_t is not None else None
  E = router_w.shape[0]; d_model = router_w.shape[1]
  T = args.seqlen; k = args.topk

  x = torch.randn(T, d_model, dtype=torch.float32)

  # Router logits -> topk indices/probs
  logits = x @ router_w.T + (router_b if router_b is not None else 0.0)  # [T,E]
  topk_scores, topk_idx = torch.topk(logits, k, dim=-1)
  topk_probs = torch.softmax(topk_scores, dim=-1)  # [T,k]

  # Compute expert outputs for *all* experts (for golden y)
  y_per_e = []
  for e in all_expert_ids:
    gw = sd[f"model.layers.{args.layer}.mlp.experts.{e}.gate_proj.weight"].float()
    uw = sd[f"model.layers.{args.layer}.mlp.experts.{e}.up_proj.weight"].float()
    dw = sd[f"model.layers.{args.layer}.mlp.experts.{e}.down_proj.weight"].float()
    gb = sd.get(f"model.layers.{args.layer}.mlp.experts.{e}.gate_proj.bias", None)
    ub = sd.get(f"model.layers.{args.layer}.mlp.experts.{e}.up_proj.bias", None)
    db = sd.get(f"model.layers.{args.layer}.mlp.experts.{e}.down_proj.bias", None)
    y_e = expert_forward_torch(x, gw, gb.float() if gb is not None else None,
                                  uw, ub.float() if ub is not None else None,
                                  dw, db.float() if db is not None else None)
    y_per_e.append(y_e.unsqueeze(1))  # [T,1,D]
  expert_outputs = torch.cat(y_per_e, dim=1)  # [T,E,D]

  # Build gating tensor [T,E] with only top-k probs placed
  gating = torch.zeros_like(logits)
  for i in range(k):
    idx = topk_idx[:, i:i+1]   # [T,1]
    prob = topk_probs[:, i:i+1]  # [T,1]
    gating.scatter_(1, idx, prob)
  gating = gating.unsqueeze(-1)  # [T,E,1]

  y = (gating * expert_outputs).sum(dim=1)  # [T,D]

  # Save
  np.save(args.outbase + ".x.npy", x.cpu().numpy().astype(np.float32))
  np.save(args.outbase + ".y.npy", y.cpu().numpy().astype(np.float32))

  # union-of-topk expert ids
  union_ids = sorted(set(topk_idx.cpu().numpy().reshape(-1).tolist()))

  with open(args.outbase + ".experts.txt", "w") as f:
    for e in union_ids: f.write(f"{e}\n")

  manifest = {
    "layer": args.layer,
    "T": int(T),
    "d_model": int(d_model),
    "E_total": int(E),
    "topk": int(k),
    "experts_union": union_ids,
    "router": {"weight": rk_w, "bias": rk_b},
  }
  with open(args.outbase + ".manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

  # Helpful instructions
  print(f"Wrote {args.outbase}.x.npy / .y.npy and experts list ({len(union_ids)} ids).")
  print("Export options:")
  print("  # Minimal (router + union-of-topk experts):")
  print(f"  python3 export.py --model {args.model} --out l{args.layer}_moe.bin "
        f"--layer {args.layer} --part mlp --experts " + ",".join(map(str, union_ids)) + " --quant none")
  print("  # Full layer (router + ALL experts):")
  print(f"  python3 export.py --model {args.model} --out l{args.layer}_moe_all.bin "
        f"--layer {args.layer} --part mlp --experts all --quant none")

if __name__ == "__main__":
  main()
