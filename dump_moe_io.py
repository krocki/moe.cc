
#!/usr/bin/env python3
import argparse, json, numpy as np, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoConfig

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
  ap.add_argument("--seqlen", type=int, default=1)
  ap.add_argument("--seed", type=int, default=123)
  ap.add_argument("--topk", type=int, default=8)
  ap.add_argument("--route", choices=["konly","full"], default="konly")
  ap.add_argument("--outbase", required=True)
  ap.add_argument("--all-experts", action="store_true")
  args = ap.parse_args()

  torch.manual_seed(args.seed); np.random.seed(args.seed)
  model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, device_map="cpu")
  sd = model.state_dict()

  # Router
  r_w = None
  for name in [f"model.layers.{args.layer}.mlp.gate.weight",
               f"model.layers.{args.layer}.mlp.router.gate.weight"]:
    if name in sd: r_w = name; break
  if r_w is None: raise KeyError("router weight not found")
  r_b = None
  for name in [f"model.layers.{args.layer}.mlp.gate.bias",
               f"model.layers.{args.layer}.mlp.router.gate.bias"]:
    if name in sd: r_b = name; break
  Wroute = sd[r_w].float()
  Brow = sd[r_b].float() if r_b in sd else None
  E, d_model = Wroute.shape

  # Collect experts
  def key(e, t): return f"model.layers.{args.layer}.mlp.experts.{e}.{t}"
  present = []
  for e in range(E):
    gw = key(e, "gate_proj.weight")
    uw = key(e, "up_proj.weight")
    dw = key(e, "down_proj.weight")
    if gw in sd and uw in sd and dw in sd:
      present.append(e)
  if len(present) < E and args.all_experts:
    print(f"Warning: only {len(present)} expert weight triplets out of {E} found.")

  # Input
  x = torch.randn(args.seqlen, d_model, dtype=torch.float32)

  # Router logits
  logits = x @ Wroute.T + (Brow if Brow is not None else 0.0)

  if args.route == "konly":
    topk_scores, topk_idx = torch.topk(logits, args.topk, dim=-1)
    topk_probs = torch.softmax(topk_scores, dim=-1)
  else:
    probs = torch.softmax(logits, dim=-1)
    topk_probs, topk_idx = torch.topk(probs, args.topk, dim=-1)

  # Precompute all expert outs (for simplicity)
  y_e = [None]*E
  for e in present:
    bg = sd.get(key(e,"gate_proj.bias"))
    bu = sd.get(key(e,"up_proj.bias"))
    bd = sd.get(key(e,"down_proj.bias"))
    y_e[e] = expert_forward_torch(
      x,
      sd[key(e,"gate_proj.weight")].float(), bg.float() if bg is not None else None,
      sd[key(e,"up_proj.weight")].float(),   bu.float() if bu is not None else None,
      sd[key(e,"down_proj.weight")].float(), bd.float() if bd is not None else None
    )

  # Aggregate
  y = torch.zeros_like(x)
  for t in range(args.seqlen):
    for i in range(args.topk):
      e = int(topk_idx[t,i]); p = float(topk_probs[t,i])
      y[t] += p * y_e[e][t]

  # Save goldens
  np.save(args.outbase + ".x.npy", x.cpu().numpy().astype(np.float32))
  np.save(args.outbase + ".y.npy", y.cpu().numpy().astype(np.float32))

  # expert list
  if args.all_experts:
    experts_list = list(range(E))
  else:
    experts_list = sorted(set(int(e) for e in topk_idx.reshape(-1).tolist()))
  with open(args.outbase + ".experts.txt","w") as f:
    f.write(",".join(str(e) for e in experts_list))

  # manifest
  man = {
    "layer": args.layer, "E": int(E), "d_model": int(d_model),
    "k": int(args.topk), "route": args.route, "seqlen": int(args.seqlen)
  }
  with open(args.outbase + ".manifest.json","w") as f:
    json.dump(man, f, indent=2)

  print("Suggested export:")
  print(f"  python export.py --model {args.model} --out l{args.layer}_moe.bin "
        f"--layer {args.layer} --part mlp --experts {','.join(map(str,experts_list))} --quant none")
  print("Suggested export (all):")
  print(f"  python export.py --model {args.model} --out l{args.layer}_moe_all.bin "
        f"--layer {args.layer} --part mlp --experts all --quant none")

if __name__ == "__main__":
  main()

