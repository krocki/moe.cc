#!/usr/bin/env python3
# scripts/export_weights.py (2-space indent)
import argparse, json, struct, torch, re
from transformers import AutoModelForCausalLM, AutoConfig

MAGIC = b"QW3W\x00\x01"  # magic + version

def _save_tensor(f, name, t):
  t = t.contiguous()
  name_b = name.encode()
  f.write(struct.pack("<I", len(name_b))); f.write(name_b)
  # dtype enum: 0=f32, 1=f16, 2=i8, 3=i4
  if   t.dtype == torch.float32: dt = 0
  elif t.dtype == torch.float16: dt = 1
  elif t.dtype == torch.int8:    dt = 2
  else: raise ValueError(f"dtype {t.dtype} not supported")
  f.write(struct.pack("<I", dt))
  f.write(struct.pack("<I", t.dim()))
  f.write(struct.pack("<" + "I"*t.dim(), *t.shape))
  f.write(t.cpu().numpy().tobytes())

def _rowwise_q8(w):  # (out,in) -> (scales, q)
  w = w.float().contiguous()
  scales = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)/127.0
  q = torch.round(w / scales).to(torch.int8)
  return scales.squeeze(1).to(torch.float32), q

def _rowwise_q4(w):  # pack 2x int4 per byte
  w = w.float().contiguous()
  s = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)/7.0
  q = torch.round(w / s).clamp_(-8,7).to(torch.int8)  # store -8..7
  q = (q + 8).to(torch.uint8)  # 0..15
  lo = q[:, ::2]; hi = q[:, 1::2]
  packed = (lo | (hi << 4)).contiguous()
  return s.squeeze(1).to(torch.float32), packed

def _want_key_for_layer_part(key, L, part):
  pref = f"model.layers.{L}."
  if not key.startswith(pref): return False
  if part == "attn":   return ".self_attn." in key
  if part == "mlp":    return ".mlp." in key
  if part == "norms":  return ".layernorm" in key
  return False

def _expert_index_from_key(key):
  m = re.search(r"\.experts\.(\d+)\.", key)
  return int(m.group(1)) if m else None

def _parse_experts_arg(experts_arg, sd, L):
  """
  Returns:
    None  -> no filtering (keep all, used when experts == 'all')
    set() -> empty set (no experts)
    set of ints -> explicit expert ids
  Accepts: 'all', comma list, or space-separated via nargs.
  """
  if experts_arg is None:
    return None
  # normalize possibly list of tokens
  if isinstance(experts_arg, list):
    if len(experts_arg) == 1 and ("," in experts_arg[0] or experts_arg[0] == "all"):
      experts_arg = experts_arg[0]
    else:
      # treat as space-separated ids
      if any(tok == "all" for tok in experts_arg):
        return None
      return {int(e) for e in experts_arg}
  if isinstance(experts_arg, str):
    if experts_arg.strip().lower() == "all":
      return None
    parts = [p for p in experts_arg.replace(",", " ").split() if p]
    return {int(p) for p in parts}
  return None

def _all_experts_in_layer(sd, L):
  pref = f"model.layers.{L}.mlp.experts."
  ids = set()
  for k in sd.keys():
    if k.startswith(pref):
      e = _expert_index_from_key(k)
      if e is not None: ids.add(e)
  return ids

def iter_subset(sd, args, cfg):
  # Whole model
  if getattr(args, "all", False):
    for k, t in sd.items():
      yield k, t
    return

  # Embeds / head
  if getattr(args, "embeds", False):
    yield "model.embed_tokens.weight", sd["model.embed_tokens.weight"]
  if getattr(args, "lm_head", False) and "lm_head.weight" in sd:
    yield "lm_head.weight", sd["lm_head.weight"]

  # Layer/part (with optional experts filter)
  if args.layer is not None:
    L = args.layer
    part = args.part
    experts_filter = _parse_experts_arg(getattr(args, "experts", None), sd, L)
    # When part=mlp, always include router weights
    router_candidates = [
      f"model.layers.{L}.mlp.gate.weight",
      f"model.layers.{L}.mlp.gate.bias",
      f"model.layers.{L}.mlp.router.gate.weight",
      f"model.layers.{L}.mlp.router.gate.bias",
    ]

    if part == "mlp":
      for rk in router_candidates:
        if rk in sd: 
          yield rk, sd[rk]

    for k, t in sd.items():
      if not _want_key_for_layer_part(k, L, part): 
        continue
      # Skip router (already included above)
      if ".mlp.gate.weight" in k or ".mlp.gate.bias" in k or ".mlp.router.gate." in k:
        continue
      # Filter by expert set when provided (None means "all experts")
      if ".experts." in k and experts_filter is not None:
        eidx = _expert_index_from_key(k)
        if eidx is None or eidx not in experts_filter:
          continue
      yield k, t
    return

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", required=True)
  ap.add_argument("--out",   required=True)
  ap.add_argument("--all", action="store_true")
  ap.add_argument("--embeds", action="store_true")
  ap.add_argument("--lm_head", action="store_true")
  ap.add_argument("--layer", type=int)
  ap.add_argument("--part", choices=["attn","mlp","norms"])
  # accept "all", "1,2,3", or "1 2 3"
  ap.add_argument("--experts", nargs="+", help="Expert IDs: 'all' or list (space/comma)")
  ap.add_argument("--quant", choices=["none","q8","q4"], default="none")
  args = ap.parse_args()

  model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="cpu")
  cfg = AutoConfig.from_pretrained(args.model)
  sd = model.state_dict()

  selected = list(iter_subset(sd, args, cfg))
  with open(args.out, "wb") as f:
    f.write(MAGIC)
    f.write(struct.pack("<I", len(selected)))
    for name, t in selected:
      if args.quant == "none" or t.dim() != 2:
        _save_tensor(f, name, t.float())
      else:
        if args.quant == "q8":
          s, q = _rowwise_q8(t)
          _save_tensor(f, name + ".scale", s)
          _save_tensor(f, name + ".q8", q)
        elif args.quant == "q4":
          s, q = _rowwise_q4(t)
          _save_tensor(f, name + ".scale", s)
          _save_tensor(f, name + ".q4", q)

if __name__ == "__main__":
  main()
