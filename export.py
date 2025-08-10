#!/usr/bin/env python3
# scripts/export_weights.py
import argparse, json, struct, torch
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
  # pack low/high nibble (offset by +8 to 0..15)
  q = (q + 8).to(torch.uint8)
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
  # expects "...experts.<E>." pattern; returns int or None
  tag = ".experts."
  if tag not in key: return None
  try:
    after = key.split(tag, 1)[1]
    e = int(after.split(".", 1)[0])
    return e
  except Exception:
    return None

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
  # Layer-scoped
  if args.layer is not None:
    L = args.layer
    part = args.part
    experts_filter = None
    if getattr(args, "experts", None):
      experts_filter = {int(e) for e in str(args.experts).split(",") if e.strip() != ""}
    for k, t in sd.items():
      if not _want_key_for_layer_part(k, L, part): continue
      if experts_filter is not None:
        eidx = _expert_index_from_key(k)
        # keep only expert-specific tensors that match; allow non-expert MLP tensors to be skipped
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
  ap.add_argument("--experts")  # "3,7,42"
  ap.add_argument("--quant", choices=["none","q8","q4"], default="none")
  args = ap.parse_args()

  # Load on CPU to avoid GPU spikes; fp16 weights are fine, we cast as needed
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

