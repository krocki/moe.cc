#!/usr/bin/env python4
# export.py — dump weights from HF to our tiny-bin format
import argparse, json, os, struct, torch
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
  elif t.dtype == torch.uint8:   dt = 3
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
  q = torch.round(w / s).clamp_(-8,7).to(torch.int8)  # -8..7
  q = (q + 8).to(torch.uint8)
  lo = q[:, ::2]; hi = q[:, 1::2]
  packed = (lo | (hi << 4)).contiguous()
  return s.squeeze(1).to(torch.float32), packed

def _is_norm_key(k: str) -> bool:
  # Qwen3: input_layernorm / post_attention_layernorm or rms_1 / rms_2
  return (".layernorm" in k) or (".rms_1" in k) or (".rms_2" in k) or k.endswith(".norm.weight")

def _is_expert_weight(k: str) -> bool:
  """Check if tensor is an expert weight matrix that should be quantized."""
  return (".experts." in k and 
          (k.endswith(".gate_proj.weight") or 
           k.endswith(".up_proj.weight") or 
           k.endswith(".down_proj.weight")))

def _should_quantize(k: str, args) -> bool:
  """Determine if a tensor should be quantized based on name and args."""
  if args.quant == "none":
    return False
  # Only quantize 2D expert weight matrices
  return _is_expert_weight(k)

def iter_subset(sd, args):
  # whole model
  if getattr(args, "all", False):
    for k,t in sd.items():
      yield k, t
    return

  # embeds / head
  if getattr(args, "embeds", False):
    yield "model.embed_tokens.weight", sd["model.embed_tokens.weight"]
  if getattr(args, "lm_head", False) and "lm_head.weight" in sd:
    yield "lm_head.weight", sd["lm_head.weight"]

  # layer-scoped
  if args.layer is not None:
    L = args.layer
    part = args.part
    pref = f"model.layers.{L}."
    # normalize experts filter
    experts_set = None
    if part == "mlp" and getattr(args, "experts", None):
      toks = []
      src = args.experts
      if isinstance(src, list):
        for item in src:
          toks.extend(s.strip() for s in item.split(",") if s.strip())
      else:
        toks = [s.strip() for s in str(src).split(",") if s.strip()]
      if len(toks) == 1 and toks[0].lower() == "all":
        experts_set = None  # keep all
      else:
        experts_set = set(int(x) for x in toks)

    for k,t in sd.items():
      if not k.startswith(pref): continue
      if part == "attn" and ".self_attn." in k:
        yield k, t
      elif part == "norms" and _is_norm_key(k):
        yield k, t
      elif part == "mlp":
        if ".mlp.gate." in k or ".mlp.router.gate." in k:
          yield k, t
        elif ".experts." in k:
          if experts_set is None:
            yield k, t
          else:
            try:
              e = int(k.split(".experts.")[1].split(".")[0])
            except Exception:
              continue
            if e in experts_set:
              yield k, t

def _sanitize(name: str) -> str:
  # filesystem-friendly
  s = name.replace("/", "_").replace("..", ".")
  s = s.replace(" ", "_")
  return s

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", required=True)
  ap.add_argument("--out",   help="single .bin output (legacy mode)")
  ap.add_argument("--outdir", help="directory to dump per-tensor files + manifest")
  ap.add_argument("--all", action="store_true")
  ap.add_argument("--embeds", action="store_true")
  ap.add_argument("--lm_head", action="store_true")
  ap.add_argument("--layer", type=int)
  ap.add_argument("--part", choices=["attn","mlp","norms"])
  ap.add_argument("--experts", nargs="+")  # e.g., 5,19 34
  ap.add_argument("--quant", choices=["none","q8","q4"], default="none")
  args = ap.parse_args()

  if not args.out and not args.outdir:
    ap.error("require --out or --outdir")

  model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="cpu")
  cfg = AutoConfig.from_pretrained(args.model)
  sd = model.state_dict()

  # collect tensors
  selected = list(iter_subset(sd, args))
  if not selected:
    print("No tensors selected.")
    return

  if args.outdir:
    os.makedirs(args.outdir, exist_ok=True)
    manifest = {"model": args.model, "quant": args.quant, "tensors": []}
    # save config.json for metadata like rope_theta, heads, etc.
    with open(os.path.join(args.outdir, "config.json"), "w") as f:
      json.dump(cfg.to_dict(), f, indent=2)

    for name, t in selected:
      path = os.path.join(args.outdir, _sanitize(name) + ".bin")
      with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", 1))  # one tensor per file
        
        # Selective quantization: only quantize 2D expert weight matrices
        should_quant = _should_quantize(name, args) and t.dim() == 2
        
        if not should_quant:
          _save_tensor(f, name, t.float())
          qinfo = None
        else:
          if args.quant == "q8":
            s,q = _rowwise_q8(t)
            _save_tensor(f, name+".scale", s)
            _save_tensor(f, name+".q8", q)
            qinfo = {"scheme":"rowwise_q8"}
          else:  # q4
            s,q = _rowwise_q4(t)
            _save_tensor(f, name+".scale", s)
            _save_tensor(f, name+".q4", q)
            qinfo = {"scheme":"rowwise_q4"}
      
      # keep a light index
      shp = tuple(int(x) for x in t.shape)
      manifest["tensors"].append({
        "name": name,
        "file": os.path.basename(path),
        "shape": shp,
        "dtype": "f32" if not should_quant else "quant",
        "quant": qinfo
      })

    with open(os.path.join(args.outdir, "manifest.json"), "w") as f:
      json.dump(manifest, f, indent=2)
    print(f"Wrote directory: {args.outdir} with {len(manifest['tensors'])} tensors")
    return

  # legacy single-file path
  with open(args.out, "wb") as f:
    f.write(MAGIC)
    f.write(struct.pack("<I", len(selected)))
    for name, t in selected:
      # Selective quantization: only quantize 2D expert weight matrices
      should_quant = _should_quantize(name, args) and t.dim() == 2
      
      if not should_quant:
        _save_tensor(f, name, t.float())
      else:
        if args.quant == "q8":
          s,q = _rowwise_q8(t)
          _save_tensor(f, name+".scale", s)
          _save_tensor(f, name+".q8", q)
        elif args.quant == "q4":
          s,q = _rowwise_q4(t)
          _save_tensor(f, name+".scale", s)
          _save_tensor(f, name+".q4", q)

if __name__ == "__main__":
  main()
