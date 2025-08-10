
#!/usr/bin/env python3
# scripts/export_weights.py (robust --experts handling)
import argparse, struct, torch
from transformers import AutoModelForCausalLM, AutoConfig

MAGIC = b"QW3W\x00\x01"

def _save_tensor(f, name, t):
  t = t.contiguous()
  name_b = name.encode()
  f.write(struct.pack("<I", len(name_b))); f.write(name_b)
  if   t.dtype == torch.float32: dt = 0
  elif t.dtype == torch.float16: dt = 1
  elif t.dtype == torch.int8:    dt = 2
  else: raise ValueError(f"dtype {t.dtype} not supported")
  f.write(struct.pack("<I", dt))
  f.write(struct.pack("<I", t.dim()))
  f.write(struct.pack("<" + "I"*t.dim(), *t.shape))
  f.write(t.cpu().numpy().tobytes())

def iter_subset(sd, args):
  if getattr(args, "all", False):
    for k,t in sd.items(): yield k,t; return
  if getattr(args, "embeds", False):
    yield "model.embed_tokens.weight", sd["model.embed_tokens.weight"]
  if getattr(args, "lm_head", False) and "lm_head.weight" in sd:
    yield "lm_head.weight", sd["lm_head.weight"]
  if args.layer is not None:
    L=args.layer; part=args.part
    # normalize experts
    experts_str = getattr(args,"experts",None)
    experts_set = None
    if experts_str:
      if experts_str=="all":
        experts_set = None  # include all experts
      else:
        # allow space/comma
        if isinstance(experts_str, list):
          flat = []
          for item in experts_str:
            flat.extend(item.split(","))
          experts_str = ",".join(flat)
        ids = [s for s in experts_str.split(",") if s.strip()!=""]
        experts_set = set(int(x) for x in ids)
    pref = f"model.layers.{L}."
    for k,t in sd.items():
      if not k.startswith(pref): continue
      if part=="attn" and ".self_attn." in k:
        yield k,t
      elif part=="norms" and ".layernorm" in k:
        yield k,t
      elif part=="mlp":
        if ".experts." in k:
          if experts_set is None:  # all
            yield k,t
          else:
            # filter by expert id
            try:
              e = int(k.split(".experts.")[1].split(".")[0])
            except Exception:
              continue
            if e in experts_set: yield k,t
        elif ".mlp.gate." in k or ".mlp.router.gate." in k:
          # always keep router weights with mlp
          yield k,t

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--all", action="store_true")
  ap.add_argument("--embeds", action="store_true")
  ap.add_argument("--lm_head", action="store_true")
  ap.add_argument("--layer", type=int)
  ap.add_argument("--part", choices=["attn","mlp","norms"])
  ap.add_argument("--experts", nargs="+")
  ap.add_argument("--quant", choices=["none","q8","q4"], default="none")
  args = ap.parse_args()

  model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="cpu")
  sd = model.state_dict()

  selected = list(iter_subset(sd, args))
  with open(args.out, "wb") as f:
    f.write(MAGIC)
    f.write(struct.pack("<I", len(selected)))
    for name,t in selected:
      _save_tensor(f, name, t.float())

if __name__=="__main__":
  main()

