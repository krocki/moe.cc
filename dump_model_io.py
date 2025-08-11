#!/usr/bin/env python3
import argparse, numpy as np, torch, json
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

@torch.no_grad()
def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", required=True)
  ap.add_argument("--seqlen", type=int, default=4)
  ap.add_argument("--seed", type=int, default=123)
  ap.add_argument("--outbase", required=True)
  args = ap.parse_args()

  torch.manual_seed(args.seed); np.random.seed(args.seed)
  tok = AutoTokenizer.from_pretrained(args.model)
  cfg = AutoConfig.from_pretrained(args.model)
  mdl = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, device_map="cpu")
  vocab = int(cfg.vocab_size)

  # Make a short token sequence deterministically (avoid special tokens conflicts)
  # Use tokenizer's bos_token_id if available, then a few random-but-valid ids.
  ids = [tok.bos_token_id if tok.bos_token_id is not None else 1]
  while len(ids) < args.seqlen:
    ids.append(int(np.random.randint(low=5, high=vocab-5)))
  ids = torch.tensor([ids], dtype=torch.long)  # [1,T]

  out = mdl(input_ids=ids)
  logits = out.logits.squeeze(0).to(torch.float32).cpu().numpy()  # [T,vocab]

  np.save(args.outbase + ".ids.npy", ids.squeeze(0).cpu().numpy().astype(np.int32))
  np.save(args.outbase + ".logits.npy", logits.astype(np.float32))
  meta = {
    "vocab": vocab,
    "n_layer": int(cfg.num_hidden_layers),
    "n_q": int(getattr(cfg, "num_attention_heads", 0)),
    "n_kv": int(getattr(cfg, "num_key_value_heads", 0) or getattr(cfg, "num_kv_heads", 0)),
    "hidden_size": int(cfg.hidden_size),
    "rope_theta": float(getattr(cfg, "rope_theta", 10000.0)),
    "rms_norm_eps": float(getattr(cfg, "rms_norm_eps", 1e-6)),
  }
  with open(args.outbase + ".meta.json", "w") as f:
    json.dump(meta, f, indent=2)

  print("Saved:")
  print(" ", args.outbase + ".ids.npy")
  print(" ", args.outbase + ".logits.npy")
  print(" ", args.outbase + ".meta.json")

if __name__ == "__main__":
  main()
