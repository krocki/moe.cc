# dump_head_io.py
import argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

@torch.inference_mode()
def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", required=True)
  ap.add_argument("--text", default="hello")
  ap.add_argument("--outbase", required=True)
  args = ap.parse_args()

  tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
  ids = tok(args.text, return_tensors="pt").input_ids  # [1, T]

  model = AutoModelForCausalLM.from_pretrained(
    args.model, torch_dtype=torch.float32, device_map="cpu"
  ).eval()

  # Forward through the transformer to get hidden states
  out = model.model(input_ids=ids, output_hidden_states=True, use_cache=False)
  last_hidden = out.hidden_states[-1][0]  # [T, d_model], pre-final-norm

  # Final RMSNorm + lm_head to get logits
  final_norm = model.model.norm(last_hidden)            # [T, d_model]
  logits = model.lm_head(final_norm)                    # [T, vocab]

  np.save(f"{args.outbase}.x.npy", last_hidden.contiguous().cpu().numpy().astype("float32"))
  np.save(f"{args.outbase}.logits.npy", logits.contiguous().cpu().numpy().astype("float32"))
  print(f"Saved {args.outbase}.x.npy and {args.outbase}.logits.npy")

if __name__ == "__main__":
  main()
