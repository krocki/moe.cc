# dump_embed_io.py
import argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", required=True)
  ap.add_argument("--text", default="hello")
  ap.add_argument("--outbase", default="embed_test")
  args = ap.parse_args()

  tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
  ids = tok(args.text, return_tensors="pt").input_ids[0]  # [T]
  T = ids.shape[0]

  model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, device_map="cpu")
  Wemb = model.model.embed_tokens.weight.detach().float()  # [V, D]
  x = Wemb[ids].contiguous().cpu().numpy().astype("float32")  # [T, D]

  np.save(f"{args.outbase}.ids.npy", ids.cpu().numpy().astype("int32"))
  np.save(f"{args.outbase}.emb.npy", x)
  print(f"Saved {args.outbase}.ids.npy and {args.outbase}.emb.npy (T={T}, D={x.shape[1]})")

if __name__ == "__main__":
  main()
