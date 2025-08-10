#!/usr/bin/env python3
import argparse, numpy as np, torch
from transformers import AutoModelForCausalLM

def infer_head_dim(Dq, Dkv):
  for d in [128, 96, 80, 64, 48, 40, 32]:
    if Dq % d == 0 and Dkv % d == 0: return d
  from math import gcd
  g = gcd(Dq, Dkv)
  return g if g > 0 else Dq

@torch.no_grad()
def apply_qk_norm(Q, K, qn, kn, n_q, n_kv, head_dim):
  T = Q.shape[0]
  Dq, Dkv = Q.shape[1], K.shape[1]
  if qn is not None:
    if qn.numel() == n_q:
      Q = Q.view(T, n_q, head_dim) * qn.view(1, n_q, 1)
      Q = Q.view(T, Dq)
    elif qn.numel() == Dq:
      Q = Q * qn
  if kn is not None:
    if kn.numel() == n_kv:
      K = K.view(T, n_kv, head_dim) * kn.view(1, n_kv, 1)
      K = K.view(T, Dkv)
    elif kn.numel() == Dkv:
      K = K * kn
  return Q, K

@torch.no_grad()
def rope_apply_torch_gqa(Q, K, n_q, n_kv, head_dim, pos0, theta):
  T = Q.shape[0]
  i = torch.arange(0, head_dim, 2, dtype=Q.dtype)
  base = theta ** (-i / head_dim)  # [D/2]
  p = torch.arange(pos0, pos0 + T, dtype=Q.dtype).unsqueeze(1)  # [T,1]
  ang = p * base  # [T,D/2]
  c = torch.cos(ang).unsqueeze(-1)
  s = torch.sin(ang).unsqueeze(-1)

  def apply(x, n_heads):
    x = x.view(T, n_heads, head_dim)
    xe = x[:, :, 0::2].transpose(1,2)  # [T,D/2,H]
    xo = x[:, :, 1::2].transpose(1,2)
    xe2 =  xe * c - xo * s
    xo2 =  xo * c + xe * s
    xe2 = xe2.transpose(1,2)
    xo2 = xo2.transpose(1,2)
    y = torch.empty_like(x)
    y[:, :, 0::2] = xe2
    y[:, :, 1::2] = xo2
    return y.view(T, n_heads * head_dim)

  return apply(Q, n_q), apply(K, n_kv)

@torch.no_grad()
def attn_forward_torch_gqa_with_rope(x, Wq,bq, Wk,bk, Wv,bv, Wo,bo, qn,kn, causal, rope, rope_theta):
  T, d_model = x.shape
  Dq, Dkv = Wq.shape[0], Wk.shape[0]
  head_dim = infer_head_dim(Dq, Dkv)
  n_q = Dq // head_dim
  n_kv = Dkv // head_dim
  scale = (1.0 / (head_dim ** 0.5))
  def proj(x,W,b): y = x @ W.t();  return y + b if b is not None else y

  Q = proj(x,Wq,bq); K = proj(x,Wk,bk); V = proj(x,Wv,bv)
  Q, K = apply_qk_norm(Q, K, qn, kn, n_q, n_kv, head_dim)
  if rope:
    Q, K = rope_apply_torch_gqa(Q, K, n_q, n_kv, head_dim, pos0=0, theta=rope_theta)

  Hcat = torch.zeros(T, Dq, dtype=x.dtype)
  Qh = Q.view(T, n_q, head_dim)
  Kh = K.view(T, n_kv, head_dim)
  Vh = V.view(T, n_kv, head_dim)
  for h in range(n_q):
    kvh = h % n_kv
    qh = Qh[:, h, :]
    kh = Kh[:, kvh, :]
    vh = Vh[:, kvh, :]
    S = (qh @ kh.t()) * scale
    if causal:
      mask = torch.triu(torch.ones(T,T, dtype=torch.bool), diagonal=1)
      S.masked_fill_(mask, float('-inf'))
    P = torch.softmax(S, dim=-1)
    Hcat[:, h*head_dim:(h+1)*head_dim] = P @ vh

  y = Hcat @ Wo.t()
  if bo is not None: y = y + bo
  return y

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", required=True)
  ap.add_argument("--layer", type=int, required=True)
  ap.add_argument("--seqlen", type=int, default=1)
  ap.add_argument("--seed", type=int, default=123)
  ap.add_argument("--causal", type=int, default=1)
  ap.add_argument("--rope", type=int, default=1)
  ap.add_argument("--rope-theta", type=float, default=10000.0)
  ap.add_argument("--outbase", required=True)
  args = ap.parse_args()

  torch.manual_seed(args.seed); np.random.seed(args.seed)
  model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, device_map="cpu")
  sd = model.state_dict()
  L = args.layer

  Wq = sd[f"model.layers.{L}.self_attn.q_proj.weight"].float()
  Wk = sd[f"model.layers.{L}.self_attn.k_proj.weight"].float()
  Wv = sd[f"model.layers.{L}.self_attn.v_proj.weight"].float()
  Wo = sd[f"model.layers.{L}.self_attn.o_proj.weight"].float()
  bq = sd.get(f"model.layers.{L}.self_attn.q_proj.bias"); bq = bq.float() if bq is not None else None
  bk = sd.get(f"model.layers.{L}.self_attn.k_proj.bias"); bk = bk.float() if bk is not None else None
  bv = sd.get(f"model.layers.{L}.self_attn.v_proj.bias"); bv = bv.float() if bv is not None else None
  bo = sd.get(f"model.layers.{L}.self_attn.o_proj.bias"); bo = bo.float() if bo is not None else None

  qn = sd.get(f"model.layers.{L}.self_attn.q_norm.weight")
  kn = sd.get(f"model.layers.{L}.self_attn.k_norm.weight")
  qn = qn.float() if qn is not None else None
  kn = kn.float() if kn is not None else None

  d_model = Wq.shape[1]
  x = torch.randn(args.seqlen, d_model, dtype=torch.float32)
  y = attn_forward_torch_gqa_with_rope(x, Wq,bq, Wk,bk, Wv,bv, Wo,bo, qn,kn, args.causal, args.rope, args.rope_theta)

  np.save(args.outbase + ".x.npy", x.numpy().astype(np.float32))
  np.save(args.outbase + ".y.npy", y.numpy().astype(np.float32))

  print("Suggested export:")
  print(f"  python export.py --model {args.model} --out l{L}_attn.bin --layer {L} --part attn --quant none")

if __name__ == "__main__":
  main()
