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
def apply_qk_rmsnorm(Q, K, qn, kn, n_q, n_kv, head_dim, eps=1e-6):
    """
    Per-head RMSNorm with epsilon, then apply learned scales.
    Supports three layouts for qn/kn: [head_dim], [#heads], or [#heads*head_dim].
    """
    def _one(Tmat, H, scale):
        # reshape to [T, H, D]
        T, Dtot = Tmat.shape
        x = Tmat.view(T, H, head_dim).float()
        # RMS over last dim
        msq = (x * x).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(msq + eps)
        if scale is not None:
            s = scale.float()
            if s.numel() == head_dim:
                x = x * s.view(1, 1, head_dim)
            elif s.numel() == H:
                x = x * s.view(1, H, 1)
            elif s.numel() == H * head_dim:
                x = x * s.view(1, H, head_dim)
            else:
                raise ValueError(f"Unexpected scale len {s.numel()} for H={H}, D={head_dim}")
        return x.view(T, H * head_dim).to(Tmat.dtype)

    Q = _one(Q, n_q,  qn)
    K = _one(K, n_kv, kn)
    return Q, K

@torch.no_grad()
def rope_apply_torch_gqa(Q, K, n_q, n_kv, head_dim, pos0, theta):
  T = Q.shape[0]
  i = torch.arange(0, head_dim, 2, dtype=Q.dtype, device=Q.device)
  base = theta ** (-i / head_dim)           # [D/2]
  p = torch.arange(pos0, pos0 + T, dtype=Q.dtype, device=Q.device).unsqueeze(1)  # [T,1]
  ang = p * base                            # [T,D/2]
  c = torch.cos(ang).unsqueeze(-1)
  s = torch.sin(ang).unsqueeze(-1)

  def apply(x, H):
    x = x.view(T, H, head_dim)
    xe = x[:, :, 0::2].transpose(1,2)       # [T,D/2,H]
    xo = x[:, :, 1::2].transpose(1,2)
    xe2 =  xe * c - xo * s
    xo2 =  xo * c + xe * s
    xe2 = xe2.transpose(1,2)
    xo2 = xo2.transpose(1,2)
    y = torch.empty_like(x)
    y[:, :, 0::2] = xe2
    y[:, :, 1::2] = xo2
    return y.view(T, H * head_dim)

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
  Q, K = apply_qk_rmsnorm(Q, K, qn, kn, n_q, n_kv, head_dim)  # <- was apply_qk_norm(...)
  if rope:
      Q, K = rope_apply_torch_gqa(Q, K, n_q, n_kv, head_dim, pos0=0, theta=rope_theta)

  Hcat = torch.zeros(T, Dq, dtype=x.dtype)
  Qh = Q.view(T, n_q, head_dim)
  Kh = K.view(T, n_kv, head_dim)
  Vh = V.view(T, n_kv, head_dim)
  for h in range(n_q):
    kvh = h % n_kv
    S = (Qh[:,h,:] @ Kh[:,kvh,:].t()) * scale
    if causal:
      mask = torch.triu(torch.ones(T,T, dtype=torch.bool, device=S.device), diagonal=1)
      S = S.masked_fill(mask, float('-inf'))
    P = torch.softmax(S, dim=-1)
    Hcat[:, h*head_dim:(h+1)*head_dim] = P @ Vh[:,kvh,:]

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
  sd = model.state_dict(); L = args.layer

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

  print(f"Saved {args.outbase}.x.npy and {args.outbase}.y.npy (T={args.seqlen}, D={d_model})")
  print("Export:")
  print(f"  python export.py --model {args.model} --out l{L}_attn.bin --layer {L} --part attn --quant none")

if __name__ == "__main__":
  main()
