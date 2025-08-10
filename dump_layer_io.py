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
def rmsnorm(x, w, eps):
  # x: [T, D]; w: [D]
  var = (x * x).mean(dim=-1, keepdim=True)
  xhat = x * torch.rsqrt(var + eps)
  return xhat * w

@torch.no_grad()
def proj(x, W, b):  # x @ W^T + b
  y = x @ W.t()
  return y + b if b is not None else y

@torch.no_grad()
def apply_qk_norm(Q, K, qn, kn, n_q, n_kv, head_dim):
  T = Q.shape[0]; Dq, Dkv = Q.shape[1], K.shape[1]
  if qn is not None:
    if qn.numel() == n_q:
      Q = Q.view(T,n_q,head_dim) * qn.view(1,n_q,1); Q = Q.view(T,Dq)
    elif qn.numel() == Dq:
      Q = Q * qn
  if kn is not None:
    if kn.numel() == n_kv:
      K = K.view(T,n_kv,head_dim) * kn.view(1,n_kv,1); K = K.view(T,Dkv)
    elif kn.numel() == Dkv:
      K = K * kn
  return Q,K

@torch.no_grad()
def rope(Q, K, n_q, n_kv, head_dim, pos0, theta):
  T = Q.shape[0]
  i = torch.arange(0, head_dim, 2, dtype=Q.dtype)
  base = theta ** (-i / head_dim)
  p = torch.arange(pos0, pos0 + T, dtype=Q.dtype).unsqueeze(1)
  ang = p * base
  c = torch.cos(ang).unsqueeze(-1)
  s = torch.sin(ang).unsqueeze(-1)

  def apply(x, H):
    x = x.view(T,H,head_dim)
    xe = x[:,:,0::2].transpose(1,2)
    xo = x[:,:,1::2].transpose(1,2)
    xe2 =  xe * c - xo * s
    xo2 =  xo * c + xe * s
    xe2 = xe2.transpose(1,2)
    xo2 = xo2.transpose(1,2)
    y = torch.empty_like(x)
    y[:,:,0::2] = xe2; y[:,:,1::2] = xo2
    return y.view(T, H*head_dim)

  return apply(Q, n_q), apply(K, n_kv)

@torch.no_grad()
def attn_gqa_block(x, Wq,bq, Wk,bk, Wv,bv, Wo,bo, qn,kn, causal, theta):
  T, D = x.shape
  Dq, Dkv = Wq.shape[0], Wk.shape[0]
  d = infer_head_dim(Dq, Dkv)
  n_q = Dq // d; n_kv = Dkv // d
  Q = proj(x,Wq,bq); K = proj(x,Wk,bk); V = proj(x,Wv,bv)
  Q,K = apply_qk_norm(Q,K,qn,kn,n_q,n_kv,d)
  Q,K = rope(Q,K,n_q,n_kv,d, pos0=0, theta=theta)
  scale = 1.0 / (d**0.5)
  Hcat = torch.zeros(T, Dq, dtype=x.dtype)
  Qh = Q.view(T,n_q,d); Kh = K.view(T,n_kv,d); Vh = V.view(T,n_kv,d)
  for h in range(n_q):
    kvh = h % n_kv
    S = (Qh[:,h,:] @ Kh[:,kvh,:].t()) * scale
    if causal:
      mask = torch.triu(torch.ones(T,T, dtype=torch.bool), diagonal=1)
      S.masked_fill_(mask, float('-inf'))
    P = torch.softmax(S, dim=-1)
    Hcat[:, h*d:(h+1)*d] = P @ Vh[:,kvh,:]
  y = Hcat @ Wo.t()
  return y + (bo if bo is not None else 0)

@torch.no_grad()
def swi_glu_expert(x, Wg,bg, Wu,bu, Wd,bd):
  g = x @ Wg.t();  g = g + (bg if bg is not None else 0);  g = torch.nn.functional.silu(g)
  u = x @ Wu.t();  u = u + (bu if bu is not None else 0)
  h = g * u
  y = h @ Wd.t();  y = y + (bd if bd is not None else 0)
  return y

@torch.no_grad()
def moe_block(x, gate_w, gate_b, experts, k=8):
  # gate logits: [T, E]
  logits = x @ gate_w.t()
  if gate_b is not None: logits = logits + gate_b
  topk = torch.topk(logits, k=k, dim=-1)
  idx = topk.indices  # [T,k]
  prob = torch.softmax(topk.values, dim=-1)  # softmax over k only
  T, D = x.shape
  y = torch.zeros_like(x)
  for t in range(T):
    for i in range(k):
      e = idx[t,i].item()
      p = prob[t,i].item()
      y[t] += p * experts[e](x[t:t+1]).squeeze(0)
  return y

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", required=True)
  ap.add_argument("--layer", type=int, default=0)
  ap.add_argument("--seqlen", type=int, default=1)
  ap.add_argument("--seed", type=int, default=123)
  ap.add_argument("--causal", type=int, default=1)
  ap.add_argument("--rope-theta", type=float, default=10000.0)
  ap.add_argument("--outbase", required=True)
  args = ap.parse_args()

  torch.manual_seed(args.seed); np.random.seed(args.seed)
  model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, device_map="cpu")
  sd = model.state_dict(); L = args.layer

  d_model = sd[f"model.layers.{L}.self_attn.q_proj.weight"].shape[1]
  x = torch.randn(args.seqlen, d_model, dtype=torch.float32)

  # Norm1
  w1 = sd.get(f"model.layers.{L}.input_layernorm.weight")
  if w1 is None:
    w1 = sd.get(f"model.layers.{L}.rms_1.weight")

  x1 = rmsnorm(x, w1.float(), 1e-6)

  # Attn
  Wq = sd[f"model.layers.{L}.self_attn.q_proj.weight"].float()
  Wk = sd[f"model.layers.{L}.self_attn.k_proj.weight"].float()
  Wv = sd[f"model.layers.{L}.self_attn.v_proj.weight"].float()
  Wo = sd[f"model.layers.{L}.self_attn.o_proj.weight"].float()
  bq = sd.get(f"model.layers.{L}.self_attn.q_proj.bias")
  bk = sd.get(f"model.layers.{L}.self_attn.k_proj.bias")
  bv = sd.get(f"model.layers.{L}.self_attn.v_proj.bias")
  bo = sd.get(f"model.layers.{L}.self_attn.o_proj.bias")
  bq = bq.float() if bq is not None else None
  bk = bk.float() if bk is not None else None
  bv = bv.float() if bv is not None else None
  bo = bo.float() if bo is not None else None

  qn = sd.get(f"model.layers.{L}.self_attn.q_norm.weight")
  kn = sd.get(f"model.layers.{L}.self_attn.k_norm.weight")
  qn = qn.float() if qn is not None else None
  kn = kn.float() if kn is not None else None

  aout = attn_gqa_block(x1, Wq,bq, Wk,bk, Wv,bv, Wo,bo, qn,kn, args.causal, args.rope_theta)
  x2 = x + aout  # residual

  # Norm2
  w2 = sd.get(f"model.layers.{L}.post_attention_layernorm.weight")
  if w2 is None:
    w2 = sd.get(f"model.layers.{L}.rms_2.weight")
  x3 = rmsnorm(x2, w2.float(), 1e-6)

  # Prepare experts
  # Gate
  gate_w = sd.get(f"model.layers.{L}.mlp.gate.weight")
  if gate_w is None:
    gate_w = sd.get(f"model.layers.{L}.mlp.router.gate.weight")
  gate_b = sd.get(f"model.layers.{L}.mlp.gate.bias")
  if gate_b is None:
    gate_b = sd.get(f"model.layers.{L}.mlp.router.gate.bias")
  gate_w = gate_w.float()
  gate_b = gate_b.float() if gate_b is not None else None

  # Expert functions
  experts = []
  e = 0
  while True:
    try:
      Wg = sd[f"model.layers.{L}.mlp.experts.{e}.gate_proj.weight"].float()
      Wu = sd[f"model.layers.{L}.mlp.experts.{e}.up_proj.weight"].float()
      Wd = sd[f"model.layers.{L}.mlp.experts.{e}.down_proj.weight"].float()
      bg = sd.get(f"model.layers.{L}.mlp.experts.{e}.gate_proj.bias")
      bu = sd.get(f"model.layers.{L}.mlp.experts.{e}.up_proj.bias")
      bd = sd.get(f"model.layers.{L}.mlp.experts.{e}.down_proj.bias")
      bg = bg.float() if bg is not None else None
      bu = bu.float() if bu is not None else None
      bd = bd.float() if bd is not None else None
      experts.append(lambda z, Wg=Wg,bg=bg,Wu=Wu,bu=bu,Wd=Wd,bd=bd: swi_glu_expert(z, Wg,bg, Wu,bu, Wd,bd))
      e += 1
    except KeyError:
      break

  y = moe_block(x3, gate_w, gate_b, experts, k=8) + x2  # residual

  np.save(args.outbase + ".x.npy", x.numpy().astype(np.float32))
  np.save(args.outbase + ".y.npy", y.numpy().astype(np.float32))

  print("Export suggestion (dump all layer weights):")
  print(f"  python export.py --model {args.model} --out l{L}_layer.bin --layer {L} --part norms --quant none")
  print(f"  python export.py --model {args.model} --out l{L}_attn.bin  --layer {L} --part attn  --quant none")
  print(f"  python export.py --model {args.model} --out l{L}_moe.bin   --layer {L} --part mlp   --experts all --quant none")
  print("Then you can either cat them or extend export.py to do a single pass including all three parts.")
if __name__ == "__main__":
  main()
