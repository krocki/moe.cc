#!/usr/bin/env python3
import argparse, numpy as np, torch

@torch.no_grad()
def rope_apply_torch_gqa(Q, K, n_q, n_kv, head_dim, pos0, theta):
  # Q: [T, n_q*head_dim], K: [T, n_kv*head_dim]
  T = Q.shape[0]
  def angles(head_dim, T, pos0, theta):
    # φ(p,i) with i over pairs, p over positions
    i = torch.arange(0, head_dim, 2, dtype=Q.dtype)
    base = theta ** (-i / head_dim)
    p = torch.arange(pos0, pos0 + T, dtype=Q.dtype).unsqueeze(1)
    return p * base  # [T, head_dim/2]
  ang = angles(head_dim, T, pos0, theta)
  c = torch.cos(ang).unsqueeze(-1)  # [T, D/2, 1]
  s = torch.sin(ang).unsqueeze(-1)  # [T, D/2, 1]

  def apply(x, n_heads):
    x = x.view(T, n_heads, head_dim)
    x_even = x[:, :, 0::2]  # [T, H, D/2]
    x_odd  = x[:, :, 1::2]  # [T, H, D/2]
    # broadcast T,D/2 over heads
    xe = x_even.transpose(1,2)  # [T, D/2, H]
    xo = x_odd.transpose(1,2)   # [T, D/2, H]
    # rotate
    xe2 =  xe * c - xo * s
    xo2 =  xo * c + xe * s
    # fold back
    xe2 = xe2.transpose(1,2)  # [T, H, D/2]
    xo2 = xo2.transpose(1,2)  # [T, H, D/2]
    y = torch.empty_like(x)
    y[:, :, 0::2] = xe2
    y[:, :, 1::2] = xo2
    return y.view(T, n_heads * head_dim)

  YQ = apply(Q, n_q)
  YK = apply(K, n_kv)
  return YQ, YK

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--seqlen", type=int, default=4)
  ap.add_argument("--n_q", type=int, default=32)
  ap.add_argument("--n_kv", type=int, default=4)
  ap.add_argument("--head_dim", type=int, default=128)
  ap.add_argument("--pos0", type=int, default=0)
  ap.add_argument("--theta", type=float, default=10000.0)
  ap.add_argument("--seed", type=int, default=123)
  ap.add_argument("--outbase", required=True)
  args = ap.parse_args()

  torch.manual_seed(args.seed); np.random.seed(args.seed)

  Dq  = args.n_q  * args.head_dim
  Dkv = args.n_kv * args.head_dim
  Q = torch.randn(args.seqlen, Dq,  dtype=torch.float32)
  K = torch.randn(args.seqlen, Dkv, dtype=torch.float32)

  YQ, YK = rope_apply_torch_gqa(Q.clone(), K.clone(),
                                args.n_q, args.n_kv, args.head_dim,
                                args.pos0, args.theta)

  np.save(args.outbase + ".Q.npy",  Q.numpy())
  np.save(args.outbase + ".K.npy",  K.numpy())
  np.save(args.outbase + ".YQ.npy", YQ.numpy())
  np.save(args.outbase + ".YK.npy", YK.numpy())

  print("Wrote:", args.outbase + ".Q.npy/.K.npy/.YQ.npy/.YK.npy")
  print("Params:", dict(T=args.seqlen, n_q=args.n_q, n_kv=args.n_kv,
                        head_dim=args.head_dim, pos0=args.pos0, theta=args.theta))

if __name__ == "__main__":
  main()
