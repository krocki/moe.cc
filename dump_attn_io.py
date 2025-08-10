#!/usr/bin/env python3
import argparse, numpy as np, torch
from transformers import AutoModelForCausalLM

def infer_head_dim(Dq, Dkv):
    # prefer common dims used by Qwen/LLaMA
    for d in [128, 96, 80, 64, 48, 40, 32]:
        if Dq % d == 0 and Dkv % d == 0:
            return d
    # fallback: gcd
    from math import gcd
    g = gcd(Dq, Dkv)
    return g if g > 0 else Dq

@torch.no_grad()
def attn_forward_torch_gqa(x, Wq,bq, Wk,bk, Wv,bv, Wo,bo, qn,kn, causal):
    T, d_model = x.shape
    Dq, Dkv = Wq.shape[0], Wk.shape[0]
    head_dim = infer_head_dim(Dq, Dkv)
    n_q = Dq // head_dim
    n_kv = Dkv // head_dim
    scale = (1.0 / (head_dim ** 0.5))

    def proj(x, W, b):
        y = x @ W.t()
        if b is not None: y = y + b
        return y

    Q = proj(x, Wq, bq)   # [T, Dq]
    K = proj(x, Wk, bk)   # [T, Dkv]
    V = proj(x, Wv, bv)   # [T, Dkv]

    # Apply per-head or per-channel norms
    if qn is not None:
        if qn.numel() == Dq:
            Q = Q * qn
        elif qn.numel() == n_q:
            Q = Q.view(T, n_q, head_dim) * qn.view(1, n_q, 1)
            Q = Q.view(T, Dq)
        elif qn.numel() == head_dim:
            Q = Q.view(T, n_q, head_dim) * qn.view(1, 1, head_dim)
            Q = Q.view(T, Dq)
    if kn is not None:
        if kn.numel() == Dkv:
            K = K * kn
        elif kn.numel() == n_kv:
            K = K.view(T, n_kv, head_dim) * kn.view(1, n_kv, 1)
            K = K.view(T, Dkv)
        elif kn.numel() == head_dim:
            K = K.view(T, n_kv, head_dim) * kn.view(1, 1, head_dim)
            K = K.view(T, Dkv)

    # GQA attention: each Q head attends to K/V head (h % n_kv)
    Hcat = torch.zeros(T, Dq, dtype=x.dtype)
    Qh = Q.view(T, n_q, head_dim)
    Kh = K.view(T, n_kv, head_dim)
    Vh = V.view(T, n_kv, head_dim)

    for h in range(n_q):
        kvh = h % n_kv
        qh = Qh[:, h, :]              # [T, d]
        kh = Kh[:, kvh, :]            # [T, d]
        vh = Vh[:, kvh, :]            # [T, d]
        S = (qh @ kh.t()) * scale     # [T, T]
        if causal:
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
            S.masked_fill_(mask, float('-inf'))
        P = torch.softmax(S, dim=-1)  # [T, T]
        out = P @ vh                   # [T, d]
        Hcat[:, h*head_dim:(h+1)*head_dim] = out

    y = Hcat @ Wo.t()
    if bo is not None: y = y + bo
    return y, n_q, n_kv, head_dim

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--seqlen", type=int, default=1)  # keep 1 for now (no RoPE)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--causal", type=int, default=1)
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

    d_model = Wq.shape[1]
    x = torch.randn(args.seqlen, d_model, dtype=torch.float32)
    y, n_q, n_kv, head_dim = attn_forward_torch_gqa(x, Wq,bq, Wk,bk, Wv,bv, Wo,bo, qn,kn, args.causal)

    np.save(args.outbase + ".x.npy", x.numpy().astype(np.float32))
    np.save(args.outbase + ".y.npy", y.numpy().astype(np.float32))

    print("Heads:", dict(n_q=int(n_q), n_kv=int(n_kv), head_dim=int(head_dim)))
    print("Suggested export:")
    print(f"  python export.py --model {args.model} --out l{L}_attn.bin --layer {L} --part attn --quant none")

if __name__ == "__main__":
    main()
