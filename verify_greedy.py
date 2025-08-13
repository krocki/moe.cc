#!/usr/bin/env python3
import argparse, json, os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

# =========================
# 1) Model (reference-style)
# =========================

def maybe_force_f32(model, enable: bool):
    if not enable:
        return
    for p in model.parameters():
        with torch.no_grad():
            p.data = p.data.to(torch.float32)
    for bname, buf in model.named_buffers():
        if buf.dtype != torch.float32:
            setattr(model, bname, buf.to(torch.float32))

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)
    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = torch.nn.functional.silu(x_fc1) * x_fc2
        return self.fc3(x)

class MoEFeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_experts_per_tok = cfg["num_experts_per_tok"]
        self.num_experts = cfg["num_experts"]
        self.gate = nn.Linear(cfg["emb_dim"], cfg["num_experts"], bias=False, dtype=cfg["dtype"])
        self.register_buffer("gate_bias", None, persistent=False)

        meta = torch.device("meta")
        self.fc1 = nn.ModuleList([
            nn.Linear(cfg["emb_dim"], cfg["moe_intermediate_size"], bias=False, dtype=cfg["dtype"], device=meta)
            for _ in range(cfg["num_experts"])
        ])
        self.fc2 = nn.ModuleList([
            nn.Linear(cfg["emb_dim"], cfg["moe_intermediate_size"], bias=False, dtype=cfg["dtype"], device=meta)
            for _ in range(cfg["num_experts"])
        ])
        self.fc3 = nn.ModuleList([
            nn.Linear(cfg["moe_intermediate_size"], cfg["emb_dim"], bias=False, dtype=cfg["dtype"], device=meta)
            for _ in range(cfg["num_experts"])
        ])

    def forward(self, x):
        scores = self.gate(x)
        if self.gate_bias is not None:
            scores = scores + self.gate_bias
        topk_scores, topk_idx = torch.topk(scores, self.num_experts_per_tok, dim=-1)
        topk_probs = torch.softmax(topk_scores, dim=-1)

        B,T,D = x.shape
        E = self.num_experts
        outs = []
        for e in range(E):
            h = torch.nn.functional.silu(self.fc1[e](x)) * self.fc2[e](x)
            y = self.fc3[e](h)
            outs.append(y.unsqueeze(-2))             # [B,T,1,D]
        expert_out = torch.cat(outs, dim=-2)         # [B,T,E,D]

        gating = torch.zeros_like(scores)            # [B,T,E]
        for i in range(self.num_experts_per_tok):
            idx = topk_idx[..., i:i+1]
            pr  = topk_probs[..., i:i+1]
            gating.scatter_(dim=-1, index=idx, src=pr)
        return (gating.unsqueeze(-1) * expert_out).sum(dim=-2)  # [B,T,D]

class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6, bias=False, qwen3_compatible=True):
        super().__init__()
        self.eps = eps
        self.qwen3_compatible = qwen3_compatible
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None
    def forward(self, x):
        inp_dtype = x.dtype
        if self.qwen3_compatible:
            x = x.to(torch.float32)
        var = (x * x).mean(dim=-1, keepdim=True)
        xhat = x * torch.rsqrt(var + self.eps)
        xhat = xhat * self.scale
        if self.shift is not None:
            xhat = xhat + self.shift
        return xhat.to(inp_dtype)

def compute_rope_params(head_dim, theta_base=10_000, context_length=4096, dtype=torch.float32):
    assert head_dim % 2 == 0
    inv = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: head_dim//2].float() / head_dim))
    pos = torch.arange(context_length, dtype=dtype)
    ang = pos[:, None] * inv[None, :]
    ang = torch.cat([ang, ang], dim=1)   # [ctx, d]
    return torch.cos(ang), torch.sin(ang)

def apply_rope(x, cos, sin):
    # x: [B,H,T,d]
    B,H,T,d = x.shape
    d2 = d//2
    x1, x2 = x[..., :d2], x[..., d2:]
    cos = cos[:T, :].unsqueeze(0).unsqueeze(0)
    sin = sin[:T, :].unsqueeze(0).unsqueeze(0)
    rot = torch.cat([-x2, x1], dim=-1)
    return (x * cos + rot * sin).to(x.dtype)

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False, dtype=None):
        super().__init__()
        assert num_heads % num_kv_groups == 0
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups
        if head_dim is None:
            assert d_in % num_heads == 0
            head_dim = d_in // num_heads
        self.head_dim = head_dim
        self.d_out = num_heads * head_dim

        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.W_key   = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.out_proj= nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

    def forward(self, x, mask, cos, sin):
        B,T,D = x.shape
        d = self.head_dim
        Hq = self.num_heads
        Hkv = self.num_kv_groups

        Q = self.W_query(x).view(B,T,Hq,d).transpose(1,2)  # [B,Hq,T,d]
        K = self.W_key(x).view(B,T,Hkv,d).transpose(1,2)   # [B,Hkv,T,d]
        V = self.W_value(x).view(B,T,Hkv,d).transpose(1,2)

        if self.q_norm is not None:
            Q = self.q_norm(Q)
            K = self.k_norm(K)

        Q = apply_rope(Q, cos, sin)
        K = apply_rope(K, cos, sin)

        K = K.repeat_interleave(self.group_size, dim=1)
        V = V.repeat_interleave(self.group_size, dim=1)

        S = (Q @ K.transpose(2,3)) / (d ** 0.5)            # [B,Hq,T,T]
        S = S.masked_fill(mask, float('-inf'))
        P = torch.softmax(S, dim=-1)
        ctx = (P @ V).transpose(1,2).reshape(B,T,self.d_out)
        return self.out_proj(ctx)

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            head_dim=cfg["head_dim"],
            num_kv_groups=cfg["n_kv_groups"],
            qk_norm=cfg["qk_norm"],
            dtype=cfg["dtype"]
        )
        if cfg["num_experts"] > 0:
            self.ff = MoEFeedForward(cfg)
        else:
            self.ff = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.norm2 = RMSNorm(cfg["emb_dim"], eps=1e-6)

    def forward(self, x, mask, cos, sin):
        x = self.att(self.norm1(x), mask, cos, sin) + x
        x = self.ff(self.norm2(x)) + x
        return x

class Qwen3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        head_dim = cfg["head_dim"] if cfg["head_dim"] is not None else (cfg["emb_dim"] // cfg["n_heads"])
        cos, sin = compute_rope_params(head_dim=head_dim, theta_base=cfg["rope_base"],
                                       context_length=cfg["context_length"], dtype=torch.float32)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cfg = cfg

    def forward(self, in_idx):
        x = self.tok_emb(in_idx)  # [B,T,D]
        B,T,_ = x.shape
        mask = torch.triu(torch.ones(T,T, device=x.device, dtype=torch.bool), diagonal=1).view(1,1,T,T)
        for blk in self.blocks:
            x = blk(x, mask, self.cos, self.sin)
        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg["dtype"]))
        return logits

# =========================
# 2) Loading weights
# =========================

def load_weights_into_qwen(model, cfg, params, qkn_mode):
    def assign(left, right, name):
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch: {name}: {left.shape} vs {right.shape}")
        return torch.nn.Parameter(right.clone().detach())

    model.tok_emb.weight = assign(model.tok_emb.weight, params["model.embed_tokens.weight"], "tok_emb")

    for l in range(cfg["n_layers"]):
        blk = model.blocks[l]
        att = blk.att

        att.W_query.weight = assign(att.W_query.weight, params[f"model.layers.{l}.self_attn.q_proj.weight"], f"q.{l}")
        att.W_key.weight   = assign(att.W_key.weight,   params[f"model.layers.{l}.self_attn.k_proj.weight"], f"k.{l}")
        att.W_value.weight = assign(att.W_value.weight, params[f"model.layers.{l}.self_attn.v_proj.weight"], f"v.{l}")
        att.out_proj.weight= assign(att.out_proj.weight,params[f"model.layers.{l}.self_attn.o_proj.weight"], f"o.{l}")

        qn = params.get(f"model.layers.{l}.self_attn.q_norm.weight", None)
        kn = params.get(f"model.layers.{l}.self_attn.k_norm.weight", None)
        if qkn_mode == "off":
            att.q_norm = None
            att.k_norm = None
        elif qkn_mode == "on":
            if qn is None or kn is None:
                raise RuntimeError("qk-norm='on' but weights missing")
            if att.q_norm is None: att.q_norm = RMSNorm(att.head_dim, eps=1e-6)
            if att.k_norm is None: att.k_norm = RMSNorm(att.head_dim, eps=1e-6)
            att.q_norm.scale = assign(att.q_norm.scale, qn, f"q_norm.{l}")
            att.k_norm.scale = assign(att.k_norm.scale, kn, f"k_norm.{l}")
        else:  # auto
            if qn is not None and kn is not None:
                if att.q_norm is None: att.q_norm = RMSNorm(att.head_dim, eps=1e-6)
                if att.k_norm is None: att.k_norm = RMSNorm(att.head_dim, eps=1e-6)
                att.q_norm.scale = assign(att.q_norm.scale, qn, f"q_norm.{l}")
                att.k_norm.scale = assign(att.k_norm.scale, kn, f"k_norm.{l}")
            else:
                att.q_norm = None
                att.k_norm = None

        n1 = params.get(f"model.layers.{l}.input_layernorm.weight", None)
        if n1 is None:
            n1 = params[f"model.layers.{l}.rms_1.weight"]
        blk.norm1.scale = assign(blk.norm1.scale, n1, f"norm1.{l}")

        gate_w = params.get(f"model.layers.{l}.mlp.gate.weight", None)
        if gate_w is None:
            gate_w = params[f"model.layers.{l}.mlp.router.gate.weight"]
        blk.ff.gate.weight = assign(blk.ff.gate.weight, gate_w, f"router.w.{l}")
        gate_b = params.get(f"model.layers.{l}.mlp.gate.bias", None)
        if gate_b is None:
            gate_b = params.get(f"model.layers.{l}.mlp.router.gate.bias", None)
        if gate_b is not None:
            blk.ff.gate_bias = gate_b.clone().detach().view(1,1,-1)

        for e in range(cfg["num_experts"]):
            base = f"model.layers.{l}.mlp.experts.{e}"
            blk.ff.fc1[e].weight = assign(blk.ff.fc1[e].weight, params[f"{base}.gate_proj.weight"], f"{base}.gate")
            blk.ff.fc2[e].weight = assign(blk.ff.fc2[e].weight, params[f"{base}.up_proj.weight"],   f"{base}.up")
            blk.ff.fc3[e].weight = assign(blk.ff.fc3[e].weight, params[f"{base}.down_proj.weight"], f"{base}.down")
            blk.ff.fc1[e] = blk.ff.fc1[e].to("cpu")
            blk.ff.fc2[e] = blk.ff.fc2[e].to("cpu")
            blk.ff.fc3[e] = blk.ff.fc3[e].to("cpu")

        n2 = params.get(f"model.layers.{l}.post_attention_layernorm.weight", None)
        if n2 is None:
            n2 = params[f"model.layers.{l}.rms_2.weight"]
        blk.norm2.scale = assign(blk.norm2.scale, n2, f"norm2.{l}")

    fn = params.get("model.norm.weight", None)
    if fn is None:
        fn = params["model.final_layernorm.weight"]
    model.final_norm.scale = assign(model.final_norm.scale, fn, "final_norm")

    if "lm_head.weight" in params:
        model.out_head.weight = assign(model.out_head.weight, params["lm_head.weight"], "lm_head")
    else:
        model.out_head.weight = assign(model.out_head.weight, params["model.embed_tokens.weight"], "lm_head<-emb")

# =========================
# 3) Tokenizer + generator
# =========================

import re
from tokenizers import Tokenizer

class Qwen3Tokenizer:
    _SPECIALS = [
        "<|endoftext|>",
        "<|im_start|>", "<|im_end|>",
        "<|object_ref_start|>", "<|object_ref_end|>",
        "<|box_start|>", "<|box_end|>",
        "<|quad_start|>", "<|quad_end|>",
        "<|vision_start|>", "<|vision_end|>",
        "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>",
    ]
    _SPLIT_RE = re.compile(r"(<\|[^>]+?\|>)")

    def __init__(self, tokenizer_file_path, repo_id=None,
                 apply_chat_template=True, add_generation_prompt=True, add_thinking=False):
        self.apply_chat_template = apply_chat_template
        self.add_generation_prompt = add_generation_prompt
        self.add_thinking = add_thinking

        self._tok = Tokenizer.from_file(str(tokenizer_file_path))
        self._special_to_id = {t: self._tok.token_to_id(t) for t in self._SPECIALS}

        self.pad_token_id = self._special_to_id.get("<|endoftext|>")
        self.eos_token_id = self.pad_token_id
        if repo_id and "Base" not in repo_id:
            eos_token = "<|im_end|>"
        else:
            eos_token = "<|endoftext|>"
        if eos_token in self._special_to_id:
            self.eos_token_id = self._special_to_id[eos_token]

    def _wrap_chat(self, user_msg):
        s = f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        if self.add_generation_prompt:
            s += "<|im_start|>assistant"
            if self.add_thinking:
                s += "\n"
            else:
                s += "\n<think>\n\n</think>\n\n"
        return s

    def encode(self, text, chat_wrapped=None):
        if chat_wrapped is None:
            chat_wrapped = self.apply_chat_template
        stripped = text.strip()
        if stripped in self._special_to_id and "\n" not in stripped:
            return [self._special_to_id[stripped]]
        if chat_wrapped:
            text = self._wrap_chat(text)

        ids = []
        for part in filter(None, self._SPLIT_RE.split(text)):
            if part in self._special_to_id:
                ids.append(self._special_to_id[part])
            else:
                ids.extend(self._tok.encode(part).ids)
        return ids

    def decode(self, ids):
        return self._tok.decode(ids, skip_special_tokens=False)

@torch.no_grad()
def generate_text_basic_stream(model, token_ids, max_new_tokens, eos_token_id=None):
    model.eval()
    for _ in range(max_new_tokens):
        out = model(token_ids)[:, -1]
        next_token = torch.argmax(out, dim=-1, keepdim=True)
        if eos_token_id is not None and torch.all(next_token == eos_token_id):
            break
        yield next_token
        token_ids = torch.cat([token_ids, next_token], dim=1)

# =========================
# 4) Dump utilities (C parity)
# =========================

def build_mask(T, device):
    return torch.triu(torch.ones(T,T, device=device, dtype=torch.bool), diagonal=1).view(1,1,T,T)

@torch.no_grad()
def _trace_one_layer(model, cfg, h, L, mask, outbase, step):
    blk = model.blocks[L]
    att = blk.att
    T = h.shape[1]
    D = h.shape[2]

    # 1) norm1 (float32 for parity)
    x_norm1 = blk.norm1(h.to(torch.float32))
    np.save(f"{outbase}.step{step}.L{L}.x_norm1.npy", x_norm1.squeeze(0).cpu().numpy().astype(np.float32))

    # 2) attention (explicit, mirrors kernels and Raschka)
    d = att.head_dim
    Hq = att.num_heads
    Hkv = att.num_kv_groups
    group = att.group_size

    Q = torch.nn.functional.linear(x_norm1, att.W_query.weight).view(1,T,Hq,d).transpose(1,2)
    K = torch.nn.functional.linear(x_norm1, att.W_key.weight  ).view(1,T,Hkv,d).transpose(1,2)
    V = torch.nn.functional.linear(x_norm1, att.W_value.weight).view(1,T,Hkv,d).transpose(1,2)

    np.save(f"{outbase}.step{step}.L{L}.Q_proj_flat.npy", Q.squeeze(0).reshape(T, Hq*d).cpu().numpy().astype(np.float32))
    np.save(f"{outbase}.step{step}.L{L}.K_proj_flat.npy", K.squeeze(0).reshape(T, Hkv*d).cpu().numpy().astype(np.float32))
    np.save(f"{outbase}.step{step}.L{L}.V_proj_flat.npy", V.squeeze(0).reshape(T, Hkv*d).cpu().numpy().astype(np.float32))

    if att.q_norm is not None:
        Q = att.q_norm(Q)
        K = att.k_norm(K)

    np.save(f"{outbase}.step{step}.L{L}.Q_qknorm_flat.npy", Q.squeeze(0).reshape(T, Hq*d).cpu().numpy().astype(np.float32))
    np.save(f"{outbase}.step{step}.L{L}.K_qknorm_flat.npy", K.squeeze(0).reshape(T, Hkv*d).cpu().numpy().astype(np.float32))

    # RoPE using the SAME cos/sin buffers (we will save them once from model)
    def _apply_rope(x):
        d2 = d // 2
        cos = model.cos[:T,:].unsqueeze(0).unsqueeze(0)
        sin = model.sin[:T,:].unsqueeze(0).unsqueeze(0)
        x1, x2 = x[..., :d2], x[..., d2:]
        rot = torch.cat([-x2, x1], dim=-1)
        return (x * cos + rot * sin).to(x.dtype)

    Q = _apply_rope(Q)
    K = _apply_rope(K)

    np.save(f"{outbase}.step{step}.L{L}.Q_rope_flat.npy", Q.squeeze(0).reshape(T, Hq*d).cpu().numpy().astype(np.float32))
    np.save(f"{outbase}.step{step}.L{L}.K_rope_flat.npy", K.squeeze(0).reshape(T, Hkv*d).cpu().numpy().astype(np.float32))

    K = K.repeat_interleave(group, dim=1)
    V = V.repeat_interleave(group, dim=1)

    S = (Q @ K.transpose(2,3)) / (d ** 0.5)
    S = S.masked_fill(mask, float('-inf'))
    P = torch.softmax(S, dim=-1)
    np.save(f"{outbase}.step{step}.L{L}.S.npy", S.squeeze(0).cpu().numpy().astype(np.float32))
    np.save(f"{outbase}.step{step}.L{L}.P.npy", P.squeeze(0).cpu().numpy().astype(np.float32))

    ctx = (P @ V).transpose(1,2).reshape(1,T,Hq*d)
    attn_out = torch.nn.functional.linear(ctx, att.out_proj.weight)
    np.save(f"{outbase}.step{step}.L{L}.attn_out.npy", attn_out.squeeze(0).cpu().numpy().astype(np.float32))

    x_after_attn = (h.to(torch.float32) + attn_out.to(torch.float32))
    np.save(f"{outbase}.step{step}.L{L}.x_after_attn.npy", x_after_attn.squeeze(0).cpu().numpy().astype(np.float32))

    x_norm2 = blk.norm2(x_after_attn.to(torch.float32))
    np.save(f"{outbase}.step{step}.L{L}.x_norm2.npy", x_norm2.squeeze(0).cpu().numpy().astype(np.float32))

    ff = blk.ff
    logits = torch.nn.functional.linear(x_norm2, ff.gate.weight)
    if ff.gate_bias is not None:
        logits = logits + ff.gate_bias
    np.save(f"{outbase}.step{step}.L{L}.router_logits.npy", logits.squeeze(0).cpu().numpy().astype(np.float32))

    k = ff.num_experts_per_tok
    topk_scores, topk_idx = torch.topk(logits, k, dim=-1)
    topk_p = torch.softmax(topk_scores, dim=-1)
    np.save(f"{outbase}.step{step}.L{L}.router_topk_idx.npy", topk_idx.squeeze(0).cpu().numpy().astype(np.int32))
    np.save(f"{outbase}.step{step}.L{L}.router_topk_p.npy", topk_p.squeeze(0).cpu().numpy().astype(np.float32))

    E = ff.num_experts
    outs = []
    for e in range(E):
        g = torch.nn.functional.linear(x_norm2, ff.fc1[e].weight)
        u = torch.nn.functional.linear(x_norm2, ff.fc2[e].weight)
        g = torch.nn.functional.silu(g)
        h_ = g * u
        y_e = torch.nn.functional.linear(h_, ff.fc3[e].weight)
        outs.append(y_e)
    expert_out = torch.stack(outs, dim=2)
    gating = torch.zeros_like(logits)
    gating.scatter_(dim=-1, index=topk_idx, src=topk_p)
    moe_out = (gating.unsqueeze(-1) * expert_out).sum(dim=2)
    np.save(f"{outbase}.step{step}.L{L}.moe_out.npy", moe_out.squeeze(0).cpu().numpy().astype(np.float32))

    y = x_after_attn + moe_out
    np.save(f"{outbase}.step{step}.L{L}.y.npy", y.squeeze(0).cpu().numpy().astype(np.float32))
    return y

@torch.no_grad()
def run_and_dump(model, cfg, steps, ids, outbase):
    device = ids.device
    V = cfg["vocab_size"]
    D = cfg["emb_dim"]

    # --- NEW: dump cos/sin once so C can reuse the exact buffers
    np.save(f"{outbase}.cos.npy", model.cos.cpu().numpy().astype(np.float32))
    np.save(f"{outbase}.sin.npy", model.sin.cpu().numpy().astype(np.float32))

    if steps == 0:
        np.save(f"{outbase}.ids.npy", ids.squeeze(0).cpu().numpy().astype(np.int32))
        np.save(f"{outbase}.logits.npy", np.zeros((0, V), dtype=np.float32))
        np.save(f"{outbase}.probs.npy",  np.zeros((0, V), dtype=np.float32))
        print(f"[dump] steps=0: wrote only ids to {outbase}.ids.npy (logits/probs empty)")
        return

    per_step_logits, per_step_probs = [], []
    for s in range(steps):
        T = ids.shape[-1]
        x = model.tok_emb(ids).to(torch.float32)
        np.save(f"{outbase}.step{s}.x.npy", x.squeeze(0).cpu().numpy().astype(np.float32))

        mask = torch.triu(torch.ones(T,T, device=device, dtype=torch.bool), diagonal=1).view(1,1,T,T)
        h = x
        for L in range(len(model.blocks)):
            h = _trace_one_layer(model, cfg, h, L, mask, outbase, s)

        y_norm = model.final_norm(h.to(torch.float32))
        logits = torch.nn.functional.linear(y_norm.to(model.cfg["dtype"]), model.out_head.weight)
        last = logits[:, -1, :].to(torch.float32).contiguous()
        probs = torch.softmax(last, dim=-1)
        per_step_logits.append(last.cpu())
        per_step_probs.append(probs.cpu())

        nxt = int(torch.argmax(probs[0]).item())
        ids = torch.cat([ids, torch.tensor([[nxt]], device=device, dtype=torch.long)], dim=1)

    np.save(f"{outbase}.ids.npy",    ids.squeeze(0).cpu().numpy().astype(np.int32))
    np.save(f"{outbase}.logits.npy", torch.cat(per_step_logits, dim=0).numpy().astype(np.float32))
    np.save(f"{outbase}.probs.npy",  torch.cat(per_step_probs,  dim=0).numpy().astype(np.float32))
    print(f"[dump] saved ids/logits/probs and full per-layer traces for {steps} step(s).")

# =========================
# 5) Main
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", type=int, default=48)
    ap.add_argument("--seqlen", type=int, default=1)
    ap.add_argument("--steps",  type=int, default=2)
    ap.add_argument("--seed",   type=int, default=123)
    ap.add_argument("--rope-theta", type=float, default=10_000_000.0)
    ap.add_argument("--qk-norm", choices=["auto","on","off"], default="auto")
    ap.add_argument("--outbase", required=True)
    ap.add_argument("--first-id", type=int, default=None)
    ap.add_argument("--force-f32", action="store_true")
    ap.add_argument("--trace-layer", type=int, default=None)
    ap.add_argument("--prompt", type=str, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--no-chat-template", action="store_true")
    ap.add_argument("--add-thinking", action="store_true")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    QCFG = {
        "vocab_size": 151_936,
        "context_length": 262_144,
        "emb_dim": 2048,
        "n_heads": 32,
        "n_layers": args.layers,
        "head_dim": 128,
        "qk_norm": (args.qk_norm == "on"),
        "n_kv_groups": 4,
        "rope_base": float(args.rope_theta),
        "dtype": torch.float32,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 768,
        "hidden_dim": 768,
    }

    model = Qwen3Model(QCFG).eval()
    model.TRACE_LAYER = args.trace_layer

    repo_id = args.model
    local_dir = Path(repo_id).parts[-1]
    repo_dir = snapshot_download(repo_id=repo_id, local_dir=local_dir)
    index_path = os.path.join(repo_dir, "model.safetensors.index.json")
    with open(index_path, "r") as f:
        index = json.load(f)

    weights = {}
    for filename in sorted(set(index["weight_map"].values())):
        shard_path = os.path.join(repo_dir, filename)
        sd = load_file(shard_path)
        weights.update(sd)

    load_weights_into_qwen(model, QCFG, weights, qkn_mode=args.qk_norm)
    maybe_force_f32(model, enable=args.force_f32)
    model.eval()

    torch.manual_seed(args.seed)
    V = QCFG["vocab_size"]
    ids = torch.randint(low=0, high=V, size=(1, args.seqlen), dtype=torch.long)
    if args.first_id is not None and args.seqlen > 0:
        ids[0, 0] = int(args.first_id)

    if args.steps >= 0:
        run_and_dump(model, QCFG, args.steps, ids, args.outbase)

    if args.prompt is not None:
        tok_path = os.path.join(Path(args.model).parts[-1], "tokenizer.json")
        tokenizer = Qwen3Tokenizer(
            tokenizer_file_path=tok_path,
            repo_id=args.model,
            apply_chat_template=(not args.no_chat_template),
            add_generation_prompt=True,
            add_thinking=args.add_thinking,
        )
        in_ids = tokenizer.encode(args.prompt)
        token_ids = torch.tensor(in_ids, dtype=torch.long).unsqueeze(0)
        print("\n--- generation (greedy) ---")
        print("user"); print(args.prompt)
        print("assistant")
        for t in generate_text_basic_stream(model, token_ids, args.max_new_tokens,
                                            eos_token_id=tokenizer.eos_token_id):
            piece = tokenizer.decode(t.squeeze(0).tolist())
            print(piece, end="", flush=True)
        print("\n---------------------------")

if __name__ == "__main__":
    main()
