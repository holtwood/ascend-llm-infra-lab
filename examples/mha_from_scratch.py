#!/usr/bin/env python3
"""Multi-head attention with KV cache, from scratch (no nn.MultiheadAttention).

Implements:
  1. naive full-attention forward over the whole sequence every step
  2. incremental decode with an explicit KV cache

Then verifies both produce identical logits and identical greedy outputs,
and prints the per-step complexity/memory difference for interview use.

Run: python3 examples/mha_from_scratch.py
"""

import math
import time

import torch

torch.manual_seed(0)


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        # fused QKV projection — one matmul, split afterwards
        self.w_qkv = torch.nn.Linear(d_model, 3 * d_model, bias=False)
        self.w_o = torch.nn.Linear(d_model, d_model, bias=False)

    def _split(self, x, B, T):
        # (B, T, d_model) -> (B, n_heads, T, head_dim)
        return x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

    def _attend(self, q, k, v):
        # q: (B,H,Tq,d)  k/v: (B,H,Tk,d)  — causal mask over the full Tk axis
        B, H, Tq, d = q.shape
        Tk = k.shape[2]
        scores = q @ k.transpose(-1, -2) / math.sqrt(d)          # (B,H,Tq,Tk)
        # query position i (absolute = Tk-Tq+i) may attend keys 0..Tk-Tq+i
        i = torch.arange(Tq).unsqueeze(1)                        # (Tq,1)
        j = torch.arange(Tk).unsqueeze(0)                        # (1,Tk)
        mask = j <= (Tk - Tq) + i                                # (Tq,Tk) lower-tri
        scores = scores.masked_fill(~mask, float("-inf"))
        return torch.softmax(scores, dim=-1) @ v                 # (B,H,Tq,d)

    def forward(self, x):
        """Naive: recompute attention over the entire sequence."""
        B, T, _ = x.shape
        q, k, v = self.w_qkv(x).chunk(3, dim=-1)
        q, k, v = (self._split(t, B, T) for t in (q, k, v))
        out = self._attend(q, k, v)
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.w_o(out)

    def forward_cached(self, x, cache):
        """Incremental: x is only the NEW tokens; cache holds past K/V.

        cache: dict with 'k','v' of shape (B,H,Tpast,d) or None.
        Returns (out, new_cache). One token at a time in real decoding.
        """
        B, T, _ = x.shape
        q, k, v = self.w_qkv(x).chunk(3, dim=-1)
        q, k, v = (self._split(t, B, T) for t in (q, k, v))
        if cache is not None:
            k = torch.cat([cache["k"], k], dim=2)                # append to cache
            v = torch.cat([cache["v"], v], dim=2)
        out = self._attend(q, k, v)
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.w_o(out), {"k": k, "v": v}


class TinyGPT(torch.nn.Module):
    """Embedding + N pre-norm attention blocks + LM head. Just enough to
    exercise the cache; FFN omitted — attention behavior is what matters."""

    def __init__(self, vocab=512, d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, d_model)
        self.blocks = torch.nn.ModuleList(
            MultiHeadAttention(d_model, n_heads) for _ in range(n_layers))
        self.norm = torch.nn.LayerNorm(d_model)
        self.head = torch.nn.Linear(d_model, vocab, bias=False)

    def forward(self, ids):
        h = self.embed(ids)
        for blk in self.blocks:
            h = h + blk(h)
        return self.head(self.norm(h))

    def forward_cached(self, ids, caches):
        h = self.embed(ids)
        new_caches = []
        for blk, c in zip(self.blocks, caches):
            a, nc = blk.forward_cached(h, c)
            h = h + a
            new_caches.append(nc)
        return self.head(self.norm(h)), new_caches


def naive_generate(model, ids, n_new):
    """Recompute the whole sequence every step: O(T^2) per token."""
    out = ids
    for _ in range(n_new):
        logits = model(out)[:, -1, :]
        out = torch.cat([out, logits.argmax(-1, keepdim=True)], dim=1)
    return out


def cached_generate(model, ids, n_new):
    """KV-cached: O(T) per token; prefill once, then 1-token steps."""
    caches = [None] * len(model.blocks)
    logits, caches = model.forward_cached(ids, caches)           # prefill
    nxt = logits[:, -1, :].argmax(-1, keepdim=True)
    out = torch.cat([ids, nxt], dim=1)
    for _ in range(n_new - 1):
        logits, caches = model.forward_cached(nxt, caches)       # decode step
        nxt = logits[:, -1, :].argmax(-1, keepdim=True)
        out = torch.cat([out, nxt], dim=1)
    return out


def main():
    B, T_prompt, T_new = 2, 32, 16
    model = TinyGPT().eval()
    ids = torch.randint(0, 512, (B, T_prompt))

    with torch.no_grad():
        # ---- correctness: identical logits on the prompt ----
        full = model(ids)
        caches = [None] * len(model.blocks)
        inc, caches = model.forward_cached(ids, caches)
        err = (full - inc).abs().max().item()
        assert err < 1e-5, f"prompt logits mismatch: {err}"
        print(f"[check] prefill logits match naive forward, max|diff|={err:.2e}")

        # ---- correctness: identical greedy continuations ----
        a = naive_generate(model, ids, T_new)
        b = cached_generate(model, ids, T_new)
        assert torch.equal(a, b), "cached decode diverged from naive"
        print(f"[check] greedy decode: naive and KV-cached produce identical "
              f"{T_new} tokens (B={B})")

        # ---- edge case: single-token prompt ----
        ids1 = ids[:, :1]
        assert torch.equal(naive_generate(model, ids1, 4),
                           cached_generate(model, ids1, 4))
        print("[check] T=1 prompt edge case OK")

        # ---- cost demonstration: wall time scaling ----
        for T in (64, 128, 256):
            ids_t = torch.randint(0, 512, (1, T))
            t0 = time.time(); naive_generate(model, ids_t, 8); t_naive = time.time() - t0
            t0 = time.time(); cached_generate(model, ids_t, 8); t_cached = time.time() - t0
            print(f"[cost] prompt={T:4d} +8 tok: naive {t_naive*1e3:7.1f} ms  "
                  f"cached {t_cached*1e3:7.1f} ms  ({t_naive/t_cached:.1f}x)")

        # ---- memory: KV cache bytes for a realistic shape ----
        kv_bytes_per_tok = 2 * 32 * 4 * 256 * 2   # Qwen3.5-4B: 8 full-attn layers
        # (see results/e8 — only full_attention layers hold KV on hybrid models)
        print(f"[note] KV/token for 8 GQA layers (4 heads x 256d, bf16) "
              f"= {kv_bytes_per_tok} B = 32 KiB")


if __name__ == "__main__":
    main()
