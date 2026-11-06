#!/usr/bin/env python3
"""Gated DeltaNet core kernels, pure torch (no fla / triton / flash-attn).

Two mathematically equivalent evaluations of the gated delta rule:

  recurrent_gated_delta_rule  — per-token loop; used for single-token decode
                                and as the numerical reference.
  chunked_gated_delta_rule    — blockwise (chunk_size=64) blocked form; used
                                for prefill. Matches the FLA chunk kernel that
                                transformers falls back to on CPU.

Update rule for one token (S: (d_k, d_v), all fp32):
    S_t    = S_{t-1} * exp(g_t)                       # input-dependent decay
    delta  = (v_t - S_t^T k_t) * beta_t               # NOTE: uses the DECAYED S_t
    S_t   += k_t (x) delta                            # rank-1 correction
    o_t    = S_t^T q_t                                # read AFTER the update
with q,k l2-normalised per head beforehand and q pre-scaled by 1/sqrt(d_k).

Shapes in/out (B, T, H, D) for q/k/v, (B, T, H) for g/beta, exactly the layout
transformers passes to its kernel. Returned output is (B, T, H, Dv); the
final state is (B, H, d_k, d_v) float32.

Run: python3 examples/engine_np/gdn_kernels.py
"""

import torch
import torch.nn.functional as F

CHUNK_SIZE = 64    # FLA default; chunked prefill is built around it


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-head L2 normalisation, aligned with the FLA l2norm (eps inside sqrt)."""
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def recurrent_gated_delta_rule(query, key, value, g, beta,
                               initial_state=None, output_final_state=True,
                               use_qk_l2norm_in_kernel=True):
    """Per-token reference implementation (decode path).

    query/key/value: (B, T, H, D);  g/beta: (B, T, H)
    Returns (out (B, T, H, Dv), final_state (B, H, d_k, d_v) fp32).
    """
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query, key = l2norm(query), l2norm(key)
    # everything moves to fp32: A_log.exp() and exp(g) overflow in fp16/bf16
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]
    batch, heads, seq_len, d_k = key.shape
    d_v = value.shape[-1]
    scale = d_k ** -0.5
    query = query * scale

    state = (torch.zeros(batch, heads, d_k, d_v, dtype=torch.float32,
                         device=query.device)
             if initial_state is None else initial_state.to(torch.float32))
    out = torch.empty_like(value)
    for t in range(seq_len):
        q_t, k_t, v_t = query[:, :, t], key[:, :, t], value[:, :, t]
        decay = g[:, :, t].exp().unsqueeze(-1).unsqueeze(-1)   # (B,H,1,1)
        # 1) decay the old state
        state = state * decay
        # 2) delta rule against the DECAYED state
        kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)       # S_t^T k_t
        delta = (v_t - kv_mem) * beta[:, :, t].unsqueeze(-1)
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)  # + k (x) delta
        # 3) read from the updated state
        out[:, :, t] = (state * q_t.unsqueeze(-1)).sum(dim=-2)
    out = out.transpose(1, 2).contiguous().to(initial_dtype)
    return out, state if output_final_state else None


def chunked_gated_delta_rule(query, key, value, g, beta,
                             initial_state=None, output_final_state=True,
                             use_qk_l2norm_in_kernel=True, chunk_size=CHUNK_SIZE):
    """Blocked prefill implementation, ported 1:1 from the transformers
    torch_chunk_gated_delta_rule (their CPU fallback for the FLA chunk kernel).

    Numerically equivalent to the recurrent form; the blocked algebra folds the
    per-token decay/updates into intra- and inter-chunk matrix products.
    """
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query, key = l2norm(query), l2norm(key)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]
    batch, heads, seq_len, d_k = key.shape
    pad = (chunk_size - seq_len % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad))
    key = F.pad(key, (0, 0, 0, pad))
    value = F.pad(value, (0, 0, 0, pad))
    beta = F.pad(beta, (0, pad))
    g = F.pad(g, (0, pad))
    total_len = seq_len + pad
    scale = d_k ** -0.5
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    diag_mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool,
                                      device=query.device), diagonal=0)

    # chunk-internal decay masks: exp(g_i - g_j) for i >= j within the chunk
    g = g.cumsum(dim=-1)
    decay_mask = (g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float().tril()
    # invert (I - tril(k_beta @ k^T * decay)) by forward substitution
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(diag_mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

    state = (torch.zeros(batch, heads, d_k, value.shape[-1], dtype=torch.float32,
                         device=query.device)
             if initial_state is None else initial_state.to(torch.float32))
    out = torch.zeros_like(value)
    diag_mask_strict = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool,
                                             device=query.device), diagonal=1)
    for i in range(total_len // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        # intra-chunk attention scaled by pairwise decay
        attn_i = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        # what the current chunk would remember of the inter-chunk state
        v_prime = k_cumdecay[:, :, i] @ state
        v_new = v_i - v_prime
        # inter-chunk contribution, decayed to the chunk boundary
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ state
        out[:, :, i] = attn_inter + attn_i @ v_new
        # roll the state forward across the whole chunk
        state = (state * g[:, :, i, -1, None, None].exp()
                 + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None])
                 .transpose(-1, -2) @ v_new)

    out = out.reshape(batch, heads, -1, out.shape[-1])
    out = out[:, :, :seq_len].transpose(1, 2).contiguous().to(initial_dtype)
    return out, state if output_final_state else None


def _self_check() -> None:
    """The two kernels must agree bit-tight on fp32."""
    torch.manual_seed(0)
    b, t, h, dk, dv = 2, 100, 4, 16, 16
    q = torch.randn(b, t, h, dk)
    k = torch.randn(b, t, h, dk)
    v = torch.randn(b, t, h, dv)
    g = -torch.rand(b, t, h) * 0.1           # negative log-decay rates
    beta = torch.sigmoid(torch.randn(b, t, h))

    o_r, s_r = recurrent_gated_delta_rule(q, k, v, g, beta)
    o_c, s_c = chunked_gated_delta_rule(q, k, v, g, beta)
    err_o = (o_r - o_c).abs().max().item()
    err_s = (s_r - s_c).abs().max().item()
    assert err_o < 1e-4, f"outputs disagree: {err_o}"
    assert err_s < 1e-4, f"states disagree: {err_s}"
    print(f"[check] chunked vs recurrent delta rule (T={t}): "
          f"max|out diff|={err_o:.2e}  max|state diff|={err_s:.2e}")

    # sequence length 1 == decode path must equal one recurrent step
    o_1, s_1 = chunked_gated_delta_rule(q[:, :1], k[:, :1], v[:, :1],
                                        g[:, :1], beta[:, :1])
    o_1r, s_1r = recurrent_gated_delta_rule(q[:, :1], k[:, :1], v[:, :1],
                                            g[:, :1], beta[:, :1])
    assert (o_1 - o_1r).abs().max().item() < 1e-5
    assert (s_1 - s_1r).abs().max().item() < 1e-5
    print("[check] T=1 chunked == recurrent (decode path)")

    # chunk boundary: non-multiple-of-64 lengths exercise the padding branch
    o_p, s_p = chunked_gated_delta_rule(q, k, v, g, beta)
    o_pr, s_pr = recurrent_gated_delta_rule(q, k, v, g, beta)
    assert (o_p - o_pr).abs().max().item() < 1e-4
    print("[check] T=100 (padded to 128) chunked == recurrent")


if __name__ == "__main__":
    _self_check()
