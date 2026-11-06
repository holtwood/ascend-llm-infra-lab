#!/usr/bin/env python3
"""Qwen3.5 (text tower) re-implemented from scratch in pure torch.nn.

Hybrid architecture, 3:1 — 24 GatedDeltaNet (linear attention) layers and 8
GQA gated-attention layers, per config.json:

  hidden 2560, 32 layers, layer_types[i] in {"linear_attention", "full_attention"}
  GDN  layer: 32 v-heads / 16 k-heads, head_dim 128; depthwise causal conv(k=4)
              + SiLU -> q/k l2norm -> gated delta rule -> SiLU-gated RMSNorm
  attn layer: 16 q-heads / 4 kv-heads (GQA), head_dim 256; q/k RMSNorm on the
              head dim, partial RoPE (64 of 256 dims, theta=1e7), sigmoid output
              gate fused into q_proj
  MLP: SwiGLU, intermediate 9216; RMSNorm zero-init weight, applied as (1+w)

Everything is math-faithful to transformers 5.x modeling_qwen3_5.py; every
forward carries the formulas it implements. Weights load straight from the
released safetensors (see load_weights for the checkpoint quirks).

Run: python3 examples/engine_np/model_qwen35.py
"""

import json
import math
import os
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from gdn_kernels import chunked_gated_delta_rule, recurrent_gated_delta_rule


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #

@dataclass
class Qwen35Config:
    """Only the fields the text engine needs; names match Qwen3_5TextConfig."""
    vocab_size: int = 248320
    hidden_size: int = 2560
    intermediate_size: int = 9216
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 256
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    layer_types: list = field(default_factory=list)
    # GatedDeltaNet
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    # RoPE (text path; the released 3D mrope collapses to plain 1D rope for text)
    rope_theta: float = 10000000.0
    partial_rotary_factor: float = 0.25
    mrope_section: list = field(default_factory=lambda: [11, 11, 10])
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if not self.layer_types:
            self.layer_types = ["full_attention"] * self.num_hidden_layers
        self.rope_dim = int(self.head_dim * self.partial_rotary_factor)
        assert self.hidden_size % self.num_attention_heads == 0
        assert self.num_attention_heads % self.num_key_value_heads == 0
        assert self.linear_num_value_heads % self.linear_num_key_heads == 0

    @property
    def num_key_value_groups(self):
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def linear_key_dim(self):
        return self.linear_num_key_heads * self.linear_key_head_dim

    @property
    def linear_value_dim(self):
        return self.linear_num_value_heads * self.linear_value_head_dim

    @classmethod
    def from_json(cls, path: str) -> "Qwen35Config":
        """Read model config.json (VL wrapper) and keep the text_config."""
        with open(path) as f:
            raw = json.load(f)["text_config"]
        rope = raw.get("rope_parameters", {})
        return cls(
            vocab_size=raw["vocab_size"],
            hidden_size=raw["hidden_size"],
            intermediate_size=raw["intermediate_size"],
            num_hidden_layers=raw["num_hidden_layers"],
            num_attention_heads=raw["num_attention_heads"],
            num_key_value_heads=raw["num_key_value_heads"],
            head_dim=raw["head_dim"],
            rms_norm_eps=raw["rms_norm_eps"],
            attention_bias=raw.get("attention_bias", False),
            layer_types=list(raw["layer_types"]),
            linear_num_key_heads=raw["linear_num_key_heads"],
            linear_num_value_heads=raw["linear_num_value_heads"],
            linear_key_head_dim=raw["linear_key_head_dim"],
            linear_value_head_dim=raw["linear_value_head_dim"],
            linear_conv_kernel_dim=raw["linear_conv_kernel_dim"],
            rope_theta=rope.get("rope_theta", 1e7),
            partial_rotary_factor=rope.get("partial_rotary_factor", 1.0),
            mrope_section=list(rope.get("mrope_section", [11, 11, 10])),
            tie_word_embeddings=raw.get("tie_word_embeddings", True),
        )


# --------------------------------------------------------------------------- #
# norms
# --------------------------------------------------------------------------- #

class RMSNorm(nn.Module):
    """RMSNorm with a ZERO-INIT delta weight: out = rmsnorm(x) * (1 + w).

    Deliberately different from Llama/Qwen2 (one-init w, plain multiply) —
    this is how transformers 5.x defines Qwen3_5RMSNorm, and the released
    checkpoint stores w in delta form. Computed in fp32, cast back.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        out = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        out = out * (1.0 + self.weight.float())
        return out.type_as(x)


class RMSNormGated(nn.Module):
    """SiLU-gated RMSNorm on the GDN output: out = rmsnorm(o)*w * silu(z).

    Order matters (norm BEFORE gate, per modeling_qwen3_5.py):
        h  = rmsnorm_fp32(o)                 (fp32, eps inside rsqrt)
        h  = w * h.to(input_dtype)           (weight multiply in model dtype)
        out= h * silu(z_fp32)                (gate math in fp32, then cast)
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        h = hidden_states.to(torch.float32)
        variance = h.pow(2).mean(-1, keepdim=True)
        h = h * torch.rsqrt(variance + self.eps)
        h = self.weight * h.to(input_dtype)
        h = h * F.silu(gate.to(torch.float32))
        return h.to(input_dtype)


# --------------------------------------------------------------------------- #
# RoPE (text-only path)
# --------------------------------------------------------------------------- #

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """cat(-x2, x1) over the last dim — the rotate used inside apply_rotary."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Rotate the first rope_dim dims of q/k; the rest passes through.

    q/k: (B, H, T, D);  cos/sin: (B, T, rope_dim)  ->  broadcast over heads.
    """
    cos, sin = cos.to(q.dtype), sin.to(q.dtype)
    q_rot, q_pass = q[..., :cos.shape[-1]], q[..., cos.shape[-1]:]
    k_rot, k_pass = k[..., :cos.shape[-1]], k[..., cos.shape[-1]:]
    q_out = q_rot * cos.unsqueeze(1) + rotate_half(q_rot) * sin.unsqueeze(1)
    k_out = k_rot * cos.unsqueeze(1) + rotate_half(k_rot) * sin.unsqueeze(1)
    return torch.cat([q_out, q_pass], dim=-1), torch.cat([k_out, k_pass], dim=-1)


class RotaryEmbedding(nn.Module):
    """Partial RoPE, text path.

    transformers builds a 3-grid (T/H/W) mrope and interleaves the frequency
    slots (mrope_section). For pure text all three grids carry identical
    positions, so the interleaved layout collapses to plain rope over
    freqs_t — that is all we implement here.
    """

    def __init__(self, config: Qwen35Config):
        super().__init__()
        dim = config.rope_dim                    # 64 of 256 head dims
        inv_freq = 1.0 / (config.rope_theta ** (
            torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """position_ids: (B, T) -> (cos, sin) of shape (B, T, rope_dim)."""
        freqs = position_ids[:, :, None].float() * self.inv_freq  # (B,T,rope/2)
        emb = torch.cat((freqs, freqs), dim=-1)                   # (B,T,rope)
        return emb.cos(), emb.sin()


# --------------------------------------------------------------------------- #
# token mixers
# --------------------------------------------------------------------------- #

class GatedDeltaNet(nn.Module):
    """Linear-attention mixer. Submodule names mirror the checkpoint keys
    (model.layers.N.linear_attn.*).

    Per token t, with q/k l2-normalised per head, beta=sigmoid(b_t) and the
    input-dependent decay g_t = -exp(A_log) * softplus(a_t + dt_bias):

        mixed = SiLU(DWConv1d([q;k;v]))            depthwise, kernel 4, no bias
        S_t   = S_{t-1} * exp(g_t)                 (d_k x d_v state, fp32)
        delta = (v_t - S_t^T k_t) * beta_t         against the DECAYED state
        S_t  += k_t (x) delta
        o_t   = S_t^T q_t / sqrt(d_k)              (scale folded into q)
        out   = out_proj( RMSNormGated(o, z) )     z = in_proj_z(x) gate
    """

    def __init__(self, config: Qwen35Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = config.linear_key_dim
        self.value_dim = config.linear_value_dim
        self.conv_kernel = config.linear_conv_kernel_dim

        self.conv_dim = self.key_dim * 2 + self.value_dim   # 2*k + v fused qkv
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, bias=False,
                                kernel_size=self.conv_kernel,
                                groups=self.conv_dim)       # depthwise

        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        # A in [0.01, 16) keeps exp(A_log) away from 0 so g never saturates
        self.A_log = nn.Parameter(torch.log(torch.empty(
            self.num_v_heads).uniform_(0.01, 16)))

        self.norm = RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, config.hidden_size, bias=False)

        self.in_proj_qkv = nn.Linear(config.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(config.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(config.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(config.hidden_size, self.num_v_heads, bias=False)

    def forward(self, hidden_states: torch.Tensor, cache=None) -> torch.Tensor:
        """hidden_states: (B, T, hidden) pre-normed input; cache is a
        RecurrentState or None. Returns (B, T, hidden)."""
        batch, seq_len, _ = hidden_states.shape

        mixed = self.in_proj_qkv(hidden_states).transpose(1, 2)   # (B, conv_dim, T)
        z = self.in_proj_z(hidden_states).view(batch, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)                          # (B, T, n_v)
        a = self.in_proj_a(hidden_states)

        # ---- causal depthwise conv + SiLU -----------------------------------
        # cache.conv_input returns x itself on the first call (prefill, zero
        # left context) and cat(conv_state, x) afterwards (decode).
        has_history = cache is not None and cache.has_history
        conv_in = cache.conv_input(mixed) if cache is not None else mixed
        weight = self.conv1d.weight                                # (conv_dim,1,K)
        # conv_state is stored fp32 alongside the recurrent state; the conv
        # itself runs in the weight (hidden) dtype like the reference impl.
        conv_in = conv_in.to(weight.dtype)
        if has_history:
            conv_out = F.conv1d(conv_in, weight, padding=0,
                                groups=self.conv_dim)[..., -seq_len:]
        else:
            conv_out = F.conv1d(conv_in, weight, padding=self.conv_kernel - 1,
                                groups=self.conv_dim)[..., :seq_len]
        mixed = F.silu(conv_out).transpose(1, 2)                   # (B, T, conv_dim)

        q, k, v = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim],
                              dim=-1)
        q = q.view(batch, seq_len, self.num_k_heads, self.head_k_dim)
        k = k.view(batch, seq_len, self.num_k_heads, self.head_k_dim)
        v = v.view(batch, seq_len, self.num_v_heads, self.head_v_dim)

        # beta/gate/decay — computed on the v heads, in fp32 where it matters
        beta = b.sigmoid()
        # .float() before exp is load-bearing: A_log.exp() overflows in bf16
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())

        # 2 v-heads share each k-head (repeat_interleave, not GQA-strided)
        if self.num_v_heads // self.num_k_heads > 1:
            rep = self.num_v_heads // self.num_k_heads
            q = q.repeat_interleave(rep, dim=2)
            k = k.repeat_interleave(rep, dim=2)

        # ---- gated delta rule -----------------------------------------------
        initial_state = cache.read_state() if cache is not None else None
        if seq_len == 1:
            core_attn_out, final_state = recurrent_gated_delta_rule(
                q, k, v, g=g, beta=beta, initial_state=initial_state,
                output_final_state=cache is not None)
        else:
            core_attn_out, final_state = chunked_gated_delta_rule(
                q, k, v, g=g, beta=beta, initial_state=initial_state,
                output_final_state=cache is not None)
        if cache is not None:
            cache.write_state(final_state)

        # ---- output gate + projections --------------------------------------
        out = core_attn_out.reshape(-1, self.head_v_dim)
        out = self.norm(out, z.reshape(-1, self.head_v_dim))
        out = out.reshape(batch, seq_len, -1)
        return self.out_proj(out)


class Attention(nn.Module):
    """GQA attention with fused sigmoid output gate. Checkpoint keys live under
    model.layers.N.self_attn.*.

        q, gate = q_proj(x) split in half           (2*n_h*head_dim out)
        q = RMSNorm(q) per head; k = RMSNorm(k_proj(x)) per head   (before RoPE)
        q,k = partial-RoPE(q, k)                    64 of 256 dims, theta=1e7
        attn = softmax(q k^T / sqrt(head_dim)) v    causal, GQA-expanded
        out  = o_proj(attn * sigmoid(gate))
    """

    def __init__(self, config: Qwen35Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.scaling = config.head_dim ** -0.5

        # 2x on the output dim: q and its sigmoid gate ride one projection
        self.q_proj = nn.Linear(config.hidden_size,
                                self.num_heads * self.head_dim * 2,
                                bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size,
                                self.num_kv_heads * self.head_dim,
                                bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size,
                                self.num_kv_heads * self.head_dim,
                                bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim,
                                config.hidden_size, bias=config.attention_bias)
        self.q_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, cos, sin, cache=None) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape
        hidden_shape = (batch, seq_len, -1, self.head_dim)

        q, gate = torch.chunk(
            self.q_proj(hidden_states).view(batch, seq_len, self.num_heads,
                                            self.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(batch, seq_len, -1)          # (B, T, n_h*head_dim)

        q = self.q_norm(q).transpose(1, 2)               # (B, H, T, D)
        k = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if cache is not None:
            cache.append(k, v)
            res = cache.read()                            # full history incl. new
            k, v = res[0], res[1]
            valid = res[2] if len(res) > 2 else None      # L3 slot cache: (B,L) pad mask
        else:
            valid = None
        kv_len = k.shape[2]

        # causal mask aligned on ABSOLUTE positions so any T_q <= T_k works
        # (single-token decode and L5 chunked prefill are both covered)
        if seq_len == kv_len and (valid is None or bool(valid.all())):
            attn_mask, is_causal = None, True
        elif seq_len == 1 and valid is not None:
            # L3 heterogeneous decode: each slot attends its own history prefix;
            # q is each slot's newest position, so only pad columns need masking.
            attn_mask = valid[:, None, None, :]           # (B,1,1,L), True=attend
            is_causal = False
        else:
            q_pos = torch.arange(kv_len - seq_len, kv_len, device=q.device)
            k_pos = torch.arange(kv_len, device=q.device)
            attn_mask = (k_pos[None, :] <= q_pos[:, None])   # bool, True=attend
            is_causal = False

        attn = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=is_causal,
            scale=self.scaling, enable_gqa=True)

        attn = attn.transpose(1, 2).reshape(batch, seq_len, -1)
        attn = attn * torch.sigmoid(gate)                # output gate, no norm
        return self.o_proj(attn)


class MLP(nn.Module):
    """SwiGLU: down(silu(gate(x)) * up(x))."""

    def __init__(self, config: Qwen35Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size,
                                   bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size,
                                 bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size,
                                   bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# --------------------------------------------------------------------------- #
# blocks
# --------------------------------------------------------------------------- #

class DecoderLayer(nn.Module):
    """Pre-norm residual block. The mixer attribute is named linear_attn or
    self_attn per layer type, matching the checkpoint keys exactly."""

    def __init__(self, config: Qwen35Config, layer_idx: int):
        super().__init__()
        self.block_type = config.layer_types[layer_idx]
        if self.block_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(config, layer_idx)
        elif self.block_type == "full_attention":
            self.self_attn = Attention(config, layer_idx)
        else:
            raise ValueError(f"unknown layer type {self.block_type!r}")
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings, cache=None) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.block_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states, cache=cache)
        else:
            cos, sin = position_embeddings
            hidden_states = self.self_attn(hidden_states, cos, sin, cache=cache)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Qwen35TextModel(nn.Module):
    """Embedding + 32 decoder layers + final norm. Checkpoint prefix: model."""

    def __init__(self, config: Qwen35Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            DecoderLayer(config, i) for i in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config)

    def forward(self, input_ids, caches=None, position_ids=None,
                logits_to_keep: int = 0) -> torch.Tensor:
        """input_ids: (B, T). caches: HybridCaches or None (no-cache forward).
        Returns logits (B, T', vocab) where T' = logits_to_keep or T."""
        past_len = caches.seq_len if caches is not None else 0
        if position_ids is None:
            position_ids = torch.arange(past_len, past_len + input_ids.shape[1],
                                        device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(input_ids.shape[0], -1)

        hidden_states = self.embed_tokens(input_ids)
        position_embeddings = self.rotary_emb(position_ids)

        # attention layers need RoPE; GDN layers ignore it
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, position_embeddings,
                                  cache=caches.layers[i] if caches is not None else None)
        hidden_states = self.norm(hidden_states)

        if logits_to_keep > 0:
            hidden_states = hidden_states[:, -logits_to_keep:, :]
        return hidden_states


class Qwen35ForCausalLM(nn.Module):
    """Text model + tied LM head. Checkpoint prefix: model.* with NO lm_head
    entry (tied weights live only in embed_tokens)."""

    def __init__(self, config: Qwen35Config):
        super().__init__()
        self.config = config
        self.model = Qwen35TextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.tie_weights()

    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids, caches=None, position_ids=None,
                logits_to_keep: int = 0) -> torch.Tensor:
        hidden = self.model(input_ids, caches=caches, position_ids=position_ids,
                            logits_to_keep=logits_to_keep)
        return self.lm_head(hidden)


# --------------------------------------------------------------------------- #
# checkpoint loading
# --------------------------------------------------------------------------- #

# released files: model.safetensors-00001-of-00002.safetensors (hyphenated, NOT
# the HF-standard model-00001-of-...). Always resolve shards via the index.
INDEX_NAME = "model.safetensors.index.json"
KEY_PREFIX = "model.language_model."      # VL wrapper around the text tower
IGNORED_PREFIXES = ("model.visual.", "mtp.")   # vision tower + MTP head


def _remap_key(key: str) -> str | None:
    """Checkpoint key -> engine state_dict key, or None to skip."""
    if key.startswith(IGNORED_PREFIXES):
        return None
    if key.startswith(KEY_PREFIX):
        key = "model." + key[len(KEY_PREFIX):]
    if key == "lm_head.weight":           # tied: materialised from embed_tokens
        return None
    return key


def load_weights(model: nn.Module, model_dir: str,
                 dtype: torch.dtype | None = None,
                 device: str = "cpu") -> list[str]:
    """Stream the released safetensors shards into `model`.

    Copies tensor-by-tensor (no full state_dict copy in memory); Tensor.copy_
    handles dtype/device conversion, so the model should already live on its
    target device/dtype. Returns the checkpoint keys that were not consumed
    (vision tower / MTP), for callers that want to sanity-check what was dropped.
    """
    from safetensors import safe_open

    index_path = os.path.join(model_dir, INDEX_NAME)
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    skipped: list[str] = []
    shards = sorted(set(weight_map.values()))
    for shard in shards:
        with safe_open(os.path.join(model_dir, shard), framework="pt") as f:
            for key in sorted(k for k, s in weight_map.items() if s == shard):
                dst = _remap_key(key)
                if dst is None:
                    skipped.append(key)
                    continue
                param = model.get_parameter(dst)
                param.data.copy_(f.get_tensor(key))
    return skipped


def from_pretrained(model_dir: str, device: str = "cpu",
                    dtype: torch.dtype = torch.bfloat16) -> Qwen35ForCausalLM:
    """Config + weights in one call (the usual entry point)."""
    config = Qwen35Config.from_json(os.path.join(model_dir, "config.json"))
    model = Qwen35ForCausalLM(config).to(device=device, dtype=dtype).eval()
    skipped = load_weights(model, model_dir, device=device)
    n_visual = sum(1 for k in skipped if k.startswith("model.visual."))
    n_mtp = sum(1 for k in skipped if k.startswith("mtp."))
    print(f"[load] {model_dir}: text weights loaded, skipped "
          f"{n_visual} vision + {n_mtp} MTP tensors")
    return model


# --------------------------------------------------------------------------- #
# shape smoke test on a tiny random config (no checkpoint needed)
# --------------------------------------------------------------------------- #

def _smoke_test() -> None:
    torch.manual_seed(0)
    config = Qwen35Config(
        vocab_size=512, hidden_size=64, intermediate_size=128,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=8, linear_value_head_dim=8,
        layer_types=["linear_attention", "linear_attention",
                     "full_attention", "linear_attention"])
    model = Qwen35ForCausalLM(config).eval()
    ids = torch.randint(0, 512, (2, 37))

    from cache import HybridCaches

    with torch.no_grad():
        # no-cache forward must equal cache-warm forward on the same tokens
        logits_ref = model(ids)
        caches = HybridCaches(config, batch=2, max_seq_len=64, dtype=torch.float32)
        logits_cached = model(ids, caches=caches)
        err = (logits_ref - logits_cached).abs().max().item()
        assert err < 1e-5, f"cache-warm prefill diverged: {err}"
        print(f"[check] prefill with/without cache identical, max|diff|={err:.2e}")

        # greedy decode from cache; keep every step's logits for comparison
        t0 = ids.shape[1]
        tokens = [logits_ref[:, -1].argmax(-1)]
        cached_logits = [logits_ref[:, -1]]
        for _ in range(8):
            logits = model(tokens[-1][:, None], caches=caches, logits_to_keep=1)
            tokens.append(logits[:, -1].argmax(-1))
            cached_logits.append(logits[:, -1])

        # recompute the whole continuation without cache; every position must
        # reproduce the cached-step logits. 1e-4 (not 1e-5): the GDN state is
        # re-recurrented every step, so fp32 rounding accumulates.
        full = torch.cat([ids] + [t[:, None] for t in tokens[1:]], dim=1)
        logits_naive = model(full)
        max_err = max((logits_naive[:, t0 - 1 + s] - cached_logits[s]).abs().max().item()
                      for s in range(9))
        assert max_err < 1e-4, f"decode steps diverged from no-cache: {max_err}"
        greedy_ok = all(
            torch.equal(logits_naive[:, t0 - 1 + s].argmax(-1), tokens[s])
            for s in range(9))
        assert greedy_ok, "greedy tokens diverged between cache and no-cache"
        print(f"[check] 8 decode steps match no-cache forward, "
              f"max|diff|={max_err:.2e}, greedy tokens identical")


if __name__ == "__main__":
    _smoke_test()
