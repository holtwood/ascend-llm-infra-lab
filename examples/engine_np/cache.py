#!/usr/bin/env python3
"""Per-layer inference caches for the Qwen3.5 hybrid architecture.

Two cache kinds (one per token-mixer type):
  - KVCache:        8 full-attention layers, classic K/V ring buffer (append + read).
  - RecurrentState: 24 GatedDeltaNet layers, conv tail + recurrent S matrix.

The public surface (allocate / append / read) is deliberately narrow so that L4
can swap KVCache for a paged (block-pool + block-table) implementation without
touching the model code: the model only ever calls append(k, v) / read() on
attention layers and conv_input / read_state / write_state on GDN layers.

State semantics (aligned with transformers 5.x LinearAttentionLayer):
  - conv_state keeps the LAST conv_kernel_size (=4) raw conv inputs, i.e. the
    SiLU-less mixed qkv, NOT kernel_size-1. Short prefills are left-padded
    with zeros up to the kernel size before being remembered.
  - the recurrent S starts at zero before the first prefill and is replaced
    (not accumulated) by every forward that runs a GDN layer.

Run: python3 examples/engine_np/cache.py   (shape/semantics self-check)
"""

import torch


class KVCache:
    """Append-only K/V buffer for one full-attention layer.

    Pre-allocates max_seq_len up front (L2 semantics). L4 will replace this
    with a block pool + block table; append/read keep the same signatures.
    """

    def __init__(self, batch: int, n_kv_heads: int, head_dim: int, max_seq_len: int,
                 device: str = "cpu", dtype: torch.dtype = torch.bfloat16):
        self.batch = batch
        self.max_seq_len = max_seq_len
        self.seq_len = 0                       # absolute position of next slot
        # (B, H_kv, max_len, D) — allocated lazily so cache.py stays import-safe on npu
        self.k = torch.zeros(batch, n_kv_heads, max_seq_len, head_dim,
                             device=device, dtype=dtype)
        self.v = torch.zeros_like(self.k)

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        """Record new k/v of shape (B, H_kv, T, D) at the tail."""
        t = k_new.shape[2]
        self.k[:, :, self.seq_len:self.seq_len + t].copy_(k_new)
        self.v[:, :, self.seq_len:self.seq_len + t].copy_(v_new)
        self.seq_len += t

    def read(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the used portion (B, H_kv, seq_len, D) — views, not copies."""
        return self.k[:, :, :self.seq_len], self.v[:, :, :self.seq_len]


class RecurrentState:
    """Cache for one GatedDeltaNet layer: conv tail + recurrent S matrix.

    conv_state: (B, conv_dim, conv_kernel) — last K raw mixed-qkv columns.
    state:      (B, n_v_heads, d_k, d_v) float32 recurrent matrix S.
    """

    def __init__(self, batch: int, conv_dim: int, conv_kernel: int,
                 n_v_heads: int, d_k: int, d_v: int,
                 device: str = "cpu", dtype: torch.dtype = torch.float32):
        self.batch = batch
        self.conv_kernel = conv_kernel
        self.conv_state: torch.Tensor | None = None   # None = no history yet
        self.state: torch.Tensor | None = None        # None = S still implicit zero
        # allocate eagerly so decode steps never allocate
        self._conv_buf = torch.zeros(batch, conv_dim, conv_kernel, device=device)
        self._state_buf = torch.zeros(batch, n_v_heads, d_k, d_v,
                                      device=device, dtype=dtype)
        self._conv_ready = False

    @property
    def has_history(self) -> bool:
        """True once a first forward has primed the conv tail. The GDN layer
        uses this to pick the padded-conv (prefill) vs cat-conv (decode) path."""
        return self._conv_ready

    def conv_input(self, x: torch.Tensor) -> torch.Tensor:
        """Return the tensor to feed into the depthwise causal conv and
        remember the tail.

        x is the new raw conv input (B, conv_dim, T).
        - first call (prefill): return x unchanged; the caller convolves with
          left-zero padding (equivalent to a zero conv history) and we store
          x's tail, zero-padded on the left up to the kernel size.
        - later calls (decode): return cat(conv_state, x); the caller convolves
          with padding=0 and keeps the last T outputs.
        """
        k = self.conv_kernel
        if not self._conv_ready:
            tail = x[..., -k:]
            tail = torch.nn.functional.pad(tail, (k - tail.shape[-1], 0))
            self._conv_buf.copy_(tail)
            self._conv_ready = True
            return x
        full = torch.cat([self._conv_buf, x], dim=-1)
        self._conv_buf.copy_(full[..., -k:])
        return full

    def read_state(self) -> torch.Tensor | None:
        """Current recurrent S (fp32) or None if the next forward starts fresh."""
        return self.state

    def write_state(self, s: torch.Tensor) -> None:
        """Store the new recurrent S (cast to the buffer dtype, fp32)."""
        self._state_buf.copy_(s)
        self.state = self._state_buf


class HybridCaches:
    """One cache per decoder layer, ordered by layer index.

    layers[i] is a KVCache or a RecurrentState according to layer_types[i].
    seq_len reflects the number of tokens already ingested (tracked on the
    first full-attention layer; GDN layers are position-agnostic).
    """

    def __init__(self, config, batch: int, max_seq_len: int,
                 device: str = "cpu", dtype: torch.dtype = torch.bfloat16):
        # duck-typed on config: needs layer_types + the linear_* / attention
        # shape fields (Qwen35Config provides all of them)
        self.layers = []
        self.max_seq_len = max_seq_len
        for layer_idx, ltype in enumerate(config.layer_types):
            if ltype == "full_attention":
                self.layers.append(KVCache(
                    batch, config.num_key_value_heads, config.head_dim,
                    max_seq_len, device, dtype))
            else:
                self.layers.append(RecurrentState(
                    batch,
                    conv_dim=config.linear_key_dim * 2 + config.linear_value_dim,
                    conv_kernel=config.linear_conv_kernel_dim,
                    n_v_heads=config.linear_num_value_heads,
                    d_k=config.linear_key_head_dim,
                    d_v=config.linear_value_head_dim,
                    device=device))
        self._kv_idx = next((i for i, c in enumerate(self.layers)
                             if isinstance(c, KVCache)), None)
        if self._kv_idx is None:
            raise NotImplementedError("model with no full-attention layer")

    @property
    def seq_len(self) -> int:
        """Tokens already cached — authoritative position offset for prefill."""
        return self.layers[self._kv_idx].seq_len

    def reset(self) -> None:
        """Drop all state (reuse the buffers for a new request)."""
        for c in self.layers:
            if isinstance(c, KVCache):
                c.seq_len = 0
            else:
                c._conv_ready = False
                c.state = None


def _self_check() -> None:
    """Conv-window semantics: cached stepwise conv must match a full conv."""
    torch.manual_seed(0)
    b, d, k, t_total = 2, 8, 4, 9
    weight = torch.randn(d, 1, k)              # depthwise kernel
    x_full = torch.randn(b, d, t_total)

    # ground truth: causal conv over the whole sequence, no history
    def causal_conv(x):
        full = torch.nn.functional.conv1d(x, weight, padding=k - 1, groups=d)
        return full[..., :x.shape[-1]]

    ref = causal_conv(x_full)

    # cached: prefill the first 5 tokens, then decode one at a time
    state = RecurrentState(b, conv_dim=d, conv_kernel=k, n_v_heads=1, d_k=1, d_v=1)
    outs = []
    x_pre = x_full[:, :, :5]
    full_in = state.conv_input(x_pre)
    outs.append(torch.nn.functional.conv1d(full_in, weight, padding=k - 1,
                                           groups=d)[..., :5])
    for t in range(5, t_total):
        full_in = state.conv_input(x_full[:, :, t:t + 1])
        outs.append(torch.nn.functional.conv1d(full_in, weight, padding=0,
                                               groups=d)[..., -1:])
    got = torch.cat(outs, dim=-1)
    err = (ref - got).abs().max().item()
    assert err < 1e-6, f"cached conv diverged from full conv: {err}"
    print(f"[check] cached causal conv == full-sequence conv, max|diff|={err:.2e}")

    # short prefill (T < kernel) must also line up
    state2 = RecurrentState(b, conv_dim=d, conv_kernel=k, n_v_heads=1, d_k=1, d_v=1)
    outs2 = []
    for t in range(t_total):                   # "prefill" of length 1 every step
        full_in = state2.conv_input(x_full[:, :, t:t + 1])
        if t == 0:                             # first call: causal conv, take pos0
            outs2.append(torch.nn.functional.conv1d(
                full_in, weight, padding=k - 1, groups=d)[..., :1])
        else:                                  # decode: padded-0 conv, take tail
            outs2.append(torch.nn.functional.conv1d(
                full_in, weight, padding=0, groups=d)[..., -1:])
    got2 = torch.cat(outs2, dim=-1)
    err2 = (ref - got2).abs().max().item()
    assert err2 < 1e-6, f"single-token prefill path diverged: {err2}"
    print(f"[check] T=1-prefill path also matches, max|diff|={err2:.2e}")


if __name__ == "__main__":
    _self_check()
