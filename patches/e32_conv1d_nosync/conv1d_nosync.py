# SPDX-License-Identifier: Apache-2.0
"""Sync-free causal conv1d state update (decode path) for Ascend.

Why this exists (lab repo E32 / upstream issue #18978):

* vllm-ascend's PyTorch fallback for ``causal_conv1d_update`` walks the batch in
  Python and calls ``conv_state_indices[i].item()`` per sequence. On NPU that is
  a host read-back **per request per layer**: it serialises the stream, blocks
  ACL-graph capture at decode-FULL and taxes steady-state ITL.
* The decode path (one new token per sequence) does not need any data-dependent
  branching: gather the states, run a width-tap depthwise conv as a broadcast
  multiply + sum, scatter the shifted states back. All ops stay on device.

Scope: this module implements the **decode** case (``x`` of shape ``[B, dim]``,
optionally with ``query_start_loc`` whose per-sequence length is 1). Multi-token
(varlen prefill) updates are left to the existing implementation; callers should
dispatch on ``x.dim() == 2``.

Numerics are checked against ``vllm_ascend.ops.causal_conv1d.causal_conv1d_ref``
in ``test_conv1d_nosync.py``.
"""

from __future__ import annotations

import torch

__all__ = ["causal_conv1d_update_nosync"]

PAD_SLOT_ID = -1


def _activation(x: torch.Tensor, activation) -> torch.Tensor:
    if activation is True:
        activation = "silu"
    if activation in (None, False):
        return x
    if activation in ("silu", "swish"):
        return torch.nn.functional.silu(x)
    raise ValueError(f"unsupported activation: {activation!r}")


def causal_conv1d_update_nosync(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation=None,
    conv_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    pad_slot_id: int = PAD_SLOT_ID,
) -> torch.Tensor:
    """Update conv state and return the new-token convolution output.

    Args:
        x: ``[B, dim]`` new tokens (decode step).
        conv_state: ``[num_slots, dim, width-1]`` (or transposed) rolling states.
        weight: ``[dim, width]`` depthwise kernel.
        bias: ``[dim]`` optional.
        activation: ``None`` | ``"silu"`` | ``"swish"`` | ``True``.
        conv_state_indices: ``[B]`` int32 slot ids; ``pad_slot_id`` marks slots
            that must be neither read nor written (their output is zero).
        num_accepted_tokens / query_start_loc: accepted for signature
            compatibility with the fallback; this path assumes one token per
            sequence (asserted below).

    Returns:
        ``[B, dim]`` output for the new tokens.
    """
    if x.dim() != 2:
        raise ValueError(
            f"causal_conv1d_update_nosync handles the decode case only, got x.dim()={x.dim()}"
        )
    if query_start_loc is not None and query_start_loc.numel() > 1:
        # varlen prefill: caller must fall back to the general implementation
        raise ValueError("varlen inputs are not supported by the sync-free decode path")

    dim, width = weight.shape
    if conv_state.shape[-2] != dim and conv_state.shape[-1] == dim:
        conv_state = conv_state.transpose(-1, -2)
    if x.shape[-1] != dim:
        raise ValueError(f"x dim {x.shape[-1]} != weight dim {dim}")

    b = x.shape[0]
    device = x.device
    if conv_state_indices is None:
        idx = torch.arange(b, device=device, dtype=torch.long)
    else:
        idx = conv_state_indices.to(torch.long)
    valid = idx != pad_slot_id
    safe_idx = torch.where(valid, idx, torch.zeros_like(idx))

    state = conv_state.index_select(0, safe_idx)                    # [B, dim, width-1]
    seq = torch.cat([state.to(x.dtype), x.unsqueeze(-1)], dim=-1)   # [B, dim, width]

    # depthwise width-tap conv: out[b, d] = sum_w weight[d, w] * seq[b, d, w]
    out = (seq * weight.unsqueeze(0)).sum(dim=-1)
    if bias is not None:
        out = out + bias
    out = _activation(out, activation)

    new_state = seq[..., 1:]                                        # [B, dim, width-1]
    if valid.any():
        conv_state.index_copy_(0, safe_idx[valid], new_state[valid].to(conv_state.dtype))

    out = torch.where(valid.unsqueeze(-1), out, torch.zeros_like(out))
    return out
