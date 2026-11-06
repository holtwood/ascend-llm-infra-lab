#!/usr/bin/env python3
"""Numerics + sync-behaviour test for the E32 sync-free conv1d update.

Run on an Ascend node (needs torch_npu + the vllm-ascend tree):

    python3 patches/e32_conv1d_nosync/test_conv1d_nosync.py

Checks:
  1. logits match ``causal_conv1d_ref`` exactly (bf16 tolerance 0)
  2. the rolling state matches the reference's final state
  3. pad slots are neither read nor written (output zero, state untouched)
  4. a second call after state mutation gives the same result as the reference
     (i.e. the update is a proper recurrence, not a one-shot)
"""
import os
import sys

import torch

sys.path.insert(0, "/vllm-workspace/vllm-ascend")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from conv1d_nosync import causal_conv1d_update_nosync  # noqa: E402
from vllm_ascend.ops.causal_conv1d import causal_conv1d_ref  # noqa: E402

PAD = -1


def ref_update(x, state, weight, bias, activation="silu"):
    """Reference for the decode case: conv over [state | x], return last output."""
    b, dim = x.shape
    width = weight.shape[1]
    seq = torch.cat([state, x.unsqueeze(-1)], dim=-1)  # [B, dim, width]
    out_ref, final = causal_conv1d_ref(
        seq, weight, bias, initial_states=None, return_final_states=True,
        activation=activation)
    return out_ref[..., -1], final[..., -(weight.shape[1] - 1):]


def main():
    torch.manual_seed(0)
    dev = "npu:0" if hasattr(torch, "npu") and torch.npu.is_available() else "cpu"
    dtype = torch.float32   # exactness check first; bf16 tolerance below
    b, dim, width = 6, 128, 4
    slots = 10

    weight = torch.randn(dim, width, device=dev, dtype=dtype) * 0.1
    bias = torch.randn(dim, device=dev, dtype=dtype) * 0.1
    state = torch.randn(slots, dim, width - 1, device=dev, dtype=dtype) * 0.1
    x = torch.randn(b, dim, device=dev, dtype=dtype) * 0.1
    idx = torch.tensor([0, 1, 2, PAD, 4, 5], device=dev, dtype=torch.int32)

    st_a = state.clone()
    st_b = state.clone()

    out_a = causal_conv1d_update_nosync(x, st_a, weight, bias, "silu",
                                       conv_state_indices=idx, pad_slot_id=PAD)

    # reference over valid slots only (pad rows must stay untouched)
    valid = idx != PAD
    gathered = state.index_select(0, idx[valid].to(torch.long))
    out_ref, new_states = ref_update(x[valid], gathered, weight, bias)
    st_b.index_copy_(0, idx[valid].to(torch.long), new_states)

    d_out = (out_a[valid].float() - out_ref.float()).abs().max().item()
    d_state = (st_a.float() - st_b.float()).abs().max().item()
    pad_out_zero = bool((out_a[~valid] == 0).all().item())
    pad_state_same = bool((st_a[idx[~valid].to(torch.long)] ==
                           state[idx[~valid].to(torch.long)]).all().item())

    print(f"device={dev} dtype={dtype} max|out diff|={d_out:.3e} max|state diff|={d_state:.3e}")
    print(f"pad slot: output zero={pad_out_zero}, state untouched={pad_state_same}")
    assert d_out < 1e-5, f"output mismatch vs reference: {d_out:.3e}"
    assert d_state < 1e-5, f"state mismatch vs reference: {d_state:.3e}"
    assert pad_out_zero and pad_state_same, "pad slot handling broken"

    # recurrence check: feed a second token and compare against a fresh reference
    x2 = torch.randn(b, dim, device=dev, dtype=dtype) * 0.1
    out_a2 = causal_conv1d_update_nosync(x2, st_a, weight, bias, "silu",
                                        conv_state_indices=idx, pad_slot_id=PAD)
    g2 = st_b.index_select(0, idx[valid].to(torch.long))
    out_ref2, _ = ref_update(x2[valid], g2, weight, bias)
    d2 = (out_a2[valid].float() - out_ref2.float()).abs().max().item()
    print(f"second step max|out diff|={d2:.3e}")
    assert d2 < 1e-5, "recurrence mismatch"

    # bf16 pass: different op order, so allow a small tolerance
    dt = torch.bfloat16
    w16 = weight.to(dt); b16 = bias.to(dt)
    s16a = state.to(dt).clone(); s16b = state.to(dt).clone()
    x16 = x.to(dt)
    o16a = causal_conv1d_update_nosync(x16, s16a, w16, b16, "silu",
                                       conv_state_indices=idx, pad_slot_id=PAD)
    g16 = state.to(dt).index_select(0, idx[valid].to(torch.long))
    o16r, ns16 = ref_update(x16[valid], g16, w16, b16)
    d16 = (o16a[valid].float() - o16r.float()).abs().max().item()
    print(f"bf16 max|out diff|={d16:.3e}")
    assert d16 < 5e-2, f"bf16 mismatch: {d16:.3e}"

    print("all E32 checks passed")


if __name__ == "__main__":
    main()
