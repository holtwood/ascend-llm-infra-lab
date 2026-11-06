#!/usr/bin/env python3
"""Numerical verification: our from-scratch Qwen3.5 vs transformers 5.x.

Two modes:

  default (mini)  — build a 2-layer mini model (1 GDN + 1 full-attention) from
                    Qwen3_5TextConfig, copy the HF state_dict into our model,
                    and compare in fp32:
                      1. prefill logits + final GDN recurrent state
                      2. single-token decode steps against the HF DynamicCache
                    Threshold: max|diff| < 1e-4 (fp32, same kernels both sides).

  --real          — load the released Qwen3.5-4B checkpoint twice (HF + ours,
                    bf16) and compare greedy generation on a real prompt.
                    Needs ~2x8GB host RAM in fp32/bf16 on CPU; meant for the
                    NPU host (see README), works on CPU too, just slowly.

Requires transformers >= 5.16 (first release with qwen3_5).

Run:  python3 examples/engine_np/verify_logits.py
      python3 examples/engine_np/verify_logits.py --real \
          --model-dir /model/Qwen3.5-4B --device npu
"""

import argparse
import sys

import torch

import cache as cache_mod
from model_qwen35 import Qwen35Config, Qwen35ForCausalLM

SEED = 0


def build_mini_pair(device="cpu", dtype=torch.float32):
    """Same hyperparameters through both config paths, same weights."""
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    hf_config = Qwen3_5TextConfig(
        vocab_size=1000, hidden_size=256, intermediate_size=512,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=64, rms_norm_eps=1e-6, attention_bias=False,
        linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=16, linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 0.25, "mrope_section": [4, 4, 0],
                         "mrope_interleaved": True},
        # Qwen3_5TextConfig DEFAULTS THIS TO FALSE; the released 4B uses True.
        # Must match on both sides or the two state_dict keys write over each
        # other through the shared lm_head Parameter on ours.
        tie_word_embeddings=True,
    )
    our_config = Qwen35Config(
        vocab_size=1000, hidden_size=256, intermediate_size=512,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=64, rms_norm_eps=1e-6, attention_bias=False,
        linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=16, linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "full_attention"],
        rope_theta=10000.0, partial_rotary_factor=0.25,
        mrope_section=[4, 4, 0],
        tie_word_embeddings=True,
    )

    torch.manual_seed(SEED)
    hf = Qwen3_5ForCausalLM(hf_config).to(device=device, dtype=dtype).eval()
    ours = Qwen35ForCausalLM(our_config).to(device=device, dtype=dtype).eval()

    # copy weights over — the state_dict keys are IDENTICAL by construction
    # (same submodule names); lm_head.weight is tied on both sides.
    missing, unexpected = ours.load_state_dict(hf.state_dict(), strict=False)
    assert not unexpected, f"unexpected keys: {unexpected}"
    assert not missing, f"missing keys: {missing}"
    return hf, ours


def compare(name, a, b, tol=1e-4):
    err = (a.float() - b.float()).abs().max().item()
    status = "OK " if err < tol else "FAIL"
    print(f"  [{status}] {name:34s} max|diff|={err:.3e}")
    assert err < tol, f"{name}: {err} >= tol {tol}"
    return err


def verify_mini(device="cpu"):
    print(f"== mini model (1 GDN + 1 attention layer, fp32, device={device}) ==")
    hf, ours = build_mini_pair(device=device)
    batch, seq = 2, 130                      # 130 = 2 chunks + 2, exercises padding
    torch.manual_seed(42)
    ids = torch.randint(0, 1000, (batch, seq), device=device)

    from transformers import DynamicCache

    # ---- 1. prefill: logits + final GDN recurrent state -------------------
    with torch.no_grad():
        hf_cache = DynamicCache(config=hf.config)
        hf_out = hf(ids, use_cache=True, past_key_values=hf_cache)
        our_caches = cache_mod.HybridCaches(ours.config, batch=batch,
                                            max_seq_len=seq * 2, device=device,
                                            dtype=hf_out.logits.dtype)
        our_logits = ours(ids, caches=our_caches)
    compare("prefill logits", hf_out.logits, our_logits)
    compare("GDN final state (layer 0)",
            hf_cache.layers[0].recurrent_states[0], our_caches.layers[0].state)

    # ---- 2. decode: feed the same next tokens one by one ------------------
    steps = 8
    torch.manual_seed(43)
    next_ids = torch.randint(0, 1000, (batch, steps), device=device)
    hf_c = hf_cache
    our_c = our_caches
    max_err = 0.0
    with torch.no_grad():
        for s in range(steps):
            tok = next_ids[:, s:s + 1]
            hf_logits = hf(tok, use_cache=True, past_key_values=hf_c).logits
            our_logits = ours(tok, caches=our_c, logits_to_keep=1)
            max_err = max(max_err, (hf_logits - our_logits).abs().max().item())
    print(f"  [{'OK ' if max_err < 1e-4 else 'FAIL'}] decode {steps} steps"
          f"{'':17s} max|diff|={max_err:.3e}")
    assert max_err < 1e-4

    # greedy continuations must agree token-for-token
    hf_gen = hf.generate(ids, max_new_tokens=10, do_sample=False,
                         attention_mask=torch.ones_like(ids))
    our_c2 = cache_mod.HybridCaches(ours.config, batch=batch, max_seq_len=seq + 16,
                                    device=device, dtype=torch.float32)
    engine_ids = ids
    with torch.no_grad():
        nxt = ours(engine_ids, caches=our_c2, logits_to_keep=1)[:, -1].argmax(-1)
        gen = [nxt]
        for _ in range(9):
            nxt = ours(nxt[:, None], caches=our_c2, logits_to_keep=1)[:, -1].argmax(-1)
            gen.append(nxt)
    our_gen = torch.cat([ids] + [g[:, None] for g in gen], dim=1)
    assert torch.equal(hf_gen, our_gen), "greedy generations disagree"
    print("  [OK ] greedy generation (10 tokens)     token-identical")


def verify_real(model_dir, device, dtype, prompt_len=24, steps=8):
    print(f"== released checkpoint {model_dir} ({device}, {dtype}) ==")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    text = "昇腾 NPU 上做 LLM 推理，最需要理解的是 prefill 与 decode 的算子差异。"
    ids = tok(text, return_tensors="pt").input_ids[:, :prompt_len].to(device)
    print(f"prompt: {prompt_len} tokens")

    hf = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=dtype, attn_implementation="sdpa").to(device).eval()

    our_config = Qwen35Config.from_json(f"{model_dir}/config.json")
    ours = Qwen35ForCausalLM(our_config).to(device=device, dtype=dtype).eval()
    from model_qwen35 import load_weights
    load_weights(ours, model_dir, device=device)

    with torch.no_grad():
        hf_logits = hf(ids).logits[:, -1, :].float()
        caches = cache_mod.HybridCaches(our_config, batch=1,
                                        max_seq_len=prompt_len + steps + 4,
                                        device=device, dtype=dtype)
        our_logits = ours(ids, caches=caches, logits_to_keep=1)[:, -1, :].float()

    err = (hf_logits - our_logits).abs().max().item()
    rel = err / hf_logits.abs().max().item()
    agree = torch.equal(hf_logits.argmax(-1), our_logits.argmax(-1))
    print(f"  prefill last-position logits: max|diff|={err:.4e} "
          f"(rel {rel:.2e}), top-1 agree={agree}")
    assert agree, "top-1 token diverges from transformers"

    with torch.no_grad():
        hf_ids = hf.generate(ids, max_new_tokens=steps, do_sample=False)
        nxt = our_logits.argmax(-1)
        ours_new = [int(nxt)]
        for _ in range(steps - 1):
            nxt = ours(nxt[:, None], caches=caches,
                       logits_to_keep=1)[:, -1].argmax(-1)
            ours_new.append(int(nxt))
    hf_new = hf_ids[0, prompt_len:].tolist()
    print(f"  transformers: {hf_new}")
    print(f"  our engine  : {ours_new}")
    assert hf_new == ours_new, "greedy generation diverges"
    print("  [OK ] greedy generation identical")
    print("  decoded:", repr(tok.decode(hf_ids[0, prompt_len:])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="verify against the 4B checkpoint")
    ap.add_argument("--model-dir", default="/model/Qwen3.5-4B")
    ap.add_argument("--device", default="cpu", help="cpu | npu")
    ap.add_argument("--dtype", default=None, help="e.g. bfloat16 (real mode default)")
    args = ap.parse_args()

    if args.real:
        dtype = getattr(torch, args.dtype) if args.dtype else torch.bfloat16
        if args.device == "npu":
            import torch_npu                      # noqa: F401
            torch.npu.set_device(0)
        verify_real(args.model_dir, args.device, dtype)
    else:
        verify_mini(args.device)
        print("\nall mini-model checks passed: hand-written forward is"
              " numerically aligned with transformers 5.x")


if __name__ == "__main__":
    sys.exit(main())
