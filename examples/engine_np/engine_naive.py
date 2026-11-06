#!/usr/bin/env python3
"""L1 engine: prefill-once + incremental decode over the hand-written model.

This is the ladder rung between L0 (transformers generate(), framework
everything) and L2 (KV cache on/off benchmarks): we own the decode loop,
position bookkeeping and sampling, and the cache is explicit.

  prefill(prompt)        -> run all prompt tokens once, ingest into caches
  decode_step(token)     -> run ONE token against the caches
  generate(prompt, n)    -> the loop, greedy or temperature sampling

generate() returns only the NEW token ids (the prompt is the caller's).

Run: python3 examples/engine_np/engine_naive.py
"""

import torch

from cache import HybridCaches


class NaiveEngine:
    """Single-request engine around a Qwen35ForCausalLM. batch=1 semantics;
    the caches already support B>1 but scheduling that is L3's job."""

    def __init__(self, model, max_seq_len: int = 4096,
                 device: str | None = None, dtype: torch.dtype | None = None):
        self.model = model
        self.config = model.config
        self.device = device or next(model.parameters()).device
        self.dtype = dtype or next(model.parameters()).dtype

    def _sample(self, logits: torch.Tensor, temperature: float,
                generator: torch.Generator | None) -> torch.Tensor:
        """logits (B, vocab) -> token ids (B,). temperature 0 = greedy argmax."""
        if temperature <= 0.0:
            return logits.argmax(dim=-1)
        probs = torch.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)

    @torch.no_grad()
    def prefill(self, prompt_ids: torch.Tensor, caches: HybridCaches,
                temperature: float = 0.0,
                generator: torch.Generator | None = None) -> torch.Tensor:
        """Ingest the whole prompt in one forward; return the FIRST new token.

        prompt_ids: (B, T) — T is the prefill length, positions 0..T-1.
        """
        logits = self.model(prompt_ids, caches=caches, logits_to_keep=1)
        return self._sample(logits[:, -1, :], temperature, generator)

    @torch.no_grad()
    def decode_step(self, token_ids: torch.Tensor, caches: HybridCaches,
                    temperature: float = 0.0,
                    generator: torch.Generator | None = None) -> torch.Tensor:
        """Run one token (B, 1); return the NEXT token (B,)."""
        logits = self.model(token_ids, caches=caches, logits_to_keep=1)
        return self._sample(logits[:, -1, :], temperature, generator)

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int,
                 temperature: float = 0.0, eos_token_id: int | None = None,
                 seed: int = 0) -> list[int]:
        """Full loop: prefill then decode. Returns the new tokens (list, B=1).

        Each decode step costs one forward over 1 token + a cache append —
        O(1) in prompt length for attention, O(1) always for GDN layers.
        """
        assert prompt_ids.shape[0] == 1, "single request only (batching is L3)"
        prompt_ids = prompt_ids.to(self.device)
        generator = None
        if temperature > 0.0:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        caches = HybridCaches(self.config, batch=1,
                              max_seq_len=prompt_ids.shape[1] + max_new_tokens,
                              device=str(self.device), dtype=self.dtype)
        nxt = self.prefill(prompt_ids, caches, temperature, generator)
        out = [int(nxt[0])]
        for _ in range(max_new_tokens - 1):
            if eos_token_id is not None and out[-1] == eos_token_id:
                break
            nxt = self.decode_step(nxt[:, None], caches, temperature, generator)
            out.append(int(nxt[0]))
        return out

    @torch.no_grad()
    def generate_no_cache(self, prompt_ids: torch.Tensor, max_new_tokens: int,
                          temperature: float = 0.0,
                          seed: int = 0) -> list[int]:
        """The L0-style control: recompute the FULL sequence every step.

        Same sampling decisions as generate() (greedy is deterministic;
        temperature uses the same generator seed), just O(T) work per token —
        this is the honest baseline for the L2 cache benchmark.
        """
        assert prompt_ids.shape[0] == 1
        prompt_ids = prompt_ids.to(self.device)
        generator = None
        if temperature > 0.0:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        seq = prompt_ids
        out: list[int] = []
        for _ in range(max_new_tokens):
            logits = self.model(seq, logits_to_keep=1)
            nxt = self._sample(logits[:, -1, :], temperature, generator)
            out.append(int(nxt[0]))
            seq = torch.cat([seq, nxt[:, None]], dim=1)
        return out


def _self_check() -> None:
    """Tiny random model: cached generate == no-cache generate, both greedy."""
    torch.manual_seed(0)
    from model_qwen35 import Qwen35Config, Qwen35ForCausalLM

    config = Qwen35Config(
        vocab_size=512, hidden_size=64, intermediate_size=128,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=8, linear_value_head_dim=8,
        layer_types=["linear_attention", "full_attention", "linear_attention",
                     "linear_attention"])
    model = Qwen35ForCausalLM(config).eval()
    engine = NaiveEngine(model)
    prompt = torch.randint(0, 512, (1, 33))

    a = engine.generate(prompt, max_new_tokens=12, temperature=0.0)
    b = engine.generate_no_cache(prompt, max_new_tokens=12, temperature=0.0)
    assert a == b, f"cached and no-cache greedy runs disagree:\n{a}\n{b}"
    print(f"[check] greedy: cached == no-cache for 12 tokens: {a}")

    # temperature sampling must at least stay in-range and reproducible
    s1 = engine.generate(prompt, max_new_tokens=8, temperature=0.7, seed=3)
    s2 = engine.generate(prompt, max_new_tokens=8, temperature=0.7, seed=3)
    assert s1 == s2, "sampling not reproducible for a fixed seed"
    assert all(0 <= t < 512 for t in s1)
    print(f"[check] temperature=0.7 reproducible for fixed seed: {s1}")

    # eos early-stop
    eos = 7
    s3 = engine.generate(prompt, max_new_tokens=50, temperature=0.0,
                         eos_token_id=eos)
    assert len(s3) <= 50 and (eos not in s3 or s3[-1] == eos)
    print(f"[check] eos_token_id={eos} stops early after {len(s3)} tokens")


if __name__ == "__main__":
    _self_check()
