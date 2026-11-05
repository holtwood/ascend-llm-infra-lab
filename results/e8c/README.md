# E8c: Hand-written Speculative Decoding — Draft/Verify Mechanism Study

**Environment**: pytorch container (new env), `torch_npu` + HF transformers, Qwen3-0.6B bf16, Ascend 910B1 (NPU 2), `scripts/e8c_specdecode.py`.

## Question

Production E3 showed vLLM ngram speculative decoding returned **negative** on real
chat traffic (~10% acceptance → net slowdown). Why does speculation ever win, and
what are the real terms? This experiment rebuilds the draft/verify loop in ~60
lines of pure PyTorch to expose the mechanism.

## Implementation (correct draft/verify, one forward per iteration)

- **Draft**: prompt-lookup ngram proposer (the same family as vLLM `--speculative-config '{"method":"ngram"}'`). A table maps the last-n-token context (n=4,3,2) of the prompt+generated prefix to the token that followed it.
- **Verify**: feed `[bonus, d1..dk]` in **one** forward. `bonus` is the already-committed token whose KV entry doesn't exist yet; `logits[j]` is the target's distribution after `in_ids[j]`, so the accepted prefix is the longest run where `argmax(logits[j]) == props[j]`.
- **KV bookkeeping**: `DynamicCache.crop()` discards KV slots of rejected draft tokens — no extra forward needed. This was the key fix: an earlier version recomputed KV per iteration (2 forwards/step), which ate all gains even at 57% acceptance.
- **Greedy everywhere**: accepted tokens equal what the target would have produced — exactness by construction.

## Results (Qwen3-0.6B, n_new=128 decode tokens, bs=1, NPU)

| Workload | spec k | accept rate | draft coverage | tok/s | vs baseline |
|---|---|---|---|---|---|
| repetitive (model asked to repeat a sentence) | — | — | — | 13.1 | — |
| | 2 | 0.25 | 8/126 iters | 13.6 | +4% |
| | 4 | 0.25 | 16/124 iters | 14.2 | **+8%** |
| | 8 | 0.25 | 32/120 iters | 14.2 | +8% |
| random (open-ended essay) | — | — | — | 14.1 | — |
| | 2 | 0.00 | 2/128 | 14.2 | ~0% |
| | 4 | 0.00 | 4/128 | 13.3 | **−6%** |
| | 8 | 0.00 | 8/128 | 13.2 | **−6%** |

Raw JSON: `spec.json`.

## What it demonstrates

`net_gain ≈ f(draft_coverage × accept_streak, verify_overhead)`:

1. **Coverage is the first gate.** On the repetitive workload the ngram table only
   fired in ~13% of iterations (16/124) — the model's own continuation drifts away
   from literal prompt ngrams even when asked to repeat. 25% per-token acceptance
   with rare firing → single-digit net gain.
2. **Negative return is mechanical, not a tuning failure.** On open text, every
   verify costs a slightly wider forward and returns exactly 1 committed token →
   pure overhead (−6%). This is the same shape as production E3's ngram result
   (~10% acceptance → net loss): on general chat traffic, prompt-lookup speculation
   is a net cost.
3. **Verify-overhead dominated before KV-crop.** An earlier implementation
   rebuilt KV state with a second forward per iteration — at 57% acceptance it
   still produced zero speedup. The cost model must include draft proposal AND
   KV bookkeeping, not just the verify batch.
4. **k beyond the coverage ceiling buys nothing.** k=8 vs k=4 identical on
   repetitive — proposals couldn't fill 8 slots because ngram lookups broke early.

## Interview narrative

> "I implemented draft/verify speculative decoding by hand on Ascend NPU to
> explain a negative production result. Two things decide return: whether the
> draft can even propose (coverage) and how long the accepted streak runs. On
> cyclic output we measured 25% token acceptance but only ~13% iteration coverage
> → +8%; on open text 0% acceptance → −6% pure overhead. That's why vLLM ngram
> spec lost on our chat traffic, and why real deployments use a learned draft
> model (EAGLE/medusa) that keeps both terms high — ngram only pays on retrieval/
> copy-heavy code-like workloads."

## Limits

- bs=1, 0.6B model, eager attention, python-loop orchestration — absolute tok/s
  is dominated by Python overhead; the *relative* accept/overhead structure is the
  finding, not the throughput level.
- Production systems amortize draft cost (small draft LM) and overlap; this study
  isolates the acceptance term.
