#!/usr/bin/env python3
"""L3: continuous batching engine on real hardware (NPU/CPU).

Combines the verified pieces of this repo into one runnable engine:
  - scheduler.py      : token-budget + KV-reservation scheduling (CPU semantics)
  - model_qwen35.py   : numerically-verified hybrid forward (24 GDN + 8 attn)
  - N2b evidence      : decode batching is ~free (host-bound), BUT left-padding
                        breaks GDN semantics -> this engine NEVER pads prompts.

Design: a fixed pool of MAX_B slots. Each slot owns (in big pooled tensors)
its attention K/V rows, its GDN conv tail + recurrent S, and its own seq_len.
Every engine step runs TWO forwards over slot subsets:
  1. prefill sub-batch: equal-length bucket of waiting prompt chunks (bucketed
     so no padding exists anywhere — varlen/cu_seqlens is the L4/L5 route);
  2. decode sub-batch: ALL running slots, one token each, heterogeneous
     history lengths handled by a per-slot (B,1,1,L) attention mask.
The gather/scatter copies in _AttnSlot.read() are deliberately naive — their
cost is exactly the L4 paged-KV motivation (block tables remove them).

Run correctness self-check (CPU, tiny random model):
    python3 examples/engine_np/batch_engine.py --selftest

KNOWN ISSUE (NPU, environment reclamation deadline hit before resolution):
the heterogeneous-decode path (per-slot (B,1,1,L) mask) fails on torch_npu
2.9.0 in two distinct ways depending on mask format:
  (a) bool mask  -> aclnnFlashAttentionScore inner error, and wrapping the
      call in sdpa_kernel([SDPBackend.MATH]) does NOT bypass it (the NPU
      dispatcher appears not to honor the backend override);
  (b) additive float mask -> aclnnInplaceAdd broadcast error "[1,16,1,256]
      vs [1,1,1,257]" inside SDPA — and (c) manually expanding KV / dropping
      enable_gqa does NOT clear it, so the fault is not the gqa path either.
      257 = max_len+1 and the mismatch surviving fix (c) suggest the async
      stack is misleading (aclnnInplaceAdd may be fired by an earlier op);
      the real culprit needs ASCEND_LAUNCH_BLOCKING=1 to pin down.
CPU selftest passes fully (batched==single, all mask paths). Untried:
(d) custom attention math (no SDPA at all) for the decode path, (e) rerun
demo with ASCEND_LAUNCH_BLOCKING=1 for the truthful stack.
"""

import argparse
import torch

from model_qwen35 import Qwen35Config, Qwen35ForCausalLM
from scheduler import ContinuousBatchScheduler, Sequence


class _AttnSlot:
    """Attention-layer view over pooled rows for the active slot subset."""

    def __init__(self, pool_k, pool_v, seq_lens, slots):
        self.pk, self.pv, self.seq_lens, self.slots = pool_k, pool_v, seq_lens, slots

    def append(self, k_new, v_new):
        t = k_new.shape[2]
        for i, s in enumerate(self.slots):                 # per-slot positions
            pos = int(self.seq_lens[s])
            self.pk[s, :, pos:pos + t] = k_new[i].to(self.pk.dtype)
            self.pv[s, :, pos:pos + t] = v_new[i].to(self.pv.dtype)
            self.seq_lens[s] += t

    def read(self):
        idx = torch.tensor(self.slots, device=self.pk.device)
        lens = self.seq_lens[idx]
        L = int(lens.max())
        k = self.pk[idx][:, :, :L]                         # naive gather copy:
        v = self.pv[idx][:, :, :L]                         # L4 paged KV removes it
        valid = (torch.arange(L, device=lens.device)[None, :]
                 < lens[:, None])                          # (B_sub, L) True=real
        return k, v, valid


class _GDNSlot:
    """GDN-layer view: conv tail + recurrent S for the active slot subset."""

    def __init__(self, conv_pool, state_pool, slots, kernel, phase):
        self.conv_pool, self.state_pool = conv_pool, state_pool
        self.slots, self.kernel, self.phase = slots, kernel, phase

    @property
    def has_history(self):
        return self.phase == "decode"                      # prefill/decode never mix

    def conv_input(self, mixed):
        # mixed: (B_sub, conv_dim, T). Store the tail NOW (decode consumes it
        # immediately), return cat(state, mixed) for the conv window.
        idx = torch.tensor(self.slots, device=mixed.device)
        hist = self.conv_pool[idx].to(mixed.dtype)         # (B_sub, conv_dim, K)
        self.conv_pool[idx] = mixed[:, :, -self.kernel:].to(self.conv_pool.dtype)
        return torch.cat([hist, mixed], dim=2)

    def read_state(self):
        idx = torch.tensor(self.slots, device=self.state_pool.device)
        return self.state_pool[idx]                        # fp32 (B_sub,H,dk,dv)

    def write_state(self, fs):
        idx = torch.tensor(self.slots, device=fs.device)
        self.state_pool[idx] = fs.to(self.state_pool.dtype)


class SlotHybridCaches:
    """Pooled caches for all layers over MAX_B slots. `begin(slots, phase)`
    binds a subset for one forward; model accesses caches.layers[i] and the
    per-layer objects expose append/read (attention) and conv_input /
    read_state / write_state (GDN) exactly as model_qwen35 expects."""

    def __init__(self, config: Qwen35Config, max_b: int, max_len: int,
                 device: str, dtype=torch.bfloat16):
        self.config, self.max_b, self.seq_len = config, max_b, 0
        self.slot_seq_lens = torch.zeros(max_b, dtype=torch.long, device=device)
        conv_dim = config.linear_key_dim * 2 + config.linear_value_dim
        self._pools = {}                                   # layer -> pooled tensors
        for i in range(config.num_hidden_layers):
            if config.layer_types[i] == "full_attention":
                pk = torch.zeros(max_b, config.num_key_value_heads, max_len,
                                 config.head_dim, device=device, dtype=dtype)
                self._pools[i] = (pk, torch.zeros_like(pk))          # (B,H,L,D)
            else:
                conv = torch.zeros(max_b, conv_dim,
                                   config.linear_conv_kernel_dim,
                                   device=device, dtype=torch.float32)
                st = torch.zeros(max_b, config.linear_num_value_heads,
                                 config.linear_key_head_dim,
                                 config.linear_value_head_dim,
                                 device=device, dtype=torch.float32)
                self._pools[i] = (conv, st)
        self.layers = [None] * config.num_hidden_layers

    def begin(self, slots: list, phase: str) -> "SlotHybridCaches":
        att_ids = {i for i, p in self._pools.items() if p[0].dim() == 4}
        for i, pool in self._pools.items():
            if i in att_ids:
                self.layers[i] = _AttnSlot(pool[0], pool[1], self.slot_seq_lens, slots)
            else:
                self.layers[i] = _GDNSlot(pool[0], pool[1], slots,
                                          self.config.linear_conv_kernel_dim, phase)
        self.slots, self.phase = slots, phase
        if slots:
            sidx = torch.tensor(slots, device=self.slot_seq_lens.device)
            self.seq_len = int(self.slot_seq_lens[sidx].max())
        else:
            self.seq_len = 0                               # model reads it for
        return self                                        # default position_ids


class BatchEngine:
    """Scheduler + slotted batch forward. Greedy; tracks last token per seq."""

    def __init__(self, model, config, max_b=16, max_len=2048, device="cpu",
                 dtype=torch.bfloat16):
        self.model, self.cfg, self.dev = model, config, device
        self.max_b = max_b
        self.caches = SlotHybridCaches(config, max_b, max_len, device, dtype)
        self.sch = ContinuousBatchScheduler(kv_blocks=(max_b * max_len) // 16)
        self.slot_of = {}                                  # seq_id -> pool slot
        self.free_slots = list(range(max_b))
        self.next_tok = {}                                 # seq_id -> token id
        self.prompts = {}                                  # seq_id -> tensor
        self._n_sub = 0

    def submit(self, prompt: list, max_new: int, arrival: int = 0):
        s = Sequence(self._n_sub, prompt_len=len(prompt),
                     max_new_tokens=max_new, arrival_step=arrival)
        self._n_sub += 1
        self.sch.waiting.append(s)
        self.prompts[s.seq_id] = torch.tensor(prompt, device=self.dev)

    def run(self, max_steps=1000):
        done, t = {}, 0
        total = len(self.sch.waiting)
        while len(done) < total and t < max_steps:
            t += 1
            info = self.sch.step()
            self._prefill_phase(info)
            self._decode_phase(info, done)
        return done, t

    def _seqs(self):
        return {s.seq_id: s for s in list(self.sch.waiting) + list(self.sch.running)}

    def _prefill_phase(self, info):
        seqs = self._seqs()
        # (seq_id, take): scheduler already advanced num_cached by `take`.
        # Equal-take bucketing => equal tensor widths => zero padding.
        by_take = {}
        for sid, take in info["prefill"]:
            s = seqs[sid]
            end = s.num_cached
            if sid not in self.slot_of:
                if not self.free_slots:
                    continue                               # slot starved; skip
                self.slot_of[sid] = self.free_slots.pop(0)
            by_take.setdefault(take, []).append((s, end))

        for take, group in by_take.items():
            slots = [self.slot_of[s.seq_id] for s, _ in group]
            ids = torch.stack([self.prompts[s.seq_id][end - take:end]
                               for s, end in group])
            pos = torch.stack([torch.arange(end - take, end, device=self.dev)
                               for s, end in group])
            self.caches.begin(slots, "prefill")
            with torch.no_grad():
                logits = self.model(ids, caches=self.caches, position_ids=pos,
                                    logits_to_keep=1)
            for (s, _), n in zip(group, logits[:, -1].argmax(-1).tolist()):
                self.next_tok[s.seq_id] = n

    def _decode_phase(self, info, done):
        if not info["decode_n"]:
            return
        group = [s for s in self.sch.running if s.seq_id in self.next_tok
                 and s.seq_id in self.slot_of]
        if not group:
            return
        slots = [self.slot_of[s.seq_id] for s in group]
        ids = torch.tensor([[self.next_tok[s.seq_id]] for s in group],
                           device=self.dev)
        pos = torch.tensor([[s.num_cached + s.num_generated - 1] for s in group],
                           device=self.dev)                # token being generated
        self.caches.begin(slots, "decode")
        with torch.no_grad():
            logits = self.model(ids, caches=self.caches, position_ids=pos,
                                logits_to_keep=1)
        for s, n in zip(group, logits[:, -1].argmax(-1).tolist()):
            self.next_tok[s.seq_id] = n
            if s.done:
                done[s.seq_id] = n
                self.free_slots.append(self.slot_of.pop(s.seq_id))


# =============================================================================
# selftest: tiny random model — heterogeneous batch path == single path
# =============================================================================
def build_tiny(device):
    c = Qwen35Config(
        vocab_size=512, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, layer_types=["linear_attention", "full_attention"],
        num_attention_heads=2, num_key_value_heads=1, head_dim=32,
        rope_theta=1e6,
        linear_num_value_heads=2, linear_num_key_heads=1,
        linear_key_head_dim=16, linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
    )
    torch.manual_seed(0)
    m = Qwen35ForCausalLM(c).to(device).to(torch.float32).eval()
    return m, c


def selftest():
    dev = "cpu"
    model, cfg = build_tiny(dev)
    prompts = [[1, 2, 3, 4, 5], [9, 8, 7, 6, 5, 4, 3], [11, 12, 13, 14]]

    def single(prompt, max_new):
        e = BatchEngine(model, cfg, max_b=1, max_len=64, device=dev,
                        dtype=torch.float32)
        e.submit(prompt, max_new)
        done, _ = e.run(max_steps=64)
        return done

    refs = [single(p, 6)[0] for p in prompts]              # each run's seq_id=0

    e = BatchEngine(model, cfg, max_b=4, max_len=64, device=dev,
                    dtype=torch.float32)
    for p in prompts:
        e.submit(p, max_new=6)
    done, steps = e.run(max_steps=64)
    assert len(done) == 3, f"expected 3 finished, got {len(done)}: {done}"
    for i in range(3):
        assert done[i] == refs[i], \
            f"seq {i}: batched last-token {done[i]} != single {refs[i]}"
    print(f"[selftest] 3 heterogeneous prompts ({[len(p) for p in prompts]} tok) "
          f"batched==single per-request, finished in {steps} engine steps  OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
