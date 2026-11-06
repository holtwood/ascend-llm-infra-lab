#!/usr/bin/env python3
"""N9 part3: KV-offload correctness demo — Qwen3.5-4B hybrid DynamicCache roundtrip (card 1).

Timeline:
  1. request A prefill (1600 tok)            -> cache_stay
  2. stay branch: greedy decode 32 tok       -> ref_tokens          (the no-offload truth)
  3. request A prefill again (same prompt)   -> cache_off           (identical prefill, E83-verified determinism)
  4. OFFLOAD: every cache tensor .cpu() (pageable) — KV + conv + GDN recurrent (the "S-matrix" state)
  5. HBM actually freed? check torch.npu.memory_allocated()
  6. run request B (different prompt, prefill+decode) in the freed memory
  7. RESTORE: tensors back to npu, time it ("swap-in stall")
  8. decode 32 tok from restored cache       -> back_tokens; compare with ref_tokens bit-exact
  9. pinned-buffer offload/restore timing on request B's cache (production-style DMA path)

Usage:  python3 n9_offload_demo.py
Output: stdout + /data/results/n9_offload_demo.md
"""
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

M = "/model/Qwen3.5-4B"
DEV = "npu"
OUT_MD = "/data/results/n9_offload_demo.md"
N_DECODE = 32
N_DECODE_B = 16

SYSTEM = ('你是"黑帆号"海盗船的船长，名号"铁须"。每一句话都必须用海盗黑话腔调：'
          '自称"俺老铁"，称呼对方"老伙计"，多用"扬帆""起锚""宝贝儿"这类词。')
FACTS = (
    "【机密交接清单】\n"
    "第一条：机柜编号是 A7。\n第二条：接入密钥是 4F2C。\n第三条：值班日是星期四。\n"
    "第四条：机房楼层在 12 层。\n第五条：巡检口令是 8Z8Z。\n第六条：备用电源是 3 号柴油机组。\n"
    "第七条：联络人是老周。\n第八条：网段是 10.77.0.0/16。\n第九条：空调编号是 AC-05。\n"
    "第十条：交接密码本是蓝皮。\n\n"
)
PAD = ("航海日志补记：夜色沉沉，黑帆号正沿着暗礁水道缓缓东行，桅杆上的风灯被海风吹得明明灭灭。"
       "俺老铁把舵轮交给大副，独自在甲板上数着星星，盘算着这一票宝贝儿该怎么分。"
       "船舱底下的淡水桶咕咚作响，见鬼，这鬼天气连海鸥都懒得叫唤一声。"
       "老伙计们围着火堆擦拭弯刀，哼着走调的水手号子，盼着早日靠岸分金。")
TAIL_A = ("\n现在起锚答话：第一问，接入密钥是什么？第二问，报上你的名号和身份。\n")
USER_B = "老伙计，用一句话说说今晚的海况。"


def build_ids(tok, user, target_tok=0):
    """Assemble chat ids; pad with accumulated log-book pages until target_tok reached (0 = no pad)."""
    pages = []
    ids = None
    for page in range(1, 60):
        pages.append(f"航海日志第{page}页：{PAD}\n")
        user_full = FACTS + ("".join(pages) if target_tok else "") + user
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_full}]
        try:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        except (TypeError, KeyError):
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            text += "\n<think>\n\n</think>\n\n"
        ids = tok(text, return_tensors="pt").input_ids
        if target_tok == 0 or ids.shape[1] >= target_tok - 40:
            break
    return ids.to(DEV)


def offload(cache):
    """Move every cache tensor to pageable host memory by attribute reassignment."""
    t0 = time.perf_counter()
    n = 0
    for layer in cache.layers:
        if hasattr(layer, "recurrent_states"):
            layer.conv_states = {k: v.cpu() for k, v in layer.conv_states.items()}
            layer.recurrent_states = {k: v.cpu() for k, v in layer.recurrent_states.items()}
            n += len(layer.conv_states) + len(layer.recurrent_states)
        else:
            layer.keys = layer.keys.cpu()
            layer.values = layer.values.cpu()
            n += 2
    torch.npu.synchronize()
    return time.perf_counter() - t0, n


def cache_bytes(cache):
    total = 0
    dtypes = set()
    for layer in cache.layers:
        if hasattr(layer, "recurrent_states"):
            for t in list(layer.conv_states.values()) + list(layer.recurrent_states.values()):
                total += t.numel() * t.element_size()
                dtypes.add(str(t.dtype))
        else:
            for t in (layer.keys, layer.values):
                total += t.numel() * t.element_size()
                dtypes.add(str(t.dtype))
    return total, sorted(dtypes)


def restore(cache):
    """Bring every cache tensor back to npu (copies from pinned or pageable host transparently)."""
    t0 = time.perf_counter()
    for layer in cache.layers:
        if hasattr(layer, "recurrent_states"):
            layer.conv_states = {k: v.to(DEV, non_blocking=False) for k, v in layer.conv_states.items()}
            layer.recurrent_states = {k: v.to(DEV, non_blocking=False) for k, v in layer.recurrent_states.items()}
        else:
            layer.keys = layer.keys.to(DEV, non_blocking=False)
            layer.values = layer.values.to(DEV, non_blocking=False)
    torch.npu.synchronize()
    return time.perf_counter() - t0


@torch.no_grad()
def decode(model, cache, first_logits, n):
    toks = []
    nxt = torch.argmax(first_logits[0, -1]).view(1, 1)
    t0 = time.perf_counter()
    for _ in range(n):
        toks.append(int(nxt.item()))
        out = model(nxt, past_key_values=cache, use_cache=True, logits_to_keep=1)
        nxt = torch.argmax(out.logits[0, -1]).view(1, 1)
    ms = (time.perf_counter() - t0) * 1000
    return toks, ms


@torch.no_grad()
def prefill(model, ids):
    out = model(ids, use_cache=True, logits_to_keep=1)
    return out.past_key_values, out.logits


def to_pinned(t):
    buf = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
    buf.copy_(t)
    torch.npu.synchronize()
    return buf


def main():
    tok = AutoTokenizer.from_pretrained(M)
    model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16).to(DEV).eval()
    print(f"[load] {M} on {DEV}", flush=True)

    ids_a = build_ids(tok, TAIL_A, target_tok=1500)
    ids_b = build_ids(tok, USER_B)
    print(f"[prompts] A={ids_a.shape[1]} tok, B={ids_b.shape[1]} tok", flush=True)

    L = ["# N9 part3: DynamicCache 换出/换回正确性 demo（Qwen3.5-4B hybrid，含 GDN recurrent state）\n"]

    # --- stay branch: the no-offload truth ---
    cache_stay, lg_stay = prefill(model, ids_a)
    ref, ref_ms = decode(model, cache_stay, lg_stay, N_DECODE)
    print(f"[stay] decoded {len(ref)} tok in {ref_ms:.0f} ms", flush=True)

    # --- offload branch: identical prefill, then evict ---
    cache_off, lg_off = prefill(model, ids_a)
    mem_before = torch.npu.memory_allocated()
    off_ms, n_tensors = offload(cache_off)
    mem_after = torch.npu.memory_allocated()
    payload, dtypes = cache_bytes(cache_off)
    print(f"[offload] {n_tensors} tensors, {payload/1e6:.1f} MB ({dtypes}) in {off_ms*1000:.0f} ms; "
          f"HBM {mem_before/1e6:.0f} -> {mem_after/1e6:.0f} MB "
          f"(freed {(mem_before-mem_after)/1e6:.1f} MB)", flush=True)

    # --- request B runs in the freed HBM ---
    cache_b, lg_b = prefill(model, ids_b)
    b_toks, b_ms = decode(model, cache_b, lg_b, N_DECODE_B)
    b_text = tok.decode(b_toks, skip_special_tokens=True)
    print(f"[reqB] decoded {len(b_toks)} tok in {b_ms:.0f} ms: {b_text[:80]}", flush=True)
    del cache_b, lg_b

    # --- restore + verify ---
    restore_ms = restore(cache_off)
    print(f"[restore] back on npu in {restore_ms*1000:.0f} ms", flush=True)
    back, back_ms = decode(model, cache_off, lg_off, N_DECODE)
    match = ref == back
    first_div = next((i for i, (a, b) in enumerate(zip(ref, back)) if a != b), None)
    print(f"[verify] ref==back: {match} (first divergence at {first_div})", flush=True)
    print(f"[verify] ref  text: {tok.decode(ref, skip_special_tokens=True)[:120]}", flush=True)
    print(f"[verify] back text: {tok.decode(back, skip_special_tokens=True)[:120]}", flush=True)

    # --- pinned (production-style) offload/restore timing on a fresh B cache ---
    cache_b2, _ = prefill(model, ids_b)
    for layer in cache_b2.layers:
        if hasattr(layer, "recurrent_states"):
            layer.conv_states = {k: to_pinned(v) for k, v in layer.conv_states.items()}
            layer.recurrent_states = {k: to_pinned(v) for k, v in layer.recurrent_states.items()}
        else:
            layer.keys = to_pinned(layer.keys)
            layer.values = to_pinned(layer.values)
    torch.npu.synchronize()
    t0 = time.perf_counter()
    restore_pinned_ms = restore(cache_b2)
    del cache_b2

    # --- report ---
    L.append(f"- 模型 Qwen3.5-4B（24×GDN + 8×attn），prompt A {ids_a.shape[1]} tok / prompt B {ids_b.shape[1]} tok，greedy")
    L.append(f"- 换出 payload：{n_tensors} 个张量，{payload/1e6:.1f} MB，dtype {dtypes}"
             f"（attn KV bf16 + GDN conv/recurrent state 一起交接）")
    L.append(f"- 换出（pageable .cpu()）{off_ms*1000:.0f} ms；HBM 释放 {mem_before/1e6:.0f}→{mem_after/1e6:.0f} MB"
             f"（实释 {(mem_before-mem_after)/1e6:.1f} MB）")
    L.append(f"- 请求 B 在腾出的 HBM 里 prefill+decode 正常完成（{b_ms:.0f} ms，{N_DECODE_B} tok）：「{b_text[:60]}」")
    L.append(f"- 换入（restore .to('npu')）{restore_ms*1000:.0f} ms"
             f"（≈ {restore_ms*1000/(back_ms/N_DECODE):.1f} 个 decode 步）；pinned 路径 restore {restore_pinned_ms*1000:.0f} ms（对照）")
    verdict = ("**BIT-EXACT**：换出→跑 B→换回后的 greedy 32 tok 与不换出逐 token 一致"
               if match else
               f"**DIVERGED**：首个分叉在第 {first_div} token")
    L.append(f"- 正确性：ref vs back = {match}，{verdict}")
    L.append(f"- decode 32 tok：stay {ref_ms:.0f} ms vs roundtrip {back_ms:.0f} ms")
    L.append("\n```\nref : " + tok.decode(ref, skip_special_tokens=True).replace("\n", "\\n")
             + "\nback: " + tok.decode(back, skip_special_tokens=True).replace("\n", "\\n") + "\n```")
    with open(OUT_MD, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"[N9] wrote {OUT_MD}", flush=True)


if __name__ == "__main__":
    main()
