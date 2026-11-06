#!/usr/bin/env python3
"""N2b: batch-dimension scan on Qwen3.5-4B hybrid (Ascend 910B2, card 1).

Interview question: why does decode scale ~free with batching while prefill doesn't?
Theory (N2/N1b): decode step = 2935 op launches, 79% host-bound, device busy only 24ms
-> GEMV(GEM) as B grows, weights read once per step -> step time ~flat, tok/s ~linear.
Prefill is compute-bound -> time scales ~linearly with B.

Per B in {1,2,4,8,16,32}:
  - decode : 30 greedy steps after the common prefill, steady-state p50 ms/step
             (skip first 5), aggregate tok/s, amortized ms/tok
  - prefill: 256-tok prompt x B, 5 reps p50
  - correctness (two groups):
      equal-length  : all slots carry the identical prompt, no pad -> strongest check,
                      every slot must equal the B=1 reference bit-for-bit
      left-padded   : slot i gets i left-pad tokens + same prompt (mask 0 on pads).
                      Hybrid GDN layers have NO attention-mask concept: pads enter the
                      conv/recurrent recurrence (and right-side completion pads enter it
                      during prefill). If outputs diverge -> that IS the finding.

Usage:  python3 n2b_batch_scan.py
Output: stdout + /data/results/n2b_batch_scan.md + /data/results/n2b_batch_scan.json
"""
import json
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

M = "/model/Qwen3.5-4B"
DEV = "npu"
BATCHES = [1, 2, 4, 8, 16, 32]
N_STEPS = 30
SKIP = 5
PREFILL_REPS = 5
OUT_MD = "/data/results/n2b_batch_scan.md"
OUT_JSON = "/data/results/n2b_batch_scan.json"

SYSTEM = '你是「黑帆号」海盗船的船长，名号「铁须」。回答必须用海盗黑话腔调，自称「俺老铁」，称呼对方「老伙计」，多用「扬帆」「起锚」「宝贝儿」这类词，句尾常带「哈！」。'
USER = (
    "【航行简报】俺老铁的船队今夜出港：机柜编号是 A7，接入密钥是 4F2C，值班日是星期四，"
    "机房楼层在 12 层，巡检口令是 8Z8Z，备用电源是 3 号柴油机组，联络人是老周，网段是 10.77.0.0/16。"
    "航海日志补记：夜色沉沉，黑帆号正沿着暗礁水道缓缓东行，桅杆上的风灯被海风吹得明明灭灭。"
    "俺老铁把舵轮交给大副，独自在甲板上数着星星，盘算着这一票宝贝儿该怎么分。"
    "船舱底下的淡水桶咕咚作响，见鬼，这鬼天气连海鸥都懒得叫唤一声。"
    "老伙计们围着火堆擦拭弯刀，哼着走调的水手号子，盼着早日靠岸分金。"
    "海图上用炭笔画了个大叉，标记的就是那处藏宝的废弃机楼，潮涨潮落都要留神。"
    "现在老伙计，起锚答话：请用两句话说明今晚的航行计划，并报出接入密钥。"
)


def build_base_ids(tok):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}]
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except (TypeError, KeyError):
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        text += "\n<think>\n\n</think>\n\n"
    ids = tok(text, return_tensors="pt").input_ids.to(DEV)
    return ids


def make_batch(base, pad_counts, pad_id):
    """Row i = pad_counts[i] left pads + prompt + right-side completion pads, equal total length."""
    L = base.shape[1]
    maxlen = max(pad_counts) + L
    rows, masks = [], []
    for p in pad_counts:
        left = torch.full((1, p), pad_id, dtype=base.dtype, device=DEV)
        row = torch.cat([left, base], dim=1)
        if row.shape[1] < maxlen:
            right = torch.full((1, maxlen - row.shape[1]), pad_id, dtype=base.dtype, device=DEV)
            row = torch.cat([row, right], dim=1)
        rows.append(row)
        masks.append([0] * p + [1] * L + [0] * (maxlen - p - L))
    ids = torch.cat(rows, dim=0)
    mask = torch.tensor(masks, dtype=torch.long, device=DEV)
    return ids, mask


@torch.no_grad()
def bench_decode(model, ids, mask, B):
    cache_out = model(ids, attention_mask=mask, use_cache=True, logits_to_keep=1)
    cache = cache_out.past_key_values
    toks = torch.argmax(cache_out.logits[:, -1, :], dim=-1).view(B, 1)
    m = mask
    times, gens = [], [toks.clone()]
    for _ in range(N_STEPS):
        t0 = time.perf_counter()
        out = model(toks, attention_mask=m, past_key_values=cache,
                    use_cache=True, logits_to_keep=1)
        torch.npu.synchronize()
        times.append(time.perf_counter() - t0)
        m = torch.cat([m, torch.ones(B, 1, dtype=m.dtype, device=DEV)], dim=1)
        toks = torch.argmax(out.logits[:, -1, :], dim=-1).view(B, 1)
        gens.append(toks.clone())
    del cache, cache_out
    gen = torch.cat(gens, dim=1)  # (B, N_STEPS+1) including the first post-prefill token
    steady = sorted(times[SKIP:])
    return steady[len(steady) // 2] * 1000, gen


@torch.no_grad()
def bench_prefill(model, ids, mask):
    times = []
    for _ in range(PREFILL_REPS):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        out = model(ids, attention_mask=mask, use_cache=True, logits_to_keep=1)
        torch.npu.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
        del out
    return statistics.median(times)


def main():
    tok = AutoTokenizer.from_pretrained(M)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16).to(DEV).eval()
    base = build_base_ids(tok)
    print(f"[load] {M} on {DEV}, base prompt {base.shape[1]} tok, pad_id={pad_id}", flush=True)

    rows = []
    ref = None
    for B in BATCHES:
        r = {"B": B}

        # equal-length group: performance + strongest consistency check
        ids0, mask0 = make_batch(base, [0] * B, pad_id)
        dec_ms, gen0 = bench_decode(model, ids0, mask0, B)
        if B == 1:
            ref = gen0[0]
        r["decode_ms"] = round(dec_ms, 1)
        r["tok_s"] = round(B * 1000 / dec_ms, 0)
        r["amortized_ms"] = round(dec_ms / B, 2)
        r["eq_match_all_slots"] = bool(all(torch.equal(gen0[i], ref) for i in range(B)))

        # prefill scaling
        r["prefill_ms"] = round(bench_prefill(model, ids0, mask0), 1)

        # left-padded group: does left padding preserve per-slot semantics?
        if B > 1:
            ids1, mask1 = make_batch(base, list(range(B)), pad_id)
            _, gen1 = bench_decode(model, ids1, mask1, B)
            bad = [i for i in range(B) if not torch.equal(gen1[i], ref)]
            r["pad_match_all_slots"] = not bad
            r["pad_mismatch_slots"] = bad[:8]
            if bad:
                i = bad[0]
                div = next(j for j in range(ref.numel()) if gen1[i, j] != ref[j])
                r["pad_first_div_slot"] = i
                r["pad_first_div_step"] = div
            del ids1, mask1
        else:
            r["pad_match_all_slots"] = None

        rows.append(r)
        print(f"[B={B:2d}] decode {r['decode_ms']:7.1f} ms/step | {r['tok_s']:7.0f} tok/s | "
              f"amort {r['amortized_ms']:6.2f} ms | prefill {r['prefill_ms']:7.1f} ms | "
              f"eq_match={r['eq_match_all_slots']} pad_match={r['pad_match_all_slots']}", flush=True)
        del ids0, mask0, gen0
        torch.npu.empty_cache()

    b1 = rows[0]
    L = [f"# N2b: batch 维度扫描 — Qwen3.5-4B hybrid（Ascend 910B2 卡1）\n",
         f"- 统一 prompt {base.shape[1]} tok（海盗简报），greedy；decode 30 步取稳态 p50（跳过前 5 步）；prefill 5 次 p50",
         f"- 参照：b1 decode {b1['decode_ms']} ms/步（与 N1b 的 89ms/步同量级）；prefill 单条 {b1['prefill_ms']} ms\n"]
    L.append("| B | decode ms/步 | 每步涨 vs b1 | 聚合 tok/s | 吞吐 vs b1 | 均摊 ms/tok | prefill ms | prefill vs b1 | 等长输出一致 | 左pad输出一致 |")
    L.append("|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for r in rows:
        L.append(f"| {r['B']} | {r['decode_ms']} | +{(r['decode_ms']/b1['decode_ms']-1)*100:.0f}% | "
                 f"{r['tok_s']:.0f} | {r['tok_s']/b1['tok_s']:.2f}× | {r['amortized_ms']} | "
                 f"{r['prefill_ms']} | {r['prefill_ms']/b1['prefill_ms']:.2f}× | "
                 f"{'YES' if r['eq_match_all_slots'] else 'NO'} | "
                 f"{'—' if r['pad_match_all_slots'] is None else ('YES' if r['pad_match_all_slots'] else 'NO ' + str(r.get('pad_mismatch_slots', [])))} |")

    last = rows[-1]
    L.append(f"\n## 结论要点\n")
    L.append(f"- decode：b1→b{last['B']} 每步 {b1['decode_ms']}→{last['decode_ms']} ms"
             f"（+{(last['decode_ms']/b1['decode_ms']-1)*100:.0f}%），聚合吞吐 {b1['tok_s']:.0f}→{last['tok_s']:.0f} tok/s"
             f"（{last['tok_s']/b1['tok_s']:.1f}×）——接近免费午餐的{'实锤' if last['tok_s']/b1['tok_s'] > 0.7*last['B'] else '偏离（见下）'}。")
    L.append(f"- prefill：b1→b{last['B']} {b1['prefill_ms']}→{last['prefill_ms']} ms"
             f"（{last['prefill_ms']/b1['prefill_ms']:.1f}×，理想线性 {last['B']}×）——compute-bound，batch 无白嫖。")
    pad_rows = [r for r in rows if r.get("pad_match_all_slots") is False]
    if pad_rows:
        r = pad_rows[0]
        L.append(f"- **left-padding 语义检查失败**（最小 B={r['B']} 即出现，首个分叉槽 {r.get('pad_first_div_slot')} 第 "
                 f"{r.get('pad_first_div_step')} 步）：hybrid 的 GDN 层没有 attention-mask 概念，"
                 "左 pad 进入 conv 滑窗与 recurrent 状态递推（prefill 阶段右侧补齐 pad 同样污染状态），"
                 "attention_mask 只挡住全注意力 8 层。对纯 Transformer 这是位置对齐小坑，对 hybrid 是语义性破坏——"
                 "batch 化 serving 必须 varlen/无 pad 路径（vLLM-ascend 混合模型走 cu_seqlens 即为此因）。")
    else:
        L.append("- left-padding 语义检查全部通过：padding 侧槽位输出与 b1 逐 token 一致。")
    if not all(r["eq_match_all_slots"] for r in rows):
        L.append("- 等长组一致性也有失败（见上表），batch 数值等价本身破裂，需查 attention mask 路径。")
    else:
        L.append("- 等长组（无 pad）全 B 逐 token 一致：batch 数值等价成立，性能数字可比。")

    with open(OUT_MD, "w") as f:
        f.write("\n".join(L) + "\n")
    with open(OUT_JSON, "w") as f:
        json.dump({"base_prompt_tok": base.shape[1], "rows": rows}, f, ensure_ascii=False, indent=1)
    print(f"[N2B] wrote {OUT_MD} + {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
