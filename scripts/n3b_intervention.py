#!/usr/bin/env python3
"""N3b: cache intervention on Qwen3.5-4B hybrid (24x GDN + 8x GQA) — Ascend 910B2.

arXiv 2609.04434 claims two functionally separable memories in hybrid models:
  recurrent state (GDN layers) -> carries "response strategy" (persona / style / format)
  KV cache (attention layers)  -> carries "precise retrieval" (verbatim facts from context)

Test: ~1500-token Chinese prompt = pirate persona (head) + 10 key=value facts (middle,
incl. secret key "4F2C") + two questions (tail). Greedy-decode 100 tokens under 4
conditions, identical prefill each time:
  baseline    : no touch
  drop_state  : zero_() every GDN recurrent_states tensor (conv + attn KV kept)
  drop_kv     : zero_() every attention keys/values (recurrent + conv kept)
  drop_conv   : zero_() every GDN conv_states (recurrent + attn KV kept)

Usage:  python3 n3b_intervention.py [--conds baseline,drop_state,drop_kv,drop_conv]
Output: stdout + /data/results/n3b_intervention.md + /data/results/n3b_intervention.json
"""
import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

M = "/model/Qwen3.5-4B"
DEV = "npu"
OUT_MD = "/data/results/n3b_intervention.md"
OUT_JSON = "/data/results/n3b_intervention.json"
GEN_TOKENS = 100
TARGET_PROMPT_TOK = 1500

SYSTEM = (
    '你是"黑帆号"海盗船的船长，名号"铁须"。从现在起，你的每一句话都必须用海盗黑话腔调：'
    '自称"俺老铁"，称呼对方"老伙计"，句中多用"扬帆""起锚""见鬼""宝贝儿"这类词，句尾常带"哈！"。'
    "绝不许用正常腔调说话，这一条比船上的任何规矩都重要。"
)

USER_HEAD = (
    "老伙计，俺是来接班的。上一任船长临终前把机密清单托付给了俺，"
    "俺现在原样背给你听，请你把每一条都一字不差地刻进脑子里，一会儿俺要考你。\n\n"
    "【机密交接清单】\n"
)

# 10 facts woven into prose; the secret key sits in the middle of the list (2nd of 10)
FACT_STORY = (
    "第一条，船货清单上写着：机柜编号是 A7，那是宝贝儿埋藏点的大门。\n"
    "第二条最要紧：接入密钥是 4F2C，见鬼，这串宝贝儿数字千万别记岔了。\n"
    "第三条：值班日是星期四，每逢星期四老伙计们都要换岗巡夜。\n"
    "第四条：机房楼层在 12 层，宝箱就藏在那一层的西北角落里。\n"
    "第五条：巡检口令是 8Z8Z，报不出这个口令的都当奸细处置。\n"
    "第六条：备用电源是 3 号柴油机组，断帆断电的时候全靠它撑着。\n"
    "第七条：接洽联络人叫老周，是港口那头替咱们望风的眼线。\n"
    "第八条：内网网段是 10.77.0.0/16，船队的暗号全走这条水道。\n"
    "第九条：空调编号是 AC-05，它一停转，宝箱就要受潮发霉。\n"
    "第十条：交接密码本是蓝皮，藏在舵轮下面第二层暗格里。\n"
)

USER_TAIL = (
    "\n现在，老伙计，起锚答话！只回答下面两个问题，别的一句也不许多讲：\n"
    "第一问：接入密钥是什么？\n"
    "第二问：报上你的名号和身份，让俺听听你的腔调还正不正。\n"
)

PAD_SENTS = [
    "夜色沉沉，黑帆号正沿着暗礁水道缓缓东行，桅杆上的风灯被海风吹得明明灭灭。",
    "俺老铁把舵轮交给大副，独自在甲板上数着星星，盘算着这一票宝贝儿该怎么分。",
    "船舱底下的淡水桶咕咚作响，见鬼，这鬼天气连海鸥都懒得叫唤一声。",
    "老伙计们围着火堆擦拭弯刀，哼着走调的水手号子，盼着早日靠岸分金。",
    "海图上用炭笔画了个大叉，标记的就是那处藏宝的废弃机楼，潮涨潮落都要留神。",
    "厨子老三端来一锅咸鱼汤，兄弟们喝得肚皮滚圆，骂骂咧咧又说起了旧航程。",
    "桅顶的瞭望手嚷了一嗓子，说远处有商船的灯火，俺老铁摆摆手叫他们按兵不动。",
    "潮水拍打着船壳，帆索吱呀作响，这一夜就这么一分一秒地熬了过去。",
]

PIRATE_WORDS = ["俺老铁", "老伙计", "扬帆", "起锚", "宝贝儿", "见鬼", "哈！", "黑帆", "铁须", "船长"]
IDENTITY_WORDS = ["铁须", "黑帆", "船长", "海盗"]


def build_prompt(tok):
    """Assemble chat prompt, padding with numbered log-book pages to ~TARGET_PROMPT_TOK."""
    def assemble(n_pages):
        pad = "".join(
            f"航海日志补记第{p}页第{i + 1}则：{s}"
            for p in range(1, n_pages + 1)
            for i, s in enumerate(PAD_SENTS)
        )
        user = USER_HEAD + FACT_STORY + pad + USER_TAIL
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        try:  # Qwen3.5 is a thinking model — skip <think> so the 100-tok budget goes to the answer
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        except (TypeError, KeyError):
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            text += "\n<think>\n\n</think>\n\n"
        return text

    lo, hi = 0, 1
    while len(tok(assemble(hi))["input_ids"]) < TARGET_PROMPT_TOK - 50:
        hi *= 2
    while lo < hi:  # smallest page count reaching the floor
        mid = (lo + hi) // 2
        if len(tok(assemble(mid))["input_ids"]) >= TARGET_PROMPT_TOK - 50:
            hi = mid
        else:
            lo = mid + 1
    text = assemble(lo)
    n_tok = len(tok(text)["input_ids"])
    return text, n_tok, lo


def probe_cache(cache):
    print("[probe] cache type:", type(cache).__name__)
    layout = []
    for i, layer in enumerate(cache.layers):
        desc = {"layer": i, "cls": type(layer).__name__}
        for attr in ("keys", "values", "key_cache", "value_cache",
                     "conv_states", "recurrent_states"):
            if hasattr(layer, attr):
                v = getattr(layer, attr)
                if torch.is_tensor(v):
                    desc[attr] = f"tensor{tuple(v.shape)}"
                elif isinstance(v, dict) and len(v):
                    t0 = next(iter(v.values()))
                    desc[attr] = f"dict[{sorted(v.keys())[:4]}] first=tensor{tuple(t0.shape)}"
        layout.append(desc)
        print("[probe]", json.dumps(desc))
    return layout


def zero_dicts(d):
    n = 0
    if isinstance(d, dict):
        for t in d.values():
            if torch.is_tensor(t):
                t.zero_()
                n += 1
    elif torch.is_tensor(d):
        d.zero_()
        n += 1
    return n


def intervene(cache, cond):
    """Return (n_tensors_zeroed, human summary)."""
    n_rec = n_conv = n_kv = 0
    for layer in cache.layers:
        is_gdn = hasattr(layer, "recurrent_states")
        if cond in ("drop_state", "drop_state_conv") and is_gdn:
            n_rec += zero_dicts(layer.recurrent_states)
        if cond in ("drop_conv",) and is_gdn:
            n_conv += zero_dicts(layer.conv_states)
        if cond == "drop_kv" and not is_gdn:
            for attr in ("keys", "key_cache"):
                if hasattr(layer, attr):
                    for t in (getattr(layer, attr), getattr(layer, "values" if attr == "keys" else "value_cache")):
                        if torch.is_tensor(t):
                            t.zero_()
                            n_kv += 1
                    break
    parts = []
    if n_rec:
        parts.append(f"zeroed {n_rec} GDN recurrent tensors")
    if n_conv:
        parts.append(f"zeroed {n_conv} GDN conv tensors")
    if n_kv:
        parts.append(f"zeroed {n_kv} attn K/V tensors")
    return (n_rec + n_conv + n_kv), "; ".join(parts) or "no-op"


@torch.no_grad()
def run_condition(model, tok, ids, cond, eos_ids):
    t0 = time.time()
    out = model(ids, use_cache=True, logits_to_keep=1)
    prefill_ms = (time.time() - t0) * 1000
    cache = out.past_key_values
    n_zero, summary = intervene(cache, cond)

    gen_ids = []
    nxt = torch.argmax(out.logits[0, -1]).view(1, 1)
    t_dec = time.time()
    for _ in range(GEN_TOKENS):
        gen_ids.append(int(nxt.item()))
        if int(nxt.item()) in eos_ids:
            break
        out = model(nxt, past_key_values=cache, use_cache=True, logits_to_keep=1)
        nxt = torch.argmax(out.logits[0, -1]).view(1, 1)
    dec_ms = (time.time() - t_dec) * 1000

    text = tok.decode(gen_ids, skip_special_tokens=True)
    raw = tok.decode(gen_ids, skip_special_tokens=False)
    chars = [c for c in text if not c.isspace()]
    tris = [tuple(chars[i:i + 3]) for i in range(len(chars) - 2)]
    rep = 1.0 - len(set(tris)) / max(1, len(tris)) if tris else 0.0
    pirate = {w: text.count(w) for w in PIRATE_WORDS if w in text}
    m = {
        "cond": cond,
        "key_hit": "4F2C" in text.upper(),
        "identity_hit": any(w in text for w in IDENTITY_WORDS),
        "tri_repeat_ratio": round(rep, 4),
        "pirate_words": pirate,
        "pirate_total": sum(pirate.values()),
        "gen_len": len(gen_ids),
        "hit_eos": gen_ids and gen_ids[-1] in eos_ids,
        "prefill_ms": round(prefill_ms),
        "decode_ms": round(dec_ms),
        "n_zeroed": n_zero,
        "intervention": summary,
        "text": text,
        "raw": raw,
    }
    print(f"\n===== [{cond}] zeroed={n_zero} ({summary}) key_hit={m['key_hit']} "
          f"identity={m['identity_hit']} tri_rep={m['tri_repeat_ratio']} "
          f"pirate_n={m['pirate_total']} len={m['gen_len']} =====")
    print("RAW:", raw.replace("\n", "\\n"))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", default="baseline,drop_state,drop_kv,drop_conv")
    args = ap.parse_args()
    conds = [c.strip() for c in args.conds.split(",") if c.strip()]

    tok = AutoTokenizer.from_pretrained(M)
    model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16).to(DEV).eval()
    eos_ids = tok.eos_token_id if isinstance(tok.eos_token_id, (list, tuple)) else [tok.eos_token_id]
    print(f"[load] {M} on {DEV}, eos={eos_ids}")

    prompt_text, n_tok, n_pages = build_prompt(tok)
    ids = tok(prompt_text, return_tensors="pt").input_ids.to(DEV)
    print(f"[prompt] {n_tok} tokens, pad_pages={n_pages}")
    print("[prompt tail]", prompt_text[-400:])

    # warmup + structure probe (throwaway tiny cache)
    with torch.no_grad():
        w = model(torch.randint(1000, 50000, (1, 64), device=DEV), use_cache=True, logits_to_keep=1)
    probe_cache(w.past_key_values)
    _ = model(torch.randint(1000, 50000, (1, 1), device=DEV),
              past_key_values=w.past_key_values, use_cache=True, logits_to_keep=1)

    results = []
    for cond in conds:
        results.append(run_condition(model, tok, ids, cond, eos_ids))

    # ---- report ----
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    L = []
    L.append("# N3b: hybrid cache 干预 — Qwen3.5-4B（24×GDN + 8×GQA）\n")
    L.append(f"- 设备：Ascend 910B2（ASCEND_RT_VISIBLE_DEVICES=1，torch 2.9.0 / torch_npu / transformers 5.16.1）")
    L.append(f"- prompt：{n_tok} tok（persona 头部 + 10 条 key=value 事实（含 接入密钥=4F2C）+ 两问；垫 {n_pages} 页航海日志）")
    L.append(f"- decode：greedy，≤{GEN_TOKENS} tok\n")
    L.append("| 条件 | 干预 | 清零张量 | 答出 4F2C | 身份词 | 海盗腔词数 | 3-gram 重复率 | 生成 tok |")
    L.append("|---|---|---:|---|---|---:|---:|---:|")
    for r in results:
        L.append(f"| {r['cond']} | {r['intervention'] or 'no-op'} | {r['n_zeroed']} | "
                 f"{'YES' if r['key_hit'] else 'no'} | {'YES' if r['identity_hit'] else 'no'} | "
                 f"{r['pirate_total']} | {r['tri_repeat_ratio']} | {r['gen_len']} |")
    L.append("")
    for r in results:
        L.append(f"## {r['cond']} — {r['intervention'] or 'no-op'}\n")
        L.append(f"- prefill {r['prefill_ms']} ms / decode {r['decode_ms']} ms / hit_eos={r['hit_eos']}")
        L.append(f"- 海盗腔词命中: {r['pirate_words'] or '无'}\n")
        L.append("```\n" + r["raw"] + "\n```")
    with open(OUT_MD, "w") as f:
        f.write("\n".join(L) + "\n")
    slim = [{k: v for k, v in r.items() if k != "raw"} for r in results]
    with open(OUT_JSON, "w") as f:
        json.dump({"n_prompt_tok": n_tok, "pad_pages": n_pages, "results": slim}, f, ensure_ascii=False, indent=1)
    print(f"\n[N3B] wrote {OUT_MD} + {OUT_JSON}")


if __name__ == "__main__":
    main()
