#!/usr/bin/env python3
"""E6 QA 精度评测：自构造题库 -> OpenAI completions API -> 规则打分
用法: python3 e6_qa_eval.py <base_url> <model> <tag> [concurrency]
产出: results/e6/qa_{tag}.json
"""
import json, re, sys, urllib.request, concurrent.futures as cf

SET = "/data/llm-infra-lab/results/e6/qa_set.jsonl"
OUT = "/data/llm-infra-lab/results/e6"

MCQ_FMT = "以下是一道单项选择题，请直接回答正确选项的字母，不要输出其他内容。\n\n{q}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n\n答案："
NUM_FMT = "请解答下面的数学题，只输出最终数值答案，不要输出过程。\n\n{q}\n\n答案："

def call(base, model, prompt, max_tokens):
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
            "temperature": 0}
    req = urllib.request.Request(base + "/v1/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)["choices"][0].get("text", "")

def pred_mcq(text):
    t = text.strip().upper()
    m = re.search(r"[A-D]", t)
    return m.group(0) if m else None

def pred_num(text):
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None

def run_item(base, model, it):
    if it["type"] == "mcq":
        prompt = MCQ_FMT.format(q=it["q"], **it["choices"])
        text = call(base, model, prompt, 8)
        pred = pred_mcq(text)
        ok = (pred == it["ans"])
    else:
        prompt = NUM_FMT.format(q=it["q"])
        text = call(base, model, prompt, 24)
        pred = pred_num(text)
        try:
            ok = pred is not None and abs(float(pred) - float(it["ans"])) < 1e-4
        except ValueError:
            ok = False
    return {"id": it["id"], "cat": it["cat"], "ans": it["ans"],
            "pred": pred, "ok": ok, "gen": text.strip()[:200]}

def main():
    base, model, tag = sys.argv[1], sys.argv[2], sys.argv[3]
    conc = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    items = [json.loads(l) for l in open(SET)]
    results = []
    with cf.ThreadPoolExecutor(max_workers=conc) as ex:
        futs = {ex.submit(run_item, base, model, it): it for it in items}
        done = 0
        for f in cf.as_completed(futs):
            results.append(f.result()); done += 1
            if done % 20 == 0:
                print(f"progress {done}/{len(items)}", flush=True)
    by_cat, by_all = {}, {"n": 0, "ok": 0}
    for r in results:
        c = by_cat.setdefault(r["cat"], {"n": 0, "ok": 0})
        c["n"] += 1; c["ok"] += int(r["ok"]); by_all["n"] += 1; by_all["ok"] += int(r["ok"])
    acc = {k: f"{v['ok']}/{v['n']}={v['ok']/v['n']:.1%}" for k, v in sorted(by_cat.items())}
    acc["ALL"] = f"{by_all['ok']}/{by_all['n']}={by_all['ok']/by_all['n']:.1%}"
    out = {"model": model, "tag": tag, "accuracy": acc, "items": results}
    fn = f"{OUT}/qa_{tag}.json"
    json.dump(out, open(fn, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(acc, ensure_ascii=False, indent=1))
    print("saved:", fn)

if __name__ == "__main__":
    main()
