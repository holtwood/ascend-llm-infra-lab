#!/usr/bin/env python3
"""E6 QA·选项 logprob 打分臂（serve/CPU 统一方法，供 W8A8 vs BF16 同法对比）
对每道 MCQ：计算 P(option_letter | prompt+答案：)，取 argmax 为预测。
- serve 模式: python3 e6_qa_score.py serve <base_url> <model> <tokenizer_path> <tag>
  每题 4 请求（prompt+" A"/" B"/" C"/" D"），读 prompt_logprobs 末位
- cpu 模式:   TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 e6_qa_score.py cpu <model_path> x <tag>
  每题 1 次 forward，比较 4 个字母 token 的 logit
产出: results/e6/qascore_{tag}.json（仅 mcq 题；num 题无法单 token 打分，跳过）
"""
import json, sys

SET = "/data/llm-infra-lab/results/e6/qa_set.jsonl"
OUT = "/data/llm-infra-lab/results/e6"
MCQ_FMT = "以下是一道单项选择题，请直接回答正确选项的字母，不要输出其他内容。\n\n{q}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n\n答案："
LETTERS = ["A", "B", "C", "D"]


def letter_ids(tok):
    """' A'/' B'/' C'/' D' 的首 token id（Qwen 系分词字母带前导空格单 token）"""
    ids = []
    for ch in LETTERS:
        e = tok(" " + ch)["input_ids"]
        ids.append(e[-1] if len(e) == 1 else tok(ch)["input_ids"][-1])
    return ids


def run_serve(base, model, tok_path, tag):
    import urllib.request
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_path)
    lids = {str(i): ch for i, ch in zip(letter_ids(tok), LETTERS)}
    items = [json.loads(l) for l in open(SET) if json.loads(l)["type"] == "mcq"]
    results = []
    for n, it in enumerate(items):
        prompt = MCQ_FMT.format(q=it["q"], **it["choices"])
        scores = {}
        for ch in LETTERS:
            body = {"model": model, "prompt": prompt + " " + ch,
                    "max_tokens": 1, "temperature": 0, "prompt_logprobs": 5}
            req = urllib.request.Request(base + "/v1/completions",
                data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            lp = json.load(urllib.request.urlopen(req, timeout=120))["choices"][0].get("prompt_logprobs") or []
            last = lp[-1] or {}
            lid = str(dict((v, k) for k, v in lids.items())[ch])
            scores[ch] = last.get(lid, {}).get("logprob")
        pred = max(scores, key=lambda c: scores[c] if scores[c] is not None else -1e9)
        results.append({"id": it["id"], "cat": it["cat"], "ans": it["ans"],
                        "pred": pred, "ok": pred == it["ans"],
                        "letter_logprobs": scores})
        if (n + 1) % 20 == 0:
            print(f"progress {n+1}/{len(items)}", flush=True)
    _save(model, tag, results)


def run_cpu(model_path, tag):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    lids = letter_ids(tok)
    torch.set_num_threads(int(__import__("os").environ.get("TORCH_THREADS", "160")))
    print("loading bf16 on CPU ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.eval()
    print("loaded", flush=True)
    items = [json.loads(l) for l in open(SET) if json.loads(l)["type"] == "mcq"]
    results = []
    for n, it in enumerate(items):
        prompt = MCQ_FMT.format(q=it["q"], **it["choices"])
        x = torch.tensor([tok(prompt)["input_ids"]])
        with torch.no_grad():
            logits = model(input_ids=x).logits[0, -1]
        lp = torch.log_softmax(logits.float(), dim=-1)
        scores = {ch: round(lp[i].item(), 4) for ch, i in zip(LETTERS, lids)}
        pred = max(scores, key=scores.get)
        results.append({"id": it["id"], "cat": it["cat"], "ans": it["ans"],
                        "pred": pred, "ok": pred == it["ans"],
                        "letter_logprobs": scores})
        if (n + 1) % 10 == 0:
            print(f"progress {n+1}/{len(items)}", flush=True)
        json.dump({"model": model_path, "tag": tag, "partial": True,
                   "items": results}, open(f"{OUT}/qascore_{tag}.json", "w"),
                  ensure_ascii=False, indent=1)
    _save(model_path, tag, results)


def _save(model, tag, results):
    by_cat, alln, allok = {}, 0, 0
    for r in results:
        c = by_cat.setdefault(r["cat"], [0, 0])
        c[0] += 1; c[1] += int(r["ok"]); alln += 1; allok += int(r["ok"])
    acc = {k: f"{v[1]}/{v[0]}={v[1]/v[0]:.1%}" for k, v in sorted(by_cat.items())}
    acc["ALL"] = f"{allok}/{alln}={allok/alln:.1%}"
    out = {"model": model, "tag": tag, "method": "letter_logprob_argmax",
           "accuracy": acc, "items": results}
    fn = f"{OUT}/qascore_{tag}.json"
    json.dump(out, open(fn, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(acc, ensure_ascii=False, indent=1))
    print("saved:", fn)


if __name__ == "__main__":
    if sys.argv[1] == "serve":
        run_serve(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    elif sys.argv[1] == "cpu":
        run_cpu(sys.argv[2], sys.argv[4])
