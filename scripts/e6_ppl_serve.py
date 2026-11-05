#!/usr/bin/env python3
"""E6 PPL 臂·server 侧：对固定语料逐段算 token 级 NLL（prompt_logprobs）
用法: python3 e6_ppl_serve.py <base_url> <model> <tokenizer_path> <tag> [chunk_tokens]
产出: results/e6/ppl_{tag}.json —— 每段 mean_nll/ppl + 汇总 + top1 预测 token 序列
"""
import json, math, sys, urllib.request

SET = "/data/llm-infra-lab/results/e6/ppl_corpus.txt"
OUT = "/data/llm-infra-lab/results/e6"

def main():
    base, model, tok_path, tag = sys.argv[1:5]
    chunk_n = int(sys.argv[5]) if len(sys.argv) > 5 else 512
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_path)
    corpus = [p.strip() for p in open(SET).read().split("\n\n") if len(p.strip()) > 50]

    chunks = []
    for p in corpus:
        ids = tok(p)["input_ids"]
        for i in range(0, len(ids) - 1, chunk_n):
            seg = ids[i:i + chunk_n]
            if len(seg) >= 64:
                chunks.append(tok.decode(seg))
    print(f"corpus: {len(corpus)} passages -> {len(chunks)} chunks x<={chunk_n}tok", flush=True)

    all_nll, n_tok = [], 0
    per_chunk, top1_seq = [], []
    for ci, text in enumerate(chunks):
        ids = tok(text)["input_ids"]
        body = {"model": model, "prompt": text, "max_tokens": 1,
                "temperature": 0, "prompt_logprobs": 20}
        req = urllib.request.Request(base + "/v1/completions",
            data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            lp = json.load(r)["choices"][0].get("prompt_logprobs") or []
        if len(lp) == len(ids) + 1:  # 服务端多出一个前导位（BOS等）则对齐
            lp = lp[1:]
        nll_c, agree_ids = [], []
        for pos, ent in enumerate(lp):
            if ent is None or pos >= len(ids):
                continue
            actual = str(ids[pos])
            d = {k: v for k, v in ent.items()}
            if actual in d:
                nll_c.append(-d[actual]["logprob"])
            top = [int(k) for k, v in d.items() if v.get("rank") == 1]
            agree_ids.append(top[0] if top else -1)
        if nll_c:
            m = sum(nll_c) / len(nll_c)
            per_chunk.append({"chunk": ci, "n_tokens": len(nll_c),
                              "mean_nll": round(m, 4), "ppl": round(math.exp(m), 2)})
            all_nll += nll_c; n_tok += len(nll_c); top1_seq += agree_ids
        print(f"chunk {ci}: {len(nll_c)} tok", flush=True)

    mean_nll = sum(all_nll) / n_tok
    res = {"model": model, "tag": tag, "n_tokens": n_tok,
           "mean_nll": round(mean_nll, 5), "ppl": round(math.exp(mean_nll), 3),
           "per_chunk": per_chunk, "top1_token_ids": top1_seq}
    fn = f"{OUT}/ppl_{tag}.json"
    json.dump(res, open(fn, "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "top1_token_ids"}, ensure_ascii=False)[:800])
    print("saved:", fn)

if __name__ == "__main__":
    main()
