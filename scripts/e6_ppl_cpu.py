#!/usr/bin/env python3
"""E6 PPL 臂·CPU 参照侧：bf16 权重 transformers 纯 CPU forward 算 token 级 NLL
用法: TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 e6_ppl_cpu.py <model_path> <tag> [chunk_tokens]
产出: results/e6/ppl_{tag}.json —— 同 serve 侧格式（含 top1_token_ids 供一致性对比）
注意: 27B bf16 ~71G，M1 内存 2TB 可装；192 核 Kunpeng forward 较慢，nohup 跑
"""
import json, math, sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SET = "/data/llm-infra-lab/results/e6/ppl_corpus.txt"
OUT = "/data/llm-infra-lab/results/e6"

def main():
    model_path, tag = sys.argv[1], sys.argv[2]
    chunk_n = int(sys.argv[3]) if len(sys.argv) > 3 else 512
    torch.set_num_threads(int(__import__("os").environ.get("TORCH_THREADS", "160")))
    tok = AutoTokenizer.from_pretrained(model_path)
    print("loading bf16 on CPU ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.eval()
    print("loaded", flush=True)

    corpus = [p.strip() for p in open(SET).read().split("\n\n") if len(p.strip()) > 50]
    chunks = []
    for p in corpus:
        ids = tok(p)["input_ids"]
        for i in range(0, len(ids) - 1, chunk_n):
            seg = ids[i:i + chunk_n]
            if len(seg) >= 64:
                chunks.append(ids[i:i + chunk_n])
    print(f"{len(chunks)} chunks", flush=True)

    all_nll, per_chunk, top1_seq = [], [], []
    for ci, ids in enumerate(chunks):
        x = torch.tensor([ids])
        with torch.no_grad():
            logits = model(input_ids=x).logits[0]  # [T, V]
        lp = torch.log_softmax(logits.float(), dim=-1)
        nll, tops = [], []
        for pos in range(len(ids) - 1):
            nll.append(-lp[pos, ids[pos + 1]].item())
            tops.append(int(lp[pos].argmax()))
        m = sum(nll) / len(nll)
        per_chunk.append({"chunk": ci, "n_tokens": len(nll),
                          "mean_nll": round(m, 4), "ppl": round(math.exp(m), 2)})
        all_nll += nll; top1_seq += tops
        print(f"chunk {ci}: {len(nll)} tok nll={m:.4f}", flush=True)
        json.dump({"model": model_path, "tag": tag, "partial": True,
                   "per_chunk": per_chunk, "top1_token_ids": top1_seq},
                  open(f"{OUT}/ppl_{tag}.json", "w"), ensure_ascii=False, indent=1)

    mean_nll = sum(all_nll) / len(all_nll)
    res = {"model": model_path, "tag": tag, "n_tokens": len(all_nll),
           "mean_nll": round(mean_nll, 5), "ppl": round(math.exp(mean_nll), 3),
           "per_chunk": per_chunk, "top1_token_ids": top1_seq}
    fn = f"{OUT}/ppl_{tag}.json"
    json.dump(res, open(fn, "w"), ensure_ascii=False, indent=1)
    print(f"DONE mean_nll={mean_nll:.4f} ppl={math.exp(mean_nll):.2f} -> {fn}")

if __name__ == "__main__":
    main()
