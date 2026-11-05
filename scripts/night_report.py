#!/usr/bin/env python3
"""Build a draft results/night/README.md from the collected JSONL files.

Cross-node aggregation the collector does not do:
  * cluster phase: per-node steady-state mean (last 20% of waves) and the
    aggregate = sum over nodes (they ran concurrently), plus drift
  * M1 soak drift, M2 grids, M3 ctx/wall/fatigue, D quant matrix
Usage: python3 scripts/night_report.py [--local results/night]
"""
import argparse, glob, json, os, statistics as st


def load_jsonl(path):
    out, buf = [], ""
    if not os.path.exists(path):
        return out
    for raw in open(path, errors="ignore"):
        line = raw.rstrip("\n").strip()
        if not buf and not line.startswith("{"):
            continue
        buf = (buf + " " + line) if buf else line
        try:
            out.append(json.loads(buf)); buf = ""
        except Exception:
            if len(buf) > 4000:
                buf = ""
    return out


def mean(xs):
    return round(st.mean(xs), 1) if xs else None


def tag_group(summ, prefixes):
    return [s for s in summ if s.get("tag", "").startswith(prefixes) and "tok_s" in s]


def drift(xs, k=10):
    if len(xs) < 2 * k:
        return None
    return mean(xs[:k]), mean(xs[-k:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", default="results/night")
    args = ap.parse_args()
    nodes = sorted([d for d in glob.glob(os.path.join(args.local, "*")) if os.path.isdir(d)])
    data = {}
    for d in nodes:
        name = os.path.basename(d)
        data[name] = {
            "summ": load_jsonl(os.path.join(d, "summary.jsonl")),
            "samples": load_jsonl(os.path.join(d, "samples.jsonl")),
            "fatigue": load_jsonl(os.path.join(d, "fatigue.jsonl")),
        }
    L = ["# 通宵实验（自动草稿）\n",
         "> 由 `scripts/night_report.py` 从各节点 JSONL 生成；数字为实测，"
         "未跑完/跳过的项在文末列出。\n"]

    # ---- cluster phase ----
    L.append("\n## 一、集群段：4 卡协同（四机同为客户端 → D:8080）\n")
    L.append("| 节点 | 波数 | 稳态单机 tok/s | TTFT | TPOT | 首10→末10 漂移 | 错误 |\n")
    L.append("|---|---|---|---|---|---|---|\n")
    steady = []
    for n, d in data.items():
        w = tag_group(d["summ"], "clu_w")
        if not w:
            continue
        tail = w[int(len(w) * 0.8):] or w
        sm = mean([x["tok_s"] for x in tail])
        steady.append(sm or 0)
        dr = drift([x["tok_s"] for x in w])
        L.append(f"| {n} | {len(w)} | **{sm}** | {mean([x['ttft'] for x in w])} | "
                 f"{mean([x['tpot'] for x in w])} | "
                 f"{dr[0]} → {dr[1]} | {sum(x.get('err', 0) for x in w)} |\n")
    if steady:
        L.append(f"\n**聚合（稳态）= {round(sum(steady), 1)} tok/s**，"
                 f"≈ {round(sum(steady) / max(1, len(steady)), 1)} × 单机。\n")

    # ---- M1 soak ----
    L.append("\n## 二、M1 长稳 soak（c16/n512 连续）\n")
    d = data.get("m1", {})
    soak = tag_group(d.get("summ", []), "soak_w")
    if soak:
        dr = drift([x["tok_s"] for x in soak])
        dttft = drift([x["ttft"] for x in soak])
        L.append(f"- {len(soak)} 波，吞吐 {dr[0]} → {dr[1]} tok/s（末10 vs 首10），"
                 f"TTFT {dttft[0]} → {dttft[1]} ms\n")
        L.append(f"- 累计错误 {sum(x.get('err', 0) for x in soak)}，"
                 f"TPOT {mean([x['tpot'] for x in soak])} ms\n")
    else:
        L.append("- 无数据（未到独立段或失败）\n")

    # ---- M2 grids ----
    L.append("\n## 三、M2 性能网格\n")
    d = data.get("m2", {})
    for title, pref in (("3D 网格 c×in×out", "g_"), ("mnb 扫描", "mnb"),
                        ("输出长度扫描", "out")):
        rows = tag_group(d.get("summ", []), pref)
        if not rows:
            continue
        L.append(f"\n### {title}\n\n| tag | tok/s | TTFT | TPOT | ok |\n|---|---|---|---|---|\n")
        for r in sorted(rows, key=lambda x: x["tag"]):
            L.append(f"| {r['tag']} | {r['tok_s']} | {r['ttft']} | {r['tpot']} | {r['ok']} |\n")

    # ---- M3 ctx + fatigue + wall ----
    L.append("\n## 四、M3 长上下文 / 换租疲劳 / 准入墙\n")
    d = data.get("m3", {})
    ctx = tag_group(d.get("summ", []), "ctx")
    if ctx:
        L.append("\n### ctx × 并发\n\n| tag | tok/s | TTFT | TPOT |\n|---|---|---|---|\n")
        for r in sorted(ctx, key=lambda x: x["tag"]):
            L.append(f"| {r['tag']} | {r['tok_s']} | {r['ttft']} | {r['tpot']} |\n")
    wall = tag_group(d.get("summ", []), "wall_")
    if wall:
        L.append("\n### 准入墙（ctx16k）\n\n| tag | tok/s | TTFT | TPOT |\n|---|---|---|---|\n")
        for r in sorted(wall, key=lambda x: x["tag"]):
            L.append(f"| {r['tag']} | {r['tok_s']} | {r['ttft']} | {r['tpot']} |\n")
    fat = d.get("fatigue", [])
    if fat:
        sleeps = [f["sleep_ms"] for f in fat]
        wakes = [f["wake_ms"] for f in fat]
        reqs = [f["req_ms"] for f in fat]
        k = max(1, min(10, len(fat) // 2))
        L.append(f"\n### sleep/wake 疲劳（{len(fat)} 轮）\n\n"
                 f"- sleep {mean(sleeps)}ms（首{k} {mean(sleeps[:k])} → 末{k} {mean(sleeps[-k:])}）\n"
                 f"- wake {mean(wakes)}ms（首{k} {mean(wakes[:k])} → 末{k} {mean(wakes[-k:])}）\n"
                 f"- 唤醒后首请求 {mean(reqs)}ms\n")

    # ---- D quant ----
    L.append("\n## 五、D 量化 × 模型规模矩阵\n")
    d = data.get("d", {})
    q = tag_group(d.get("summ", []), "q_")
    if q:
        L.append("\n| tag | tok/s | TTFT | TPOT |\n|---|---|---|---|\n")
        for r in sorted(q, key=lambda x: x["tag"]):
            L.append(f"| {r['tag']} | {r['tok_s']} | {r['ttft']} | {r['tpot']} |\n")
    ds = tag_group(d.get("summ", []), "dsoak_w")
    if ds:
        dr = drift([x["tok_s"] for x in ds])
        L.append(f"\nD soak：{len(ds)} 波，{dr[0]} → {dr[1]} tok/s\n")

    L.append("\n## 六、失败 / 跳过项\n\n"
             "（收尾时人工补：serve 失败、显存不足、线路中断等）\n")

    out = os.path.join(args.local, "README.md")
    open(out, "w").write("".join(L))
    print("draft written:", out)


if __name__ == "__main__":
    main()
