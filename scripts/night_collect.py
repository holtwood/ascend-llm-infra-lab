#!/usr/bin/env python3
"""Morning collector for the overnight shift: pull results from all four
nodes and build a single SUMMARY.md with the key tables/trends.

Usage: python3 scripts/night_collect.py [--local results/night]
"""
import argparse, json, os, re, subprocess, sys, glob

KEY = os.path.expanduser("~/ascend-lab/keys/npu_nodes")
HOSTS_ENV = os.path.expanduser("~/ascend-lab/hosts.env")
ALIASES = {"m3": "M3", "m1": "M1", "m2": "M2", "d": "D"}


def hosts():
    txt = open(HOSTS_ENV).read()
    out = {}
    for a, u in ALIASES.items():
        pm = re.search(rf"{u}_PORT=(\d+)", txt)
        hm = re.search(rf"{u}_HOST=([\d.]+)", txt)
        if pm and hm:
            out[a] = (hm.group(1), pm.group(1))
    return out


def pull(node, host, port, local):
    d = os.path.join(local, node)
    os.makedirs(d, exist_ok=True)
    subprocess.run(["scp", "-i", KEY, "-o", "StrictHostKeyChecking=no",
                    "-P", port, f"root@{host}:/data/llm-infra-lab/results/night/*",
                    d + "/"], capture_output=True)
    return d


def load_jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    for line in open(path, errors="ignore"):
        line = line.strip()
        if line.startswith("{"):
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def mean(xs):
    return round(sum(xs) / len(xs), 1) if xs else None


def trend(xs, k=10):
    if len(xs) < 2 * k:
        return None
    head, tail = xs[:k], xs[-k:]
    return mean(head), mean(tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", default="results/night")
    args = ap.parse_args()
    H = hosts()
    lines = ["# 通宵实验汇总（自动生成）\n"]
    for node, (host, port) in H.items():
        d = pull(node, host, port, args.local)
        summ = load_jsonl(os.path.join(d, "summary.jsonl"))
        samples = load_jsonl(os.path.join(d, "samples.jsonl"))
        fatigue = load_jsonl(os.path.join(d, "fatigue.jsonl"))
        lines.append(f"\n## {node.upper()} — {len(summ)} bench 点 / {len(samples)} 采样\n")

        clu = [s for s in summ if s.get("tag", "").startswith("clu_w") and "tok_s" in s]
        if clu:
            lines.append(f"- **集群段**: {len(clu)} 波，单机均值 {mean([c['tok_s'] for c in clu])} tok/s"
                         f"（4 机聚合 ≈ 4×），TTFT {mean([c['ttft'] for c in clu])}ms，"
                         f"错误 {sum(c.get('err', 0) for c in clu)}\n")
            trc = trend([c["tok_s"] for c in clu])
            if trc:
                lines.append(f"- 集群段漂移：首10 {trc[0]} → 末10 {trc[1]} tok/s\n")

        soak = [s for s in summ if s.get("tag", "").startswith(("soak_", "dsoak_", "ctxsoak_"))
                and "tok_s" in s]
        if soak:
            tr = trend([s["tok_s"] for s in soak])
            errs = sum(s.get("err", 0) for s in soak)
            lines.append(f"- **soak**: {len(soak)} 波，吞吐首10均值 {tr[0] if tr else '-'} → "
                         f"末10均值 {tr[1] if tr else '-'} tok/s，累计错误 {errs}\n")
            ttr = trend([s["ttft"] for s in soak])
            if ttr:
                lines.append(f"- soak TTFT 漂移：{ttr[0]} → {ttr[1]} ms\n")

        grid = [s for s in summ if s.get("tag", "").startswith(("g_", "mnb", "out")) and "tok_s" in s]
        if grid:
            lines.append("\n| tag | tok/s | TTFT | TPOT | ok |\n|---|---|---|---|---|\n")
            for s in sorted(grid, key=lambda x: x["tag"]):
                lines.append(f"| {s['tag']} | {s['tok_s']} | {s['ttft']} | {s['tpot']} | {s['ok']} |\n")

        ctx = [s for s in summ if s.get("tag", "").startswith(("ctx", "wall_")) and "tok_s" in s]
        if ctx:
            lines.append("\n| tag | tok/s | TTFT | TPOT | ok |\n|---|---|---|---|---|\n")
            for s in sorted(ctx, key=lambda x: x["tag"]):
                lines.append(f"| {s['tag']} | {s['tok_s']} | {s['ttft']} | {s['tpot']} | {s['ok']} |\n")

        quant = [s for s in summ if s.get("tag", "").startswith("q_") and "tok_s" in s]
        if quant:
            lines.append("\n| tag | tok/s | TTFT | TPOT |\n|---|---|---|---|\n")
            for s in sorted(quant, key=lambda x: x["tag"]):
                lines.append(f"| {s['tag']} | {s['tok_s']} | {s['ttft']} | {s['tpot']} |\n")

        if fatigue:
            sleeps = [f["sleep_ms"] for f in fatigue]
            wakes = [f["wake_ms"] for f in fatigue]
            reqs = [f["req_ms"] for f in fatigue]
            tr = trend(sleeps, k=min(10, len(sleeps) // 2) or 1)
            lines.append(f"\n- **sleep/wake 疲劳**: {len(fatigue)} 轮，sleep {mean(sleeps)}ms，"
                         f"wake {mean(wakes)}ms，首请求 {mean(reqs)}ms\n")
            if tr:
                lines.append(f"- 漂移：sleep 首10 {tr[0]}ms → 末10 {tr[1]}ms\n")
        if samples:
            kv = [float(s["kv"]) for s in samples if s.get("kv") not in (None, "", "None")]
            pre = [float(s["pre"]) for s in samples if s.get("pre") not in (None, "", "None")]
            if kv:
                lines.append(f"- KV 占用峰值 {max(kv)*100:.1f}%，抢占累计 {max(pre) if pre else 0:.0f}\n")

    out = os.path.join(args.local, "SUMMARY.md")
    os.makedirs(args.local, exist_ok=True)
    open(out, "w").write("".join(lines))
    print("written:", out)


if __name__ == "__main__":
    main()
