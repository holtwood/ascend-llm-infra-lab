#!/usr/bin/env python3
"""E23: client-count sweep — is the ceiling the injector or the replicas?

Fixed aggregate concurrency, varying number of client *processes*:
  1 client  c128   (D)
  2 clients c64 x2 (D + M3)
  4 clients c32 x4 (D + M3 + M1 + M2)
...same total load, more processes. If aggregate throughput rises with
process count at fixed concurrency, the single-process client (aiohttp
event loop + GIL) is the bottleneck, not the four replicas.

Also samples each client process's CPU% (ps) during the run — direct
evidence of where the injection cost lands.

Runs locally, drives all nodes over ssh (same pattern as cluster_bench).
"""
import argparse, json, re, subprocess, threading, time, os

KEY = os.path.expanduser("~/ascend-lab/keys/npu_nodes")
HOSTS_ENV = os.path.expanduser("~/ascend-lab/hosts.env")
ALIASES = {"m3": ("M3_PORT", "M3_HOST"), "m1": ("M1_PORT", "M1_HOST"),
           "m2": ("M2_PORT", "M2_HOST"), "d": ("D_PORT", "D_HOST")}


def load_hosts():
    txt = open(HOSTS_ENV).read()
    out = {}
    for a, (pk, hk) in ALIASES.items():
        pm = re.search(rf"{pk}=(\d+)", txt); hm = re.search(rf"{hk}=([\d.]+)", txt)
        if pm and hm:
            out[a] = (hm.group(1), pm.group(1))
    return out


HOSTS = load_hosts()
BENCH = ("cd /data/llm-infra-lab && /usr/local/python3.12.13/bin/python3 "
         "scripts/bench_openai.py --base-url http://127.0.0.1:8080 "
         "--endpoint /v1/chat/completions --model qwen35-4b "
         "--concurrency {c} --num-requests {n} --input-tokens 256 "
         "--output-tokens 96 --out {out}")


def ssh(a, cmd, timeout=900):
    host, port = HOSTS[a]
    return subprocess.run(["ssh", "-i", KEY, "-o", "StrictHostKeyChecking=no",
                           "-o", "BatchMode=yes", "-p", port, f"root@{host}", cmd],
                          capture_output=True, text=True, timeout=timeout)


def sample_cpu(aliases, stop_ev, out):
    while not stop_ev.is_set():
        for a in aliases:
            r = ssh(a, "ps -C python3 -o %cpu=,args= | grep bench_openai | "
                       "awk '{s+=$1} END {print s+0}'", timeout=20)
            try:
                out.setdefault(a, []).append(float(r.stdout.strip() or 0))
            except ValueError:
                pass
        time.sleep(1.0)


def run_arm(tag, plan, c, n):
    """plan: list of (alias, per_client_concurrency)."""
    stop_ev = threading.Event(); cpu = {}
    sampler = threading.Thread(target=sample_cpu,
                               args=(list({a for a, _ in plan}), stop_ev, cpu),
                               daemon=True)
    sampler.start()
    t0 = time.time()
    ths = []
    for a, cc in plan:
        out = f"/data/llm-infra-lab/results/cluster/e23_{tag}_{a}.json"
        cmd = BENCH.format(c=cc, n=n, out=out)
        ths.append(threading.Thread(target=ssh, args=(a, cmd)))
    [t.start() for t in ths]
    [t.join() for t in ths]
    wall = time.time() - t0
    stop_ev.set(); sampler.join(timeout=10)

    agg = 0.0; per = {}
    for a, cc in plan:
        r = ssh(a, f"cat /data/llm-infra-lab/results/cluster/e23_{tag}_{a}.json")
        try:
            d = json.loads(r.stdout[r.stdout.index("{"):])["summary"]
            per[a] = {"c": cc, "tok_s": d["output_tok_per_s"],
                      "ttft": d["ttft_mean_ms"], "tpot": d["tpot_mean_ms"],
                      "ok": d["completed"], "err": d["errors"]}
            agg += d["output_tok_per_s"]
        except Exception:
            per[a] = {"c": cc, "error": r.stdout[-120:]}
    rep = {"tag": tag, "n_clients": len(plan), "wall_s": round(wall, 1),
           "aggregate_tok_s": round(agg, 1), "per_client": per,
           "client_cpu_pct": {a: {"mean": round(sum(v)/len(v), 1), "max": max(v)}
                              for a, v in cpu.items() if v}}
    print(json.dumps(rep, indent=1))
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/debian/ascend-lab/repo/ascend-llm-infra-lab/results/e23")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    reps = []
    # aggregate c128: 1x128 / 2x64 / 4x32
    reps.append(run_arm("c128_1x", [("d", 128)], 128, 256))
    reps.append(run_arm("c128_2x", [("d", 64), ("m3", 64)], 64, 128))
    reps.append(run_arm("c128_4x", [("d", 32), ("m3", 32), ("m1", 32), ("m2", 32)], 32, 64))
    # aggregate c256: 1x256 / 2x128 / 4x64
    reps.append(run_arm("c256_1x", [("d", 256)], 256, 512))
    reps.append(run_arm("c256_2x", [("d", 128), ("m3", 128)], 128, 256))
    reps.append(run_arm("c256_4x", [("d", 64), ("m3", 64), ("m1", 64), ("m2", 64)], 64, 128))
    json.dump(reps, open(f"{args.out}/e23_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
