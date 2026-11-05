#!/usr/bin/env python3
"""Cluster orchestrator for the 4-NPU lab (stdlib only, runs on the local box).

Why this exists: with no RDMA between nodes (E18), the only valid ways to
use all four NPUs together are (a) four replicas behind a gateway and
(b) running experiments *in parallel* across nodes instead of serially on
one. This tool implements (b) plus utilization sampling so every run shows
what the NPUs were actually doing.

Modes
-----
  util                        sample AICore%/HBM for N seconds on all nodes
  bench-all                   same bench_openai.py run on every node against
                              its local :8000, started simultaneously
  fanout --spec spec.json     per-node commands: {"m3": "...", "m1": "..."}
  collect --dir DIR --glob G  scp result files back from every node

Node aliases: m3, m1, m2, d (parsed from ~/ascend-lab/hosts.env).
"""
import argparse, json, os, re, subprocess, sys, threading, time

HOSTS_ENV = os.path.expanduser("~/ascend-lab/hosts.env")
KEY = os.path.expanduser("~/ascend-lab/keys/npu_nodes")
ALIASES = {"m3": ("M3_PORT", "M3_HOST"), "m1": ("M1_PORT", "M1_HOST"),
           "m2": ("M2_PORT", "M2_HOST"), "d": ("D_PORT", "D_HOST")}
ENVSRC = "source /data/llm-infra-lab/scripts/env.sh 2>/dev/null;"


def load_hosts():
    txt = open(HOSTS_ENV).read()
    out = {}
    for alias, (pk, hk) in ALIASES.items():
        pm = re.search(rf"{pk}=(\d+)", txt)
        hm = re.search(rf"{hk}=([\d.]+)", txt)
        if pm and hm:
            out[alias] = (hm.group(1), pm.group(1))
    return out


HOSTS = load_hosts()


def ssh(alias, cmd, timeout=1800, check=False):
    host, port = HOSTS[alias]
    full = ["ssh", "-i", KEY, "-o", "StrictHostKeyChecking=no",
            "-o", "BatchMode=yes", "-p", port, f"root@{host}", cmd]
    r = subprocess.run(full, capture_output=True, text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"[{alias}] rc={r.returncode}: {r.stderr[-400:]}")
    return r


def sample_util(aliases, seconds, interval=2.0, stop_event=None):
    """Background sampler: AICore% + HBM per node every `interval` s.

    Stops after `seconds` or when `stop_event` is set by the caller."""
    samples = {a: [] for a in aliases}
    stop = stop_event or threading.Event()

    def loop(a):
        while not stop.is_set():
            r = ssh(a, ENVSRC + " npu-smi info", timeout=30)
            m = re.search(r"\|\s*(\d+)\s*\|[^|]*\|\s*(\d+)\s+(\d+)\s*/\s*(\d+)\s+(\d+)\s*/\s*(\d+)",
                          r.stdout)
            if m:
                samples[a].append({"t": round(time.time(), 1),
                                   "aicore": int(m.group(2)),
                                   "hbm_mb": int(m.group(5))})
            time.sleep(interval)

    ths = [threading.Thread(target=loop, args=(a,), daemon=True) for a in aliases]
    [t.start() for t in ths]
    deadline = time.time() + seconds
    while time.time() < deadline and not stop.is_set():
        time.sleep(0.5)
    stop.set()
    [t.join(timeout=8) for t in ths]
    out = {}
    for a, xs in samples.items():
        if xs:
            ac = [x["aicore"] for x in xs]
            out[a] = {"n": len(xs), "aicore_mean": round(sum(ac)/len(ac), 1),
                      "aicore_max": max(ac), "hbm_mb_last": xs[-1]["hbm_mb"]}
    return out


BENCH = ("cd /data/llm-infra-lab && /usr/local/python3.12.13/bin/python3 "
         "scripts/bench_openai.py --base-url http://127.0.0.1:{port} "
         "--endpoint /v1/chat/completions --model qwen35-4b "
         "--concurrency {c} --num-requests {n} --input-tokens {i} "
         "--output-tokens {o} --out {out}")

# D serves its replica on 8010 (8000 is free there); the NPU nodes use 8000.
NODE_PORT = {"d": 8010}


def run_parallel(tasks):
    """tasks: {alias: shell_cmd}. Runs all concurrently, returns {alias: (rc, out)}."""
    res = {}

    def go(a, cmd):
        r = ssh(a, ENVSRC + " " + cmd)
        res[a] = (r.returncode, r.stdout)

    ths = [threading.Thread(target=go, args=(a, c)) for a, c in tasks.items()]
    t0 = time.time()
    [t.start() for t in ths]
    [t.join() for t in ths]
    return res, time.time() - t0


def parse_bench(out):
    try:
        d = json.loads(out[out.index("{"):])
        return {k: d.get(k) for k in ("wall_s", "ttft_mean_ms", "ttft_p99_ms",
                                      "tpot_mean_ms", "tpot_p99_ms",
                                      "output_tok_per_s", "completed", "errors")}
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["util", "bench-all", "fanout", "collect"])
    ap.add_argument("--seconds", type=int, default=10)
    ap.add_argument("--nodes", default="m3,m1,m2,d")
    ap.add_argument("--c", type=int, default=8)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--i", type=int, default=256)
    ap.add_argument("--o", type=int, default=96)
    ap.add_argument("--outdir", default="/data/llm-infra-lab/results/cluster")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--spec", help="fanout spec json {alias: cmd}")
    ap.add_argument("--dir", help="collect: remote dir")
    ap.add_argument("--glob", default="*.json")
    ap.add_argument("--local", help="collect: local dir")
    args = ap.parse_args()
    nodes = args.nodes.split(",")

    if args.mode == "util":
        print(json.dumps(sample_util(nodes, args.seconds), indent=1))
        return

    if args.mode == "bench-all":
        prep = f"mkdir -p {args.outdir}"
        outs = {a: f"{args.outdir}/{args.tag}_{a}.json" for a in nodes}
        tasks = {a: f"{prep}; " + BENCH.format(c=args.c, n=args.n, i=args.i,
                                               o=args.o, out=outs[a],
                                               port=NODE_PORT.get(a, 8000))
                 for a in nodes}
        sampler, stop_ev = {}, threading.Event()

        def _sample():
            sampler.update(sample_util(nodes, 600, stop_event=stop_ev))

        th = threading.Thread(target=_sample, daemon=True)
        th.start()
        res, wall = run_parallel(tasks)
        stop_ev.set()
        th.join(timeout=30)
        rep = {"tag": args.tag, "wall_s": round(wall, 1), "per_node": {},
               "util": sampler}
        for a in nodes:
            rc, out = res.get(a, (1, ""))
            rep["per_node"][a] = {"rc": rc, "bench": parse_bench(out)}
        print(json.dumps(rep, indent=1))
        return

    if args.mode == "fanout":
        spec = json.load(open(args.spec))
        res, wall = run_parallel(spec)
        print(json.dumps({"wall_s": round(wall, 1),
                          "per_node": {a: {"rc": rc, "tail": out[-300:]}
                                       for a, (rc, out) in res.items()}}, indent=1))
        return

    if args.mode == "collect":
        os.makedirs(args.local, exist_ok=True)
        for a in nodes:
            host, port = HOSTS[a]
            subprocess.run(["scp", "-i", KEY, "-o", "StrictHostKeyChecking=no",
                            "-P", port,
                            f"root@{host}:{args.dir}/{args.glob}", args.local])
        print("collected ->", args.local)


if __name__ == "__main__":
    main()
