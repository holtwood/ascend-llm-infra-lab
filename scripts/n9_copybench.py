#!/usr/bin/env python3
"""N9 part1+2: host<->NPU copy bandwidth microbench + KV offload crossover ledger (card 1).

Bandwidth: npu<->cpu, sizes 64/256/1024/4096 MB, 50 reps, p10/p50/p90.
  - pageable host buffer (torch.empty)   vs  pin_memory buffer
NUMA: card-1 domain is node0 (cores 0-23), card-0 domain is node6 (144-167);
container cpuset is 8-49,155-175 -> right=8..23, wrong=155..167. D2H pinned 1GB re-run
under both affinities to price the cross-NUMA tax.

Ledger: with KV=32 KB/token (N3), prefill 2765 tok/s (N2), inter-NPU 16.3 GB/s (N4),
compare offload-to-host roundtrip vs recompute vs cross-card migration for 256MB..8GB.

Usage:  python3 n9_copybench.py --bench && python3 n9_copybench.py --ledger
Output: /data/results/n9_copybench.json + /data/results/n9_ledger.md
"""
import argparse
import json
import os
import subprocess
import sys
import time

import torch

DEV = "npu"
MB = 1024 * 1024
SIZES_MB = [64, 256, 1024, 4096]
REPS = 50
WARMUP = 3
OUT_JSON = "/data/results/n9_copybench.json"
OUT_MD = "/data/results/n9_ledger.md"

# constants from earlier experiments
KV_BYTES_PER_TOK = 32 * 1024          # N3: hybrid KV+GDN state, 32 KB/token
PREFILL_TPS = 2765                    # N2: prefill throughput
INTER_NPU_BW = 16.3e9                 # N4: card-to-card 16.3 GB/s


def set_affinity(cores):
    allowed = os.sched_getaffinity(0)
    target = set(cores) & allowed
    os.sched_setaffinity(0, target)
    return sorted(target)


def copy_times(src, dst, reps):
    for _ in range(WARMUP):
        dst.copy_(src, non_blocking=False)
    torch.npu.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        dst.copy_(src, non_blocking=False)
        torch.npu.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    nbytes = src.numel() * src.element_size()
    return {
        "p10_gbs": nbytes / ts[int(len(ts) * 0.10)] / 1e9,
        "p50_gbs": nbytes / ts[len(ts) // 2] / 1e9,
        "p90_gbs": nbytes / ts[int(len(ts) * 0.90)] / 1e9,
        "p50_ms": ts[len(ts) // 2] * 1e3,
    }


def run_sweep(tag):
    rows = []
    for mb in SIZES_MB:
        nbytes = mb * MB
        t_npu = torch.empty(nbytes, dtype=torch.uint8, device=DEV)
        t_page = torch.empty(nbytes, dtype=torch.uint8)
        t_pin = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        r = {"size_mb": mb}
        r["d2h_pageable"] = copy_times(t_npu, t_page, REPS)
        r["d2h_pinned"] = copy_times(t_npu, t_pin, REPS)
        r["h2d_pageable"] = copy_times(t_page, t_npu, REPS)
        r["h2d_pinned"] = copy_times(t_pin, t_npu, REPS)
        rows.append(r)
        print(f"[{tag}] {mb:5d} MB  d2h pageable {r['d2h_pageable']['p50_gbs']:6.2f} | "
              f"d2h pinned {r['d2h_pinned']['p50_gbs']:6.2f} | "
              f"h2d pageable {r['h2d_pageable']['p50_gbs']:6.2f} | "
              f"h2d pinned {r['h2d_pinned']['p50_gbs']:6.2f}  GB/s (p50)")
        del t_npu, t_page, t_pin
    return rows


def numa_probe_standalone():
    """Child process entry (--numa-probe): 1GB pinned D2H under whatever affinity taskset gave us."""
    t = torch.empty(1024 * MB, dtype=torch.uint8, device=DEV)
    dst = torch.empty(1024 * MB, dtype=torch.uint8, pin_memory=True)
    r = copy_times(t, dst, 30)
    print("NUMAPROBE", json.dumps({"cores": sorted(os.sched_getaffinity(0)), **r}))


def numa_probe_via_taskset(cores, timeout=900):
    """Fresh process under taskset: sched_setaffinity from inside a torch_npu process
    that already set a mask can EINVAL; an independent process is also the cleaner
    experiment (binding affects all threads)."""
    core_str = ",".join(str(c) for c in cores)
    cmd = ["taskset", "-c", core_str, sys.executable, "-u", os.path.abspath(__file__),
           "--numa-probe"]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    for line in out.stdout.splitlines():
        if line.startswith("NUMAPROBE"):
            return json.loads(line.split(" ", 1)[1])
    raise RuntimeError(f"numa probe failed: {out.stderr[-400:]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--ledger", action="store_true")
    ap.add_argument("--numa-probe", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--numa-reps", type=int, default=30)
    args = ap.parse_args()

    os.makedirs("/data/results", exist_ok=True)

    if args.numa_probe:
        numa_probe_standalone()
        return

    orig_affinity = sorted(os.sched_getaffinity(0))
    print(f"[env] cpuset={orig_affinity}", flush=True)

    if args.bench:
        res = {"cpuset": orig_affinity}
        try:
            res["sweep"] = run_sweep("affinity-orig")

            # NUMA tax via independent taskset processes: card-1 domain node0 (0-23)
            # vs card-0 domain node6 (144-167), restricted to container cpuset 8-49,155-175
            numa = {}
            for label, lo, hi in (("right_node0", 8, 24), ("wrong_node6", 155, 168)):
                r = numa_probe_via_taskset(range(lo, hi))
                numa[label] = r
                print(f"[numa] {label} cores={r['cores']} 1GB D2H pinned "
                      f"{r['p50_gbs']:.2f} GB/s", flush=True)
            res["numa_1gb_d2h_pinned"] = numa
        finally:
            with open(OUT_JSON, "w") as f:
                json.dump(res, f, indent=1)
            print(f"[N9] wrote {OUT_JSON}", flush=True)

    if args.ledger:
        with open(OUT_JSON) as f:
            res = json.load(f)
        bw_pin_d2h = res["sweep"][-1]["d2h_pinned"]["p50_gbs"]      # use 4GB (steady-state) numbers
        bw_pin_h2d = res["sweep"][-1]["h2d_pinned"]["p50_gbs"]
        bw_pg_d2h = res["sweep"][-1]["d2h_pageable"]["p50_gbs"]
        bw_pg_h2d = res["sweep"][-1]["h2d_pageable"]["p50_gbs"]
        numa = res["numa_1gb_d2h_pinned"]
        bw_numa_right = numa["right_node0"]["p50_gbs"]
        bw_numa_wrong = numa["wrong_node6"]["p50_gbs"]

        recompute_bw = PREFILL_TPS * KV_BYTES_PER_TOK / 1e9  # GB/s equivalent: tokens/s * bytes/tok
        L = []
        L.append("# N9: KV offload crossover 账本 — 换出 host / 重算 / 卡间迁移\n")
        L.append(f"- 常数（实测）：KV {KV_BYTES_PER_TOK//1024} KB/token（N3）、prefill {PREFILL_TPS} tok/s（N2）、"
                 f"卡间 {INTER_NPU_BW/1e9:.1f} GB/s（N4）")
        L.append(f"- host 带宽（4 GB 档 p50）：pinned D2H {bw_pin_d2h:.2f} / H2D {bw_pin_h2d:.2f} GB/s；"
                 f"pageable D2H {bw_pg_d2h:.2f} / H2D {bw_pg_h2d:.2f} GB/s")
        L.append(f"- NUMA（1GB D2H pinned）：卡1域核 {numa['right_node0']['cores']} → {bw_numa_right:.2f} GB/s；"
                 f"卡0域核 {numa['wrong_node6']['cores']} → {bw_numa_wrong:.2f} GB/s\n")
        L.append(f"**等效重算带宽** = prefill {PREFILL_TPS} tok/s × 32 KB/tok = **{recompute_bw*1000:.0f} MB/s**。"
                 "换出/换入是线性搬运、重算也是线性（对固定吞吐模型），故分界不随 X 变化："
                 f"只要换入通道带宽 > {recompute_bw*1000:.0f} MB/s，搬回就永远比重算快。实测 pinned 往返带宽 "
                 f"{2/(1/bw_pin_d2h+1/bw_pin_h2d):.2f} GB/s（调和平均）是分界线的 "
                 f"{2/(1/bw_pin_d2h+1/bw_pin_h2d)/recompute_bw:.0f} 倍。\n")
        L.append("| X | tokens | 重算 prefill | 换出 D2H(pinned) | 换入 H2D(pinned) | host 往返 | 换出(pageable 往返) | 卡间迁移 | host 往返 vs 重算 |")
        L.append("|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for mb in [256, 512, 1024, 2048, 4096, 8192]:
            nbytes = mb * MB
            toks = nbytes / KV_BYTES_PER_TOK
            t_rec = toks / PREFILL_TPS * 1000
            t_d2h = nbytes / (bw_pin_d2h * 1e9) * 1000
            t_h2d = nbytes / (bw_pin_h2d * 1e9) * 1000
            t_pg_rt = nbytes / (bw_pg_d2h * 1e9) * 1000 + nbytes / (bw_pg_h2d * 1e9) * 1000
            t_inter = nbytes / INTER_NPU_BW * 1000
            verdict = "offload 赢" if (t_d2h + t_h2d) < t_rec else "重算赢"
            L.append(f"| {mb//1024 if mb>=1024 else mb}{'GB' if mb>=1024 else 'MB'} | {toks/1000:.0f}K | "
                     f"{t_rec:.0f} ms | {t_d2h:.0f} ms | {t_h2d:.0f} ms | {t_d2h+t_h2d:.0f} ms | "
                     f"{t_pg_rt:.0f} ms | {t_inter:.0f} ms | {verdict}（{t_rec/(t_d2h+t_h2d):.0f}×）|")
        L.append("\n三方结论（restore 视角，X=1GB）：")
        t1g_rec = 1024 * MB / KV_BYTES_PER_TOK / PREFILL_TPS * 1000
        t1g_host = 1024 * MB / (bw_pin_d2h * 1e9) * 1000 + 1024 * MB / (bw_pin_h2d * 1e9) * 1000
        t1g_pg = 1024 * MB / (bw_pg_d2h * 1e9) * 1000 + 1024 * MB / (bw_pg_h2d * 1e9) * 1000
        t1g_inter = 1024 * MB / INTER_NPU_BW * 1000
        L.append(f"- 重算 {t1g_rec:.0f} ms ｜ host pinned 往返 {t1g_host:.0f} ms ｜ host pageable 往返 {t1g_pg:.0f} ms ｜ 卡间 {t1g_inter:.0f} ms")
        L.append(f"- 跨 NUMA 税：绑错域（{numa['wrong_node6']['cores']}）1GB D2H 掉到 {bw_numa_wrong:.2f} GB/s"
                 f"（绑对 {bw_numa_right:.2f}），慢 {bw_numa_right/bw_numa_wrong:.1f}×"
                 f"——offload 线程必须绑卡所属 NUMA 域。")
        L.append("- offload 的隐藏收益不在本次账本内：释放 HBM 后可容纳更多并发请求（本机 2TB DRAM vs 每卡 64GB HBM），"
                 "以及 H2D 是 DMA 不占 AI Core，重算则排挤 decode 算力。")
        with open(OUT_MD, "w") as f:
            f.write("\n".join(L) + "\n")
        print("\n".join(L))
        print(f"[N9] wrote {OUT_MD}")


if __name__ == "__main__":
    main()
