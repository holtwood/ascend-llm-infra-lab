# L3 NPU demo: continuous batching vs serial serving on Qwen3.5-4B (real weights)
# 场景: 8 个异构请求（4 短 64tok + 4 长 192tok prompt, max_new 16/32 交替）
# A) 串行: 逐请求单独跑 (max_b=1)  B) continuous: 一个 BatchEngine 全提交
# 输出: makespan / 吞吐 / 每步 batch 轨迹 + 正确性抽样 (batched==serial 末token)
import argparse, time, torch, sys, os
sys.path.insert(0, os.path.dirname(__file__))

from model_qwen35 import Qwen35Config, from_pretrained
from batch_engine import BatchEngine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/model/Qwen3.5-4B")
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--n-req", type=int, default=8)
    args = ap.parse_args()
    dev = args.device

    cfg = Qwen35Config.from_json(f"{args.model_dir}/config.json")
    model = from_pretrained(args.model_dir, device=dev)
    model = model.to(torch.bfloat16).eval()

    torch.manual_seed(7)
    reqs = []                                          # (prompt, max_new)
    for i in range(args.n_req):
        plen = 64 if i % 2 == 0 else 192
        max_new = 16 if i % 4 in (0, 1) else 32
        reqs.append((torch.randint(1000, 50000, (plen,)).tolist(), max_new))

    # warmup (compile kernels)
    e = BatchEngine(model, cfg, max_b=2, max_len=256, device=dev)
    e.submit(reqs[0][0][:32], max_new=2)
    e.run(max_steps=8)
    del e
    torch.npu.synchronize() if dev.startswith("npu") else None

    # ---- A) serial ----
    t0 = time.time()
    serial_done = {}
    for i, (p, mn) in enumerate(reqs):
        e = BatchEngine(model, cfg, max_b=1, max_len=256, device=dev)
        e.submit(p, max_new=mn)
        d, _ = e.run(max_steps=256)
        serial_done.update(d)
        del e
    serial_t = time.time() - t0
    serial_tokens = sum(mn for _, mn in reqs)

    # ---- B) continuous ----
    t0 = time.time()
    e = BatchEngine(model, cfg, max_b=args.n_req, max_len=256, device=dev)
    for p, mn in reqs:
        e.submit(p, max_new=mn)
    batch_done, steps = e.run(max_steps=512)
    cont_t = time.time() - t0

    # report
    total_out = sum(mn for _, mn in reqs)
    print(f"\n| 模式 | makespan s | 吞吐 tok/s | engine steps |")
    print(f"|---|---:|---:|---:|")
    print(f"| 串行 x{args.n_req} | {serial_t:.1f} | {total_out/serial_t:.1f} | - |")
    print(f"| continuous (max_b={args.n_req}) | {cont_t:.1f} | {total_out/cont_t:.1f} | {steps} |")
    print(f"\nspeedup = {serial_t/cont_t:.2f}x")

    agree = sum(1 for k in batch_done if batch_done[k] == serial_done.get(k))
    print(f"correctness: {agree}/{len(batch_done)} last-tokens match serial "
          f"({'OK' if agree == len(batch_done) else 'DIVERGED (bf16 batch shape sensitivity, see N2b)'})")
    print("L3_DEMO_OK")


if __name__ == "__main__":
    main()
