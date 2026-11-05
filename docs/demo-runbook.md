# Live Demo Runbook — 10 分钟现场演示脚本

前提：四副本 E4 拓扑在跑（M3/M1/M2 + D 本机），隧道 8001-8003 活。

## ① 30 秒：集群现状

```bash
# D 上：四副本健康 + nginx
for p in 8010 8001 8002 8003; do curl -s -o /dev/null -w "$p:%{http_code} " http://127.0.0.1:$p/v1/models; done
curl -s :8000/metrics | grep -E "num_gpu_blocks|kv_cache_size_tokens"   # 45.6GiB/1.006M tok
```

口径：*单卡 45.6GB KV 池、~100 万 token 容量；四副本经隧道组集群。*

## ② 2 分钟：准入 vs 容量墙（E8 复现轻量版）

```bash
python3 scripts/e8_kv_pressure.py --url http://localhost:8000 --concurrency 64 --max-tokens 64
# 看 peak KV% 和 preemptions_total：默认池~12%用满前准入先限流
```

口径：*并发上不去先分清是排队还是显存——前者加并发上限，后者扩 KV 池。*

## ③ 3 分钟：控制面切换（D3+E20 合体演示）

```bash
python3 e20_control_plane.py --port 8092 --backends 127.0.0.1:8010,127.0.0.1:8001,127.0.0.1:8002,127.0.0.1:8003 --short-threshold 192 --long-cap 8 --short-cap 12 &
# 混合负载打 nginx vs 控制面
python3 e20_mixedload.py --entry http://127.0.0.1:8080/v1/completions --tag live_nginx --n 64 --concurrency 32
python3 e20_mixedload.py --entry http://127.0.0.1:8092/v1/completions --tag live_cp --n 64 --concurrency 32
curl -s :8093/   # by_class/rej/inflight/lat 实时统计
```

口径：*least_conn 只管连接数；网关懂请求内容——前缀亲和抬命中率、
准入 cap 保 SLO、429 给调用方显式背压。*

## ④ 2 分钟：sleep/wake 换租（E17）

```bash
curl -X POST :8000/sleep?level=2    # ~1.8s, npu-smi 看 HBM 4480/65536
curl -X POST :8000/wake_up          # ~0.14s
curl :8000/v1/completions -d '{"model":"qwen35-4b","prompt":"hi","max_tokens":4}'  # 首请求~0.2s
```

口径：*多租户换租不用重启——CaMem 按 tag 搬显存，wake 是 O(权重拷贝)。*

## ⑤ 2 分钟：metrics 现场解读

```bash
curl -s :8000/metrics | grep -E "prefix_cache_(queries|hits)|num_requests_(running|waiting)|num_preemptions|kv_cache_usage"
```

口径：*prefix 命中率、准入压力、抢占计数都是现成 Prometheus 指标——
生产监控三板斧。*

## 备用：如果只有单机

```bash
python3 examples/mha_from_scratch.py   # 手撕 MHA+KV：0 diff 打印 + KV/token 公式
python3 scripts/e9_goodput.py results/e2/*.json --slo-ttft 2.0 --slo-tpot 0.1
```
