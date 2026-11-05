# D3 — prefix-aware 路由 vs least-conn（E4 四副本拓扑）

> 面试题：**"多副本后面挂了负载均衡，怎么让 prefix cache 真正生效？"**
> 数据回答：哈希亲和路由让 token 级 prefix 命中率 **53.8% → 84.0%**（~6k tok 前缀），
> 小前缀下 42.5% → 62.2%；但 TTFT 未显著下降——获益被两件事稀释。

## 拓扑与实现

复用 E4 四副本：`D:8010`（本机副本）+ `D:8001→M3` + `D:8002→M2` + `D:8003→M1`（SSH 隧道）。

- **基线**：nginx `least_conn`（E4 原配置），`http://D:8080`
- **实验臂**：`scripts/d3_router.py`——纯 stdlib HTTP 代理（D 上无 fastapi），
  对请求取 **前 2048 字符的 sha1** 做前缀指纹，维护 `prefix → backend` 亲和表；
  亲和后端若 inflight 高于最少者 +2 则降级 least-load。`http://D:8090`，统计口 `:8091`
- 负载：`scripts/d3_sharedprefix.py`——T 个模板共享长前缀 + 每请求随机 30 词后缀，
  保证同模板请求前缀字节级一致。**两臂用不同 seed**，否则第二臂吃到第一臂缓存

## 数据

### 小前缀（~900 tok，T=8×8 请求，c=16）

| 臂 | 命中率 | 客户端 p50/p90 | 后端分布 |
|---|---|---|---|
| nginx least_conn | **42.5%** | 1.64 / 2.10 s | 13/14/22/15 |
| LPR | **62.2%** | 1.55 / 2.28 s | 32/8/16/8 |

### 大前缀（~6k tok，T=6×8 请求，c=16）

| 臂 | 命中率 | 服务端 TTFT 均值 | 客户端 p50/p90 |
|---|---|---|---|
| nginx least_conn | **53.8%** | 0.69–1.36 s | 2.51 / 3.80 s |
| LPR | **84.0%** | 0.62–1.19 s | 1.87 / 3.99 s |

LPR 后端分布倾斜（如 24/8/8/8）——亲和性天然不均，靠降级阈值兜底。
router stats 累计 112 请求路由决策，`by_backend` 与 per-backend metrics 吻合。

## 机制解释

1. **least_conn 为什么有 42–54% 命中**：8 个请求同模板散到 4 副本，
   同一后端平均仍见到该模板 2 次（8/4），首 miss 次 hit——所以基线不是 0%，
   稀释了 LPR 的相对优势。命中率天然上界 = `(per_template-1)/per_template × 前缀占比`。
2. **LPR 为什么也到不了 ~95%**：c=16 下同模板的 8 个请求**并发到达**，
   前面的还没把前缀写进缓存，后面的亲和路由过去照样 miss——亲和性只能让
   "第二波及以后"命中，这是命中率上界比理论低的主因。
3. **TTFT 为什么没有同步跳水**：命中省的只是 prefill 计算（~6k tok ≈ 0.6–1.2s），
   但并发 16 下排队占 TTFT 大头；且亲和倾斜让热点后端排队更长，部分抵消收益。
   净效应：p50 改善 26%（2.51→1.87s），p90 持平。
4. **亲和 vs 均衡的根本权衡**：prefix 命中收益 ∝ 前缀长度×复用次数，
   倾斜代价 ∝ 排队恶化。工程取舍：亲和表加 TTL + 负载漂移阈值（本实现 +2 inflight），
   与 SGLang/vLLM production-stack 的 cache-aware routing 思路一致。

## 面试口径

- "least_conn 对 LLM 路由不是最优——它无视请求内容。共享前缀场景下
  内容感知路由把命中率从 54% 提到 84%"（有数据）
- "但命中率不是目标，TTFT 才是。排队主导时亲和倾斜会抵消收益，
  所以要带负载降级"（有机制 + 有反例）
- "前缀指纹用截断哈希，不用全 prompt——KV block 粒度 16 token，
  指纹粒度对齐 cache 粒度"（设计细节）

## 坑位

- D 无 fastapi/aiohttp → router 用 `http.server`+`http.client` 手写反向代理
- 两臂必须不同 seed 模板，否则缓存跨臂污染
- 隧道是唯一通路：路由层只能部署在 D，不能部署在 M3

## 复现

```bash
# D 上
python3 d3_router.py --port 8090 --stats-port 8091 \
  --backends 127.0.0.1:8010 127.0.0.1:8001 127.0.0.1:8002 127.0.0.1:8003 &
# 基线
python3 d3_sharedprefix.py --entry http://127.0.0.1:8080/v1/completions \
  --tag arm_nginx_big --seed 11 --templates 6 --per-template 8 \
  --concurrency 16 --prefix-words 4000
# 实验臂（不同 seed）
python3 d3_sharedprefix.py --entry http://127.0.0.1:8090/v1/completions \
  --tag arm_lpr_big --seed 13 --templates 6 --per-template 8 \
  --concurrency 16 --prefix-words 4000
```
