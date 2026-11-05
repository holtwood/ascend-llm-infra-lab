# 通宵实验方案（8.5-9 小时，4 卡满载）

> 目的：一晚上把 4 台 NPU 用满，产出**面试能讲、追问能接**的长跑/网格数据。
> 排程分两段：**前 2.5 小时 4 卡协同**（一个逻辑系统、四机同时注入），
> **后 6 小时四机各自独立线**（实验吞吐 ×4）。结束自动恢复 E4 副本。

## 零、先定义"充分利用"（本环境的口径）

| 口径 | 含义 | 实测依据 |
|---|---|---|
| **单卡吞吐上限** | 4B decode 是 memory-bound：AICore 40-53% 峰值即满负荷，`npu-smi` 的 AICore% 不能当 CPU 用 | E21：c16 负载下 AICore 峰值 40-53% |
| **4 卡协同** | 无 RDMA（E18）→ 唯一形态是 4 副本 + 网关 + 多客户端；聚合吞吐贴住 4×单副本饱和即用满 | E23：聚合 4371-5002 tok/s ≈ 4×单副本 1.1-1.25k |
| **实验吞吐** | 四条独立线并行，墙钟 ×1/4 | E21：4 路并行墙钟 9.2s vs 串行 ~33s |

因此本方案 = **协同段（证明系统级用满）+ 独立段（实验产出 ×4）**。

## 一、时长标定（实测，2026-11-05）

先测真实波次耗时，再定时长——**不靠估算**：

| 波次定义 | 实测耗时 | 吞吐 |
|---|---|---|
| 集群段波次：c32/n256/in256/out96（经 nginx） | **18-22s** | 单客户端 1.7k tok/s |
| M1 soak 波次：c16/n512/in256/out96 | **59s**（+采样/间隔 7s = 66s） | 837 tok/s |
| D soak 波次：c32/n512/in256/out96 | **40s**（+7s = 47s） | 1242 tok/s |
| M2 最慢网格点：c128/n128/in4096/out256 | **17s** | 1953 tok/s |
| M3 16k soak 波次：c4/n32/in16k/out64 | **11s**（+7s = 18s） | — |
| M3 63k 网格点：c8/n8/in63k/out64 | **6s** | — |

**总时长 = 集群段 150min（墙钟上界）+ 独立段 360min（墙钟上界）= 510min = 8.5h**
——两段都是"到点必停"的时间上界循环（`CLUSTER_MIN` / `LINE_MIN`），
不是波数估算，所以**总时长确定性落在 8.5h**；各线实际波数见下表。

## 二、总排程（假设 22:00 启动）

| 时段 | M1 | M2 | M3 | D |
|---|---|---|---|---|
| 22:00-00:30 | **集群 soak：四机同为客户端 → D:8080（c32/机，聚合 c128）**，约 430 波 | 同左 | 同左 | 同左 + nginx/控制面 |
| 00:30-06:30 | soak 约 330 波（c16/n512） | 网格 30 点 + mnb 3 档 + 输出 8 点（~20min）+ 混合 soak 约 580 轮 | ctx 网格 16 点 + 300 轮换租疲劳 + 准入墙（~25min）+ 16k soak 约 1100 波 | 量化矩阵 4 模型×3 并发（~16min）+ soak 约 440 波 |
| 06:30-07:00 | 自动恢复 E4 副本 | 同左 | 同左 | 同左 |

预计数据量：集群段 ~430 + M1 ~330 + M2 ~610 + M3 ~1120 + D ~450 ≈ **~2900 个 bench 点**。

缩短版：`CLUSTER_MIN=60 LINE_MIN=180` → 总 4h；加长：`CLUSTER_MIN=180 LINE_MIN=420` → 10h。

## 二、启动前检查（2 分钟）

```bash
source ~/ascend-lab/hosts.env

# 1) 四副本 + nginx 健康（应 5 个 200）
ssh -i $KEY -p $D_PORT root@$D_HOST \
  'for p in 8010 8001 8002 8003 8080; do curl -s -o /dev/null -w "$p:%{http_code} " http://127.0.0.1:$p/v1/models; done; echo'

# 2) 脚本同步（本次新增 night_all/night_cluster_phase，必须重传）
for spec in "$M3_PORT|$M3_HOST" "$M1_PORT|$M1_HOST" "$M2_PORT|$M2_HOST" "$D_PORT|$D_HOST"; do
  IFS='|' read -r PORT HOST <<< "$spec"
  scp -i $KEY -P $PORT scripts/night_*.sh root@$HOST:/data/llm-infra-lab/scripts/
done

# 3) 隧道检查（M1/M2/M3 → D:8080；集群段需要）
for spec in "$M1_PORT|$M1_HOST" "$M2_PORT|$M2_HOST" "$M3_PORT|$M3_HOST"; do
  IFS='|' read -r PORT HOST <<< "$spec"
  printf "%s " "$HOST"
  ssh -i $KEY -p $PORT root@$HOST 'curl -s -o /dev/null -w "%{http_code}\n" -m 5 http://127.0.0.1:8080/v1/models'
done
# 若不是 200：脚本会自动重建隧道；也可手动：
# ssh -N -i /root/.ssh/id_nodekey -L 8080:localhost:8080 -p $D_PORT root@$D_HOST &
```

## 三、一键启动（四条命令，复制即用）

```bash
source ~/ascend-lab/hosts.env

ssh -i $KEY -p $M1_PORT root@$M1_HOST \
 'cd /data/llm-infra-lab && setsid nohup bash scripts/night_all.sh m1 night_m1_soak.sh \
  > results/night/m1_run.log 2>&1 < /dev/null & echo m1_started'

ssh -i $KEY -p $M2_PORT root@$M2_HOST \
 'cd /data/llm-infra-lab && setsid nohup bash scripts/night_all.sh m2 night_m2_grid.sh \
  > results/night/m2_run.log 2>&1 < /dev/null & echo m2_started'

ssh -i $KEY -p $M3_PORT root@$M3_HOST \
 'cd /data/llm-infra-lab && setsid nohup bash scripts/night_all.sh m3 night_m3_ctx.sh \
  > results/night/m3_run.log 2>&1 < /dev/null & echo m3_started'

ssh -i $KEY -p $D_PORT root@$D_HOST \
 'cd /data/llm-infra-lab && setsid nohup bash scripts/night_all.sh d night_d_quant.sh \
  > results/night/d_run.log 2>&1 < /dev/null & echo d_started'
```

启动后 5 分钟确认进入集群段：

```bash
for spec in "$M1_PORT|$M1_HOST|m1" "$M2_PORT|$M2_HOST|m2" "$M3_PORT|$M3_HOST|m3" "$D_PORT|$D_HOST|d"; do
  IFS='|' read -r PORT HOST TAG <<< "$spec"
  printf "%-3s " "$TAG"
  ssh -i $KEY -p $PORT root@$HOST "grep -E 'cluster phase start|cluster wave' /data/llm-infra-lab/results/night/${TAG}.log | tail -1"
done
```

## 四、集群段利用率自证（睡前看一眼）

```bash
# 四机 AICore/HBM 实时（每 2s 一次）
for spec in "$M1_PORT|$M1_HOST" "$M2_PORT|$M2_HOST" "$M3_PORT|$M3_HOST" "$D_PORT|$D_HOST"; do
  IFS='|' read -r PORT HOST <<< "$spec"
  ssh -i $KEY -p $PORT root@$HOST 'source /data/llm-infra-lab/scripts/env.sh 2>/dev/null; npu-smi info | sed -n "7p;8p"'
done
# 集群吞吐（各节点 summary.jsonl 里的 clu_w* 行）
ssh -i $KEY -p $D_PORT root@$D_HOST 'grep clu_w /data/llm-infra-lab/results/night/summary.jsonl | tail -3'
```

预期：四机 AICore 峰值 45-60%、聚合吞吐 4.2-5.0k tok/s（E23 基线）。

## 五、早上收集

```bash
python3 scripts/night_collect.py --local results/night
# → results/night/SUMMARY.md：
#   · 集群段：每节点 clu_w* 吞吐（聚合 ≈ 4 倍单机）
#   · soak 漂移：首10 vs 末10 吞吐/TTFT
#   · 网格表（c×in×out / mnb / 输出长度）
#   · ctx 表（4k-63k × c1-8）
#   · 量化矩阵（0.6B/4B/8B/27B-w8a8）
#   · 换租疲劳（sleep/wake 延迟漂移）、KV 峰值/抢占
```

重点看三件事：
1. **集群段聚合吞吐是否 ≈ 4× 单机**（证明 4 卡协同无瓶颈）
2. **soak 末 10 波 vs 首 10 波**：掉 >5% 说明热/碎片问题
3. **换租 300 轮后 wake 延迟**是否漂移

## 六、中断与恢复

```bash
# 单线停止（括号技巧避免 pkill 自匹配）
ssh -i $KEY -p $M1_PORT root@$M1_HOST "pkill -f 'night_[a]ll'; pkill -f 'night_[m]1_soak'; echo stopped"

# 全部停止
for spec in "$M1_PORT|$M1_HOST|night_[a]ll" "$M2_PORT|$M2_HOST|night_[a]ll" \
            "$M3_PORT|$M3_HOST|night_[a]ll" "$D_PORT|$D_HOST|night_[a]ll"; do
  IFS='|' read -r PORT HOST PAT <<< "$spec"; ssh -i $KEY -p $PORT root@$HOST "pkill -f '$PAT'"
done

# 手动恢复（正常结束会自动做；M2/D 模型在 /data/models，D 端口 8010）
ssh -i $KEY -p $M3_PORT root@$M3_HOST \
 'cd /data/llm-infra-lab && setsid nohup vllm serve /model/Qwen3.5-4B \
  --served-model-name qwen35-4b --port 8000 --max-model-len 8192 \
  > results/night/restore_m3.log 2>&1 < /dev/null &'
```

## 七、风险清单

| 风险 | 处理 |
|---|---|
| 集群段某节点隧道断 | `night_cluster_phase.sh` 自动重建隧道；仍失败则只影响该节点客户端 |
| 27B-w8a8 显存不足 | 该模型 `serve` 失败自动跳过，继续下一个 |
| 睡眠模式起不来 | M3 phase B 失败后继续 phase C/D |
| 某节点崩 | 四条线独立，其余不受影响；看 `*_run.log` |
| `pkill -f night_` 杀自己 | 文档命令统一用 `night_[a]ll` / `night_[m]1` 括号技巧 |
| 早上忘了收 | 结果在节点 `/data/llm-infra-lab/results/night/`，跑一次 collector 即可 |

## 八、预期产出（对应面试卡）

1. 卡 1："4 卡集群聚合吞吐 ≈ 4×单机，瓶颈在注入不在副本" → 集群段
2. 卡 2："长上下文并发上限 + KV 墙" → M3 ctx 网格/准入墙
3. 卡 5："换租 300 轮无退化" → M3 疲劳曲线
4. 卡 6："部署参数 c×in×out 怎么选" → M2 网格
5. 新增："大模型量化 vs 小模型" → D 的 0.6B/4B/8B/27B-w8a8 矩阵
6. 新增："连续 8 小时服务退化吗" → M1 soak 漂移
