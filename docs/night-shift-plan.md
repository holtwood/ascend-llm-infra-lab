# 通宵实验方案（8-9 小时满载）

> 目的：用一晚上的闲置时间把 4 台 NPU 跑满，产出**面试能讲、追问能接**的
> 长跑/网格数据。四条任务线互相独立（各用本机副本），失败互不影响，
> 结束自动恢复 E4 副本配置。
>
> 关联卡片：`docs/drill-cards.md` 卡 1（扩容/瓶颈）、卡 2（KV 容量）、
> 卡 5（换租）、卡 6（网关/调度）。

## 一、总览

| 节点 | 任务线 | 阶段 | 预计时长 | 产出文件（`results/night/`） |
|---|---|---|---|---|
| **M1** | 长稳 soak | 400 波 c16/n512 连续压测 + 每波采样 | **~8.2h** | `summary.jsonl` `samples.jsonl` |
| **M2** | 性能网格 + 混合 soak | A 3D 网格 30 点 → B mnb 3 档 → C 输出长度 8 点 → D 混合负载 750 轮 | **~7.8h** | 同上 + `mix_runs.jsonl` |
| **M3** | 长上下文 + 换租疲劳 | A ctx×c 16 点 → B sleep/wake 300 轮 → C 准入墙 3 点 → D 16k soak 1000 波 | **~8.0h** | `fatigue.jsonl` + 同上 |
| **D** | 量化×规模矩阵 + soak | A 4 模型×3 并发（0.6B/4B/8B/27B-w8a8）→ B 550 波 c32 soak | **~8.3h** | 同上 |

时长估算依据（实测波次耗时）：
- M1 波次：512 请求×96 tok ≈ 49k tok @ ~750 tok/s ≈ 66s + 采样/间隔 8s ≈ 74s → 400 波 ≈ 8.2h
- M2 phase D：`e13_mixedjob.py` 96 请求×256 tok ≈ 24.6k tok @ ~700 tok/s ≈ 35s → 750 轮 ≈ 7.3h
- M3 phase D：32 请求×64 tok @ 16k ctx、c4 ≈ 25s/波 → 1000 波 ≈ 7.0h
- D phase B：512 请求×96 tok @ ~1100 tok/s ≈ 45s + 8s ≈ 53s → 550 波 ≈ 8.1h

> 想缩短/延长：改脚本里的 `WAVES` / `SOAK_RUNS` / `SOAK_WAVES` / `CYCLES`
> 环境变量即可，例如 `WAVES=200 bash scripts/night_m1_soak.sh`（~4h）。

## 二、启动前检查（2 分钟）

```bash
source ~/ascend-lab/hosts.env

# 1) 四副本健康（应为 4 个 200）
ssh -i $KEY -p $D_PORT root@$D_HOST \
  'for p in 8010 8001 8002 8003; do curl -s -o /dev/null -w "$p:%{http_code} " http://127.0.0.1:$p/v1/models; done; echo'

# 2) 脚本已同步到四台（本次改动后需重新 scp）
for spec in "$M3_PORT|$M3_HOST" "$M1_PORT|$M1_HOST" "$M2_PORT|$M2_HOST" "$D_PORT|$D_HOST"; do
  IFS='|' read -r PORT HOST <<< "$spec"
  scp -i $KEY -P $PORT scripts/night_*.sh root@$HOST:/data/llm-infra-lab/scripts/
done

# 3) 磁盘余量（soak 只写 JSONL，很小；确认 /data 有 >1GB）
ssh -i $KEY -p $M1_PORT root@$M1_HOST 'df -h /data | tail -1'
```

## 三、一键启动（复制四段，按顺序执行）

```bash
source ~/ascend-lab/hosts.env

ssh -i $KEY -p $M1_PORT root@$M1_HOST \
  'cd /data/llm-infra-lab && setsid nohup bash scripts/night_m1_soak.sh \
   > results/night/m1_run.log 2>&1 < /dev/null & echo m1_started'

ssh -i $KEY -p $M2_PORT root@$M2_HOST \
  'cd /data/llm-infra-lab && setsid nohup bash scripts/night_m2_grid.sh \
   > results/night/m2_run.log 2>&1 < /dev/null & echo m2_started'

ssh -i $KEY -p $M3_PORT root@$M3_HOST \
  'cd /data/llm-infra-lab && setsid nohup bash scripts/night_m3_ctx.sh \
   > results/night/m3_run.log 2>&1 < /dev/null & echo m3_started'

ssh -i $KEY -p $D_PORT root@$D_HOST \
  'cd /data/llm-infra-lab && setsid nohup bash scripts/night_d_quant.sh \
   > results/night/d_run.log 2>&1 < /dev/null & echo d_started'
```

启动后 3 分钟确认四条线都进入 `serve ready`：

```bash
for spec in "$M1_PORT|$M1_HOST|m1" "$M2_PORT|$M2_HOST|m2" "$M3_PORT|$M3_HOST|m3" "$D_PORT|$D_HOST|d"; do
  IFS='|' read -r PORT HOST TAG <<< "$spec"
  printf "%-3s " "$TAG"
  ssh -i $KEY -p $PORT root@$HOST "tail -1 /data/llm-infra-lab/results/night/${TAG}_run.log"
done
```

## 四、睡前抽查（可选，1 分钟）

```bash
# 进度行（每 50 波打印一次）
ssh -i $KEY -p $M1_PORT root@$M1_HOST 'grep progress /data/llm-infra-lab/results/night/m1.log | tail -2'
# 当前吞吐趋势
ssh -i $KEY -p $M1_PORT root@$M1_HOST 'tail -3 /data/llm-infra-lab/results/night/summary.jsonl'
```

## 五、早上收集（本地一条命令）

```bash
python3 scripts/night_collect.py --local results/night
# 输出 results/night/SUMMARY.md：每节点的
#   soak 吞吐首10/末10 漂移、TTFT 漂移、错误数
#   网格表、ctx 表、量化表
#   sleep/wake 疲劳均值与漂移、KV 峰值/抢占
```

重点看三件事（也是面试要讲的）：
1. **soak 漂移**：末 10 波吞吐 vs 首 10 波——掉 >5% 说明有热/碎片问题
2. **疲劳漂移**：sleep/wake 300 轮后 wake 延迟是否变大（泄漏/碎片）
3. **网格拐点**：c 增大后吞吐平台在哪、TPOT 何时破 SLO

## 六、中断与恢复

```bash
# 单条线停止（注意用括号技巧，避免 pkill 自匹配杀掉 ssh 会话）
ssh -i $KEY -p $M1_PORT root@$M1_HOST "pkill -f 'night_[m]1_soak'; echo stopped"

# 全部停止
for spec in "$M1_PORT|$M1_HOST|night_[m]1" "$M2_PORT|$M2_HOST|night_[m]2" \
            "$M3_PORT|$M3_HOST|night_[m]3" "$D_PORT|$D_HOST|night_[d]_quant"; do
  IFS='|' read -r PORT HOST PAT <<< "$spec"
  ssh -i $KEY -p $PORT root@$HOST "pkill -f '$PAT'"
done

# 手动恢复某节点 E4 副本（正常结束脚本会自动做）
ssh -i $KEY -p $M3_PORT root@$M3_HOST \
  'cd /data/llm-infra-lab && setsid nohup vllm serve /model/Qwen3.5-4B \
   --served-model-name qwen35-4b --port 8000 --max-model-len 8192 \
   > results/night/restore_m3.log 2>&1 < /dev/null &'
# 注意：M2/D 的模型路径是 /data/models/Qwen3.5-4B；D 端口是 8010
```

## 七、风险与已知坑

| 风险 | 处理 |
|---|---|
| 27B-w8a8 在 D 上显存不足 | 脚本里 `serve` 失败会 `continue` 到下一个模型，不影响后续 soak |
| M3 睡眠模式起不来（rl_config） | phase B 失败会继续到 phase C；日志 `serve_sleep.log` 可查 |
| 某节点中途崩溃 | 四条线独立；其余节点不受影响，早上看 `*_run.log` 定位 |
| 脚本被 ssh 断开杀死 | 已用 `setsid nohup ... < /dev/null` 脱离会话 |
| `pkill -f night_` 杀到自己 | 文档命令已用 `night_[m]1` 括号技巧；不要直接写 `pkill -f night_` |
| 早上忘了收结果 | 结果在节点 `/data/llm-infra-lab/results/night/`，跑一次 `night_collect.py` 即可 |

## 八、预期产出（面试价值）

1. **"你的服务能连续跑多久不退化？"** → M1 的 8 小时漂移曲线（别人没有）
2. **"部署参数怎么选？"** → M2 的 c×in×out 30 点网格 + mnb/输出长度扫描
3. **"长上下文并发上限多少？"** → M3 的 ctx×c 网格 + 16k 准入墙
4. **"大模型量化 vs 小模型怎么选？"** → D 的 0.6B/4B/8B/27B-w8a8 同负载矩阵
5. **"换租久了会不会退化？"** → M3 的 300 轮 sleep/wake 疲劳曲线
