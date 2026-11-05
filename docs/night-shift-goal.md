# 通宵实验 Goal 指令（直接复制给 agent）

> 用法：晚上启动前，把下面整段粘给 agent 作为 goal。它会按
> `docs/night-shift-plan.md` 启动四机任务线、全程巡检、早上收数据并落盘。

```
执行通宵实验：按 docs/night-shift-plan.md 启动四机 8.5-9 小时任务线，
并全程自主推进到结果落盘、文档更新、commit+push。

【启动】
1. 启动前检查（文档第二节）：四副本+nginx 健康（5 个 200）、night_*.sh 已同步
   到四台、M1/M2/M3→D:8080 隧道可用；任何一项不满足先修复再启动。
2. 按文档第三节四条命令启动 night_all.sh（m1/m2/m3/d），必须 setsid nohup
   + </dev/null 脱离会话。
3. 启动后 5 分钟确认四条线都进入 cluster phase（grep 'cluster phase start'
   各节点 results/night/<tag>.log），没进入就看 <tag>_run.log 排查。

【运行期（每小时一轮，务必做）】
4. 巡检 + 落盘推送：
   a) 各节点 results/night/<tag>.log 尾部 + summary.jsonl 最新 3 行
   b) 跑 python3 scripts/night_collect.py --local results/night（拉回增量）
   c) 把增量结果 + docs/status.md 进度行 commit+push（先 fetch+rebase），
      保证机器中途挂掉也不丢数据
   d) 记录一行进度到 docs/status.md（如 "01:30 集群段 200 波，聚合 4.4k tok/s"）
5. 某条线死亡（进程不在且未到结束时间）→ 按该节点脚本重启续跑，保留已有
   JSONL 不覆盖；连续两次起不来则停该线并在报告注明，其余三条继续。
6. 不干预正在跑的线；不要重启 vllm，除非该线已死。

【收尾（全部结束后）】
7. 跑 python3 scripts/night_collect.py --local results/night 生成 SUMMARY.md。
8. 写 results/night/README.md：
   - 集群段：单机均值、聚合（应≈4×）、首末漂移、AICore/HBM 证据
   - 四条线关键表：M1 soak 漂移、M2 c×in×out/mnb/输出长度网格、M3 ctx×c 网格
     + 换租 300 轮疲劳 + 准入墙、D 量化×规模矩阵（0.6B/4B/8B/27B-w8a8）
   - 每条结论配机制解释 + 面试口径（对照 docs/drill-cards.md 的卡 1/2/5/6）
   - 如实标注失败/跳过项（如 27B 显存不足）
9. 更新 docs/status.md（踩坑与新发现）、docs/drill-cards.md（把新数据补进
   对应卡片的 L2/L3）、README 实验矩阵加一行。
10. commit+push gitee：先 git fetch origin main && git rebase origin/main，
    不 push -f、不动 git config。
11. 确认四台已恢复 E4 副本（D 上四端口 200）；注意 M2/D 模型路径是
    /data/models/Qwen3.5-4B，D 端口 8010，M3/M1 是 /model/Qwen3.5-4B。

【规则】
- 结果文件随测随拷回本地 results/night/（远端可能被并发 git 操作清理）
- 需要决策时停下（改上游源码、花钱、破坏性操作）
- 硬件异常（npu-smi 报错/服务反复起不来）→ 停该线、报告注明、其余继续
- 时间不够时用缩短版：CLUSTER_MIN=60 + 各线 WAVES/SOAK_RUNS/SOAK_WAVES 减半
- 不要用 pkill -f night_（会杀掉执行命令的 shell）；用 night_[a]ll 括号技巧
```

## 早上验收清单（人工 5 分钟）

- [ ] `results/night/SUMMARY.md` 存在，四条线都有数据
- [ ] 集群段聚合吞吐 ≈ 4× 单机（4.2-5.0k tok/s 量级）
- [ ] M1 soak 末 10 波 vs 首 10 波漂移 < 5%
- [ ] M3 疲劳 300 轮 wake 延迟无明显上漂
- [ ] D 量化矩阵含 4 个模型 × 3 并发
- [ ] `git log` 有通宵结果提交；四副本 200
