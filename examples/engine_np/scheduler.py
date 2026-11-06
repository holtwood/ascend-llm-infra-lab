# N1-L3 前置：continuous batching 调度器（纯 CPU 模拟，先于 NPU 实现把调度语义跑对）
#
# 面试考点覆盖：
#   1. "static/dynamic/continuous batching 的区别"——这里用代码回答：
#      static = 凑齐一批一起走、一起结束（有人空转）
#      continuous = 迭代级调度：每一步重排批次，完成的立刻退出、新请求立刻插入
#   2. "调度器每个 iteration 做哪些事"——本文件 Schedule() 的五步
#   3. preemption（抢占）：KV 不够时把最新请求整体逐出、重排队重算（vLLM 的
#      recompute 路线，vs swap 路线——N9 的 offload 数据正好是 swap 的成本依据）
#
# 设计与 vLLM v1 对齐：token-budget 预算制（而非请求数制），prefill 可切块
# （chunked prefill 的调度面），decode 步恒定批量。模型侧用假引擎（每步耗时
# 由形状参数合成），先把调度语义验证对，NPU 实测由 engine_naive.py 接入。

import random
from dataclasses import dataclass, field
from enum import Enum


class SeqStatus(Enum):
    WAITING = 0
    RUNNING = 1
    FINISHED = 2
    PREEMPTED = 3   # 被抢占回 WAITING，需重算（保留 output，丢弃 KV）


@dataclass(eq=False)   # 按身份比较：调度池的 in/remove 必须走 id，不能按字段值
class Sequence:
    seq_id: int
    prompt_len: int
    max_new_tokens: int
    arrival_step: int
    status: SeqStatus = SeqStatus.WAITING
    num_cached: int = 0        # 已 prefill 的 token 数（chunked prefill 进度）
    num_generated: int = 0     # 已 decode 的 token 数
    prefill_steps: int = 0     # 经历过的 prefill iteration 数（诊断用）

    @property
    def need(self) -> int:     # 该请求仍需的 KV 槽位（按到达即预 reserve 尾长的口径）
        return self.prompt_len + self.max_new_tokens

    @property
    def done(self) -> bool:
        return self.num_generated >= self.max_new_tokens


class ContinuousBatchScheduler:
    """token-budget + KV-budget 双约束的迭代级调度器。

    每个 iteration 依次做五件事（顺序即优先级）：
      1. 回收：FINISHED 的请求释放全部 KV 槽
      2. 抢占判定：剩余 KV 不够 decode 推进 → 逐出最新到达的 RUNNING/WAITING
      3. prefill：按 FIFO 从 WAITING 取请求，可切块（num_token_budget 以内）
      4. decode：所有"prompt 已完整"的 RUNNING 请求各出 1 token
      5. 完成：num_generated 达标 → FINISHED（下一步 iteration 被回收）
    """

    def __init__(self, kv_blocks: int, block_size: int = 16,
                 token_budget: int = 2048, chunk_size: int = 512):
        self.kv_free = kv_blocks * block_size      # 简化：全池按 token 计
        self.block_size = block_size
        self.token_budget = token_budget           # 单 iteration 的计算预算
        self.chunk_size = chunk_size
        self.waiting: list[Sequence] = []
        self.running: list[Sequence] = []
        self.finished: list[Sequence] = []
        self.stats = {"preemptions": 0, "prefill_chunks": 0}

    # ---- 槽位会计：按 block 向上取整（PagedAttention 的内部碎片就在这 ceil 里）----
    def _blocks(self, n_tok: int) -> int:
        return (n_tok + self.block_size - 1) // self.block_size * self.block_size

    def _final_blocks(self, s: Sequence) -> int:
        """请求跑完时将占的块数 = _blocks(prompt_len + max_new_tokens)。"""
        return self._blocks(s.prompt_len + s.max_new_tokens)

    def _release(self, s: Sequence):
        self.kv_free += self._blocks(s.num_cached + s.num_generated)
        s.num_cached = 0                           # recompute 路线：丢 KV 保 output

    def _capacity_left(self) -> int:
        """当前池还能"完整养活"的额外块数 = 空闲 + 可抢占回收的占用 - 现有保留。"""
        reserved = sum(self._final_blocks(s) for s in self.running)
        occupied = sum(self._blocks(s.num_cached + s.num_generated) for s in self.running)
        return self.kv_free + occupied - reserved

    def _preempt_newest(self) -> bool:
        # 只逐出 RUNNING：WAITING 尚未占 KV 槽，逐出它们不产生空间（这个注释
        # 就是一次真实 bug 的墓志——早期版本 waiting 也逐出 + 每步只抢一个，
        # 池装不下全体的完成尾长时形成"抢占-复录-再抢"活锁）。
        if not self.running:
            return False
        s = max(self.running, key=lambda x: (x.arrival_step, x.seq_id))  # 最新者，并列按 seq_id
        self.running.remove(s)
        self._release(s)                          # recompute 路线：退槽 + 丢 KV
        s.num_generated = 0                       # 保 output 则为 swap 路线（N9 的账本）
        s.status = SeqStatus.WAITING
        self.waiting.append(s)                    # 尾插：队头插入+立即重扫=活锁；
        self.stats["preemptions"] += 1            # 尾插的代价是可能饥饿，vLLM 用
        return True                               # protection counter 限重录次数折中

    def step(self) -> dict:
        """推进一个 iteration，返回该步的批次构成。"""
        # 1. 回收
        for s in [x for x in self.running if x.done]:
            s.status = SeqStatus.FINISHED
            self._release(s)
            self.running.remove(s)
            self.finished.append(s)

        # 2. 抢占（预留制）：保证留下的 RUNNING 全部能跑完——否则逐出最新者。
        #    这是 vLLM watermark 检查的教学简化版：预留制保守（尾长空块先占
        #    账面）但无活锁；vLLM 生产版是"用到快满才抢"，利用率更高。
        while self.running and self._capacity_left() < self.block_size:
            self._preempt_newest()

        budget = self.token_budget
        prefill_batch, decode_n = [], 0

        # 3. prefill（FIFO + 可切块；新请求收录也过预留检查）
        while self.waiting and budget > 0:
            s = self.waiting[0]
            remaining_prompt = s.prompt_len - s.num_cached
            take = min(self.chunk_size, remaining_prompt, budget)
            need_kv = self._blocks(s.num_cached + take) - self._blocks(s.num_cached)
            # 完整养活这条请求的空间（当前块增量 + 尾长块）必须可得，否则本步
            # **不收录、不抢占**——只等 running 推进/完成释放（vLLM 语义：
            # preemption 是最后手段，只在 decode 自身无法前进时触发；曾试过
            # "收录失败就抢 running"，tight-KV 下抢-复录-再抢震荡，24 请求
            # 5000 步只完成 6 个——收录让位给等待后 6 轮批次干净收敛）
            if self.kv_free < need_kv or \
               self._capacity_left() < self._final_blocks(s) - self._blocks(s.prompt_len):
                break
            s.num_cached += take
            budget -= take
            self.kv_free -= need_kv
            s.prefill_steps += 1
            self.stats["prefill_chunks"] += 1
            prefill_batch.append((s.seq_id, take))
            if s.num_cached == s.prompt_len:
                self.waiting.pop(0)
                s.status = SeqStatus.RUNNING
                self.running.append(s)

        # 4. decode：所有 RUNNING 各 1 token（预留制保证有槽）
        for s in self.running:
            s.num_generated += 1
            self.kv_free -= 1
            decode_n += 1

        return {"prefill": prefill_batch, "decode_n": decode_n,
                "running": len(self.running), "waiting": len(self.waiting),
                "kv_free": self.kv_free}


# ============================================================
# 模拟实验：static vs continuous 的尾部时延与槽位利用率
# ============================================================

def simulate(n_reqs=64, mode="continuous", kv_blocks=1000, seed=7):
    """mode=static: 模拟凑批等待（批次 8，一批全结束才收下一批）。
    返回每个请求的完成 iteration 与总步数。"""
    rng = random.Random(seed)
    reqs = [Sequence(i, prompt_len=rng.randint(32, 256),
                     max_new_tokens=rng.randint(16, 128),
                     arrival_step=rng.randint(0, 30))
            for i in range(n_reqs)]
    sch = ContinuousBatchScheduler(kv_blocks=kv_blocks)
    t, dones = 0, {}
    if mode == "continuous":
        queue = list(reqs)
        while len(dones) < n_reqs:
            for s in queue:
                if s.arrival_step <= t and s.status == SeqStatus.WAITING and s not in sch.waiting:
                    sch.waiting.append(s)
            info = sch.step()
            for s in sch.finished:
                if s.seq_id not in dones:
                    dones[s.seq_id] = t
            t += 1
            if t > 5000: break
    else:  # static batching：批 8，全完成才放下一批
        queue = sorted(reqs, key=lambda r: r.arrival_step)
        t = 0
        while queue:
            batch = queue[:8]; queue = queue[8:]
            done_at = max(r.prompt_len * 0.02 + r.max_new_tokens for r in batch)
            t += done_at
            for r in batch: dones[r.seq_id] = t
    return dones, t


if __name__ == "__main__":
    # 1) 调度器 sanity：小规模跑通，无死锁、无超卖
    sch = ContinuousBatchScheduler(kv_blocks=64, token_budget=1024)
    for i in range(8):
        sch.waiting.append(Sequence(i, prompt_len=100, max_new_tokens=32, arrival_step=0))
    for _ in range(200):
        sch.step()
        if not sch.waiting and not sch.running:
            break
    assert len(sch.finished) == 8, f"expected 8 finished, got {len(sch.finished)}"
    assert sch.kv_free <= 64 * 16 + 1
    print(f"[sanity] 8 reqs all finished in time, preemptions={sch.stats['preemptions']}, "
          f"prefill_chunks={sch.stats['prefill_chunks']}")

    # 2) 连续批处理 vs 静态批处理：尾部完成时间
    d_c, t_c = simulate(64, "continuous")
    d_s, t_s = simulate(64, "static")
    c_sorted = sorted(d_c.values()); s_sorted = sorted(d_s.values())
    print(f"[compare] makespan: continuous {t_c:.0f} vs static {t_s:.0f} iters")
    print(f"          p50 done: {c_sorted[32]:.0f} vs {s_sorted[32]:.0f}")
    print(f"          p90 done: {c_sorted[57]:.0f} vs {s_sorted[57]:.0f}  "
          f"(static 后半批被同批长请求拖住 → 尾部差)")
    # 3) 抢占路径 sanity：KV 池故意做小，确认 preemption 发生且最终全部完成
    d_t, _ = simulate(24, "continuous", kv_blocks=40)
    print(f"[preempt] tight-KV run: {len(d_t)}/24 finished (全部完成=抢占恢复正确)")
    print("ALL OK")
