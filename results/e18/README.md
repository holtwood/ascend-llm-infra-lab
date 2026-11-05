# E18 — PD 跨节点重试：P=M3, D=M1（三层失败画像）

> 面试题：**"E7 同卡 PD 失败在建链，跨节点是不是就能成？"**
> 数据回答：**不能**。分层排障后确认——握手层可隧道化修通，
> 数据面 device-mem 传输在 NAT 容器间物理不可行；
> 且**失败后 D 照常返回垃圾 token，不是干净的错**。

## 环境硬约束（先决事实）

- 四节点全在 NAT 后：互访只放行 **SSH 端口**，其余 TCP 全不通（M3→M1:8000 实测失败）
- 容器内无 `/etc/hccn.conf`、无 hccn_tool 输出——**跨节点 NPU 互联（RDMA/HCCS）根本未配置**
- 官方文档 `pd_disaggregation_mooncake_multi_node.md` 前置：同 LAN + NPU 互联 RDMA——本环境不满足

## 实验拓扑与排障分层

```
P=M3:23700 (kv_producer, kv_port 31000)   D=M1:23701 (kv_consumer, kv_port 31100)
proxy=M1:28081（prefiller 走 ssh 隧道→M3）
```

| 层 | 现象 | 修法 | 结果 |
|---|---|---|---|
| L1 ZMQ 元数据 | P 广播容器 IP `172.17.0.5:31000`，D 连不通，3 次重试后 RuntimeError | `VLLM_HOST_IP=127.0.0.1` + ssh -L 31000 | ✅ 握手通过 |
| L2 TE-RPC | `SocketHandShakePlugin connect()127.0.0.1:16401 refused`——rpc 口是动态的（P=16401, D=16352） | 双向再补 -L 16401/16352/20778/20488 | ✅ socket 错误消失 |
| L3 数据面 | `Mooncake transfer failed, ret: -1` | — | ❌ device-mem 传输过不去 |

注：L3 阶段连 `SocketHandShakePlugin` 报错都没了——握手全通，
挂在 `transfer_kv_cache` 的 device→device 拷贝本身。

## 最危险的发现：**静默垃圾输出**

两次请求都**正常返回 200 + 完整 JSON**，但输出是垃圾：

- req1：`"user() {\n return (uid() == 0);\n}"`（文本通顺但与 prompt 无关）
- req2：`"!!!!!!!!!!!!!!!!"`

原因：KV 传输失败后 **D 侧用未初始化/错误的 KV 直接 decode**，
proxy 层拿不到传输错误 → 请求"成功"。生产上这比 crash 可怕：
**PD 链路故障表现为输出质量静默劣化，不是报警**。

## 机制解释

1. MooncakeConnectorV1 P2P 模式三段式：ZMQ 元数据握手（kv_port）
   → TransferEngine RPC（动态口）→ adxl 引擎做 device-mem 传输。
   前两段是普通 TCP 可隧道化，第三段注册的 buffer 是
   `mem type:device`（`register mem addr:0x12d526200000, length:14.9GB`），
   需要 NPU 侧 DMA 通道——SSH 隧道只是个 TCP 管道，托不起设备内存语义。
2. `get_ip()` 取的是容器内网 IP——这就是广播 172.17.0.5 的来源；
   `VLLM_HOST_IP` 覆盖只对控制面有效，改不了数据面可达性。
3. 同卡 E7 失败的 `HixlCSClientConnect` 与本实验不同根因：E7 是
   同卡双 adxl 引擎资源受限，E18 是跨机物理无链路——**PD 在
   本环境的唯一可行形态是同卡共置（且限 util≤0.35+0.35）**。

## 面试口径

- "跨节点 PD 不是配出来就行：官方前置是 NPU RDMA 互联；
  NAT/容器环境连握手 IP 都是错的（pod IP），要先过 get_ip/VLLM_HOST_IP"
- "排障要分层：元数据→RPC→数据面，每层失败模式不同；
  我修通了两层，败在 device-mem 物理层"（三层日志证据）
- "最重要的运维结论：传输失败请求仍 200，输出是垃圾——
  PD 部署必须加输出质量/传输成功率监控，不能只看 HTTP 状态码"

## 复现

```bash
# M1 侧建到 M3 的隧道（节点间仅 ssh 端口互通，需先放 key）
ssh -N -i /root/.ssh/id_nodekey -L 23700:localhost:23700 \
  -L 31000:localhost:31000 -L 16401:localhost:16401 \
  -L 20778:localhost:20778 -p <M3_SSH_PORT> root@<M3_HOST>

# P (M3): VLLM_HOST_IP=127.0.0.1 让元数据广播 localhost（走隧道）
VLLM_HOST_IP=127.0.0.1 vllm serve /model/Qwen3.5-4B --port 23700 \
  --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1",
    "kv_role":"kv_producer","kv_port":31000,...}'
# D (M1): 同法 kv_consumer kv_port=31100；proxy 指 127.0.0.1:23700/23701
```

日志摘录：`serve_p2_extract.log` / `serve_d2_extract.log`
