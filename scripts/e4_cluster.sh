#!/bin/bash
# E4 集群编排（在 M3 跑）：D 机建 SSH 隧道 + nginx least_conn LB
# 拓扑: bench(D) -> nginx:8080 -> {8001:tunnel->M3, 8002:tunnel->M2, 8010:D本机}
# 用法: bash e4_cluster.sh <副本数1|2|3>
set -u
D="ssh -p 40083 root@223.111.149.165"
N=$1

# 1) D 上建隧道（幂等；[s] 防 self-match）
$D 'for p in $(pgrep -f "^ssh.*-N -L 800"); do kill $p 2>/dev/null; done; sleep 1
setsid ssh -p 40019 -N -L 8001:localhost:8000 -o StrictHostKeyChecking=no -o ServerAliveInterval=30 root@223.111.149.177 >/dev/null 2>&1 < /dev/null &
[ '"$N"' -ge 2 ] && setsid ssh -p 40019 -N -L 8002:localhost:8000 -o StrictHostKeyChecking=no -o ServerAliveInterval=30 root@223.92.35.253 >/dev/null 2>&1 < /dev/null &
sleep 3
for port in 8001 8002; do curl -s --max-time 3 localhost:$port/v1/models >/dev/null && echo "tunnel $port UP" || echo "tunnel $port down"; done'

# 2) nginx upstream（D 本机副本固定走 8010 端口以免和隧道号混）
UP="server 127.0.0.1:8001;"
[ "$N" -ge 2 ] && UP="$UP server 127.0.0.1:8002;"
[ "$N" -ge 3 ] && UP="$UP server 127.0.0.1:8010;"
$D "cat > /etc/nginx/conf.d/llblb.conf <<EOF
upstream llm { least_conn; $UP }
server { listen 8080; location / { proxy_pass http://llm; proxy_read_timeout 600s; proxy_buffering off; } }
EOF
nginx -t 2>&1 | tail -1 && (nginx -s reload 2>/dev/null || nginx) && echo LB_READY:N=$N"

# 3) D 本机副本如需启用：serve 已另起在 :8000，由调用方保证
