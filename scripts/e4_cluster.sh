#!/bin/bash
# E4 集群编排：在 D 机建 SSH 隧道 + nginx least_conn LB，然后本机压测
# 拓扑: bench(D) -> nginx:8080 -> [tunnel->M3:8000, tunnel->M1:8000, tunnel->M2:8000]
set -u
D="ssh -p 40083 root@223.111.149.165"
N=$1  # 副本数: 1/2/3

# 1) D 上建隧道（幂等：先杀旧隧道）
$D 'for p in $(pgrep -f "[s]sh.*-L 800"); do kill $p 2>/dev/null; done
nohup ssh -p 40019 -N -L 8001:localhost:8000 -o StrictHostKeyChecking=no -o ServerAliveInterval=30 root@223.111.149.177 >/dev/null 2>&1 &
[ '"$N"' -ge 2 ] && nohup ssh -p 40010 -N -L 8002:localhost:8000 -o StrictHostKeyChecking=no -o ServerAliveInterval=30 root@223.92.35.244 >/dev/null 2>&1 &
[ '"$N"' -ge 3 ] && nohup ssh -p 40019 -N -L 8003:localhost:8000 -o StrictHostKeyChecking=no -o ServerAliveInterval=30 root@223.92.35.253 >/dev/null 2>&1 &
sleep 3
for port in 8001 8002 8003; do curl -s --max-time 2 localhost:$port/v1/models >/dev/null && echo "tunnel $port UP" || echo "tunnel $port down"; done'

# 2) nginx 配置
UP="server 127.0.0.1:8001;"
[ "$N" -ge 2 ] && UP="$UP server 127.0.0.1:8002;"
[ "$N" -ge 3 ] && UP="$UP server 127.0.0.1:8003;"
$D "cat > /etc/nginx/sites-available/llblb <<EOF
upstream llm { least_conn; $UP }
server { listen 8080; location / { proxy_pass http://llm; proxy_read_timeout 600s; proxy_buffering off; } }
EOF
ln -sf /etc/nginx/sites-available/llblb /etc/nginx/sites-enabled/llblb
rm -f /etc/nginx/sites-enabled/default
nginx -t 2>&1 | tail -1 && (systemctl reload nginx 2>/dev/null || nginx -s reload 2>/dev/null || nginx) && echo LB_READY:$N"
