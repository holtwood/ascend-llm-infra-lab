#!/bin/bash
# Cluster phase: all four NPUs serve ONE logical system (E4 cluster) while all
# four machines also inject load through D's nginx. This is the "4 cards
# cooperating" mode — the only one possible without RDMA (E18).
#
# Usage: night_cluster_phase.sh <tag> <minutes>
#   tag: m1|m2|m3|d (d uses port 8010 and /data/models)
# Samples local AICore/HBM + server KV metrics every wave, so the morning
# report can show the NPUs were actually busy.
set -u
TAG=${1:?tag}; DUR=${2:-150}
export NODE_TAG="$TAG"
source /data/llm-infra-lab/scripts/night_common.sh

if [ "$TAG" = "d" ]; then
  PORT=8010; MODEL=/data/models/Qwen3.5-4B
else
  PORT=8000; MODEL=/model/Qwen3.5-4B
  # M1/M2 may live under /data/models
  [ -d "$MODEL" ] || MODEL=/data/models/Qwen3.5-4B
fi

log "cluster phase start (${DUR}min, client -> D:8080)"
code=$(curl -s -o /dev/null -w '%{http_code}' -m 3 "http://127.0.0.1:$PORT/v1/models")
[ "$code" = "200" ] || { serve $PORT "$MODEL" cluster_base || exit 1; }

if [ "$TAG" != "d" ]; then
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 3 http://127.0.0.1:8080/v1/models)
  if [ "$code" != "200" ]; then
    log "tunnel down, recreating"
    nohup ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -N \
      -i /root/.ssh/id_nodekey -L 8080:localhost:8080 -p "${D_PORT:-40083}" \
      "root@${D_HOST:?}" > /tmp/cluster_tunnel.log 2>&1 &
    sleep 6
  fi
fi

END=$(( $(date +%s) + DUR * 60 ))
i=0
while [ "$(date +%s)" -lt "$END" ]; do
  i=$((i + 1))
  bench 8080 "clu_w$(printf %04d "$i")" 32 256 256 96
  sample "clu_w$(printf %04d "$i")" "$PORT"
  sleep 3
  [ $((i % 20)) -eq 0 ] && log "cluster wave $i elapsed $(( (DUR*60 - (END - $(date +%s)))/60 ))min"
done
log "cluster phase done: $i waves"
