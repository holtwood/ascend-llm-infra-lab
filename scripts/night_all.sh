#!/bin/bash
# Night wrapper: cluster phase (all 4 NPUs on one system) then this node's
# independent experiment line. Total ~8.5-9h.
#
# Usage: night_all.sh <m1|m2|m3|d> <line-script>
#   CLUSTER_MIN=150 (default 2.5h), line durations via each script's env vars.
set -u
TAG=${1:?tag}; LINE=${2:?line script}
export NODE_TAG="$TAG"
CLUSTER_MIN=${CLUSTER_MIN:-150}
log_start=$(date +%s)

bash /data/llm-infra-lab/scripts/night_cluster_phase.sh "$TAG" "$CLUSTER_MIN"
bash "/data/llm-infra-lab/scripts/$LINE"

echo "[$(date +%H:%M:%S)] night_all total $(( ($(date +%s)-log_start)/60 ))min" \
  >> "/data/llm-infra-lab/results/night/${TAG}_run.log"
