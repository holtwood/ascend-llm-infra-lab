#!/bin/bash
# Deploy overnight suite to remote npt and launch master.sh (nohup).
# Run from LOCAL container. Idempotent: skips finished transfers.
#
#   bash scripts/overnight/deploy.sh            # full deploy + launch
#   bash scripts/overnight/deploy.sh --no-launch # stage only
set -u
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 npt"
REMOTE_BASE=/data1/pytroch-7/overnight
DIR=$(cd "$(dirname "$0")" && pwd)

echo "[1/4] push task scripts"
tar czf - -C "$DIR" tasks master.sh | $SSH "mkdir -p $REMOTE_BASE && tar xzf - -C $REMOTE_BASE"
$SSH "chmod +x $REMOTE_BASE/master.sh $REMOTE_BASE/tasks/*.sh"

echo "[2/4] models (skip if present)"
$SSH "mkdir -p /data1/pytroch-7/models"
for m in Qwen3-0.6B Qwen3-0.6B-w8a16; do
    src=/data/models/$m
    if [ ! -d "$src" ]; then echo "  local $src missing, skip"; continue; fi
    if $SSH "[ -d /data1/pytroch-7/models/$m ]"; then
        echo "  $m already remote, skip"
    else
        echo "  copying $m (~$(du -sh $src | cut -f1)) ..."
        tar czf - -C /data/models "$m" | $SSH "tar xzf - -C /data1/pytroch-7/models"
    fi
done

echo "[3/4] verify remote state"
$SSH "du -sh /data1/pytroch-7/vllm-workspace 2>/dev/null; ls /data1/pytroch-7/models; tail -3 /tmp/stack_install.log 2>/dev/null"

if [ "${1:-}" = "--no-launch" ]; then
    echo "staged only. Launch later: ssh npt 'nohup $REMOTE_BASE/master.sh > $REMOTE_BASE/nohup.log 2>&1 &'"
    exit 0
fi

echo "[4/4] launch master (nohup, survives ssh drop)"
$SSH "if pgrep -f overnight/master.sh >/dev/null; then echo ALREADY_RUNNING; else nohup $REMOTE_BASE/master.sh > $REMOTE_BASE/nohup.log 2>&1 & echo LAUNCHED pid=\$!; fi"
