#!/bin/bash
# Pull overnight progress/results from remote npt.
#   bash scripts/overnight/observe.sh          # status snapshot
#   bash scripts/overnight/observe.sh --fetch  # rsync results back to results/
set -u
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 npt"
REMOTE_BASE=/data1/pytroch-7/overnight

if [ "${1:-}" = "--fetch" ]; then
    DIR=$(cd "$(dirname "$0")/../.." && pwd)
    last=$($SSH "ls -td $REMOTE_BASE/run_* | head -1")
    name=$(basename "$last")
    mkdir -p "$DIR/results/overnight_$name"
    $SSH "cd $REMOTE_BASE && tar czf - $name" | tar xzf - -C "$DIR/results/overnight_$name" --strip-components=1
    echo "fetched -> results/overnight_$name"
    exit 0
fi

$SSH "
last=\$(ls -td $REMOTE_BASE/run_* 2>/dev/null | head -1)
echo \"=== run: \$last ===\"
pgrep -f overnight/master.sh >/dev/null && echo MASTER_RUNNING || echo MASTER_DONE
[ -n \"\$last\" ] && tail -12 \"\$last/master.log\" 2>/dev/null
echo '--- manifest ---'
[ -n \"\$last\" ] && cat \"\$last/MANIFEST.jsonl\" 2>/dev/null
echo '--- npu ---'
npu-smi info 2>/dev/null | grep -E 'davinci|MB' | head -10
"
