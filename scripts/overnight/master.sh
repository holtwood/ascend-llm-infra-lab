#!/bin/bash
# Overnight task queue on remote npt (8x910B1, exclusive).
# Runs tasks sequentially with per-task timeout + MANIFEST.jsonl accounting.
# Self-contained: launch via nohup, check progress via manifest.
#
# Env knobs:
#   BUDGET_HOURS   total wall-clock budget (default 8.5)
#   SKIP_VLLM      set 1 to skip vllm-dependent tasks entirely
set -u
export TZ=Asia/Shanghai

# CANN env (non-login shell skips .bashrc) + all 8 NPUs visible
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
source /usr/local/Ascend/cann-9.0.0/share/info/ascendnpu-ir/bin/set_env.sh 2>/dev/null
source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null
export ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

BASE=/data1/pytroch-7/overnight
export OUT_DIR=$BASE/run_$(date +%Y%m%d_%H%M%S)
export TASKS_DIR=$BASE/tasks
mkdir -p "$OUT_DIR"
MANIFEST=$OUT_DIR/MANIFEST.jsonl
BUDGET_HOURS=${BUDGET_HOURS:-8.5}
BUDGET_S=$(python3 -c "print(int($BUDGET_HOURS*3600))")
START=$(date +%s)
export PYTHONUNBUFFERED=1

log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT_DIR/master.log"; }
left(){ echo $(( BUDGET_S - ($(date +%s)-START) )); }

run(){ # name timeout_s cmd...
    local name=$1 tmo=$2; shift 2
    local t0=$(date +%s)
    log "TASK $name start (budget ${tmo}s)"
    timeout --signal=INT --kill-after=60 "$tmo" bash -c "$*" \
        > "$OUT_DIR/${name}.log" 2>&1
    local rc=$?
    # reap orphan workers holding NPUs (mp.spawn children outlive a killed parent)
    for p in $(pgrep -f "spawn_main|api_server|hccl_pairs|hccl_soak|kv_quant|paged_gather|copy_bw"); do
        kill -9 "$p" 2>/dev/null
    done
    local dt=$(( $(date +%s)-t0 ))
    echo "{\"task\":\"$name\",\"rc\":$rc,\"wall_s\":$dt,\"ts_end\":$(date +%s)}" >> "$MANIFEST"
    log "TASK $name end rc=$rc wall=${dt}s"
}

VLLM_OK=0
vllm_gate(){ # cached check; sets VLLM_OK=1 once imports pass
    [ "$VLLM_OK" = "1" ] && return 0
    [ "${SKIP_VLLM:-0}" = "1" ] && return 1
    python3 -c "import vllm, vllm_ascend" 2>/dev/null && VLLM_OK=1
    [ "$VLLM_OK" = "1" ]
}

log "=== overnight queue start, budget ${BUDGET_HOURS}h, out=$OUT_DIR ==="

run env_probe 300 "TASKS_DIR=$TASKS_DIR OUT_DIR=$OUT_DIR bash $TASKS_DIR/env_probe.sh"

# --- Block A: vllm bring-up (bounded) --------------------------------------
if [ "${SKIP_VLLM:-0}" != "1" ]; then
    run finish_install 3600 "TASKS_DIR=$TASKS_DIR OUT_DIR=$OUT_DIR bash $TASKS_DIR/finish_install.sh"
fi

# --- Block B: torch-level matrices (guaranteed runnable) -------------------
run hccl_pairs 3000 \
    "python3 $TASKS_DIR/hccl_pairs.py --world 8 --port 29711 --out $OUT_DIR/hccl_pairs.json"
run kv_quant_matrix 5400 \
    "python3 $TASKS_DIR/kv_quant_matrix.py --out $OUT_DIR/kv_quant_matrix.json"
run paged_gather 2700 \
    "python3 $TASKS_DIR/paged_gather_matrix.py --out $OUT_DIR/paged_gather.json"
run copy_bw 2400 \
    "python3 $TASKS_DIR/copy_bw_matrix.py --out $OUT_DIR/copy_bw.json"

# --- Block C: serving-level (only if vllm came up) --------------------------
if vllm_gate; then
    run serve_smoke 2100 "TASKS_DIR=$TASKS_DIR OUT_DIR=$OUT_DIR bash $TASKS_DIR/serve_smoke.sh"
    run tp_scaling 5400 "TASKS_DIR=$TASKS_DIR OUT_DIR=$OUT_DIR bash $TASKS_DIR/tp_scaling.sh"
else
    log "vllm unavailable -> skip serving tasks"
    echo '{"task":"vllm_tasks","rc":127,"note":"vllm/vllm_ascend import failed"}' >> "$MANIFEST"
fi

# --- Block D: soak fills the remaining budget -------------------------------
REM=$(left)
SOAK=$(( REM > 600 ? REM - 300 : 0 ))
if [ "$SOAK" -gt 900 ]; then
    run hccl_soak "$SOAK" \
        "python3 $TASKS_DIR/hccl_soak.py --world 8 --duration $((SOAK-300)) --port 29721 --out $OUT_DIR/soak.jsonl"
else
    log "remaining ${REM}s too small for soak"
fi

log "=== queue done; manifest: ==="
cat "$MANIFEST" | tee -a "$OUT_DIR/master.log"
