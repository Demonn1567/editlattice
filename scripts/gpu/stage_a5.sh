#!/usr/bin/env bash
# Phase A5 (pre-registered addendum 2026-11-24): evolution from multi-step seeds with the Qwen3-14B-AWQ proposer
# (thinking on) and the Qwen3-4B-AWQ solver sharing the A4000; then, with the solver alone, the D8 gain checks and
# Phase B (statement-level tree lattices, exhaustive audit, fresh-run audit, report) of every lineage with accepted
# edits. A finished loop, gain check or lattice is never rerun. Exclusive lock; servers stopped by PID only.
set -uo pipefail
cd "$(dirname "$0")/../.."
exec 9>runs/locks_stage_a5.lock
flock -n 9 || { echo "another stage_a5 is running"; exit 1; }
export PATH=/usr/lib/wsl/lib:$PATH
. .venv-gpu/bin/activate
log() { echo "$(date '+%F %T') $*"; }
wait_ready() { for _ in $(seq 1 180); do curl -sf "localhost:$1/v1/models" > /dev/null && return 0; sleep 5; done; return 1; }
stop_servers() { for pid in $(pgrep -f "vllm serve Qwen/"); do log "stopping vllm pid $pid"; kill $pid; done
                 for i in $(seq 1 60); do pgrep -f "vllm serve" > /dev/null || break; sleep 5; done; }
W=16
JOBS=("birdreact bird 199" "birdmodular bird 199" "lcbrepair lcb 85")
# Evolution: the proposer starts first (its cap is a share of the whole card), the solver takes the rest.
stop_servers
nohup vllm serve Qwen/Qwen3-14B-AWQ --port 8100 --gpu-memory-utilization 0.68 --max-model-len 12288 --max-num-seqs 2 \
  --kv-cache-dtype fp8_e5m2 --enable-chunked-prefill --max-num-batched-tokens 1024 --enforce-eager --seed 0 \
  > runs/logs/vllm_proposer_a5.log 2>&1 < /dev/null 9>&- &
wait_ready 8100 || { log "proposer did not start"; exit 1; }
nohup vllm serve Qwen/Qwen3-4B-AWQ --port 8000 --gpu-memory-utilization 0.95 --max-model-len 8192 --max-num-seqs 32 \
  --enable-chunked-prefill --max-num-batched-tokens 2048 --seed 0 > runs/logs/vllm_solver_a5.log 2>&1 < /dev/null 9>&- &
wait_ready 8000 || { log "solver did not start"; exit 1; }
log "evolution servers up"
for job in "${JOBS[@]}"; do
  set -- $job
  log "A5 evolution $1 start"
  python scripts/run_evolution.py configs/gain_pilot5_$1.yaml > runs/logs/pilot5_$1.log 2>&1
  log "A5 evolution $1 exit $?"
done
# Gain checks and Phase B: the solver alone, as in every other phase.
stop_servers
bash scripts/gpu/serve.sh lattice 9>&- || { log "solver did not start"; exit 1; }
log "lattice solver up"
for job in "${JOBS[@]}"; do
  set -- $job
  grep -q "real-gain loops:" runs/logs/gain5_$1.log 2>/dev/null || \
    python scripts/gain_check.py configs/backend_gpu.yaml $2 runs/gain_pilot5_$1 --n-tasks $3 --replicates 4 > runs/logs/gain5_$1.log 2>&1
  log "A5 gain $1: $(tail -1 runs/logs/gain5_$1.log)"
done
for job in "${JOBS[@]}"; do
  read -r seedname bench held <<< "$job"
  for f in runs/gain_pilot5_$seedname/*.log.jsonl; do
    loop=$(basename $f .log.jsonl)
    grep -q '"accepted": true' $f || { log "$seedname $loop: no accepted edit, skipped"; continue; }
    name=phaseA5_${seedname}_$loop
    [ -f results/report_${name}_stmt.json ] && { log "$name: report exists, skipped"; continue; }
    common="$bench --log $f --harness-dir runs/harnesses --statement-level --tree --workers $W"
    python scripts/run_lattice.py configs/backend_gpu.yaml $common --name ${name}_audit --namespace $name \
      --split search --n-tasks 10 --replicates 1 --brute --brute-all > runs/logs/b_${name}_audit.log 2>&1
    log "$name audit: $(tail -1 runs/logs/b_${name}_audit.log)"
    python scripts/run_lattice.py configs/backend_gpu.yaml $common --name $name --split search --n-tasks 60 --replicates 4 \
      > runs/logs/b_${name}_search.log 2>&1
    log "$name search: $(tail -1 runs/logs/b_${name}_search.log)"
    python scripts/run_lattice.py configs/backend_gpu.yaml $common --name $name --split heldout --n-tasks $held --replicates 4 \
      > runs/logs/b_${name}_heldout.log 2>&1
    log "$name heldout: $(tail -1 runs/logs/b_${name}_heldout.log)"
    python scripts/a2_audit.py configs/backend_gpu.yaml $bench runs/lattice/${name}_stmt_heldout.json --fraction 0.1 --workers $W \
      > runs/logs/b_${name}_a2.log 2>&1
    log "$name fresh-run audit: $(tail -1 runs/logs/b_${name}_a2.log)"
    python scripts/rq_report.py runs/lattice/${name}_stmt_search.json runs/lattice/${name}_stmt_heldout.json --n-boot 2000 \
      > runs/logs/b_${name}_report.log 2>&1
    log "$name report: $(tail -1 runs/logs/b_${name}_report.log)"
  done
done
log "STAGE A5 DONE"
