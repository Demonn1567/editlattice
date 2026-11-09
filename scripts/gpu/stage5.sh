#!/usr/bin/env bash
# Stage 5 (Phase A4, pre-registered addendum 2026-11-02): Phase A3 with the Qwen3-8B-AWQ proposer on the GPU
# (18,432-token context, thinking on; DEC-15). GSM-Hard, LCB, BIRD; each followed by the gain check. Exclusive lock.
set -uo pipefail
cd "$(dirname "$0")/../.."
exec 9>runs/locks_stage5.lock
flock -n 9 || { echo "another stage5 is running"; exit 1; }
export PATH=/usr/lib/wsl/lib:$PATH
. .venv-gpu/bin/activate
log() { echo "$(date '+%F %T') $*"; }
wait_ready() { for _ in $(seq 1 180); do curl -sf "localhost:$1/v1/models" > /dev/null && return 0; sleep 5; done; return 1; }
# Free the GPU: stop any vLLM server (RQ6 is done), by PID. vLLM counts memory held by other processes
# against --gpu-memory-utilization, so the solver, started second, gets a whole-card budget (0.93).
for pid in $(pgrep -f "vllm serve Qwen/"); do log "stopping vllm pid $pid"; kill $pid; done
for i in $(seq 1 60); do pgrep -f "vllm serve" > /dev/null || break; sleep 5; done
nohup vllm serve Qwen/Qwen3-8B-AWQ --port 8100 --gpu-memory-utilization 0.62 --max-model-len 18432 --max-num-seqs 2 \
  --enforce-eager --seed 0 > runs/logs/vllm_proposer_a4.log 2>&1 < /dev/null 9>&- &
wait_ready 8100 || { log "proposer did not start"; exit 1; }
nohup vllm serve Qwen/Qwen3-4B-AWQ --port 8000 --gpu-memory-utilization 0.93 --max-model-len 8192 --max-num-seqs 32 \
  --seed 0 > runs/logs/vllm_solver_a4.log 2>&1 < /dev/null 9>&- &
wait_ready 8000 || { log "solver did not start"; exit 1; }
log "servers up"
for job in "gsmhard 200" "lcb 85" "bird 199"; do
  set -- $job
  log "A4 evolution $1 start"
  python scripts/run_evolution.py configs/gain_pilot4_$1.yaml > runs/logs/pilot4_$1.log 2>&1
  log "A4 evolution $1 exit $?"
  python scripts/gain_check.py configs/backend_gpu.yaml $1 runs/gain_pilot4_$1 --n-tasks $2 --replicates 4 > runs/logs/gain4_$1.log 2>&1
  log "A4 gain $1: $(tail -1 runs/logs/gain4_$1.log)"
done
log "STAGE5 DONE"
