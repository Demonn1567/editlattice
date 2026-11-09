#!/usr/bin/env bash
# RQ6 on the evolved real-gain lineages (preregistration addendum 2026-11-01) and the DEC-11 bf16 reruns of
# constructed LCB/BIRD with the full context. Waits for Phase B (stage_b4) to finish. Exclusive lock.
set -uo pipefail
cd "$(dirname "$0")/../.."
exec 9>runs/locks_stage_rq6c.lock
flock -n 9 || { echo "another stage_rq6c is running"; exit 1; }
export PATH=/usr/lib/wsl/lib:$PATH
. .venv-gpu/bin/activate
log() { echo "$(date '+%F %T') $*"; }
W=16
until grep -q "STAGE B4 DONE" runs/logs/stage_b4.log 2>/dev/null; do sleep 300; done
log "stage_b4 done; starting AWQ T=1.0 lattices"
LINEAGES="lcb:85:gain_pilot3_lcb/metaharness_B1_s2 bird:199:gain_pilot3_bird/metaharness_B1_s1 bird:199:gain_pilot3_bird/metaharness_B3_s1"
evolved() {  # config suffix
  for item in $LINEAGES; do
    IFS=: read bench held path <<< "$item"
    name=phaseA3_${bench}_$(basename $path)_$2
    python scripts/run_lattice.py $1 $bench --log runs/$path.log.jsonl --harness-dir runs/harnesses --name $name \
      --split heldout --n-tasks $held --replicates 4 --statement-level --tree --workers $W > runs/logs/rq6c_$name.log 2>&1
    log "$name: $(tail -1 runs/logs/rq6c_$name.log)"
  done
}
evolved configs/backend_gpu_t1.yaml t1
# Both servers stop; bf16 restarts alone with the solver's context length.
for port in 8100 8000; do
  pid=$(pgrep -f "vllm serve Qwen/Qwen3-4B[-A-Z]* --port $port" | head -1)
  [ -n "$pid" ] && { log "stopping vllm on $port (pid $pid)"; kill $pid; }
done
for i in $(seq 1 60); do pgrep -f "vllm serve" > /dev/null || break; sleep 5; done
nohup vllm serve Qwen/Qwen3-4B --port 8100 --gpu-memory-utilization 0.90 --max-model-len 8192 --max-num-seqs 16 --seed 0 > runs/logs/vllm_bf16_full.log 2>&1 < /dev/null &
for i in $(seq 1 90); do curl -sf localhost:8100/v1/models > /dev/null && break; sleep 10; done
curl -sf localhost:8100/v1/models > /dev/null || { log "bf16 server did not start"; exit 1; }
log "bf16 full-context server up"
declare -A SEED=([lcb]=lcb_oneshot [bird]=bird_oneshot)
declare -A HELD=([lcb]=85 [bird]=199)
for bench in lcb bird; do
  name=constructed_${bench}_bf16full
  python scripts/run_lattice.py configs/backend_gpu_bf16.yaml $bench --seed src/editlattice/seed_harnesses/${SEED[$bench]}.py \
    --final examples/constructed/${bench}_constructed_final.py --statement-level --tree --workers $W --split heldout \
    --n-tasks ${HELD[$bench]} --replicates 4 --name $name > runs/logs/rq6c_$name.log 2>&1
  log "$name: $(tail -1 runs/logs/rq6c_$name.log)"
done
evolved configs/backend_gpu_bf16.yaml bf16
log "STAGE RQ6C DONE"
