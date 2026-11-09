#!/usr/bin/env bash
# Exploratory cross-family setting on the GPU (preregistration addendum 2026-11-06): Llama-3.1-8B-Instruct AWQ INT4
# under vLLM, alone on the A4000. Every lineage of the transfer table, in the registered order: audit on 10 search
# tasks, then the held-out lattice and its gain. Exclusive lock.
set -uo pipefail
cd "$(dirname "$0")/../.."
exec 9>runs/locks_stage_llama_awq.lock
flock -n 9 || { echo "another stage_llama_awq is running"; exit 1; }
export PATH=/usr/lib/wsl/lib:$PATH
. .venv-gpu/bin/activate
log() { echo "$(date '+%F %T') $*"; }
W=16
MODEL=hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4
REV=db1f81ad4b8c7e39777509fac66c652eb0a52f91
CFG=configs/backend_gpu_llama_awq.yaml
for pid in $(pgrep -f "vllm serve"); do log "stopping vllm pid $pid"; kill $pid; done
for i in $(seq 1 60); do pgrep -f "vllm serve" > /dev/null || break; sleep 5; done
python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL', revision='$REV')" || { log "download failed"; exit 1; }
nohup vllm serve $MODEL --revision $REV --port 8000 --gpu-memory-utilization 0.90 --max-model-len 8192 --max-num-seqs 32 \
  --seed 0 > runs/logs/vllm_llama_awq.log 2>&1 < /dev/null 9>&- &
for i in $(seq 1 120); do curl -sf localhost:8000/v1/models > /dev/null && break; sleep 10; done
curl -sf localhost:8000/v1/models > /dev/null || { log "llama server did not start"; exit 1; }
log "llama awq server up"
G=runs
# name | benchmark | held-out tasks | lineage arguments
JOBS=(
  "planted_llama_awq|gsmhard|200|--seed src/editlattice/seed_harnesses/math_direct.py --final examples/gsm_planted_final.py"
  "constructed_gsmhard_llama_awq|gsmhard|200|--seed src/editlattice/seed_harnesses/math_direct.py --final examples/constructed/gsm_constructed_final.py"
  "constructed_lcb_llama_awq|lcb|85|--seed src/editlattice/seed_harnesses/lcb_oneshot.py --final examples/constructed/lcb_constructed_final.py"
  "constructed_bird_llama_awq|bird|199|--seed src/editlattice/seed_harnesses/bird_oneshot.py --final examples/constructed/bird_constructed_final.py"
  "phaseA3_lcb_metaharness_B1_s2_llama_awq|lcb|85|--log $G/gain_pilot3_lcb/metaharness_B1_s2.log.jsonl --harness-dir $G/harnesses"
  "phaseA3_bird_metaharness_B1_s1_llama_awq|bird|199|--log $G/gain_pilot3_bird/metaharness_B1_s1.log.jsonl --harness-dir $G/harnesses"
  "phaseA3_bird_metaharness_B3_s1_llama_awq|bird|199|--log $G/gain_pilot3_bird/metaharness_B3_s1.log.jsonl --harness-dir $G/harnesses"
)
for job in "${JOBS[@]}"; do
  IFS='|' read -r NAME BENCH HELD ARGS <<< "$job"
  if [ ! -f runs/lattice/${NAME}_audit_stmt_search.json ]; then
    python scripts/run_lattice.py $CFG $BENCH $ARGS --name ${NAME}_audit --namespace $NAME --split search --n-tasks 10 \
      --replicates 1 --statement-level --tree --brute --workers $W > runs/logs/llama_awq_${NAME}_audit.log 2>&1
    log "$NAME audit: $(tail -1 runs/logs/llama_awq_${NAME}_audit.log)"
  fi
  if [ ! -f runs/lattice/${NAME}_stmt_heldout.json ]; then
    python scripts/run_lattice.py $CFG $BENCH $ARGS --name $NAME --namespace $NAME --split heldout --n-tasks $HELD \
      --replicates 4 --statement-level --tree --workers $W > runs/logs/llama_awq_${NAME}.log 2>&1
    log "$NAME heldout: $(tail -1 runs/logs/llama_awq_${NAME}.log)"
  fi
  python scripts/lattice_gain.py runs/lattice/${NAME}_stmt_heldout.json > runs/logs/llama_awq_${NAME}_gain.log 2>&1
  log "$NAME gain: $(tail -1 runs/logs/llama_awq_${NAME}_gain.log)"
done
log "STAGE LLAMA AWQ DONE"
