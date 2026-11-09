#!/bin/zsh
# Exploratory cross-family transfer of the remaining lineages (preregistration addendum 2026-11-03) under
# Llama-3.1-8B (MLX 4-bit) on the Mac, in the registered order. Each lineage: brute-force audit on 10 search
# tasks, then the held-out lattice. A lineage whose held-out lattice exists is skipped, and cached responses
# make a rerun resume, so the script can be stopped and restarted at any time.
# Stop: touch runs/LLAMA_STOP
cd "$(dirname "$0")/.."
LOG=runs/mac_llama_transfer2.log
log() { echo "$(date '+%F %T') $*" >> $LOG; }
caffeinate -i -w $$ &
rm -f runs/LLAMA_STOP
PY=.venv/bin/python
MODEL=mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
CFG=configs/backend_mlx_llama.yaml
G=runs/gpu_pilot
server_up() { curl -sf -m 10 localhost:8081/v1/models > /dev/null; }
ensure_server() {
  server_up && return 0
  pgrep -f "mlx_server_capped.py --model $MODEL" > /dev/null || {
    log "starting server"
    nohup $PY runs/mlx_server_capped.py --model $MODEL --host 127.0.0.1 --port 8081 --prompt-cache-size 1 \
      --prompt-cache-bytes 268435456 >> runs/mlx_server_llama.log 2>&1 &
  }
  pgrep -f mem_guard_a2.sh > /dev/null || nohup zsh runs/mem_guard_a2.sh > /dev/null 2>&1 &
  for i in $(seq 1 60); do server_up && return 0; sleep 5; done
  return 1
}
# name | benchmark | held-out tasks | lineage arguments
JOBS=(
  "phaseA3_bird_metaharness_B1_s1_llama|bird|199|--log $G/gain_pilot3_bird/metaharness_B1_s1.log.jsonl --harness-dir $G/harnesses"
  "phaseA3_lcb_metaharness_B1_s2_llama|lcb|85|--log $G/gain_pilot3_lcb/metaharness_B1_s2.log.jsonl --harness-dir $G/harnesses"
  "constructed_bird_llama|bird|199|--seed src/editlattice/seed_harnesses/bird_oneshot.py --final examples/constructed/bird_constructed_final.py"
  "constructed_lcb_llama|lcb|85|--seed src/editlattice/seed_harnesses/lcb_oneshot.py --final examples/constructed/lcb_constructed_final.py"
  "constructed_gsmhard_llama|gsmhard|200|--seed src/editlattice/seed_harnesses/math_direct.py --final examples/constructed/gsm_constructed_final.py"
)
for job in $JOBS; do
  IFS='|' read -r NAME BENCH HELD ARGS <<< "$job"
  [ -f runs/lattice/${NAME}_stmt_heldout.json ] && { log "$NAME: held-out lattice exists, skipped"; continue; }
  for step in audit heldout; do
    [ $step = audit ] && [ -f runs/lattice/${NAME}_audit_stmt_search.json ] && continue
    while [ ! -f runs/LLAMA_STOP ]; do
      ensure_server || { sleep 30; continue; }
      if [ $step = audit ]; then
        .venv/bin/python scripts/run_lattice.py $CFG $BENCH ${=ARGS} --name ${NAME}_audit --namespace $NAME \
          --split search --n-tasks 10 --replicates 1 --statement-level --tree --brute --workers 1 > runs/logs_llama2_${NAME}_audit.log 2>&1
      else
        .venv/bin/python scripts/run_lattice.py $CFG $BENCH ${=ARGS} --name $NAME --namespace $NAME \
          --split heldout --n-tasks $HELD --replicates 4 --statement-level --tree --workers 1 > runs/logs_llama2_${NAME}.log 2>&1
      fi
      rc=$?
      f=runs/logs_llama2_${NAME}$([ $step = audit ] && echo _audit).log
      log "$NAME $step exit $rc: $(tail -1 $f)"
      [ $rc -eq 0 ] && break
      sleep 30
    done
    [ -f runs/LLAMA_STOP ] && { log "stop requested"; exit 0; }
  done
done
log "all done; stopping server"
for p in $(pgrep -f "mlx_server_capped.py --model $MODEL"); do kill $p; done
