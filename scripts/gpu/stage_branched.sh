#!/usr/bin/env bash
# RQ1c (preregistration addendum 2026-11-09): branched lineages, whose units sit mostly on failure paths. Per benchmark,
# BIRD first: crash check on 5 search tasks, exhaustive brute-force audit (10 search tasks x 1, 4,096 products), search
# and held-out lattices, D8 gain, fresh-run audit on 0.1% of held-out pairs, RQ3/RQ4 report. Exclusive lock; a step
# whose output exists is not rerun.
set -uo pipefail
cd "$(dirname "$0")/../.."
exec 9>runs/locks_stage_branched.lock
flock -n 9 || { echo "another stage_branched is running"; exit 1; }
export PATH=/usr/lib/wsl/lib:$PATH
. .venv-gpu/bin/activate
log() { echo "$(date '+%F %T') $*"; }
W=16
CFG=configs/backend_gpu.yaml
if ! curl -sf localhost:8000/v1/models | grep -q Qwen3-4B-AWQ; then
  for pid in $(pgrep -f "vllm serve"); do log "stopping vllm pid $pid"; kill $pid; done
  for i in $(seq 1 60); do pgrep -f "vllm serve" > /dev/null || break; sleep 5; done
  bash scripts/gpu/serve.sh lattice 9>&- || { log "solver server did not start"; exit 1; }
fi
log "solver server up"
declare -A HELD=([bird]=199 [lcb]=85)
for bench in bird lcb; do
  name=branched_$bench
  common="$bench --seed examples/branched/${bench}_branched_seed.py --final examples/branched/${bench}_branched_final.py --statement-level --tree --workers $W --namespace $name"
  if [ ! -f runs/logs/${name}_crash.log ]; then
    python scripts/seed_eval.py $CFG examples/branched/${bench}_branched_final.py $bench --split search --n-tasks 5 \
      --out runs/branched_crash_$bench.json > runs/logs/${name}_crash.log 2>&1
    log "$name crash check: $(tail -1 runs/logs/${name}_crash.log)"
  fi
  if [ ! -f runs/lattice/${name}_audit_stmt_search.json ]; then
    python scripts/run_lattice.py $CFG $common --name ${name}_audit --split search --n-tasks 10 --replicates 1 --brute > runs/logs/${name}_audit.log 2>&1
    log "$name audit: $(tail -1 runs/logs/${name}_audit.log)"
  fi
  if [ ! -f runs/lattice/${name}_stmt_search.json ]; then
    python scripts/run_lattice.py $CFG $common --name $name --split search --n-tasks 60 --replicates 4 > runs/logs/${name}_search.log 2>&1
    log "$name search: $(tail -1 runs/logs/${name}_search.log)"
  fi
  if [ ! -f runs/lattice/${name}_stmt_heldout.json ]; then
    python scripts/run_lattice.py $CFG $common --name $name --split heldout --n-tasks ${HELD[$bench]} --replicates 4 > runs/logs/${name}_heldout.log 2>&1
    log "$name heldout: $(tail -1 runs/logs/${name}_heldout.log)"
  fi
  log "$name D8 gain: $(python scripts/lattice_gain.py runs/lattice/${name}_stmt_heldout.json 2>&1 | tail -1)"
  if [ ! -f results/a2_${name}_stmt_heldout.json ]; then
    python scripts/a2_audit.py $CFG $bench runs/lattice/${name}_stmt_heldout.json --fraction 0.001 --workers $W > runs/logs/${name}_a2.log 2>&1
    log "$name fresh-run audit: $(tail -1 runs/logs/${name}_a2.log)"
  fi
  python scripts/rq_report.py runs/lattice/${name}_stmt_search.json runs/lattice/${name}_stmt_heldout.json --n-boot 2000 > runs/logs/${name}_report.log 2>&1
  log "$name report: $(tail -1 runs/logs/${name}_report.log)"
done
log "STAGE BRANCHED DONE"
