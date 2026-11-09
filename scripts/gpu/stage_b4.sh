#!/usr/bin/env bash
# Phase B for the A3 lineages of each benchmark once its loops finish (same protocol as Phase A and A2).
exec 9>runs/locks_stage_b4.lock
flock -n 9 || { echo "another stage_b4 is running"; exit 1; }
set -uo pipefail
cd "$(dirname "$0")/../.."
export PATH=/usr/lib/wsl/lib:$PATH
. .venv-gpu/bin/activate
log() { echo "$(date '+%F %T') $*"; }
W=16
lineage() {  # pilot bench held prefix
  local pilot=$1 bench=$2 held=$3 prefix=$4
  for f in runs/$pilot/*.log.jsonl; do
    loop=$(basename $f .log.jsonl)
    grep -q '"accepted": true' $f || { log "$prefix $loop: no accepted edit, skipped"; continue; }
    name=${prefix}_${loop}
    python scripts/run_lattice.py configs/backend_gpu.yaml $bench --log $f --harness-dir runs/harnesses --name ${name}_audit --namespace $name \
      --split search --n-tasks 10 --replicates 1 --statement-level --tree --brute --workers $W > runs/logs/b_${name}_audit.log 2>&1
    log "$name audit: $(tail -1 runs/logs/b_${name}_audit.log)"
    for split in "search 60" "heldout $held"; do
      set -- $split
      python scripts/run_lattice.py configs/backend_gpu.yaml $bench --log $f --harness-dir runs/harnesses --name $name \
        --split $1 --n-tasks $2 --replicates 4 --statement-level --tree --workers $W > runs/logs/b_${name}_$1.log 2>&1
      log "$name $1: $(tail -1 runs/logs/b_${name}_$1.log)"
    done
    python scripts/a2_audit.py configs/backend_gpu.yaml $bench runs/lattice/${name}_stmt_heldout.json --fraction 0.1 --workers $W > runs/logs/b_${name}_a2.log 2>&1
    log "$name A2 audit: $(tail -1 runs/logs/b_${name}_a2.log)"
    python scripts/rq_report.py runs/lattice/${name}_stmt_search.json runs/lattice/${name}_stmt_heldout.json --n-boot 2000 > /dev/null 2>&1
  done
}
pool() {  # pilot bench held name
  local pilot=$1 bench=$2 held=$3 name=$4
  logs=$(for f in runs/$pilot/*.log.jsonl; do grep -q '"accepted": true' $f && echo $f; done)
  [ -z "$logs" ] && { log "$name: no lineage with accepted edits"; return; }
  size=$(python scripts/pool_size.py runs/harnesses $logs)
  if [ "$size" -gt 500 ]; then log "$name: $size conflict-free products > 500, not run (DEC-14)"; return; fi
  for split in "search 60" "heldout $held"; do
    set -- $split
    python scripts/run_lattice.py configs/backend_gpu.yaml $bench --pool $logs --harness-dir runs/harnesses --name $name \
      --split $1 --n-tasks $2 --replicates 4 --tree --workers $W > runs/logs/b_${name}_$1.log 2>&1
    log "$name $1: $(tail -1 runs/logs/b_${name}_$1.log)"
  done
  python scripts/rq_report.py runs/lattice/${name}_search.json runs/lattice/${name}_heldout.json --n-boot 2000 > runs/logs/b_${name}_report.log 2>&1
  log "$name report: $(tail -1 runs/logs/b_${name}_report.log)"
}
for job in "gsmhard 200" "lcb 85" "bird 199"; do
  set -- $job
  until grep -q "A3 gain $1" runs/logs/stage4.log 2>/dev/null; do sleep 600; done
  lineage gain_pilot3_$1 $1 $2 phaseA3_$1
  pool gain_pilot3_$1 $1 $2 pooledA3_$1
done
log "STAGE B4 DONE"
