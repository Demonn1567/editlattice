#!/usr/bin/env bash
# DEC-19: rerun the pooled A4 GSM-Hard lattices and run the pooled A4 LCB lattices under the fixed conflict rule.
# Same commands as stage_b5's pool(); stored responses make reruns exact. Exclusive lock.
exec 9>runs/locks_fix_poolA4_dec19.lock
flock -n 9 || { echo "already running"; exit 1; }
set -uo pipefail
cd "$(dirname "$0")/../.."
export PATH=/usr/lib/wsl/lib:$PATH
. .venv-gpu/bin/activate
log() { echo "$(date '+%F %T') $*"; }
for job in "gsmhard 200" "lcb 85"; do
  set -- $job
  bench=$1 held=$2 name=pooledA4_$1
  logs=$(for f in runs/gain_pilot4_$bench/*.log.jsonl; do grep -q '"accepted": true' $f && echo $f; done)
  log "$name: $(python scripts/pool_size.py runs/harnesses $logs) conflict-free products"
  for split in "search 60" "heldout $held"; do
    set -- $split
    [ -f runs/lattice/${name}_$1.json ] && mv runs/lattice/${name}_$1.json runs/lattice/${name}_$1_pre_dec19.json
    python scripts/run_lattice.py configs/backend_gpu.yaml $bench --pool $logs --harness-dir runs/harnesses --name $name \
      --split $1 --n-tasks $2 --replicates 4 --tree --workers 16 > runs/logs/b_${name}_$1.log 2>&1
    log "$name $1: $(tail -1 runs/logs/b_${name}_$1.log)"
  done
  python scripts/rq_report.py runs/lattice/${name}_search.json runs/lattice/${name}_heldout.json --n-boot 2000 > runs/logs/b_${name}_report.log 2>&1
  log "$name report: $(tail -1 runs/logs/b_${name}_report.log)"
done
log "FIX DEC19 DONE"
