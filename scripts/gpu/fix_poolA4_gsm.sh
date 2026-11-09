#!/usr/bin/env bash
# Rerun of the pooled A4 GSM-Hard search lattice, which failed when stage_b5 started it before stage5's solver was
# ready (results_log 2026-11-05). Waits for the held-out pool lattice, then runs the search lattice with the same
# command stage_b5 uses and writes the pool report. Exclusive lock.
exec 9>runs/locks_fix_poolA4_gsm.lock
flock -n 9 || { echo "already running"; exit 1; }
set -uo pipefail
cd "$(dirname "$0")/../.."
export PATH=/usr/lib/wsl/lib:$PATH
. .venv-gpu/bin/activate
log() { echo "$(date '+%F %T') $*"; }
name=pooledA4_gsmhard
until [ -f runs/lattice/${name}_heldout.json ]; do sleep 300; done
logs=$(for f in runs/gain_pilot4_gsmhard/*.log.jsonl; do grep -q '"accepted": true' $f && echo $f; done)
if [ ! -f runs/lattice/${name}_search.json ]; then
  python scripts/run_lattice.py configs/backend_gpu.yaml gsmhard --pool $logs --harness-dir runs/harnesses --name $name \
    --split search --n-tasks 60 --replicates 4 --tree --workers 16 > runs/logs/b_${name}_search.log 2>&1
  log "$name search: $(tail -1 runs/logs/b_${name}_search.log)"
fi
python scripts/rq_report.py runs/lattice/${name}_search.json runs/lattice/${name}_heldout.json --n-boot 2000 > runs/logs/b_${name}_report.log 2>&1
log "$name report: $(tail -1 runs/logs/b_${name}_report.log)"
log "FIX POOL DONE"
