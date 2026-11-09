#!/usr/bin/env bash
# Brute-force audit of every pooled lattice (preregistration addendum 2026-11-09): 10 search tasks x 1 replicate, every
# conflict-free product run alone and compared with the tree. Same units and read model as the pooled lattices
# (stage_b2-b5, DEC-19). Waits for stage_branched to finish. Exclusive lock; an existing audit is not rerun.
set -uo pipefail
cd "$(dirname "$0")/../.."
exec 9>runs/locks_stage_pool_audit.lock
flock -n 9 || { echo "another stage_pool_audit is running"; exit 1; }
export PATH=/usr/lib/wsl/lib:$PATH
. .venv-gpu/bin/activate
log() { echo "$(date '+%F %T') $*"; }
W=16
until grep -q "STAGE BRANCHED DONE" runs/logs/stage_branched.log 2>/dev/null; do sleep 120; done
curl -sf localhost:8000/v1/models | grep -q Qwen3-4B-AWQ || { log "solver server is not up"; exit 1; }
audit() {  # pilot bench name
  local pilot=$1 bench=$2 name=$3
  logs=$(for f in runs/$pilot/*.log.jsonl; do grep -q '"accepted": true' $f && echo $f; done)
  [ -f runs/lattice/${name}_audit_search.json ] && { log "$name audit exists, skipped"; return; }
  python scripts/run_lattice.py configs/backend_gpu.yaml $bench --pool $logs --harness-dir runs/harnesses --name ${name}_audit --brute-all \
    --namespace $name --split search --n-tasks 10 --replicates 1 --tree --brute --workers $W > runs/logs/pool_audit_${name}.log 2>&1
  log "$name audit: $(tail -1 runs/logs/pool_audit_${name}.log)"
}
audit gain_pilot_gsmhard gsmhard pooledA_gsmhard
audit gain_pilot_bird bird pooledA_bird
audit gain_pilot2_gsmhard gsmhard pooledA2_gsmhard
audit gain_pilot2_lcb lcb pooledA2_lcb
audit gain_pilot2_bird bird pooledA2_bird
audit gain_pilot3_gsmhard gsmhard pooledA3_gsmhard
audit gain_pilot3_lcb lcb pooledA3_lcb
audit gain_pilot3_bird bird pooledA3_bird
audit gain_pilot4_gsmhard gsmhard pooledA4_gsmhard
audit gain_pilot4_lcb lcb pooledA4_lcb
log "STAGE POOL AUDIT DONE"
