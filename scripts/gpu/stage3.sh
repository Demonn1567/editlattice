#!/usr/bin/env bash
# Stage 3 (Phase A2, pre-registered addendum 2026-10-28): 14B proposer on the Mac via SSH reverse
# tunnel (localhost:8101 on this machine), prompt v2. GSM-Hard then LCB, each followed by the gain check.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PATH=/usr/lib/wsl/lib:$PATH
. .venv-gpu/bin/activate
log() { echo "$(date '+%F %T') $*"; }
for i in $(seq 1 60); do curl -sf http://localhost:8101/v1/models > /dev/null && break; log "waiting for the proposer tunnel"; sleep 30; done
for job in "gsmhard 200" "lcb 85"; do
  set -- $job
  log "A2 evolution $1 start"
  python scripts/run_evolution.py configs/gain_pilot2_$1.yaml > runs/logs/pilot2_$1.log 2>&1
  log "A2 evolution $1 exit $?"
  python scripts/gain_check.py configs/backend_gpu.yaml $1 runs/gain_pilot2_$1 --n-tasks $2 --replicates 4 > runs/logs/gain2_$1.log 2>&1
  log "A2 gain $1: $(tail -1 runs/logs/gain2_$1.log)"
done
log "STAGE3 DONE"
