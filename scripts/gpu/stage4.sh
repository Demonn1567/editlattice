#!/usr/bin/env bash
# Stage 4 (Phase A3, pre-registered 2026-10-29): A2 with the proposer's thinking mode on; 14B proposer
# on the Mac via the SSH reverse tunnel (localhost:8101). GSM-Hard, LCB, BIRD; each followed by the gain check.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PATH=/usr/lib/wsl/lib:$PATH
. .venv-gpu/bin/activate
log() { echo "$(date '+%F %T') $*"; }
for i in $(seq 1 120); do curl -sf http://localhost:8101/v1/models > /dev/null && break; log "waiting for the proposer tunnel"; sleep 60; done
for job in "gsmhard 200" "lcb 85" "bird 199"; do
  set -- $job
  log "A3 evolution $1 start"
  python scripts/run_evolution.py configs/gain_pilot3_$1.yaml > runs/logs/pilot3_$1.log 2>&1
  log "A3 evolution $1 exit $?"
  python scripts/gain_check.py configs/backend_gpu.yaml $1 runs/gain_pilot3_$1 --n-tasks $2 --replicates 4 > runs/logs/gain3_$1.log 2>&1
  log "A3 gain $1: $(tail -1 runs/logs/gain3_$1.log)"
done
log "STAGE4 DONE"
