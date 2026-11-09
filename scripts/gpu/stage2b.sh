#!/usr/bin/env bash
# Stage 2b (DEC-4): LCB pilot restarted under the first-line indentation rule, then BIRD. GSM-Hard is done (stage2.sh).
# For each benchmark: 8 evolution loops (B1 and B3 x 4 seeds), then the held-out gain check.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PATH=/usr/lib/wsl/lib:$PATH
. .venv-gpu/bin/activate
log() { echo "$(date '+%F %T') $*"; }
for job in "lcb 85" "bird 199"; do
  set -- $job
  log "evolution $1 start"
  python scripts/run_evolution.py configs/gain_pilot_$1.yaml > runs/logs/pilot_$1.log 2>&1
  log "evolution $1 exit $?"
  python scripts/gain_check.py configs/backend_gpu.yaml $1 runs/gain_pilot_$1 --n-tasks $2 --replicates 4 > runs/logs/gain_$1.log 2>&1
  log "gain $1: $(tail -1 runs/logs/gain_$1.log)"
done
log "STAGE2B DONE"
