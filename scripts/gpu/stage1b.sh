#!/usr/bin/env bash
# Stage 1b: both models served together, evolution smoke test. Stops for review.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PATH=/usr/lib/wsl/lib:$PATH
. .venv-gpu/bin/activate
log() { echo "$(date '+%F %T') $*"; }
bash scripts/gpu/serve.sh stop
bash scripts/gpu/serve.sh evolve || { log "evolve servers did not start"; exit 1; }
start=$(date +%s)
python scripts/run_evolution.py configs/gain_smoke_gsmhard.yaml > runs/logs/smoke.log 2>&1
log "smoke exit $? after $(( $(date +%s) - start )) s"
log "STAGE1B DONE"
