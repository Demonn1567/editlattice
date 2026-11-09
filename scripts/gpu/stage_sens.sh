#!/usr/bin/env bash
# DEC-16 sensitivity analysis: LiveCodeBench gain checks of Phases A, A2, A3 with the evolution-time solver
# output budget (2,048 tokens). Writes to results/sens_lcb2048/ (never over the registered results). Exclusive lock.
set -uo pipefail
cd "$(dirname "$0")/../.."
exec 9>runs/locks_stage_sens.lock
flock -n 9 || { echo "another stage_sens is running"; exit 1; }
. .venv-gpu/bin/activate
log() { echo "$(date '+%F %T') $*"; }
export EDITLATTICE_RESULTS_DIR=results/sens_lcb2048
mkdir -p "$EDITLATTICE_RESULTS_DIR"
until curl -sf localhost:8000/v1/models > /dev/null; do sleep 60; done
for prefix in gain_pilot gain_pilot2 gain_pilot3; do
  python scripts/gain_check.py configs/backend_gpu_lcb2048.yaml lcb runs/${prefix}_lcb --n-tasks 85 --replicates 4 --workers 16 > runs/logs/sens_${prefix}_lcb.log 2>&1
  log "$prefix lcb at 2048 tokens: $(tail -1 runs/logs/sens_${prefix}_lcb.log)"
done
log "STAGE SENS DONE"
