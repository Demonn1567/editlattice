#!/usr/bin/env bash
# Phase A2 on BIRD (pre-registered: runs because BIRD's Phase A failed the entry rule, 0 of 8),
# after stage3.sh (GSM-Hard, LCB) so the three do not compete for the single Mac proposer.
set -uo pipefail
cd "$(dirname "$0")/../.."
. .venv-gpu/bin/activate
log() { echo "$(date '+%F %T') $*"; }
until grep -q "STAGE3 DONE" runs/logs/stage3.log 2>/dev/null; do sleep 300; done
log "A2 evolution bird start"
python scripts/run_evolution.py configs/gain_pilot2_bird.yaml > runs/logs/pilot2_bird.log 2>&1
log "A2 evolution bird exit $?"
python scripts/gain_check.py configs/backend_gpu.yaml bird runs/gain_pilot2_bird --n-tasks 199 --replicates 4 > runs/logs/gain2_bird.log 2>&1
log "A2 gain bird: $(tail -1 runs/logs/gain2_bird.log)"
log "STAGE3B DONE"
