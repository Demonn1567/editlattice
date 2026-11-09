#!/usr/bin/env bash
# Stage 1 on the A4000: verified models, seed baselines (solver only), evolution smoke test (both models).
# Stops after the smoke test so its proposals and prompt lengths can be checked before the pilot.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PATH=/usr/lib/wsl/lib:$PATH
. .venv-gpu/bin/activate
log() { echo "$(date '+%F %T') $*"; }
wait_for() { until grep -q "^downloaded $1" "$2" 2>/dev/null; do grep -q "attempt 5" "$2" 2>/dev/null && { log "download of $1 FAILED"; exit 1; }; sleep 30; done; }
wait_for Qwen/Qwen3-4B-AWQ runs/logs/download_4bawq.log
python scripts/gpu/verify_models.py Qwen/Qwen3-4B-AWQ > runs/logs/verify_4bawq.log 2>&1 || { log "4B-AWQ checksum FAILED"; exit 1; }
log "4B-AWQ verified"
bash scripts/gpu/serve.sh lattice || { log "solver did not start"; exit 1; }
bash scripts/gpu/seed_baselines.sh > runs/logs/seed_baselines.log 2>&1
log "baselines: $(grep -E 'accuracy|runs$' runs/logs/seed_baselines.log | tr '\n' ' ')"
wait_for Qwen/Qwen3-14B-AWQ runs/logs/download_14b.log
python scripts/gpu/verify_models.py Qwen/Qwen3-14B-AWQ > runs/logs/verify_14b.log 2>&1 || { log "14B-AWQ checksum FAILED"; exit 1; }
log "14B-AWQ verified"
bash scripts/gpu/serve.sh stop
bash scripts/gpu/serve.sh evolve || { log "evolve servers did not start"; exit 1; }
python scripts/run_evolution.py configs/gain_smoke_gsmhard.yaml > runs/logs/smoke.log 2>&1
log "smoke exit $?"
log "STAGE1 DONE"
