#!/usr/bin/env bash
# Stage 1c (DEC-2): wait for the 8B-AWQ proposer, verify it, then stage 1b (servers + smoke test).
set -uo pipefail
cd "$(dirname "$0")/../.."
. .venv-gpu/bin/activate
log() { echo "$(date '+%F %T') $*"; }
until grep -q "^downloaded Qwen/Qwen3-8B-AWQ" runs/logs/download_8b.log 2>/dev/null; do
  grep -q "attempt 5" runs/logs/download_8b.log 2>/dev/null && { log "8B download FAILED"; exit 1; }; sleep 30; done
python scripts/gpu/verify_models.py Qwen/Qwen3-8B-AWQ > runs/logs/verify_8b.log 2>&1 || { log "8B checksum FAILED"; exit 1; }
log "8B-AWQ verified"
bash scripts/gpu/stage1b.sh
log "STAGE1C DONE"
