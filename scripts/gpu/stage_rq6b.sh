#!/usr/bin/env bash
# DEC-13: clean rerun of the planted RQ6 lattices (the first run was executed by duplicate processes).
# Exclusive lock: a second launch exits at once. Reuses the bf16 server on :8100 if it is up.
set -uo pipefail
cd "$(dirname "$0")/../.."
exec 9>runs/locks_stage_rq6b.lock
flock -n 9 || { echo "another stage_rq6b is running"; exit 1; }
export PATH=/usr/lib/wsl/lib:$PATH
. .venv-gpu/bin/activate
log() { echo "$(date '+%F %T') $*"; }
curl -sf localhost:8100/v1/models > /dev/null || { log "bf16 server not up"; exit 1; }
for f in runs/lattice/planted_gpu*_search.json runs/lattice/planted_gpu*_heldout.json results/a2_planted_gpu_stmt_heldout.json results/report_planted_gpu_stmt.json; do
  [ -f "$f" ] && mv "$f" "$(dirname $f)/dup_$(basename $f)"
done
S=src/editlattice/seed_harnesses/math_direct.py; F=examples/gsm_planted_final.py; W=16
run() {  # config name split n
  python scripts/run_lattice.py $1 gsmhard --seed $S --final $F --name $2 --split $3 --n-tasks $4 --replicates 4 --statement-level --tree --workers $W > runs/logs/rq6b_$2_$3.log 2>&1
  log "$2 $3: $(tail -1 runs/logs/rq6b_$2_$3.log)"
}
python scripts/run_lattice.py configs/backend_gpu.yaml gsmhard --seed $S --final $F --name planted_gpu_audit --namespace planted_gpu --split search --n-tasks 10 --replicates 1 --statement-level --tree --brute --workers $W > runs/logs/rq6b_audit.log 2>&1
log "audit: $(tail -1 runs/logs/rq6b_audit.log)"
run configs/backend_gpu.yaml planted_gpu search 60
run configs/backend_gpu.yaml planted_gpu heldout 200
python scripts/a2_audit.py configs/backend_gpu.yaml gsmhard runs/lattice/planted_gpu_stmt_heldout.json --fraction 0.1 --workers $W > runs/logs/rq6b_a2.log 2>&1
log "A2 audit: $(tail -1 runs/logs/rq6b_a2.log)"
python scripts/rq_report.py runs/lattice/planted_gpu_stmt_search.json runs/lattice/planted_gpu_stmt_heldout.json --n-boot 2000 > /dev/null 2>&1
run configs/backend_gpu_t1.yaml planted_gpu_t1 heldout 200
run configs/backend_gpu_bf16.yaml planted_gpu_bf16 heldout 200
python scripts/run_lattice.py configs/backend_gpu.yaml gsmhard --seed $S --final $F --name planted_gpu_namelevel --namespace planted_gpu --split search --n-tasks 60 --replicates 4 --tree --workers $W > runs/logs/rq6b_namelevel.log 2>&1
log "name-level: $(tail -1 runs/logs/rq6b_namelevel.log)"
log "STAGE RQ6B DONE"
