#!/usr/bin/env bash
# Phase B on the Phase A (null-gain) lineages, pre-registered lattices: per lineage, tree exploration
# with statement-level reads on 60 search tasks x 4 and all held-out tasks x 4, the brute-force audit
# on 10 search tasks x 1, the 10% fresh-run A2 audit on the held-out lattice. Pooled lattices are
# sized separately (DEC-7) once per-lineage costs are known. Solver only (Qwen3-4B-AWQ on :8000).
set -uo pipefail
cd "$(dirname "$0")/../.."
export PATH=/usr/lib/wsl/lib:$PATH
. .venv-gpu/bin/activate
log() { echo "$(date '+%F %T') $*"; }
W=16
for job in "gsmhard 200" "lcb 85" "bird 199"; do
  set -- $job; bench=$1; held=$2
  for f in runs/gain_pilot_$bench/*.log.jsonl; do
    loop=$(basename $f .log.jsonl)
    if ! grep -q '"accepted": true' $f; then log "$bench $loop: no accepted edit, skipped"; continue; fi
    name=phaseA_${bench}_${loop}
    python scripts/run_lattice.py configs/backend_gpu.yaml $bench --log $f --harness-dir runs/harnesses --name ${name}_audit --namespace $name \
      --split search --n-tasks 10 --replicates 1 --statement-level --tree --brute --workers $W > runs/logs/b_${name}_audit.log 2>&1
    log "$name audit: $(tail -1 runs/logs/b_${name}_audit.log)"
    python scripts/run_lattice.py configs/backend_gpu.yaml $bench --log $f --harness-dir runs/harnesses --name $name \
      --split search --n-tasks 60 --replicates 4 --statement-level --tree --workers $W > runs/logs/b_${name}_search.log 2>&1
    log "$name search: $(tail -1 runs/logs/b_${name}_search.log)"
    python scripts/run_lattice.py configs/backend_gpu.yaml $bench --log $f --harness-dir runs/harnesses --name $name \
      --split heldout --n-tasks $held --replicates 4 --statement-level --tree --workers $W > runs/logs/b_${name}_heldout.log 2>&1
    log "$name heldout: $(tail -1 runs/logs/b_${name}_heldout.log)"
    python scripts/a2_audit.py configs/backend_gpu.yaml $bench runs/lattice/${name}_stmt_heldout.json --fraction 0.1 --workers $W > runs/logs/b_${name}_a2.log 2>&1
    log "$name A2 audit: $(tail -1 runs/logs/b_${name}_a2.log)"
  done
done
log "STAGE B NULL DONE"
