#!/usr/bin/env bash
# DEC-10: rerun the two Phase A lineages whose statement-level marking was unsound, with the fix.
# The old lattices, audits and reports are kept with the prefix buggy_.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PATH=/usr/lib/wsl/lib:$PATH
. .venv-gpu/bin/activate
log() { echo "$(date '+%F %T') $*"; }
W=16
for job in "gsmhard 200 metaharness_B3_s3" "lcb 85 metaharness_B3_s2"; do
  set -- $job; bench=$1; held=$2; loop=$3; name=phaseA_${bench}_${loop}; f=runs/gain_pilot_$bench/$loop.log.jsonl
  for old in runs/lattice/${name}_audit_stmt_search.json runs/lattice/${name}_stmt_search.json runs/lattice/${name}_stmt_heldout.json \
             results/a2_${name}_stmt_heldout.json results/report_${name}_stmt.json; do
    [ -f $old ] && mv $old $(dirname $old)/buggy_$(basename $old)
  done
  python scripts/run_lattice.py configs/backend_gpu.yaml $bench --log $f --harness-dir runs/harnesses --name ${name}_audit --namespace $name \
    --split search --n-tasks 10 --replicates 1 --statement-level --tree --brute --workers $W > runs/logs/fix_${name}_audit.log 2>&1
  log "$name audit: $(tail -1 runs/logs/fix_${name}_audit.log)"
  for split in "search 60" "heldout $held"; do
    set -- $split
    python scripts/run_lattice.py configs/backend_gpu.yaml $bench --log $f --harness-dir runs/harnesses --name $name \
      --split $1 --n-tasks $2 --replicates 4 --statement-level --tree --workers $W > runs/logs/fix_${name}_$1.log 2>&1
    log "$name $1: $(tail -1 runs/logs/fix_${name}_$1.log)"
  done
  python scripts/a2_audit.py configs/backend_gpu.yaml $bench runs/lattice/${name}_stmt_heldout.json --fraction 0.1 --workers $W > runs/logs/fix_${name}_a2.log 2>&1
  log "$name A2 audit: $(tail -1 runs/logs/fix_${name}_a2.log)"
  python scripts/rq_report.py runs/lattice/${name}_stmt_search.json runs/lattice/${name}_stmt_heldout.json --n-boot 2000 > /dev/null 2>&1
done
log "STAGE FIX10 DONE"
