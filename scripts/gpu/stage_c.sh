#!/usr/bin/env bash
# Phase C (pre-registered 2026-10-29): constructed lineages. Per benchmark: brute-force audit, search and
# held-out lattices, D8 gain of the full lineage, fresh-run audit, RQ3/RQ4 report, RQ2 estimation, then
# RQ6 on held-out under AWQ T=1.0 and bf16 T=0.7 (bf16 server started by stage_rq6.sh).
set -uo pipefail
cd "$(dirname "$0")/../.."
export PATH=/usr/lib/wsl/lib:$PATH
. .venv-gpu/bin/activate
log() { echo "$(date '+%F %T') $*"; }
W=16
declare -A SEED=([gsmhard]=math_direct [lcb]=lcb_oneshot [bird]=bird_oneshot)
declare -A SHORT=([gsmhard]=gsm [lcb]=lcb [bird]=bird)
declare -A HELD=([gsmhard]=200 [lcb]=85 [bird]=199)
for bench in gsmhard lcb bird; do
  S=src/editlattice/seed_harnesses/${SEED[$bench]}.py; F=examples/constructed/${SHORT[$bench]}_constructed_final.py; name=constructed_$bench
  common="$bench --seed $S --final $F --statement-level --tree --workers $W"
  python scripts/run_lattice.py configs/backend_gpu.yaml $common --name ${name}_audit --namespace $name --split search --n-tasks 10 --replicates 1 --brute > runs/logs/c_${name}_audit.log 2>&1
  log "$name audit: $(tail -1 runs/logs/c_${name}_audit.log)"
  python scripts/run_lattice.py configs/backend_gpu.yaml $common --name $name --split search --n-tasks 60 --replicates 4 > runs/logs/c_${name}_search.log 2>&1
  log "$name search: $(tail -1 runs/logs/c_${name}_search.log)"
  python scripts/run_lattice.py configs/backend_gpu.yaml $common --name $name --split heldout --n-tasks ${HELD[$bench]} --replicates 4 > runs/logs/c_${name}_heldout.log 2>&1
  log "$name heldout: $(tail -1 runs/logs/c_${name}_heldout.log)"
  log "$name D8 gain: $(python scripts/lattice_gain.py runs/lattice/${name}_stmt_heldout.json 2>&1 | tail -1)"
  python scripts/a2_audit.py configs/backend_gpu.yaml $bench runs/lattice/${name}_stmt_heldout.json --fraction 1.0 --workers $W > runs/logs/c_${name}_a2.log 2>&1
  log "$name A2 audit: $(tail -1 runs/logs/c_${name}_a2.log)"
  python scripts/rq_report.py runs/lattice/${name}_stmt_search.json runs/lattice/${name}_stmt_heldout.json --n-boot 2000 > runs/logs/c_${name}_report.log 2>&1
  log "$name report done"
  python scripts/rq2_estimation.py runs/lattice/${name}_stmt_heldout.json results/a2_${name}_stmt_heldout.json --repetitions 20 > runs/logs/c_${name}_rq2.log 2>&1
  log "$name RQ2: $(tail -1 runs/logs/c_${name}_rq2.log)"
done
for bench in gsmhard lcb bird; do
  S=src/editlattice/seed_harnesses/${SEED[$bench]}.py; F=examples/constructed/${SHORT[$bench]}_constructed_final.py; name=constructed_$bench
  common="$bench --seed $S --final $F --statement-level --tree --workers $W --split heldout --n-tasks ${HELD[$bench]} --replicates 4"
  python scripts/run_lattice.py configs/backend_gpu_t1.yaml $common --name ${name}_t1 > runs/logs/c_${name}_t1.log 2>&1
  log "${name}_t1: $(tail -1 runs/logs/c_${name}_t1.log)"
done
until grep -q "bf16 server up" runs/logs/stage_rq6.log 2>/dev/null; do sleep 120; done
for bench in gsmhard lcb bird; do
  S=src/editlattice/seed_harnesses/${SEED[$bench]}.py; F=examples/constructed/${SHORT[$bench]}_constructed_final.py; name=constructed_$bench
  common="$bench --seed $S --final $F --statement-level --tree --workers $W --split heldout --n-tasks ${HELD[$bench]} --replicates 4"
  python scripts/run_lattice.py configs/backend_gpu_bf16.yaml $common --name ${name}_bf16 > runs/logs/c_${name}_bf16.log 2>&1
  log "${name}_bf16: $(tail -1 runs/logs/c_${name}_bf16.log)"
done
log "STAGE C DONE"
