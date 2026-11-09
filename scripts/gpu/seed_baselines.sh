#!/usr/bin/env bash
# Seed baselines on the 60 search tasks x 4 replicates the loops search on (solver only).
set -euo pipefail
cd "$(dirname "$0")/../.."
. .venv-gpu/bin/activate
for job in "gsmhard math_direct" "lcb lcb_oneshot" "bird bird_oneshot"; do
  set -- $job
  start=$(date +%s)
  python scripts/seed_eval.py configs/backend_gpu.yaml src/editlattice/seed_harnesses/$2.py $1 --n-tasks 60 --replicates 4
  echo "$1: $(( $(date +%s) - start )) s for 240 runs"
done
echo BASELINES DONE
