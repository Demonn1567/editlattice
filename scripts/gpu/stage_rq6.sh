#!/usr/bin/env bash
# RQ6 (DEC-9): planted lineage under three solver settings on the A4000.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PATH=/usr/lib/wsl/lib:$PATH
. .venv-gpu/bin/activate
log() { echo "$(date '+%F %T') $*"; }
until grep -q "^downloaded Qwen/Qwen3-4B" runs/logs/download_qwen3_4b.log 2>/dev/null; do sleep 60; done
python scripts/gpu/verify_models.py Qwen/Qwen3-4B > runs/logs/verify_qwen3_4b.log 2>&1 || { log "bf16 weights failed verification"; exit 1; }
log "bf16 weights verified"
nohup vllm serve Qwen/Qwen3-4B --port 8100 --gpu-memory-utilization 0.97 --max-model-len 1536 --max-num-batched-tokens 1536 --max-num-seqs 8 --enforce-eager > runs/logs/vllm_bf16.log 2>&1 < /dev/null &
for i in $(seq 1 90); do curl -sf localhost:8100/v1/models > /dev/null && break; sleep 10; done
curl -sf localhost:8100/v1/models > /dev/null || { log "bf16 server did not start"; exit 1; }
log "bf16 server up"
S=src/editlattice/seed_harnesses/math_direct.py; F=examples/gsm_planted_final.py; W=16
run() {  # config name split n
  python scripts/run_lattice.py $1 gsmhard --seed $S --final $F --name $2 --split $3 --n-tasks $4 --replicates 4 --statement-level --tree --workers $W > runs/logs/rq6_$2_$3.log 2>&1
  log "$2 $3: $(tail -1 runs/logs/rq6_$2_$3.log)"
}
python scripts/run_lattice.py configs/backend_gpu.yaml gsmhard --seed $S --final $F --name planted_gpu_audit --namespace planted_gpu --split search --n-tasks 10 --replicates 1 --statement-level --tree --brute --workers $W > runs/logs/rq6_audit.log 2>&1
log "audit: $(tail -1 runs/logs/rq6_audit.log)"
run configs/backend_gpu.yaml planted_gpu search 60
run configs/backend_gpu.yaml planted_gpu heldout 200
python scripts/a2_audit.py configs/backend_gpu.yaml gsmhard runs/lattice/planted_gpu_stmt_heldout.json --fraction 0.1 --workers $W > runs/logs/rq6_a2.log 2>&1
log "A2 audit: $(tail -1 runs/logs/rq6_a2.log)"
python scripts/rq_report.py runs/lattice/planted_gpu_stmt_search.json runs/lattice/planted_gpu_stmt_heldout.json --n-boot 2000 > /dev/null 2>&1
run configs/backend_gpu_t1.yaml planted_gpu_t1 heldout 200
run configs/backend_gpu_bf16.yaml planted_gpu_bf16 heldout 200
log "STAGE RQ6 DONE"
