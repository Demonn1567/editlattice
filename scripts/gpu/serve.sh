#!/usr/bin/env bash
# vLLM servers on the RTX A4000 (16 GB, shared with the Windows display).
# Usage: bash scripts/gpu/serve.sh evolve   - proposer Qwen3-8B-AWQ (port 8100) + solver Qwen3-4B-AWQ (port 8000)
#        bash scripts/gpu/serve.sh lattice  - solver Qwen3-4B-AWQ alone with most of the memory (port 8000)
#        bash scripts/gpu/serve.sh stop
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH=/usr/lib/wsl/lib:$PATH
. .venv-gpu/bin/activate
mkdir -p runs/logs
SOLVER=${SOLVER:-Qwen/Qwen3-4B-AWQ}
PROPOSER=${PROPOSER:-Qwen/Qwen3-8B-AWQ}
wait_ready() { for _ in $(seq 1 180); do curl -sf "localhost:$1/v1/models" > /dev/null && { echo "port $1 ready"; return 0; }; sleep 5; done; echo "port $1 NOT ready"; return 1; }
case "${1:-evolve}" in
  stop)
    # Never pull the server from under a running evaluation.
    while pgrep -f "scripts/(seed_eval|gain_check|run_lattice|run_evolution|a2_audit).py" > /dev/null; do sleep 30; done
    pkill -f "vllm serve" || true; sleep 5; nvidia-smi --query-gpu=memory.used --format=csv ;;
  evolve)
    # The larger model starts first so it sees the free memory it was promised.
    # Proposer Qwen3-8B-AWQ (DEC-2: the 14B does not fit next to the solver on 16 GB).
    # At 0.52 the 8B left 0.98 GiB of KV cache (1.12 GiB needed for one 8,192-token request);
    # eager mode removes CUDA-graph memory, the larger share leaves room for two requests.
    nohup vllm serve "$PROPOSER" --port 8100 --gpu-memory-utilization 0.58 --max-model-len 8192 --max-num-seqs 8 \
      --enforce-eager --seed 0 > runs/logs/vllm_proposer.log 2>&1 &
    wait_ready 8100
    nohup vllm serve "$SOLVER" --port 8000 --gpu-memory-utilization 0.38 --max-model-len 8192 --max-num-seqs 32 \
      --seed 0 > runs/logs/vllm_solver.log 2>&1 &
    wait_ready 8000 ;;
  lattice)
    nohup vllm serve "$SOLVER" --port 8000 --gpu-memory-utilization 0.92 --max-model-len 8192 --max-num-seqs 128 --seed 0 > runs/logs/vllm_solver.log 2>&1 &
    wait_ready 8000 ;;
esac
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
