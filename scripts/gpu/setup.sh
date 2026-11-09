#!/usr/bin/env bash
# One-time setup on the RTX machine (Ubuntu 24.04 under Windows WSL2, or Linux; NVIDIA driver on the host).
# Usage: bash scripts/gpu/setup.sh            (from the repository root)
set -euo pipefail
echo "== GPU"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo "== Python"; python3 -c "import sys; assert sys.version_info >= (3, 11), sys.version" || { echo "need Python 3.11+ (Ubuntu 24.04 ships 3.12)"; exit 1; }
python3 -m venv .venv-gpu
. .venv-gpu/bin/activate
pip install -q --upgrade pip
pip install -q "vllm>=0.8.5" httpx pyyaml numpy scipy pytest huggingface_hub
python -m pytest -q tests/test_variational.py tests/test_gsmhard.py
# Models: solver candidates and the proposer. AWQ keeps the 14B proposer around 10 GB.
for model in Qwen/Qwen3-4B Qwen/Qwen3-8B Qwen/Qwen3-14B-AWQ; do
  huggingface-cli download "$model" --quiet > /dev/null && echo "downloaded $model"
done
memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
echo "== GPU memory ${memory} MiB"
if [ "$memory" -ge 22000 ]; then echo "plan: solver and proposer served together (scripts/gpu/serve.sh both)"; else echo "plan: one model at a time (scripts/gpu/serve.sh solver | proposer)"; fi
