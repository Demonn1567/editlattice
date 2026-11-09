#!/usr/bin/env bash
# Install on the A4000 machine (WSL2, Ubuntu 24.04, NVIDIA driver 551 = CUDA 12.4).
# vLLM 0.8.5 is built for CUDA 12.4; newer vLLM needs a newer driver.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH=/usr/lib/wsl/lib:$PATH
python3 -m venv .venv-gpu
. .venv-gpu/bin/activate
pip install -q --upgrade pip
pip install -q "vllm==0.8.5" "transformers==4.51.3" httpx pyyaml numpy scipy pytest huggingface_hub
python -c "import torch, vllm; print(\"torch\", torch.__version__, \"cuda\", torch.version.cuda, \"available\", torch.cuda.is_available(), torch.cuda.get_device_name(0)); print(\"vllm\", vllm.__version__)"
python -m pytest -q tests/test_variational.py tests/test_gsmhard.py tests/test_attribution.py
echo INSTALL DONE
