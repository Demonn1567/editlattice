#!/usr/bin/env bash
# Models for the A4000: AWQ solver and proposer for evolution; bf16 4B solver for transfer (RQ6).
# Uses the huggingface_hub Python API (the CLI was renamed from huggingface-cli to hf).
set -euo pipefail
cd "$(dirname "$0")/../.."
. .venv-gpu/bin/activate
# The Xet transfer backend failed with "error decoding response body"; plain HTTP downloads with retries.
export HF_HUB_DISABLE_XET=1
python - <<'PY'
import time
from huggingface_hub import snapshot_download
for model in ["Qwen/Qwen3-4B-AWQ", "Qwen/Qwen3-14B-AWQ", "Qwen/Qwen3-4B"]:
    for attempt in range(1, 6):
        try:
            path = snapshot_download(model)
            print("downloaded", model, path, flush=True)
            break
        except Exception as error:  # noqa: BLE001 - network faults are retried, then re-raised
            print(f"attempt {attempt} for {model} failed: {error}", flush=True)
            if attempt == 5:
                raise
            time.sleep(30)
PY
echo DOWNLOADS DONE
