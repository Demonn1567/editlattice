#!/usr/bin/env bash
# Download one model (plain HTTP, retries); used to fetch several models in parallel.
set -euo pipefail
cd "$(dirname "$0")/../.."
. .venv-gpu/bin/activate
export HF_HUB_DISABLE_XET=1
python - "$1" <<"PY"
import sys, time
from huggingface_hub import snapshot_download
model = sys.argv[1]
for attempt in range(1, 6):
    try:
        print("downloaded", model, snapshot_download(model), flush=True)
        break
    except Exception as error:  # noqa: BLE001
        print(f"attempt {attempt} failed: {error}", flush=True)
        if attempt == 5:
            raise
        time.sleep(30)
PY
