"""mlx_lm.server with MLX's buffer cache capped (outputs unchanged; only freed buffers are released
sooner). Used for the Phase A2 proposer on the 16 GB Mac. Arguments are passed to mlx_lm.server."""
import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm import server

mx.set_cache_limit(512 * 1024 ** 2)
# Keep the model's memory resident (wired), so macOS swaps other processes instead of the weights.
# 9.5 GB is below MLX's recommended working set on this 16 GB machine (about 10.7 GB) and suits the 14B
# proposer. A smaller model needs less: runs/mlx_wired_gb (one number, in GB) overrides it, because a
# wired limit far above the model pushed other memory to swap and filled the disk (2026-11-06). The limit
# changes only residency, never the computed outputs.
override = Path(__file__).with_name("mlx_wired_gb")
wired_gb = float(override.read_text().strip()) if override.exists() else 9.5
mx.set_wired_limit(int(wired_gb * 1024 ** 3))
sys.argv = ["mlx_lm.server"] + sys.argv[1:]
server.main()
