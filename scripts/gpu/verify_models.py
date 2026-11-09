"""Verify downloaded models against the SHA-256 checksums Hugging Face publishes.

Usage: python scripts/gpu/verify_models.py Qwen/Qwen3-4B-AWQ [Qwen/Qwen3-14B-AWQ ...]
Exits non-zero if any file is missing or differs (written after the 2026-10-27 disk failure,
which interrupted downloads mid-write).
"""

import hashlib
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def sha256(path: Path) -> str:
    """SHA-256 of a file, streamed."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    """Check every LFS file of every model."""
    failures = 0
    for model in sys.argv[1:]:
        local = Path(snapshot_download(model, local_files_only=True))
        info = HfApi().model_info(model, files_metadata=True)
        for sibling in info.siblings:
            path = local / sibling.rfilename
            if not path.exists():
                print(f"MISSING {model}/{sibling.rfilename}")
                failures += 1
                continue
            if sibling.lfs is None:
                continue
            ok = sha256(path) == sibling.lfs.sha256
            failures += not ok
            print(f"{'ok ' if ok else 'BAD'} {model}/{sibling.rfilename}")
    print("ALL VERIFIED" if failures == 0 else f"{failures} FAILURES")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
