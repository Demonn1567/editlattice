"""Real-gain test (design D8) of a lattice's full product against its seed on held-out cells, with the
same paired task-cluster bootstrap as gain_check.py.

Usage: python scripts/lattice_gain.py runs/lattice/<name>_heldout.json
Writes results/gain_lattice_<name>.json.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from gain_check import MIN_GAIN, paired_gain  # noqa: E402


def main() -> None:
    path = Path(sys.argv[1])
    data = json.loads(path.read_text())
    full = ",".join(str(unit["index"]) for unit in data["units"])
    seed: dict[str, list[int]] = defaultdict(list)
    final: dict[str, list[int]] = defaultdict(list)
    for cell in data["cells"]:
        seed[cell["task_id"]].append(int(cell["outcomes"][""]))
        final[cell["task_id"]].append(int(cell["outcomes"][full]))
    entry = paired_gain(dict(seed), dict(final))
    entry["real_gain"] = entry["low"] > 0 and entry["gain"] >= MIN_GAIN
    entry.update({"lattice": path.name, "cells": len(data["cells"])})
    out = ROOT / "results" / f"gain_lattice_{path.stem}.json"
    out.write_text(json.dumps(entry, indent=1))
    print(json.dumps(entry))


if __name__ == "__main__":
    main()
