"""Conflict-free product counts of every pooled lattice (one per phase and benchmark), for the pool table and the
registered size rule (DEC-14: later phases pool only up to 500 products). Needs Python 3.12 (some evolved harnesses
use 3.12 syntax).

Usage: python scripts/pool_sizes.py   (writes results/pool_sizes.json)
"""

import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from editlattice.lineage import load_loop_lineage  # noqa: E402
from editlattice.units import combine_lineages  # noqa: E402

PHASES = [("A", "gain_pilot"), ("A2", "gain_pilot2"), ("A3", "gain_pilot3"), ("A4", "gain_pilot4")]


def main() -> None:
    out = {}
    for phase, prefix in PHASES:
        for bench in ("gsmhard", "lcb", "bird"):
            logs = [f for f in sorted((ROOT / "runs/gpu_pilot" / f"{prefix}_{bench}").glob("*.log.jsonl")) if '"accepted": true' in f.read_text()]
            if len(logs) < 2:
                continue
            lineages = [load_loop_lineage(log, ROOT / "runs/gpu_pilot/harnesses") for log in logs]
            finals = [lineage.product(frozenset(edit.index for edit in lineage.edits)) for lineage in lineages]
            pooled, _ = combine_lineages(lineages[0].seed, finals)
            products = sum(1 for r in range(pooled.m + 1) for c in itertools.combinations(range(pooled.m), r)
                           if not any(p <= frozenset(c) for p in pooled.conflicts))
            out[f"{phase}_{bench}"] = {"lineages": len(logs), "units": pooled.m, "products": products}
            print(phase, bench, out[f"{phase}_{bench}"])
    (ROOT / "results" / "pool_sizes.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
