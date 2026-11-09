"""Brute-force audits of the pooled lattices (preregistration addendum 2026-11-09): per pool, the (product, cell) pairs
checked against runs alone and the mismatches.

Usage: python scripts/pool_audit_summary.py   (reads runs/lattice/pooled*_audit_search.json; writes results/pool_audits.json)
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    out = {}
    for path in sorted((ROOT / "runs" / "lattice").glob("pooled*_audit_search.json")):
        data = json.loads(path.read_text())
        cells = data["cells"]
        out[path.name.removesuffix("_audit_search.json")] = {
            "units": len(data["units"]), "products": len(cells[0]["outcomes"]), "cells": len(cells),
            "checked": sum(c["n_checked"] for c in cells), "mismatches": sum(len(c["mismatches"]) for c in cells),
            "executions": sum(len(c["executions"]) for c in cells)}
    (ROOT / "results" / "pool_audits.json").write_text(json.dumps(out, indent=1))
    for name, entry in out.items():
        print(name, entry)


if __name__ == "__main__":
    main()
