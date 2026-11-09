"""RQ1c (preregistration addendum 2026-11-09): cost and exactness of the branched lineages, whose units sit mostly on
failure paths. Per benchmark: the brute-force audit, the cost of the search and held-out lattices against brute force,
the read widths, how often each unit is read, and the fresh-run audit (replayed minus fresh accuracy with a 95%
bootstrap interval over pairs).

Usage: python scripts/branched_summary.py   (reads runs/lattice/branched_*.json and results/a2_branched_*;
writes results/branched.json)
"""

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from editlattice.attribution import mobius  # noqa: E402

LATTICES = ROOT / "runs" / "lattice"
RESULTS = ROOT / "results"


def lattice_cost(data: dict) -> dict:
    cells = data["cells"]
    runs = [e for cell in cells for e in cell["executions"]]
    m = len(data["units"])
    leaves = [len(cell["executions"]) for cell in cells]
    return {"units": m, "cells": len(cells), "products": len(cells[0]["outcomes"]), "executions": len(runs),
            "brute_force": len(cells[0]["outcomes"]) * len(cells), "max_leaves_per_cell": max(leaves),
            "mean_leaves_per_cell": sum(leaves) / len(leaves), "max_read_width": max(len(e["reads"]) for e in runs),
            "reads": {str(u): sum(u in e["reads"] for e in runs) for u in range(m)},
            "generated_tokens": sum(int(e.get("tokens_out", 0)) for e in runs),
            "nominal_tokens": sum(int(e.get("nominal_tokens_out", 0)) for e in runs), "seconds": data.get("seconds")}


def exactly_inert(data: dict, unit: int) -> bool:
    """True if adding the unit to any product changes no outcome on any cell."""
    for cell in data["cells"]:
        outcomes = cell["outcomes"]
        for product, outcome in outcomes.items():
            subset = {int(u) for u in product.split(",") if u}
            if unit not in subset and outcomes[",".join(map(str, sorted(subset | {unit})))] != outcome:
                return False
    return True


def values(data: dict) -> dict:
    totals: dict = {}
    for cell in data["cells"]:
        for product, outcome in cell["outcomes"].items():
            totals[product] = totals.get(product, 0.0) + outcome
    return {frozenset(int(u) for u in k.split(",") if u): v / len(data["cells"]) for k, v in totals.items()}


def fresh_audit(path: Path, resamples: int = 2000, seed: int = 20261109) -> dict:
    pairs = json.loads(path.read_text())["pairs"]
    differences = [p["replayed"] - p["fresh"] for p in pairs]
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(differences) for _ in differences) / len(differences) for _ in range(resamples))
    return {"pairs": len(pairs), "replayed": sum(p["replayed"] for p in pairs) / len(pairs),
            "fresh": sum(p["fresh"] for p in pairs) / len(pairs), "difference": sum(differences) / len(differences),
            "low": means[int(0.025 * resamples)], "high": means[int(0.975 * resamples) - 1]}


def main() -> None:
    out = {}
    for bench in ("bird", "lcb"):
        name = f"branched_{bench}"
        if not (LATTICES / f"{name}_stmt_heldout.json").exists():
            continue
        audit = json.loads((LATTICES / f"{name}_audit_stmt_search.json").read_text())
        entry = {"audit": {"cells": len(audit["cells"]), "checked": sum(c["n_checked"] for c in audit["cells"]),
                           "mismatches": sum(len(c["mismatches"]) for c in audit["cells"]),
                           "executions": sum(len(c["executions"]) for c in audit["cells"])}}
        for split in ("search", "heldout"):
            data = json.loads((LATTICES / f"{name}_stmt_{split}.json").read_text())
            entry[split] = lattice_cost(data)
            entry[split]["marked_units"] = data.get("marked_units", [])
            if split == "heldout":
                held = values(data)
                players = list(range(len(data["units"])))
                unread = [u for u in players if entry[split]["reads"][str(u)] == 0]
                dividends = mobius(held, players)
                entry["unread_units"] = unread
                entry["inert_units"] = [u for u in players if exactly_inert(data, u)]
                entry["unread_max_dividend"] = max((abs(v) for k, v in dividends.items() if set(k) & set(unread)), default=0.0)
        a2 = RESULTS / f"a2_{name}_stmt_heldout.json"
        if a2.exists():
            entry["fresh_audit"] = fresh_audit(a2)
        out[bench] = entry
        print(bench, json.dumps({k: v for k, v in entry.items() if k in ("audit", "fresh_audit")}),
              {s: (entry[s]["executions"], entry[s]["brute_force"], entry[s]["max_leaves_per_cell"], entry[s]["max_read_width"])
               for s in ("search", "heldout")})
    (RESULTS / "branched.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
