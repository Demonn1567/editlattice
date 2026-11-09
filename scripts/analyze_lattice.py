"""Attribution and cost report for one lattice file.

Usage: python scripts/analyze_lattice.py runs/lattice/<name>_<split>.json [--n-boot 2000]
Writes results/attribution_<name>_<split>.json and prints a summary.
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Overridable so that integration tests never write into the real run and result folders.
RUNS = Path(os.environ.get("EDITLATTICE_RUNS_DIR", ROOT / "runs"))
RESULTS = Path(os.environ.get("EDITLATTICE_RESULTS_DIR", ROOT / "results"))
sys.path.insert(0, str(ROOT / "src"))

from editlattice.attribution import attribute, bootstrap, lattice_values  # noqa: E402


def parse_key(text: str) -> frozenset[int]:
    """Product key 'a,b' -> frozenset."""
    return frozenset(int(part) for part in text.split(",") if part != "")


def main() -> None:
    """Load, attribute, bootstrap, report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("lattice")
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()
    data = json.loads(Path(args.lattice).read_text())
    players = [unit["index"] for unit in data["units"]]
    outcomes = {(cell["task_id"], cell["replicate"]): {parse_key(k): float(v) for k, v in cell["outcomes"].items()} for cell in data["cells"]}
    values = lattice_values(outcomes)
    result = attribute(values, players)
    intervals = bootstrap(outcomes, players, n_boot=args.n_boot)
    executions = [execution for cell in data["cells"] for execution in cell["executions"]]
    n_products = len(values)
    report = {
        "name": data["name"], "split": data["split"], "n_cells": len(data["cells"]), "units": data["units"],
        "values": {",".join(map(str, sorted(s))): v for s, v in values.items()},
        "shapley": {str(p): {"value": v, **intervals[f"shapley:{p}"]} for p, v in result.shapley.items()},
        "interaction": {f"{a},{b}": {"value": v, **intervals[f"interaction:{a},{b}"]} for (a, b), v in result.interaction.items()},
        "mobius": {",".join(map(str, sorted(s))): v for s, v in result.mobius.items()},
        "loo": {str(p): v for p, v in result.loo.items()},
        "loi": {str(p): v for p, v in result.loi.items()},
        "cost": {
            "executions": len(executions), "brute_force_runs": n_products * len(data["cells"]),
            "generated_tokens": sum(e["tokens_out"] for e in executions),
            "nominal_tokens": sum(e["nominal_tokens_out"] for e in executions),
            "mismatches": sum(len(cell["mismatches"]) for cell in data["cells"]),
        },
    }
    out = RESULTS / f"attribution_{data['name']}_{data['split']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    full, empty = frozenset(players), frozenset()
    print(f"{data['name']} [{data['split']}] cells={len(data['cells'])}  seed={values[empty]:.3f}  final={values[full]:.3f}")
    for p in players:
        s = report["shapley"][str(p)]
        print(f"  unit {p} {data['units'][p]['items']}: Shapley {s['value']:+.3f} [{s['low']:+.3f},{s['high']:+.3f}]  "
              f"LOO {result.loo[p]:+.3f}  LOI {result.loi[p]:+.3f}")
    for pair, entry in report["interaction"].items():
        print(f"  interaction {pair}: {entry['value']:+.3f} [{entry['low']:+.3f},{entry['high']:+.3f}]")
    best = max(values, key=values.get)
    print(f"  best product {sorted(best)} = {values[best]:.3f}; cost {report['cost']}")


if __name__ == "__main__":
    main()
