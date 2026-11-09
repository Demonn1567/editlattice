"""RQ2: accuracy of Shapley estimators at equal budget, coupled (replay) vs independent streams.

Usage: python scripts/rq2_estimation.py runs/lattice/<name>_<split>.json results/a2_<name>_<split>.json
           [--repetitions 20]
Coupled outcomes come from the lattice (all products on shared streams); independent outcomes
from the A2 audit run at --fraction 1.0 (every product on a stream private to it). Ground truth:
exact Shapley values of the lattice on all its cells. Writes results/rq2_<name>_<split>.json.
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(os.environ.get("EDITLATTICE_RESULTS_DIR", ROOT / "results"))
sys.path.insert(0, str(ROOT / "src"))

from editlattice.attribution import lattice_values, shapley  # noqa: E402
from editlattice.estimators import compare  # noqa: E402


def main() -> None:
    """Align pools, compute truth, compare estimators."""
    parser = argparse.ArgumentParser()
    parser.add_argument("lattice")
    parser.add_argument("audit")
    parser.add_argument("--repetitions", type=int, default=20)
    args = parser.parse_args()
    lattice = json.loads(Path(args.lattice).read_text())
    audit = json.loads(Path(args.audit).read_text())
    players = [unit["index"] for unit in lattice["units"]]
    cells = [(cell["task_id"], cell["replicate"]) for cell in lattice["cells"]]
    key = lambda text: frozenset(int(x) for x in text.split(",") if x)  # noqa: E731
    coupled = {key(product): [float(cell["outcomes"][product]) for cell in lattice["cells"]] for product in lattice["cells"][0]["outcomes"]}
    fresh = {(pair["product"], pair["task_id"], pair["replicate"]): float(pair["fresh"]) for pair in audit["pairs"]}
    missing = [(p, t, r) for p in lattice["cells"][0]["outcomes"] for t, r in cells if (p, t, r) not in fresh]
    if missing:
        raise SystemExit(f"audit lacks {len(missing)} (product, cell) pairs; run a2_audit.py --fraction 1.0")
    independent = {key(product): [fresh[(product, t, r)] for t, r in cells] for product in lattice["cells"][0]["outcomes"]}
    outcomes = {(t, r): {subset: coupled[subset][i] for subset in coupled} for i, (t, r) in enumerate(cells)}
    truth = shapley(lattice_values(outcomes), players)
    full_budget = len(coupled) * len(cells)
    budgets = sorted({max(len(coupled), full_budget // divisor) for divisor in (16, 8, 4, 2, 1)})
    rows = compare(coupled, independent, players, truth, budgets, repetitions=args.repetitions)
    out = RESULTS / f"rq2_{lattice['name']}_{lattice['split']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"truth": {str(p): v for p, v in truth.items()}, "budgets": budgets, "rows": rows}, indent=1))
    for row in rows:
        print(f"{row['budget']:6} {row['regime']:11} {row['estimator']:13} MAE {row['mae']:.4f} (sd {row['mae_sd']:.4f}) runs {row['runs']:.0f}")


if __name__ == "__main__":
    main()
