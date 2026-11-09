"""Exploratory (not registered): do sampled Shapley estimates lead to the right decisions at equal cost?

For every lineage whose held-out lattice has a full fresh-run audit (every product and cell on a
private stream), and for each budget of RQ2, we repeat each sampled estimator 20 times on shared
streams (coupled) and on independent streams, and record two decisions:
* inert units: units that change no outcome on any cell for any product (exact Shapley value 0); how
  often the estimate claims an effect of at least one point (|estimate| >= 0.01);
* top unit: how often the unit with the largest estimate is the unit with the largest exact value.

Usage: python scripts/decisions_at_equal_budget.py   (writes results/decisions.json)
"""

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from editlattice.attribution import lattice_values, shapley  # noqa: E402
from editlattice.estimators import estimate_once  # noqa: E402

LINEAGES = {
    "planted (Qwen3-4B, Mac)": ("runs/lattice/planted_stmt_heldout.json", "results/a2_planted_stmt_heldout.json"),
    "constructed GSM-Hard": ("runs/lattice/constructed_gsmhard_stmt_heldout.json", "results/phaseC/a2_constructed_gsmhard_stmt_heldout.json"),
    "constructed LiveCodeBench": ("runs/lattice/constructed_lcb_stmt_heldout.json", "results/phaseC/a2_constructed_lcb_stmt_heldout.json"),
    "constructed BIRD": ("runs/lattice/constructed_bird_stmt_heldout.json", "results/phaseC/a2_constructed_bird_stmt_heldout.json"),
}
REPETITIONS = 20
THRESHOLD = 0.01


def load(lattice_path: Path, audit_path: Path):
    """Coupled and independent outcome pools, the players, the exact values and the inert units."""
    lattice = json.loads(lattice_path.read_text())
    audit = json.loads(audit_path.read_text())
    players = [unit["index"] for unit in lattice["units"]]
    cells = [(cell["task_id"], cell["replicate"]) for cell in lattice["cells"]]
    key = lambda text: frozenset(int(x) for x in text.split(",") if x)  # noqa: E731
    products = list(lattice["cells"][0]["outcomes"])
    coupled = {key(p): [float(cell["outcomes"][p]) for cell in lattice["cells"]] for p in products}
    fresh = {(pair["product"], pair["task_id"], pair["replicate"]): float(pair["fresh"]) for pair in audit["pairs"]}
    independent = {key(p): [fresh[(p, t, r)] for t, r in cells] for p in products}
    outcomes = {(t, r): {s: coupled[s][i] for s in coupled} for i, (t, r) in enumerate(cells)}
    truth = shapley(lattice_values(outcomes), players)
    inert = [u for u in players if all(coupled[s | {u}] == coupled[s] for s in coupled if u not in s)]
    return coupled, independent, players, truth, inert, len(products) * len(cells)


def main() -> None:
    report = {}
    for name, (lattice_file, audit_file) in LINEAGES.items():
        if not (ROOT / lattice_file).exists() or not (ROOT / audit_file).exists():
            continue
        coupled, independent, players, truth, inert, full = load(ROOT / lattice_file, ROOT / audit_file)
        top = max(players, key=lambda p: truth[p])
        budgets = sorted({max(len(coupled), full // divisor) for divisor in (16, 8, 4, 2, 1)})
        rows = []
        for budget in budgets:
            for regime, pool, is_coupled in (("coupled", coupled, True), ("independent", independent, False)):
                for estimator in ("permutation", "kernelshap"):
                    false_effects = top_hits = 0
                    for repetition in range(REPETITIONS):
                        estimate, _ = estimate_once(estimator, pool, is_coupled, players, budget, random.Random(repetition))
                        false_effects += sum(abs(estimate[u]) >= THRESHOLD for u in inert)
                        top_hits += max(players, key=lambda p: estimate[p]) == top
                    rows.append({"budget": budget, "regime": regime, "estimator": estimator,
                                 "inert_false_effect_rate": false_effects / (REPETITIONS * len(inert)) if inert else None,
                                 "top_unit_rate": top_hits / REPETITIONS})
        report[name] = {"players": players, "exact": {str(p): truth[p] for p in players}, "inert": inert, "top": top,
                        "budgets": budgets, "rows": rows}
        print(name, "inert", inert, "top", top)
        for row in rows:
            if row["budget"] in (budgets[0], budgets[-1]):
                print("  ", row)
    (ROOT / "results" / "decisions.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
