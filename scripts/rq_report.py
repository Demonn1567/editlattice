"""RQ3 (what matters, overfitting), RQ4 (distillation) and RQ5 (recombination) for one lineage.

Usage: python scripts/rq_report.py runs/lattice/<name>_search.json runs/lattice/<name>_heldout.json
           [--deltas 0.02 0.05] [--n-boot 2000]
RQ5 runs automatically when the lattices are pooled (run_lattice.py --pool).
Writes results/report_<name>.json. Follows notes/preregistration.md.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(os.environ.get("EDITLATTICE_RESULTS_DIR", ROOT / "results"))
sys.path.insert(0, str(ROOT / "src"))

from editlattice.attribution import attribute, bootstrap, holm, lattice_values  # noqa: E402
from editlattice.distill import all_methods, certify_non_inferior  # noqa: E402
from editlattice.stats.eprocess import paired_difference_test, to_unit_interval  # noqa: E402


def load(path: str) -> tuple[dict, dict, list[int]]:
    """Lattice data, per-cell outcomes, players."""
    data = json.loads(Path(path).read_text())
    outcomes = {(c["task_id"], c["replicate"]): {frozenset(int(x) for x in k.split(",") if x): float(v) for k, v in c["outcomes"].items()}
                for c in data["cells"]}
    return data, outcomes, [u["index"] for u in data["units"]]


def paired_sequence(outcomes: dict, first: frozenset, second: frozenset, seed: int = 0) -> list[float]:
    """Paired differences outcome(first) - outcome(second) over cells, in a fixed random order."""
    cells = sorted(outcomes)
    random.Random(seed).shuffle(cells)
    return [outcomes[cell][first] - outcomes[cell][second] for cell in cells]


def certify_superior(differences: list[float], alpha: float = 0.05) -> tuple[bool, int]:
    """Anytime-valid test of H0: E[D] <= 0."""
    test = paired_difference_test(0.0, alpha)
    for count, difference in enumerate(differences, start=1):
        test.update(to_unit_interval(difference))
        if test.rejects_null:
            return True, count
    return False, len(differences)


def main() -> None:
    """Pre-registered RQ3-RQ5 analyses."""
    parser = argparse.ArgumentParser()
    parser.add_argument("search")
    parser.add_argument("heldout")
    parser.add_argument("--deltas", nargs="*", type=float, default=[0.02, 0.05])
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()
    search_data, search, players = load(args.search)
    _, heldout, held_players = load(args.heldout)
    assert players == held_players, "search and held-out lattices must share units"
    full, empty = frozenset(players), frozenset()
    report = {"name": search_data["name"], "units": search_data["units"], "rq3": {}, "rq4": [], "rq5": None}
    pooled_with_conflicts = bool(search_data.get("conflicts"))
    # Shapley values need every subset; a pooled lattice with alternatives has none for conflicting
    # combinations, so it is used for RQ5 only (pre-registration: RQ3 and RQ4 are per lineage).
    for split, outcomes in (() if pooled_with_conflicts else (("search", search), ("heldout", heldout))):
        values = lattice_values(outcomes)
        result = attribute(values, players)
        intervals = bootstrap(outcomes, players, n_boot=args.n_boot)
        shapley_p = {str(p): intervals[f"shapley:{p}"]["p"] for p in players}
        pair_p = {f"{a},{b}": intervals[f"interaction:{a},{b}"]["p"] for (a, b) in result.interaction}
        report["rq3"][split] = {
            "seed": values[empty], "full": values[full],
            "shapley": {str(p): {"value": v, **intervals[f"shapley:{p}"]} for p, v in result.shapley.items()},
            "interaction": {f"{a},{b}": {"value": v, **intervals[f"interaction:{a},{b}"]} for (a, b), v in result.interaction.items()},
            "matters": holm(shapley_p), "interacts": holm(pair_p) if pair_p else {},
            "loo": {str(p): v for p, v in result.loo.items()}, "loi": {str(p): v for p, v in result.loi.items()},
        }
    search_values = lattice_values(search)
    heldout_values = lattice_values(heldout)
    if not pooled_with_conflicts:
        s, h = report["rq3"]["search"]["shapley"], report["rq3"]["heldout"]["shapley"]
        # Pre-registered: search CI above 0, held-out CI not above 0 (contains 0 or lies below it).
        report["rq3"]["overfitting_units"] = [p for p in s if s[p]["low"] > 0 and h[p]["low"] <= 0]
    for delta in (() if pooled_with_conflicts else args.deltas):
        for choice in all_methods(search_values, players, delta):
            certified, used = certify_non_inferior(paired_sequence(heldout, choice.product, full), delta)
            report["rq4"].append({"delta": delta, "method": choice.method, "product": sorted(choice.product), "units_kept": len(choice.product),
                                  "queries": choice.queries, "search_value": choice.value, "heldout_value": heldout_values[choice.product],
                                  "heldout_full": heldout_values[full], "certified": certified, "pairs_used": used})
    if search_data.get("origin"):
        # RQ5: in a pooled lattice each run's final harness is itself a product (the units that came
        # from that run), so every comparison below is on shared streams.
        origin = {int(unit): runs for unit, runs in search_data["origin"].items()}
        n_runs = 1 + max(run for runs in origin.values() for run in runs)
        finals = {run: frozenset(unit for unit, runs in origin.items() if run in runs) for run in range(n_runs)}
        best_single_run = max(finals, key=lambda run: search_values[finals[run]])  # chosen on search cells
        best = max(search_values, key=lambda subset: (search_values[subset], -len(subset)))
        certified, used = certify_superior(paired_sequence(heldout, best, finals[best_single_run]))
        differences = paired_sequence(heldout, best, finals[best_single_run])
        rng = random.Random(1)
        boot = sorted(sum(rng.choice(differences) for _ in differences) / len(differences) for _ in range(args.n_boot))
        report["rq5"] = {"pool": search_data.get("pool"), "best_pooled_product": sorted(best), "best_single_run": best_single_run,
                         "single_final_product": sorted(finals[best_single_run]),
                         "search": {"pooled": search_values[best], "single": search_values[finals[best_single_run]]},
                         "heldout": {"pooled": heldout_values[best], "single": heldout_values[finals[best_single_run]],
                                     "all_single": {str(run): heldout_values[product] for run, product in finals.items()}},
                         "difference": heldout_values[best] - heldout_values[finals[best_single_run]],
                         "ci": [boot[int(0.025 * args.n_boot)], boot[int(0.975 * args.n_boot) - 1]],
                         "superior_certified": certified, "pairs_used": used}
    out = RESULTS / f"report_{search_data['name']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    if pooled_with_conflicts:
        print(json.dumps({"rq5": report["rq5"]}))
    else:
        print(json.dumps({"seed_heldout": report["rq3"]["heldout"]["seed"], "full_heldout": report["rq3"]["heldout"]["full"],
                          "matters_heldout": [p for p, ok in report["rq3"]["heldout"]["matters"].items() if ok],
                          "overfitting": report["rq3"]["overfitting_units"],
                          "rq4": [(r["delta"], r["method"], r["units_kept"], r["certified"]) for r in report["rq4"]]}))


if __name__ == "__main__":
    main()
