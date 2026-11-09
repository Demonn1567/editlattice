"""Phase B summary over every evolved lineage with accepted edits without a real held-out gain (Phases A, A2 and A3): exactness audits,
executions against brute force, per-unit attribution (inert, matters, harmful, overfitting), the
fresh-run (A2) audits and the pooled lattices' recombination results.

Inputs (copied from the GPU): runs/lattice/{phaseA,phaseA2,phaseA3}_*_stmt_{search,heldout}.json and their
_audit lattices, runs/lattice/pooled*_{search,heldout}.json, results/phaseB_null/report_*.json and
a2_*.json. Writes results/null_summary.json.

A unit is exactly inert if adding it to any product changes no outcome on any held-out cell; it
matters if its held-out Shapley interval excludes 0 after Holm correction within the lineage (the
registered RQ3 rule, computed by rq_report.py); harmful if that interval lies below 0; overfitting if
its search interval lies above 0 and its held-out interval does not.

Usage: python scripts/null_summary.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LATTICES = ROOT / "runs" / "lattice"
REPORTS = ROOT / "results" / "phaseB_null"


def key_with(product: str, unit: str) -> str:
    """Product key with ``unit`` added."""
    members = {part for part in product.split(",") if part} | {unit}
    return ",".join(sorted(members, key=int))


def unit_rows(name: str) -> list[dict]:
    """Per-unit attribution of one lineage."""
    report = json.loads((REPORTS / f"report_{name}_stmt.json").read_text())
    heldout = json.loads((LATTICES / f"{name}_stmt_heldout.json").read_text())
    rows = []
    for unit, entry in report["rq3"]["heldout"]["shapley"].items():
        search = report["rq3"]["search"]["shapley"][unit]
        inert = all(cell["outcomes"][key_with(product, unit)] == value
                    for cell in heldout["cells"] for product, value in cell["outcomes"].items()
                    if unit not in product.split(","))
        executions = [execution for cell in heldout["cells"] for execution in cell["executions"]]
        rows.append({"lineage": name, "unit": unit, "items": heldout["units"][int(unit)]["items"],
                     "heldout": entry["value"], "low": entry["low"], "high": entry["high"],
                     "search": search["value"], "search_low": search["low"], "search_high": search["high"],
                     "matters": report["rq3"]["heldout"]["matters"][unit], "inert": inert,
                     "harmful": entry["high"] < 0, "overfitting": search["low"] > 0 and not entry["low"] > 0,
                     "read_fraction": sum(1 for e in executions if int(unit) in e["reads"]) / len(executions)})
    return rows


def lattice_cost(path: Path) -> dict:
    """Executions, brute-force runs and mismatches of one lattice file."""
    data = json.loads(path.read_text())
    cells = data["cells"]
    products = len(cells[0]["outcomes"]) if "outcomes" in cells[0] else 2 ** len(data["units"])
    return {"executions": sum(len(cell["executions"]) for cell in cells), "brute_force": products * len(cells),
            "mismatches": sum(len(cell["mismatches"]) for cell in cells), "checked": sum(cell.get("n_checked", 0) for cell in cells)}


def main() -> None:
    """Collect everything into results/null_summary.json."""
    lineages = sorted({path.name.rsplit("_stmt_", 1)[0] for path in LATTICES.glob("phaseA*_stmt_heldout.json")})
    lineages = [name for name in lineages if (REPORTS / f"report_{name}_stmt.json").exists()]
    # The null group is every evolved lineage without a real held-out gain; real-gain loops are reported apart.
    pilots = json.loads((ROOT / "results" / "pilot_summary.json").read_text())["loops"]
    real = {f"phase{row['phase']}_{row['benchmark']}_{row['loop']}" for row in pilots if row["real_gain"]}
    lineages = [name for name in lineages if name not in real]
    units, costs, audits, fresh = [], {}, {}, {}
    for name in lineages:
        units += unit_rows(name)
        costs[name] = {split: lattice_cost(LATTICES / f"{name}_stmt_{split}.json") for split in ("search", "heldout")}
        audits[name] = lattice_cost(LATTICES / f"{name}_audit_stmt_search.json")
        audit_file = REPORTS / f"a2_{name}_stmt_heldout.json"
        if audit_file.exists():
            fresh[name] = json.loads(audit_file.read_text())["summary"]
    pools = {}
    for report in sorted(REPORTS.glob("report_pooled*.json")):
        name = report.stem.replace("report_", "")
        rq5 = json.loads(report.read_text())["rq5"]
        pools[name] = {"rq5": {k: v for k, v in rq5.items() if k != "pool"},
                       **{split: lattice_cost(LATTICES / f"{name}_{split}.json") for split in ("search", "heldout")
                          if (LATTICES / f"{name}_{split}.json").exists()}}
    summary = {
        "lineages": len(lineages), "units": len(units),
        "inert": sum(r["inert"] for r in units), "matters": sum(r["matters"] for r in units),
        "harmful": sum(r["harmful"] for r in units), "overfitting": sum(r["overfitting"] for r in units),
        "audit_checked": sum(a["checked"] for a in audits.values()), "audit_mismatches": sum(a["mismatches"] for a in audits.values()),
        "lattice_executions": sum(c[s]["executions"] for c in costs.values() for s in c),
        "lattice_brute_force": sum(c[s]["brute_force"] for c in costs.values() for s in c),
        "fresh_pairs": sum(f["pairs"] for f in fresh.values()), "fresh_products": sum(f["products"] for f in fresh.values()),
        "fresh_significant": sum(f["products_with_p_below_0.05"] for f in fresh.values()),
        "fresh_replayed": sum(f["replayed_accuracy"] * f["pairs"] for f in fresh.values()) / max(1, sum(f["pairs"] for f in fresh.values())),
        "fresh_fresh": sum(f["fresh_accuracy"] * f["pairs"] for f in fresh.values()) / max(1, sum(f["pairs"] for f in fresh.values())),
    }
    out = ROOT / "results" / "null_summary.json"
    out.write_text(json.dumps({"summary": summary, "units": units, "costs": costs, "audits": audits, "fresh": fresh, "pools": pools}, indent=1))
    print(json.dumps(summary, indent=1))
    for name, pool in pools.items():
        print(name, pool["rq5"]["best_pooled_product"], pool["rq5"]["difference"], pool["rq5"]["ci"], pool["rq5"]["superior_certified"])


if __name__ == "__main__":
    main()
