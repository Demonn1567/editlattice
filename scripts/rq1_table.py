"""RQ1 summary: exactness and cost of variational replay per lineage, both read modes.

Usage: python scripts/rq1_table.py [prefix]   (default prefix: prior_)
Reads runs/lattice/<prefix>*_search.json (name-level) and <prefix>*_stmt_search.json; writes
results/rq1_<prefix>summary.json and prints a table.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def summarize(path: Path) -> dict:
    """Counts for one lattice file."""
    data = json.loads(path.read_text())
    executions = [execution for cell in data["cells"] for execution in cell["executions"]]
    products = len(data["cells"][0]["outcomes"]) if data["cells"] else 0
    return {
        "units": len(data["units"]), "products": products, "cells": len(data["cells"]),
        "executions": len(executions), "brute_force": products * len(data["cells"]),
        "generated_tokens": sum(e["tokens_out"] for e in executions),
        "nominal_tokens": sum(e["nominal_tokens_out"] for e in executions),
        "mismatches": sum(len(cell["mismatches"]) for cell in data["cells"]),
        "checked": sum(1 for cell in data["cells"] for _ in cell["outcomes"]) if any("mismatches" in c for c in data["cells"]) else 0,
        "marked_units": data.get("marked_units", []),
    }


def main() -> None:
    """Pair name-level and statement-level files and print the table."""
    prefix = sys.argv[1] if len(sys.argv) > 1 else "prior_"
    directory = ROOT / "runs" / "lattice"
    rows = []
    for path in sorted(directory.glob(f"{prefix}*_search.json")):
        if path.name.endswith("_stmt_search.json"):
            continue
        name = path.name.removesuffix("_search.json")
        statement = directory / f"{name}_stmt_search.json"
        row = {"lineage": name.removeprefix(prefix), "name_level": summarize(path)}
        if statement.exists():
            row["statement_level"] = summarize(statement)
        rows.append(row)
    total = lambda key, mode: sum(row[mode][key] for row in rows if mode in row)  # noqa: E731
    out = ROOT / "results" / f"rq1_{prefix}summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows}, indent=1))
    print(f"{'lineage':38} units prods  exec(name)  exec(stmt)  brute  gen/nominal tokens  mismatches")
    for row in rows:
        n, s = row["name_level"], row.get("statement_level")
        print(f"{row['lineage']:38} {n['units']:5} {n['products']:5}  {n['executions']:10}  {s['executions'] if s else '-':>10}  "
              f"{n['brute_force']:5}  {n['generated_tokens']:7}/{n['nominal_tokens']:<8}  {n['mismatches'] + (s['mismatches'] if s else 0)}")
    print(f"TOTAL executions name {total('executions', 'name_level')}, statement {total('executions', 'statement_level')}, "
          f"brute force {total('brute_force', 'name_level')}; generated {total('generated_tokens', 'name_level')} "
          f"of nominal {total('nominal_tokens', 'name_level')}; mismatches "
          f"{total('mismatches', 'name_level') + total('mismatches', 'statement_level')} over "
          f"{total('checked', 'name_level') + total('checked', 'statement_level')} checked product-cells")


if __name__ == "__main__":
    main()
