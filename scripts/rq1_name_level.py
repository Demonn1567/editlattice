"""RQ1 (registered): name-level vs statement-level reads on the GPU lineages.

For every search-split lattice explored with statement-level reads that marked at least one unit, the
same lattice is explored with name-level reads on the same stream namespace (responses replay), and
the two are compared: per-product outcomes must agree on every cell (a second exactness check), and
the executions show what statement-level reads save.

Usage: python scripts/rq1_name_level.py configs/backend_gpu.yaml
Writes results/rq1_name_vs_stmt.json.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LATTICES = ROOT / "runs" / "lattice"
SEEDS = {"gsmhard": "math_direct", "lcb": "lcb_oneshot", "bird": "bird_oneshot"}
SHORT = {"gsmhard": "gsm", "lcb": "lcb", "bird": "bird"}


def source_args(name: str) -> list[str]:
    """How the lineage of a lattice was built."""
    if name.startswith("constructed_"):
        bench = name.split("_", 1)[1]
        return ["--seed", f"src/editlattice/seed_harnesses/{SEEDS[bench]}.py", "--final", f"examples/constructed/{SHORT[bench]}_constructed_final.py"]
    if name == "planted_gpu":
        return ["--seed", "src/editlattice/seed_harnesses/math_direct.py", "--final", "examples/gsm_planted_final.py"]
    phase, bench, loop = name.split("_", 2)
    pilot = "gain_pilot" if phase == "phaseA" else "gain_pilot2"
    return ["--log", f"runs/{pilot}_{bench}/{loop}.log.jsonl", "--harness-dir", "runs/harnesses"]


def main() -> None:
    config = sys.argv[1]
    rows = []
    for path in sorted(LATTICES.glob("*_stmt_search.json")):
        if any(tag in path.name for tag in ("audit", "buggy", "invalid")):
            continue
        stmt = json.loads(path.read_text())
        if not stmt.get("marked_units"):
            continue
        name = path.name.replace("_stmt_search.json", "")
        bench = stmt["benchmark"]
        command = [sys.executable, "scripts/run_lattice.py", config, bench, *source_args(name), "--name", f"{name}_namelevel",
                   "--namespace", stmt["namespace"], "--split", "search", "--n-tasks", str(stmt["n_tasks"]),
                   "--replicates", str(stmt["replicates"]), "--tree", "--workers", "16"]
        subprocess.run(command, cwd=ROOT, check=True, capture_output=True)
        named = json.loads((LATTICES / f"{name}_namelevel_search.json").read_text())
        by_cell = {(c["task_id"], c["replicate"]): c for c in named["cells"]}
        differing = sum(1 for c in stmt["cells"] for p, v in c["outcomes"].items() if by_cell[(c["task_id"], c["replicate"])]["outcomes"][p] != v)
        row = {"lineage": name, "benchmark": bench, "units": len(stmt["units"]), "marked": stmt["marked_units"],
               "cells": len(stmt["cells"]), "products": len(stmt["cells"][0]["outcomes"]),
               "executions_statement": sum(len(c["executions"]) for c in stmt["cells"]),
               "executions_name": sum(len(c["executions"]) for c in named["cells"]),
               "outcomes_differing": differing}
        rows.append(row)
        print(json.dumps(row), flush=True)
    (ROOT / "results" / "rq1_name_vs_stmt.json").write_text(json.dumps({"rows": rows}, indent=1))


if __name__ == "__main__":
    main()
