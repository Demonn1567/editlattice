"""Cost of every real lattice of the study against brute force (executions) and against independent runs
(generated tokens), by group of lineages. Audit lattices are left out (they repeat search cells of a lattice).

Usage: python scripts/cost_by_lattice.py   (reads runs/lattice/*.json; writes results/cost_by_lattice.json)
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GROUPS = [("prior_", "prior"), ("planted", "planted"), ("constructed", "constructed"), ("phaseA", "evolved"), ("pooled", "pooled"), ("branched", "branched")]


def main() -> None:
    groups = {name: {"lattices": 0, "units_min": None, "units_max": 0, "executions": 0, "brute_force": 0,
                     "generated_tokens": 0, "nominal_tokens": 0} for _, name in GROUPS}
    for path in sorted((ROOT / "runs" / "lattice").glob("*.json")):
        name = path.name
        if name.startswith(("invalid_", "buggy_", "dup_")) or "_pre_dec19" in name or "_audit" in name:
            continue
        group = next((g for prefix, g in GROUPS if name.startswith(prefix)), None)
        if group is None:
            continue
        data = json.loads(path.read_text())
        cells = data.get("cells", [])
        if not cells or "outcomes" not in cells[0]:
            continue
        executions = [e for cell in cells for e in cell["executions"]]
        g = groups[group]
        m = len(data["units"])
        g["lattices"] += 1
        g["units_min"] = m if g["units_min"] is None else min(g["units_min"], m)
        g["units_max"] = max(g["units_max"], m)
        g["executions"] += len(executions)
        g["brute_force"] += len(cells[0]["outcomes"]) * len(cells)
        g["generated_tokens"] += sum(int(e.get("tokens_out", 0)) for e in executions)
        g["nominal_tokens"] += sum(int(e.get("nominal_tokens_out", 0)) for e in executions)
    (ROOT / "results" / "cost_by_lattice.json").write_text(json.dumps(groups, indent=1))
    for name, g in groups.items():
        print(name, g["lattices"], g["units_min"], g["units_max"], round(g["executions"] / g["brute_force"], 3),
              round(g["generated_tokens"] / g["nominal_tokens"], 3))


if __name__ == "__main__":
    main()
