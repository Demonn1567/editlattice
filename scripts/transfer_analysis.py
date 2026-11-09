"""RQ6 (DEC-9): does attribution transfer across solvers and temperatures?

For each lineage measured under several settings (held-out lattices), the exact Shapley value and
pairwise interaction of every unit per setting, the Spearman correlation of Shapley values between
every pair of settings, whether each setting ranks the same unit first, and the value of the full
lineage over the seed. Settings whose lattice file is missing are skipped.

Usage: python scripts/transfer_analysis.py
Writes results/transfer.json.
"""

import json
import sys
from pathlib import Path

from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from editlattice.attribution import shapley, shapley_interaction  # noqa: E402

LATTICES = ROOT / "runs" / "lattice"
LINEAGES = {
    "planted": {"Qwen3-4B MLX 4-bit T0.7 (Mac)": "planted_stmt_heldout.json",
                "Llama-3.1-8B MLX 4-bit T0.7 (Mac)": "planted_llama_stmt_heldout.json",
                "Qwen3-4B AWQ T0.7": "planted_gpu_stmt_heldout.json",
                "Qwen3-4B AWQ T1.0": "planted_gpu_t1_stmt_heldout.json",
                "Qwen3-4B bf16 T0.7": "planted_gpu_bf16_stmt_heldout.json"},
}
for bench in ("gsmhard", "lcb", "bird"):
    LINEAGES[f"constructed_{bench}"] = {"Qwen3-4B AWQ T0.7": f"constructed_{bench}_stmt_heldout.json",
                                        "Qwen3-4B AWQ T1.0": f"constructed_{bench}_t1_stmt_heldout.json",
                                        # DEC-11: the LCB and BIRD bf16 lattices were rerun with the full context (*_bf16full).
                                        "Qwen3-4B bf16 T0.7": f"constructed_{bench}_bf16{'' if bench == 'gsmhard' else 'full'}_stmt_heldout.json"}
# The evolved real-gain lineages (preregistration addendum 2026-11-01); Llama for B3_s1 is exploratory (2026-11-02).
for name in ("phaseA3_lcb_metaharness_B1_s2", "phaseA3_bird_metaharness_B1_s1", "phaseA3_bird_metaharness_B3_s1"):
    LINEAGES[name] = {"Qwen3-4B AWQ T0.7": f"{name}_stmt_heldout.json", "Qwen3-4B AWQ T1.0": f"{name}_t1_stmt_heldout.json",
                      "Qwen3-4B bf16 T0.7": f"{name}_bf16_stmt_heldout.json"}
# Exploratory Llama runs of the remaining lineages (addendum 2026-11-03); missing lattices are skipped.
LLAMA = "Llama-3.1-8B MLX 4-bit T0.7 (Mac, exploratory)"
for name in ("phaseA3_lcb_metaharness_B1_s2", "phaseA3_bird_metaharness_B1_s1", "phaseA3_bird_metaharness_B3_s1"):
    LINEAGES[name][LLAMA] = f"{name}_llama_stmt_heldout.json"
for bench in ("gsmhard", "lcb", "bird"):
    LINEAGES[f"constructed_{bench}"][LLAMA] = f"constructed_{bench}_llama_stmt_heldout.json"
# Second cross-family build on the GPU (addendum 2026-11-06), every lineage of the transfer tables.
LLAMA_AWQ = "Llama-3.1-8B AWQ vLLM T0.7 (GPU, exploratory)"
for name in ("planted", "constructed_gsmhard", "constructed_lcb", "constructed_bird", "phaseA3_lcb_metaharness_B1_s2",
             "phaseA3_bird_metaharness_B1_s1", "phaseA3_bird_metaharness_B3_s1"):
    LINEAGES[name][LLAMA_AWQ] = f"{name}_llama_awq_stmt_heldout.json"


def values(path: Path) -> tuple[dict[frozenset[int], float], list[int], int]:
    """Lattice values v(S) over all cells, the players and the number of cells."""
    data = json.loads(path.read_text())
    players = [unit["index"] for unit in data["units"]]
    cells = data["cells"]
    totals: dict[str, float] = {}
    for cell in cells:
        for product, outcome in cell["outcomes"].items():
            totals[product] = totals.get(product, 0.0) + outcome
    lattice = {frozenset(int(p) for p in product.split(",") if p): total / len(cells) for product, total in totals.items()}
    return lattice, players, len(cells)


def main() -> None:
    out = {}
    for lineage, settings in LINEAGES.items():
        rows = {}
        for setting, file in settings.items():
            path = LATTICES / file
            if not path.exists():
                continue
            lattice, players, cells = values(path)
            phi = shapley(lattice, players)
            rows[setting] = {"cells": cells, "seed": lattice[frozenset()], "full": lattice[frozenset(players)],
                             "shapley": {str(p): phi[p] for p in players},
                             "interaction": {f"{a},{b}": v for (a, b), v in shapley_interaction(lattice, players).items()},
                             "top_unit": max(players, key=lambda p: phi[p])}
        names = list(rows)
        pairs = []
        for i, first in enumerate(names):
            for second in names[i + 1:]:
                a = [rows[first]["shapley"][u] for u in rows[first]["shapley"]]
                b = [rows[second]["shapley"][u] for u in rows[first]["shapley"]]
                rho = spearmanr(a, b).statistic if len(set(a)) > 1 and len(set(b)) > 1 else None
                pairs.append({"first": first, "second": second, "spearman": rho})
        out[lineage] = {"settings": rows, "pairs": pairs,
                        "same_top_unit": len({row["top_unit"] for row in rows.values()}) == 1 if rows else None}
        print(lineage, {s: (round(r["full"] - r["seed"], 3), {u: round(v, 3) for u, v in r["shapley"].items()}) for s, r in rows.items()})
    (ROOT / "results" / "transfer.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
