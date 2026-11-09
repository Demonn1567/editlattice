"""Summary of every gain-pilot loop (Phases A, A2, A3 and A4): proposals, validity, accepted edits, net units,
held-out gain with its CI, and real-gain verdict. Inputs: the loop logs and stored harnesses copied
from the GPU (runs/gpu_pilot/) and results/gpu/gain_*.json. Writes results/pilot_summary.json.

Usage: python scripts/pilot_summary.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from editlattice.lineage import load_loop_lineage  # noqa: E402
from editlattice.units import build_units  # noqa: E402

PILOTS = [("A", "gain_pilot"), ("A2", "gain_pilot2"), ("A3", "gain_pilot3"), ("A4", "gain_pilot4")]
BENCHMARKS = ["gsmhard", "lcb", "bird"]


def main() -> None:
    """One row per loop, plus per (phase, benchmark) totals."""
    rows, totals = [], []
    for phase, prefix in PILOTS:
        for bench in BENCHMARKS:
            experiment = f"{prefix}_{bench}"
            if not (ROOT / "results/gpu" / f"gain_{experiment}.json").exists():
                print(f"{experiment}: no gain check yet, skipped")
                continue
            gain = {loop["loop"]: loop for loop in json.loads((ROOT / "results/gpu" / f"gain_{experiment}.json").read_text())["loops"]}
            errors: Counter = Counter()
            for log in sorted((ROOT / "runs/gpu_pilot" / experiment).glob("*.log.jsonl")):
                entries = [json.loads(line) for line in log.read_text().splitlines()]
                loop = log.name.replace(".log.jsonl", "")
                for entry in entries:
                    errors["valid" if entry["proposal_valid"] else entry["proposal_error"].split(":")[0][:40]] += 1
                lineage = load_loop_lineage(log, ROOT / "runs/gpu_pilot/harnesses")
                final = lineage.product(frozenset(edit.index for edit in lineage.edits))
                units = build_units(lineage.seed, final).m if lineage.edits else 0
                result = gain[loop]
                rows.append({"phase": phase, "benchmark": bench, "loop": loop, "proposals": len(entries),
                             "valid": sum(1 for e in entries if e["proposal_valid"]), "accepted": sum(1 for e in entries if e.get("accepted")),
                             "net_units": units, "gain": result.get("gain"), "low": result.get("low"), "high": result.get("high"),
                             "real_gain": result["real_gain"], "seed_accuracy": result.get("seed_accuracy")})
            loops = [r for r in rows if r["phase"] == phase and r["benchmark"] == bench]
            gains = [r for r in loops if r["gain"] is not None]
            best = max(gains, key=lambda r: r["gain"]) if gains else None
            totals.append({"phase": phase, "benchmark": bench, "loops": len(loops), "proposals": sum(r["proposals"] for r in loops),
                           "valid": sum(r["valid"] for r in loops), "accepted": sum(r["accepted"] for r in loops),
                           "net_units": sum(r["net_units"] for r in loops), "real_gain_loops": sum(r["real_gain"] for r in loops),
                           "harmful_loops": sum(1 for r in gains if r["high"] < 0), "exactly_zero_loops": sum(1 for r in gains if r["low"] == r["high"] == 0),
                           "best_gain": best and {k: best[k] for k in ("loop", "gain", "low", "high")}, "errors": dict(errors)})
    out = ROOT / "results" / "pilot_summary.json"
    out.write_text(json.dumps({"totals": totals, "loops": rows}, indent=1))
    for total in totals:
        print(json.dumps({k: total[k] for k in ("phase", "benchmark", "loops", "proposals", "valid", "accepted", "net_units", "real_gain_loops",
                                                "harmful_loops", "exactly_zero_loops")}), total["best_gain"])


if __name__ == "__main__":
    main()
