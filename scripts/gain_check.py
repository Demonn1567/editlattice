"""Real-gain check (pre-registration, Phase A): final harness vs seed on held-out tasks.

Usage: python scripts/gain_check.py <backend.yaml> <benchmark> <evolution_dir> [--n-tasks 200] [--replicates 4]
For every loop log in <evolution_dir> (runs/<experiment>/*.log.jsonl) whose loop accepted at least
one edit, runs the seed and the final harness on the same held-out cells (shared CRN streams, own
namespace) and reports the paired gain with a 95% task-cluster bootstrap CI. A loop is real-gain if
the CI lies above 0 and the gain is at least 3 points. Writes results/gain_<experiment>.json.
"""

import argparse
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
# Overridable so that integration tests never write into the real run and result folders.
RUNS = Path(os.environ.get("EDITLATTICE_RUNS_DIR", ROOT / "runs"))
RESULTS = Path(os.environ.get("EDITLATTICE_RESULTS_DIR", ROOT / "results"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from pilot import load_benchmark, select_tasks  # noqa: E402

from editlattice.harness_runtime.runner import HarnessVersion, run_one  # noqa: E402
from editlattice.lineage import load_loop_lineage  # noqa: E402
from editlattice.backends import make_client  # noqa: E402

MIN_GAIN = 0.03
NAMESPACE = "heldout"


def paired_gain(seed: dict[str, list[int]], final: dict[str, list[int]], n_boot: int = 2000, seed_rng: int = 0) -> dict:
    """Mean per-task difference and its task-cluster bootstrap CI."""
    tasks = sorted(seed)
    differences = [sum(final[t]) / len(final[t]) - sum(seed[t]) / len(seed[t]) for t in tasks]
    rng = random.Random(seed_rng)
    boot = sorted(sum(rng.choice(differences) for _ in differences) / len(differences) for _ in range(n_boot))
    gain = sum(differences) / len(differences)
    return {"gain": gain, "low": boot[int(0.025 * n_boot)], "high": boot[int(0.975 * n_boot) - 1],
            "seed_accuracy": sum(sum(v) / len(v) for v in seed.values()) / len(seed),
            "final_accuracy": sum(sum(v) / len(v) for v in final.values()) / len(final)}


def main() -> None:
    """Evaluate every loop's seed and final harness on held-out cells."""
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("benchmark")
    parser.add_argument("evolution_dir")
    parser.add_argument("--n-tasks", type=int, default=200)
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    client = make_client(config, RUNS / "cache")
    benchmark = load_benchmark(args.benchmark)
    tasks = select_tasks(benchmark, "heldout", args.n_tasks)
    evolution_dir = Path(args.evolution_dir)
    store = RUNS / "harnesses"
    loops = []
    for log in sorted(evolution_dir.glob("*.log.jsonl")):
        lineage = load_loop_lineage(log, store)
        if lineage.k == 0:
            loops.append({"loop": lineage.name, "accepted": 0})
            continue
        final = lineage.product(frozenset(edit.index for edit in lineage.edits))
        loops.append({"loop": lineage.name, "accepted": lineage.k, "seed": HarnessVersion.from_source(lineage.seed, store),
                      "final": HarnessVersion.from_source(final, store)})
    jobs = [(loop, role, task, replicate) for loop in loops if loop["accepted"] for role in ("seed", "final")
            for task in tasks for replicate in range(args.replicates)]

    def run(job):
        loop, role, task, replicate = job
        record = run_one(benchmark, task, loop[role], replicate, client, config["params"], True, config.get("max_llm_calls", 12), namespace=NAMESPACE)
        return loop["loop"], role, record["task_id"], record["outcome"]

    outcomes: dict[tuple[str, str], dict[str, list[int]]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for name, role, task_id, outcome in pool.map(run, jobs):
            outcomes.setdefault((name, role), {}).setdefault(task_id, []).append(outcome)
    results = []
    for loop in loops:
        entry = {"loop": loop["loop"], "accepted": loop["accepted"]}
        if loop["accepted"]:
            entry.update(paired_gain(outcomes[(loop["loop"], "seed")], outcomes[(loop["loop"], "final")]))
            entry["real_gain"] = entry["low"] > 0 and entry["gain"] >= MIN_GAIN
        else:
            entry["real_gain"] = False
        results.append(entry)
        print(json.dumps({key: (round(value, 4) if isinstance(value, float) else value) for key, value in entry.items()}))
    out = RESULTS / f"gain_{evolution_dir.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"benchmark": args.benchmark, "n_tasks": len(tasks), "replicates": args.replicates,
                               "model": config["backend"]["model"], "loops": results}, indent=1))
    print(f"real-gain loops: {sum(entry['real_gain'] for entry in results)} of {len(results)} -> {out}")


if __name__ == "__main__":
    main()
