"""Accuracy of one harness on a split (gain-pilot baseline and sanity check).

Usage: python scripts/seed_eval.py configs/<backend>.yaml <harness.py> <bird|lcb> [--split search]
           [--n-tasks 20] [--replicates 1] [--out runs/seed_eval/<name>.jsonl]
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
from editlattice.backends import make_client  # noqa: E402


def main() -> None:
    """Run the harness and print accuracy with a task-bootstrap CI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("harness")
    parser.add_argument("benchmark")
    parser.add_argument("--split", default="search")
    parser.add_argument("--n-tasks", type=int, default=20)
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    client = make_client(config, RUNS / "cache")
    benchmark = load_benchmark(args.benchmark)
    tasks = select_tasks(benchmark, args.split, args.n_tasks)
    version = HarnessVersion.from_file(args.harness, RUNS / "harnesses")
    jobs = [(task, replicate) for task in tasks for replicate in range(args.replicates)]
    with ThreadPoolExecutor(max_workers=config.get("concurrency", 2)) as pool:
        records = list(pool.map(lambda job: run_one(benchmark, job[0], version, job[1], client, config["params"], True,
                                                    config.get("max_llm_calls", 12)), jobs))
    out = Path(args.out or RUNS / "seed_eval" / f"{Path(args.harness).stem}_{args.benchmark}_{args.split}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(record) + "\n" for record in records))
    per_task: dict[str, list[int]] = {}
    for record in records:
        per_task.setdefault(record["task_id"], []).append(record["outcome"])
    means = [sum(values) / len(values) for values in per_task.values()]
    accuracy = sum(means) / len(means)
    rng = random.Random(0)
    boot = sorted(sum(rng.choice(means) for _ in means) / len(means) for _ in range(2000))
    errors = sum(1 for record in records if record["error"])
    tokens = sum(record["tokens_out"] for record in records) / len(records)
    print(f"{version.harness_hash} {args.benchmark}/{args.split}: accuracy {accuracy:.3f} "
          f"[{boot[50]:.3f}, {boot[1949]:.3f}] on {len(means)} tasks x {args.replicates}; "
          f"errors {errors}; mean generated tokens {tokens:.0f}; records {out}")


if __name__ == "__main__":
    main()
