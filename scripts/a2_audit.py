"""A2 audit (pre-registered): do replayed runs behave like fresh runs?

Usage: python scripts/a2_audit.py <backend.yaml> <benchmark> runs/lattice/<name>_<split>.json [--fraction 0.1]
Draws a random fraction of (product, cell) pairs of a lattice, runs each product on a stream
private to the product (an independent stream, so nothing is replayed from other products), and
compares, per product, the accuracy of the replayed lattice outcomes with the fresh outcomes on the
same cells (Fisher exact test). Writes results/a2_<name>_<split>.json.
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(os.environ.get("EDITLATTICE_RUNS_DIR", ROOT / "runs"))
RESULTS = Path(os.environ.get("EDITLATTICE_RESULTS_DIR", ROOT / "results"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from pilot import load_benchmark, select_tasks  # noqa: E402

from editlattice.backends import make_client  # noqa: E402
from editlattice.harness_runtime.runner import HarnessVersion, run_one  # noqa: E402
from editlattice.units import Hunk, Unit, UnitSet  # noqa: E402


def unit_set_from_lattice(data: dict, seed: str) -> UnitSet:
    """Rebuild the products from the unit hunks stored in the lattice file."""
    units = [Unit(unit["index"], tuple(Hunk(h["start"], h["end"], tuple(h["replacement"])) for h in unit["hunks"]), tuple(unit["items"]))
             for unit in data["units"]]
    return UnitSet(seed=seed, units=units, conflicts=frozenset(frozenset(pair) for pair in data.get("conflicts", [])))


def main() -> None:
    """Fresh runs on a random sample of lattice cells; compare per product."""
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("benchmark")
    parser.add_argument("lattice")
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261027)
    args = parser.parse_args()
    data = json.loads(Path(args.lattice).read_text())
    if "seed_source" not in data or "hunks" not in data["units"][0]:
        raise SystemExit("lattice file lacks seed_source/unit hunks (rerun run_lattice.py)")
    unit_set = unit_set_from_lattice(data, data["seed_source"])
    config = yaml.safe_load(Path(args.config).read_text())
    client = make_client(config, RUNS / "cache")
    benchmark = load_benchmark(args.benchmark)
    tasks = {task["task_id"]: task for task in select_tasks(benchmark, data["split"], data["n_tasks"])}
    rng = random.Random(args.seed)
    pairs = [(product, cell["task_id"], cell["replicate"], cell["outcomes"][product])
             for cell in data["cells"] for product in cell["outcomes"] if rng.random() < args.fraction]
    store = RUNS / "harnesses"

    def fresh(pair):
        product, task_id, replicate, replayed = pair
        subset = frozenset(int(part) for part in product.split(",") if part)
        version = HarnessVersion.from_source(unit_set.product(subset), store)
        # A stream namespace private to this product: no response is shared with any other product.
        record = run_one(benchmark, tasks[task_id], version, replicate, client, config["params"], True,
                         config.get("max_llm_calls", 12), namespace=f"a2:{data['namespace']}:{product}")
        return product, replayed, record["outcome"], task_id, replicate

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(fresh, pairs))
    by_product: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for product, replayed, outcome, _, _ in results:
        by_product[product].append((replayed, outcome))
    rows = []
    for product, values in sorted(by_product.items()):
        replay_hits = sum(r for r, _ in values)
        fresh_hits = sum(f for _, f in values)
        n = len(values)
        _, p_value = fisher_exact([[replay_hits, n - replay_hits], [fresh_hits, n - fresh_hits]])
        rows.append({"product": product, "n": n, "replayed": replay_hits / n, "fresh": fresh_hits / n, "fisher_p": p_value})
    total = len(results)
    summary = {"pairs": total, "replayed_accuracy": sum(r[1] for r in results) / total, "fresh_accuracy": sum(r[2] for r in results) / total,
               "products_with_p_below_0.05": sum(1 for row in rows if row["fisher_p"] < 0.05), "products": len(rows)}
    out = RESULTS / f"a2_{data['name']}_{data['split']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    # Per-pair fresh outcomes are kept: they are the independent-stream pool for RQ2.
    pairs_out = [{"product": product, "task_id": task_id, "replicate": replicate, "replayed": replayed, "fresh": outcome}
                 for product, replayed, outcome, task_id, replicate in results]
    out.write_text(json.dumps({"summary": summary, "rows": rows, "pairs": pairs_out}, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
