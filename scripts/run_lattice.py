"""Evaluate every product of a lineage's net units on shared streams (RQ1-RQ6 input).

Usage:
  python scripts/run_lattice.py <backend.yaml> <benchmark> --seed S.py --final F.py --name NAME
         [--split search] [--n-tasks 60] [--replicates 4] [--brute] [--workers 4]
  python scripts/run_lattice.py <backend.yaml> <benchmark> --log LOOP.log.jsonl --harness-dir DIR --name NAME ...

Writes runs/lattice/<name>_<split>.json: units, per-cell outcome of every product, and the
executions performed (reads, generated and nominal tokens). With --brute every product is
also run on its own in every cell and compared with the shared result (exactness check).
With --tree (design D11) cells are explored depth first over ordered reads and stored as leaves
(present units, absent units, outcome); per-product outcomes are also written when there are at
most 2^14 products. --brute then checks every product if there are at most 2^12, else 64 random
products per cell (seeded), or every product with --brute-all.
"""

import argparse
import itertools
import json
import os
import random
import sys
import time
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

from editlattice.lineage import load_loop_lineage  # noqa: E402
from editlattice.backends import make_client  # noqa: E402
from editlattice.units import build_units, combine_lineages, group_by_items  # noqa: E402
from editlattice.tree import explore_tree  # noqa: E402
from editlattice.variational import Instrumented, explore_cell, run_product  # noqa: E402

COMPARED = ("answer", "outcome", "n_llm_calls", "request_fps")


def key(subset: frozenset[int]) -> str:
    """JSON key of a product."""
    return ",".join(map(str, sorted(subset)))


def main() -> None:
    """Explore every cell and write the lattice file."""
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("benchmark")
    parser.add_argument("--seed", default="")
    parser.add_argument("--final", default="")
    parser.add_argument("--log", default="")
    parser.add_argument("--pool", nargs="*", default=[], help="several loop logs from one seed: recombination product line")
    parser.add_argument("--max-units", type=int, default=12, help="above this, units are grouped by top-level item (pre-registered)")
    parser.add_argument("--harness-dir", default="")
    parser.add_argument("--name", required=True)
    parser.add_argument("--split", default="search")
    parser.add_argument("--n-tasks", type=int, default=60)
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--brute", action="store_true")
    parser.add_argument("--brute-all", action="store_true", help="with --brute, check every feasible product even above 12 units")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--statement-level", action="store_true", help="track units by changed statements where sound")
    parser.add_argument("--tree", action="store_true", help="depth-first exploration over ordered reads; leaves stored (D11)")
    parser.add_argument("--namespace", default=None, help="stream namespace (default: the lineage name, so each lineage's cost is measured from its own streams)")
    args = parser.parse_args()
    origin = None
    grouped = False
    if args.pool:
        lineages = [load_loop_lineage(log, args.harness_dir) for log in args.pool]
        seeds = {lineage.seed for lineage in lineages}
        if len(seeds) != 1:
            raise SystemExit("pooled lineages must share one seed")
        finals = [lineage.product(frozenset(edit.index for edit in lineage.edits)) for lineage in lineages]
        unit_set, origin = combine_lineages(lineages[0].seed, finals)
    else:
        if args.log:
            lineage = load_loop_lineage(args.log, args.harness_dir)
            seed, final = lineage.seed, lineage.product(frozenset(edit.index for edit in lineage.edits))
        else:
            seed, final = Path(args.seed).read_text(), Path(args.final).read_text()
        unit_set = build_units(seed, final)
        if unit_set.m > args.max_units:
            unit_set, grouped = group_by_items(unit_set), True
    instrumented = Instrumented.build(unit_set, statement_level=args.statement_level, store_dir=RUNS / "harnesses", lazy=args.tree)
    config = yaml.safe_load(Path(args.config).read_text())
    client = make_client(config, RUNS / "cache")
    benchmark = load_benchmark(args.benchmark)
    tasks = select_tasks(benchmark, args.split, args.n_tasks)
    params, max_calls = config["params"], config.get("max_llm_calls", 12)
    cells = [(task, replicate) for task in tasks for replicate in range(args.replicates)]
    namespace = args.name if args.namespace is None else args.namespace

    small = unit_set.m <= 14
    players = [unit.index for unit in unit_set.units]

    def feasible(subset: frozenset[int]) -> bool:
        return not any(pair <= subset for pair in unit_set.conflicts)

    def explore_as_tree(cell):
        task, replicate = cell
        leaves = explore_tree(benchmark, task, instrumented, replicate, client, params, max_calls, namespace, unit_set.conflicts)
        mismatches = []
        if args.brute:
            if unit_set.m <= 12 or args.brute_all:
                checked = [s for s in (frozenset(c) for r in range(unit_set.m + 1) for c in itertools.combinations(players, r)) if feasible(s)]
            else:
                rng = random.Random(f"{task['task_id']}:{replicate}")
                checked = [s for s in (frozenset(p for p in players if rng.random() < 0.5) for _ in range(64)) if feasible(s)]
            for subset in checked:
                shared = next(leaf.record for leaf in leaves if leaf.contains(subset))
                alone = run_product(benchmark, task, instrumented, subset, replicate, client, params, max_calls, namespace)
                if any(shared[field] != alone[field] for field in COMPARED):
                    mismatches.append(key(subset))
        result = {"task_id": task["task_id"], "replicate": replicate,
                  "leaves": [{"ones": sorted(leaf.ones), "zeros": sorted(leaf.zeros), "outcome": leaf.outcome} for leaf in leaves],
                  "executions": [{"product": leaf.record["product"], "reads": leaf.record["reads"], "read_order": leaf.record["read_order"],
                                  "outcome": leaf.record["outcome"], "tokens_out": leaf.record["tokens_out"],
                                  "nominal_tokens_out": leaf.record["nominal_tokens_out"], "n_llm_calls": leaf.record["n_llm_calls"],
                                  "error": leaf.record["error"][:200]} for leaf in leaves],
                  "mismatches": mismatches, "n_checked": len(checked) if args.brute else 0}
        if small:
            result["outcomes"] = {key(s): next(leaf.outcome for leaf in leaves if leaf.contains(s))
                                  for s in (frozenset(c) for r in range(unit_set.m + 1) for c in itertools.combinations(players, r)) if feasible(s)}
        return result

    def explore(cell):
        if args.tree:
            return explore_as_tree(cell)
        task, replicate = cell
        results, executed = explore_cell(benchmark, task, instrumented, replicate, client, params, max_calls, namespace)
        mismatches = []
        if args.brute:
            for subset, shared in results.items():
                alone = run_product(benchmark, task, instrumented, subset, replicate, client, params, max_calls, namespace)
                if any(shared[field] != alone[field] for field in COMPARED):
                    mismatches.append(key(subset))
        return {"task_id": task["task_id"], "replicate": replicate,
                "outcomes": {key(subset): record["outcome"] for subset, record in results.items()},
                "executions": [{"product": record["product"], "reads": record["reads"], "outcome": record["outcome"],
                                "tokens_out": record["tokens_out"], "nominal_tokens_out": record["nominal_tokens_out"],
                                "n_llm_calls": record["n_llm_calls"], "error": record["error"][:200]} for record in executed],
                "mismatches": mismatches}

    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        cell_results = list(pool.map(explore, cells))
    output = {
        "name": args.name + ("_stmt" if args.statement_level else ""), "benchmark": args.benchmark, "split": args.split, "n_tasks": len(tasks), "replicates": args.replicates,
        "model": config["backend"]["model"], "params": params, "namespace": namespace,
        "units": [{"index": unit.index, "items": list(unit.items), "diff": unit.describe(unit_set.seed.splitlines(keepends=True)),
                   "hunks": [{"start": h.start, "end": h.end, "replacement": list(h.replacement)} for h in unit.hunks]} for unit in unit_set.units],
        "seed_source": unit_set.seed,
        "depends": {name: sorted(units) for name, units in instrumented.model.depends.items()},
        "global_units": sorted(instrumented.model.global_units), "statement_level": args.statement_level,
        "grouped_by_items": grouped, "pool": [Path(log).name for log in args.pool],
        "origin": {str(k): v for k, v in origin.items()} if origin else None,
        "conflicts": sorted(sorted(pair) for pair in unit_set.conflicts),
        "marked_units": sorted(instrumented.marked), "tree": args.tree,
        "seconds": round(time.time() - started, 1), "cells": cell_results,
    }
    out = RUNS / "lattice" / f"{args.name}{'_stmt' if args.statement_level else ''}_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=1))
    executions = sum(len(cell["executions"]) for cell in cell_results)
    products = len(instrumented.sources) if not args.tree else (len(next(iter(cell_results))["outcomes"]) if small else 2 ** unit_set.m)
    print(f"{args.name}: {unit_set.m} units, {products} products, {len(cells)} cells, {executions} executions "
          f"(brute force {products * len(cells)}), mismatches {sum(len(cell['mismatches']) for cell in cell_results)}, "
          f"{output['seconds']} s -> {out}")


if __name__ == "__main__":
    main()
