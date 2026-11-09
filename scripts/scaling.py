"""RQ1b scaling study (pre-registered in notes/preregistration.md, addendum D11).

Usage: python scripts/scaling.py [--out results/scaling.json]
A synthetic harness family with m units and route length L: task routes call a chain of L step
functions, each changed by one unit (prompt constant, post-processing statement or retry), with
data-dependent early exits. Runs use the fake LLM (seeded, request-dependent replies), so the
study measures the method's cost and exactness, not model quality. For each (m, L):
static analysis time (compositional; exhaustive and eager where feasible), executions of tree
exploration vs 2^m brute force, read width, exactness against plain runs on sampled products,
attribution time and agreement with exhaustive formulas (small m), MILP times.
"""

import argparse
import itertools
import json
import random
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from editlattice.attribution import shapley, shapley_interaction  # noqa: E402
from editlattice.harness_runtime.runner import HarnessVersion, run_one  # noqa: E402
from editlattice.llm.cache import ResponseCache  # noqa: E402
from editlattice.llm.client import LLMClient  # noqa: E402
from editlattice.llm.fake import FakeBackend  # noqa: E402
import editlattice.units as units_module  # noqa: E402
import editlattice.variational as variational_module  # noqa: E402
from editlattice.tree import TreeLattice, explore_tree  # noqa: E402
from editlattice.units import build_units  # noqa: E402
from editlattice.variational import Instrumented  # noqa: E402

COMPARED = ("answer", "outcome", "n_llm_calls", "request_fps")
WORDS = ["stop here", "alpha", "beta gamma", "ALPHA", "delta", "go on", "Gamma"]


def responder(messages, rng):
    return rng.choice(WORDS)


class Bench:
    name = "scaling"

    def public_view(self, task):
        return {k: v for k, v in task.items() if k != "gold"}

    def make_env(self, task):
        return None

    def close_env(self, env):
        return None

    def score(self, task, answer):
        correct = int(task["gold"] in str(answer).lower())
        return correct, float(correct)


def family(m: int, length: int) -> tuple[str, str, int]:
    """Seed and final harness: m steps, routes of ``length`` steps; unit k changes step k."""
    routes = [list(range(start, min(start + length, m))) for start in range(0, m, length)]
    seed, final = ['"""Synthetic harness (scaling study)."""\n\n'], ['"""Synthetic harness (scaling study)."""\n\n']
    for k in range(m):
        prompt = f'PROMPT_{k} = "Step {k}: answer."\n\n\n'
        body = (f"def step_{k}(llm, text):\n"
                f'    reply = llm.chat([{{"role": "system", "content": PROMPT_{k}}}, {{"role": "user", "content": text}}])\n'
                f"    return reply.strip()\n\n\n")
        seed += [prompt, body]
        if k % 3 == 0:
            final += [prompt.replace("answer.", "answer carefully."), body]
        elif k % 3 == 1:
            final += [prompt, body.replace("return reply.strip()", "return reply.strip().lower()")]
        else:
            retry = ("    if len(reply) < 6:\n"
                     f'        reply = llm.chat([{{"role": "system", "content": PROMPT_{k}}}, {{"role": "user", "content": "Again: " + text}}])\n')
            final += [prompt, body.replace("    return reply.strip()", retry + "    return reply.strip()")]
    for r, steps in enumerate(routes):
        lines = [f"def route_{r}(task, llm):\n", '    text = task["question"]\n']
        for k in steps:
            lines += [f"    text = step_{k}(llm, text)\n", '    if text.startswith("stop"):\n', "        return text\n"]
        lines += ["    return text\n\n\n"]
        seed += lines
        final += lines
    tail = ("def solve(task, llm, env):\n"
            f"    routes = [{', '.join(f'route_{r}' for r in range(len(routes)))}]\n"
            '    return {"answer": routes[task["route"]](task, llm)}\n')
    seed.append(tail)
    final.append(tail)
    return "".join(seed), "".join(final), len(routes)


def exhaustive_analysis(unit_set):
    """Read model and markers with the all-subsets enumeration (for comparison)."""
    saved = units_module.analysis_subsets
    units_module.analysis_subsets = variational_module.analysis_subsets = lambda u: u.all_subsets()
    try:
        started = time.perf_counter()
        Instrumented.build(unit_set, statement_level=True, lazy=True)
        return time.perf_counter() - started
    finally:
        units_module.analysis_subsets = variational_module.analysis_subsets = saved


def run_config(m: int, length: int, n_tasks: int, replicates: int, samples: int, workdir: Path, seed: int,
               exhaustive: bool = True) -> dict:
    rng = random.Random(seed)
    source, final, n_routes = family(m, length)
    client = LLMClient(backend=FakeBackend(responder), cache=ResponseCache(workdir / "cache"), model="fake")
    bench = Bench()
    started = time.perf_counter()
    unit_set = build_units(source, final)
    t_units = time.perf_counter() - started
    assert unit_set.m == m, (unit_set.m, m)
    started = time.perf_counter()
    instrumented = Instrumented.build(unit_set, statement_level=True, lazy=True, store_dir=workdir / "store")
    t_static = time.perf_counter() - started
    row = {"m": m, "length": length, "routes": n_routes, "t_build_units": round(t_units, 3), "t_static": round(t_static, 3),
           "marked": len(instrumented.marked)}
    if exhaustive and m <= 12:
        row["t_static_exhaustive"] = round(exhaustive_analysis(unit_set), 3)
    tasks = [{"task_id": f"t{i}", "route": i % n_routes, "question": f"question {i}", "gold": rng.choice(["alpha", "beta", "gamma", "delta"])}
             for i in range(n_tasks)]
    started = time.perf_counter()
    cells = {}
    for task in tasks:
        for replicate in range(replicates):
            cells[(task["task_id"], replicate)] = explore_tree(bench, task, instrumented, replicate, client, {}, 2 * m + 4)
    row["t_explore"] = round(time.perf_counter() - started, 3)
    players = [unit.index for unit in unit_set.units]
    lattice = TreeLattice(players=players, cells=cells)
    leaves = [len(v) for v in cells.values()]
    row.update({"cells": len(cells), "executions": sum(leaves), "max_leaves_per_cell": max(leaves),
                "mean_leaves_per_cell": round(sum(leaves) / len(leaves), 2), "read_width": lattice.depth(),
                "brute_force_runs": len(cells) * 2 ** m})
    # Exactness on sampled products: seed, full, and random products per cell.
    started = time.perf_counter()
    mismatches = checked = 0
    for task in tasks:
        for replicate in range(replicates):
            chosen = [frozenset(), frozenset(players)] + [frozenset(p for p in players if rng.random() < 0.5) for _ in range(samples)]
            for subset in chosen:
                leaf = [leaf for leaf in cells[(task["task_id"], replicate)] if leaf.contains(subset)]
                assert len(leaf) == 1
                version = HarnessVersion.from_source(unit_set.product(subset), workdir / "store")
                reference = run_one(bench, task, version, replicate, client, {}, crn=True, max_llm_calls=2 * m + 4)
                checked += 1
                mismatches += any(leaf[0].record[field] != reference[field] for field in COMPARED)
    row.update({"checked": checked, "mismatches": mismatches, "t_check": round(time.perf_counter() - started, 3)})
    started = time.perf_counter()
    values_shapley = lattice.shapley()
    values_interaction = lattice.interaction()
    row["t_attribution"] = round(time.perf_counter() - started, 3)
    row["shapley_abs_sum"] = round(sum(abs(v) for v in values_shapley.values()), 4)
    row["nonzero_interactions"] = sum(1 for v in values_interaction.values() if abs(v) > 1e-12)
    if m <= 10:
        started = time.perf_counter()
        values = {frozenset(c): lattice.value(frozenset(c)) for r in range(m + 1) for c in itertools.combinations(players, r)}
        exact_s = shapley(values, players)
        exact_i = shapley_interaction(values, players)
        row["t_attribution_exhaustive"] = round(time.perf_counter() - started, 3)
        row["max_diff_shapley"] = max(abs(exact_s[p] - values_shapley[p]) for p in players)
        row["max_diff_interaction"] = max(abs(exact_i[k] - values_interaction[k]) for k in exact_i)
    full_value = lattice.value(frozenset(players))
    milp_ok = m <= 12 or length < m  # the dense rows beyond m = 12 are left to enumeration (DEC-5)
    if milp_ok:
        started = time.perf_counter()
        best, best_value = lattice.optimise(method="milp")
        row["t_milp_best"] = round(time.perf_counter() - started, 3)
        started = time.perf_counter()
        small, small_value = lattice.optimise(minimise_size_with_floor=full_value - 0.05, method="milp")
        row["t_milp_distill"] = round(time.perf_counter() - started, 3)
    if m <= 24:
        started = time.perf_counter()
        best_e, best_value_e = lattice.optimise(method="enumerate")
        small_e, small_value_e = lattice.optimise(minimise_size_with_floor=full_value - 0.05, method="enumerate")
        row["t_enumerate_both"] = round(time.perf_counter() - started, 3)
        if milp_ok:
            row["milp_equals_enumeration"] = bool(abs(best_value - best_value_e) < 1e-12 and len(small) == len(small_e))
        else:
            best, best_value, small, small_value = best_e, best_value_e, small_e, small_value_e
    row.update({"v_seed": lattice.value(frozenset()), "v_full": full_value, "v_best": best_value, "best_size": len(best),
                "distilled_size": len(small), "v_distilled": small_value})
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "results" / "scaling.json"))
    parser.add_argument("--tasks", type=int, default=24)
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261027)
    args = parser.parse_args()
    grid = [(m, length) for m in (4, 8, 12, 16, 24, 32, 48, 64) for length in (2, 4, 8) if length <= m]
    grid += [(m, m) for m in (10, 12, 14, 16)]  # one route through every step: the honest worst case
    rows = []
    timed: set[int] = set()  # the exhaustive static analysis depends on the units, not the routes: time it once per m
    for m, length in grid:
        with tempfile.TemporaryDirectory() as tmp:
            row = run_config(m, length, args.tasks, args.replicates, args.samples, Path(tmp), args.seed, exhaustive=m not in timed)
        timed.add(m)
        rows.append(row)
        print(json.dumps(row), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"config": vars(args), "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
