"""Pilot: parent + mutant children under CRN; measures cost, touch rates, flakiness.

Usage: python scripts/pilot.py configs/<pilot>.yaml [--limit-tasks N]
Resumable: re-running skips finished (task, harness, replicate) runs.
"""

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from editlattice.benchmarks.bird import BirdBenchmark  # noqa: E402
from editlattice.benchmarks.gsmhard import GsmHardBenchmark  # noqa: E402
from editlattice.benchmarks.livecodebench import LiveCodeBenchmark  # noqa: E402
from editlattice.harness_runtime.runner import HarnessVersion, RecordStore, RunRequest, run_many  # noqa: E402
from editlattice.impact.diff import compute_impact  # noqa: E402
from editlattice.llm.cache import ResponseCache  # noqa: E402
from editlattice.llm.settings import backend_settings  # noqa: E402
from editlattice.llm.client import LLMClient, OpenAICompatibleBackend  # noqa: E402
from editlattice.mutation.operators import apply_operator, operators_for  # noqa: E402


def load_benchmark(name: str):
    """Instantiate a benchmark by name."""
    if name == "bird":
        return BirdBenchmark()
    if name == "lcb":
        return LiveCodeBenchmark()
    if name == "gsmhard":
        return GsmHardBenchmark()
    raise ValueError(name)


def select_tasks(benchmark, split: str, n_tasks: int) -> list[dict]:
    """First n tasks of the frozen split (nested subsets, D-007)."""
    split_file = json.loads((ROOT / "configs" / "splits" / f"{benchmark.name}.json").read_text())
    wanted = split_file["splits"][split][:n_tasks]
    by_id = {}
    for task in benchmark.load_all_tasks():
        by_id[task["task_id"]] = task
    selected = []
    for task_id in wanted:
        selected.append(by_id[task_id])
    return selected


def progress_printer(total: int):
    """Return a callback that prints one line per finished run."""
    state = {"done": 0, "start": time.time()}

    def report(record: dict) -> None:
        state["done"] += 1
        elapsed = time.time() - state["start"]
        print(
            f"[{state['done']}/{total}] {record['task_id']} h={record['harness_hash'][:6]} r={record['replicate_id']} "
            f"ok={record['outcome']} calls={record['n_llm_calls']} hits={record['n_cache_hits']} "
            f"out_tok={record['tokens_out']} wall={record['wall_time']:.0f}s err={record['error'][:60]!r} elapsed={elapsed/60:.1f}m",
            flush=True,
        )

    return report


def summarize_cost(records: list[dict]) -> dict:
    """Mean cost per run over records that actually generated tokens."""
    if not records:
        return {}
    return {
        "n_runs": len(records),
        "mean_llm_calls": statistics.mean(r["n_llm_calls"] for r in records),
        "mean_tokens_in": statistics.mean(r["tokens_in"] for r in records),
        "mean_tokens_out": statistics.mean(r["tokens_out"] for r in records),
        "mean_wall_s": statistics.mean(r["wall_time"] for r in records),
        "mean_backend_s": statistics.mean(r["backend_seconds"] for r in records),
        "accuracy": statistics.mean(r["outcome"] for r in records),
        "error_rate": statistics.mean(1 if r["error"] else 0 for r in records),
    }


def flakiness(records: list[dict]) -> dict:
    """Share of tasks whose outcome differs across replicates."""
    by_task: dict[str, list[int]] = {}
    for record in records:
        by_task.setdefault(record["task_id"], []).append(record["outcome"])
    mixed = 0
    for outcomes in by_task.values():
        if len(set(outcomes)) > 1:
            mixed += 1
    return {"n_tasks": len(by_task), "flaky_tasks": mixed, "flaky_fraction": mixed / max(1, len(by_task))}


def main() -> None:
    """Run the pilot described by a YAML config."""
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--limit-tasks", type=int, default=None)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    backend_cfg = backend_settings(config)
    experiment = config["experiment"]
    run_dir = ROOT / "runs" / experiment
    benchmark = load_benchmark(config["benchmark"])
    n_tasks = args.limit_tasks or config["n_tasks"]
    tasks = select_tasks(benchmark, config["split"], n_tasks)
    backend = OpenAICompatibleBackend(backend_cfg["base_url"], backend_cfg["name"])
    client = LLMClient(backend=backend, model=backend_cfg["model"], cache=ResponseCache(ROOT / "runs" / "cache"), seed_mode=config.get("seed_mode", "request"))
    store = RecordStore(run_dir / "records.jsonl")
    harness_store = ROOT / "runs" / "harnesses"
    parent = HarnessVersion.from_file(ROOT / config["seed_harness"], harness_store)
    replicates = list(range(config["replicates"]))
    params = config["params"]

    started = datetime.now(timezone.utc).isoformat()
    parent_requests = [RunRequest(task, parent, r, True) for task in tasks for r in replicates]
    print(f"parent {parent.harness_hash}: {len(parent_requests)} runs", flush=True)
    parent_records = run_many(benchmark, parent_requests, client, params, store, config["concurrency"], config["max_llm_calls"], progress_printer(len(parent_requests)))

    operators = {}
    for operator in operators_for(config["benchmark"]):
        operators[operator.name] = operator
    edit_summaries = []
    for edit_name in config["edits"]:
        child = HarnessVersion.from_source(apply_operator(operators[edit_name], parent.source), harness_store)
        impact = compute_impact(parent.source, child.source)
        touched_runs = 0
        affected_tasks = set()
        for record in parent_records:
            if impact.is_global or set(record["covered_functions"]) & impact.functions:
                touched_runs += 1
                affected_tasks.add(record["task_id"])
        child_requests = [RunRequest(task, child, r, True) for task in tasks for r in replicates]
        print(f"edit {edit_name} -> {child.harness_hash}: I={sorted(impact.functions)} global={impact.is_global} affected={len(affected_tasks)}/{len(tasks)}", flush=True)
        child_records = run_many(benchmark, child_requests, client, params, store, config["concurrency"], config["max_llm_calls"], progress_printer(len(child_requests)))
        line_touched_runs = 0
        diverged_runs = 0
        for parent_record, child_record in zip(parent_records, child_records):
            covered_lines = set()
            for label, number in parent_record["covered_lines"]:
                covered_lines.add((label, number))
            if impact.is_global or covered_lines & impact.lines:
                line_touched_runs += 1
            if child_record["tokens_out"] > 0 or child_record["n_cache_hits"] < child_record["n_llm_calls"]:
                diverged_runs += 1
        identical_untouched = 0
        untouched = 0
        free_untouched = 0
        for parent_record, child_record in zip(parent_records, child_records):
            touched = impact.is_global or bool(set(parent_record["covered_functions"]) & impact.functions)
            if touched:
                continue
            untouched += 1
            same = parent_record["answer"] == child_record["answer"] and parent_record["outcome"] == child_record["outcome"]
            identical_untouched += 1 if same else 0
            free_untouched += 1 if child_record["tokens_out"] == 0 else 0
        delta = statistics.mean(c["outcome"] - p["outcome"] for p, c in zip(parent_records, child_records))
        edit_summaries.append({
            "edit": edit_name,
            "child_hash": child.harness_hash,
            "impact": impact.to_dict(),
            "affected_tasks": len(affected_tasks),
            "touched_run_fraction": touched_runs / len(parent_records),
            "line_touched_run_fraction": line_touched_runs / len(parent_records),
            "diverged_run_fraction": diverged_runs / len(parent_records),
            "untouched_runs": untouched,
            "untouched_identical": identical_untouched,
            "untouched_free": free_untouched,
            "child_cost_generated": summarize_cost([r for r in child_records if r["tokens_out"] > 0]),
            "child_accuracy": statistics.mean(r["outcome"] for r in child_records),
            "mean_paired_delta": delta,
        })
    summary = {
        "experiment": experiment,
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "n_tasks": len(tasks),
        "parent_hash": parent.harness_hash,
        "parent_cost": summarize_cost(parent_records),
        "parent_flakiness": flakiness(parent_records),
        "edits": edit_summaries,
    }
    out = ROOT / "results" / f"{experiment}.json"
    out.write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["parent_cost"], indent=1))
    print(json.dumps(summary["parent_flakiness"]))
    for item in edit_summaries:
        print(item["edit"], "affected", item["affected_tasks"], "touched_frac", round(item["touched_run_fraction"], 2), "line_touched", round(item["line_touched_run_fraction"], 2), "diverged", round(item["diverged_run_fraction"], 2), "untouched identical", item["untouched_identical"], "/", item["untouched_runs"], "free", item["untouched_free"], "delta", round(item["mean_paired_delta"], 3))


if __name__ == "__main__":
    main()
