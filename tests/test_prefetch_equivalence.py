"""Planned prefetching must not change any gate decision, ledger or detail."""

from __future__ import annotations

import json
import random
from functools import partial
from pathlib import Path

from editlattice.evolution.live_pool import Executor
from editlattice.evolution.loop import LoopConfig, evolve
from editlattice.gates import baselines, ours
from editlattice.harness_runtime.runner import HarnessVersion, RecordStore
from editlattice.llm.cache import ResponseCache
from editlattice.llm.client import LLMClient
from editlattice.llm.fake import FakeBackend
from editlattice.proposers.proposers import SelfHarnessStyleProposer
from tests.test_evolution import responder
from tests.test_lemma1 import PARENT, ToyBenchmark, make_tasks

GATES = {
    "B1": partial(baselines.gate_greedy),
    "B3": partial(baselines.gate_self_harness, repeats=2),
    "B4": partial(baselines.gate_avg_at_k, k=3, max_regression=2.0),
    "B5": partial(baselines.gate_mcnemar_full),
    "B8": partial(baselines.gate_two_gate, params=baselines.TwoGateParameters(tau=0.01, delta=0.2, m=2)),
    "B10a": partial(baselines.gate_historical_caching, epsilon=0.25),
    "B11": partial(baselines.gate_pace, budget_runs=60),
    "O3": partial(ours.gate_impact_sequential, budget_runs=60, crn=False),
    "O4": partial(ours.gate_impact_sequential, budget_runs=60, futility_margin=0.2),
}


def run(tmp_path: Path, name: str, gate, batch: int) -> list[dict]:
    root = tmp_path / f"{name}_{batch}"
    client = LLMClient(backend=FakeBackend(responder), model="fake", cache=ResponseCache(root / "cache"))
    executor = Executor(ToyBenchmark(), client, RecordStore(root / "records.jsonl"), {}, 10)
    tasks = {}
    for task in make_tasks():
        if task["kind"] == "text":
            task = dict(task, gold="yes")
        tasks[task["task_id"]] = task
    seed = HarnessVersion.from_source(PARENT, root / "harnesses")
    proposer = SelfHarnessStyleProposer(client=client, params={}, stream_prefix=name)
    log = root / "log.jsonl"
    config = LoopConfig(run_id=name, total_budget_runs=500, n_streams=8, k_char=4, max_iterations=1, seed=3, prefetch_batch=batch)
    evolve(config, executor, tasks, {}, seed, root / "harnesses", proposer, gate, log)
    entries = []
    for line in log.read_text().splitlines():
        entry = json.loads(line)
        entries.append({key: entry.get(key) for key in ("accepted", "gate", "gate_task_runs", "gate_nominal_tokens", "gate_details")})
    return entries


def test_prefetch_does_not_change_decisions(tmp_path: Path) -> None:
    for name, gate in GATES.items():
        sequential = run(tmp_path, name, gate, 1)
        prefetched = run(tmp_path, name, gate, 24)
        assert sequential == prefetched, (name, sequential, prefetched)
        assert sequential and sequential[0]["gate"] is not None, name
