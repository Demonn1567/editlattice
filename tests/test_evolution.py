"""End-to-end evolution loop with a fake LLM (also acting as proposer) and live gates."""

from __future__ import annotations

import json
import random
from functools import partial
from pathlib import Path
from typing import Any

from editlattice.evolution.live_pool import Executor
from editlattice.evolution.loop import LoopConfig, evolve
from editlattice.gates import baselines, ours
from editlattice.harness_runtime.runner import HarnessVersion, RecordStore
from editlattice.llm.cache import ResponseCache
from editlattice.llm.client import LLMClient
from editlattice.llm.fake import FakeBackend
from editlattice.proposers.proposers import SelfHarnessStyleProposer
from tests.test_lemma1 import PARENT, ToyBenchmark, make_tasks

EDIT = """<<<<<<< SEARCH
TEXT_PROMPT = "Answer yes or no. Reply 'ANSWER: <word>'."
=======
TEXT_PROMPT = "Answer yes or no, thinking twice. Reply 'ANSWER: <word>'."
>>>>>>> REPLACE"""


def responder(messages: list[dict[str, Any]], rng: random.Random) -> str:
    """Proposer requests get the edit; harness requests get noisy answers."""
    system = messages[0]["content"]
    if system.startswith("You are an expert engineer"):
        if "failing runs" in messages[-1]["content"]:
            return "The text route answers at random."
        return EDIT
    if system.startswith("Compute"):
        return f"ANSWER: {rng.randrange(5)}"
    better = "thinking twice" in system
    return "ANSWER: " + ("yes" if rng.random() < (0.9 if better else 0.5) else "no")


def run_loop(tmp_path: Path, gate, name: str) -> list[dict[str, Any]]:
    client = LLMClient(backend=FakeBackend(responder), model="fake", cache=ResponseCache(tmp_path / "cache"))
    executor = Executor(ToyBenchmark(), client, RecordStore(tmp_path / f"{name}.jsonl"), {}, 10)
    tasks = {}
    for task in make_tasks():
        if task["kind"] == "text":
            task = dict(task, gold="yes")
        tasks[task["task_id"]] = task
    seed = HarnessVersion.from_source(PARENT, tmp_path / "harnesses")
    proposer = SelfHarnessStyleProposer(client=client, params={}, stream_prefix=name)
    log_path = tmp_path / f"{name}_log.jsonl"
    evolve(LoopConfig(run_id=name, total_budget_runs=400, max_iterations=2, seed=1), executor, tasks, {}, seed, tmp_path / "harnesses", proposer, gate, log_path)
    entries = []
    for line in log_path.read_text().splitlines():
        entries.append(json.loads(line))
    return entries


def test_loop_logs_candidates_with_impact_and_decisions(tmp_path: Path) -> None:
    entries = run_loop(tmp_path, baselines.gate_greedy, "greedy")
    assert entries and entries[0]["proposal_valid"]
    assert entries[0]["impact"]["functions"] == ["solve_text"]
    assert "gate_task_runs" in entries[0]


def test_loop_with_coupled_gate(tmp_path: Path) -> None:
    gate = partial(ours.gate_impact_sequential, budget_runs=200)
    entries = run_loop(tmp_path, gate, "coupled_gate")
    first = entries[0]
    assert first["gate"] == "O4_coupled_gate"
    assert first["gate_details"]["affected"] == 6


def test_proposer_outage_is_retried_not_logged_as_invalid(tmp_path: Path, monkeypatch) -> None:
    """DEC-6: a proposer outage waits and retries with the same examples; the logged loop equals an uninterrupted one."""
    import editlattice.evolution.loop as loop
    from editlattice.llm.client import TransientBackendError

    monkeypatch.setattr(loop, "PROPOSER_OUTAGE_WAIT_SECONDS", 0.0)
    reference = run_loop(tmp_path / "plain", baselines.gate_greedy, "outage")
    original = SelfHarnessStyleProposer.propose
    failures = {"left": 3}

    def flaky(self, *args, **kwargs):
        if failures["left"]:
            failures["left"] -= 1
            args[5].random()  # consume randomness before failing, as a real proposer would
            raise TransientBackendError("tunnel down")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(SelfHarnessStyleProposer, "propose", flaky)
    interrupted = run_loop(tmp_path / "flaky", baselines.gate_greedy, "outage")
    assert failures["left"] == 0
    keep = ("iteration", "proposal_valid", "proposal_error", "parent_hash", "child_hash", "accepted")
    assert [{k: e.get(k) for k in keep} for e in interrupted] == [{k: e.get(k) for k in keep} for e in reference]
