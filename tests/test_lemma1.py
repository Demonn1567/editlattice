"""Lemma 1 (coupling) on a toy harness with a fake LLM.

Under CRN replay, a child run on a (task, replicate) whose parent run never touched the
impact set must be identical to the parent run: same answer, same outcome, same coverage
(in function names), same number of LLM calls, and it must cost nothing (all responses
are cache hits). We also check the converse sanity property that some touched runs do
differ, so the test cannot pass vacuously.
"""

from __future__ import annotations

import random
import textwrap
from pathlib import Path
from typing import Any

from editlattice.harness_runtime.runner import HarnessVersion, RecordStore, RunRequest, run_many
from editlattice.impact.diff import compute_impact
from editlattice.llm.cache import ResponseCache
from editlattice.llm.client import LLMClient
from editlattice.llm.fake import FakeBackend

PARENT = textwrap.dedent(
    '''
    """Toy two-route harness."""

    MATH_PROMPT = "Compute the value. Reply 'ANSWER: <n>'."
    TEXT_PROMPT = "Answer yes or no. Reply 'ANSWER: <word>'."


    def parse(reply):
        if "ANSWER:" not in reply:
            return None
        return reply.split("ANSWER:", 1)[1].strip()


    def solve_math(task, llm):
        reply = llm.chat([{"role": "system", "content": MATH_PROMPT}, {"role": "user", "content": task["question"]}])
        answer = parse(reply)
        if answer is None:
            reply = llm.chat([{"role": "system", "content": MATH_PROMPT}, {"role": "user", "content": task["question"]}])
            answer = parse(reply)
        return answer


    def solve_text(task, llm):
        reply = llm.chat([{"role": "system", "content": TEXT_PROMPT}, {"role": "user", "content": task["question"]}])
        return parse(reply)


    def solve(task, llm, env):
        if task["kind"] == "math":
            return {"answer": solve_math(task, llm)}
        return {"answer": solve_text(task, llm)}
    '''
).lstrip("\n")

# The edit touches only the text route: a different prompt and a retry.
CHILD = PARENT.replace(
    'TEXT_PROMPT = "Answer yes or no. Reply \'ANSWER: <word>\'."',
    'TEXT_PROMPT = "Answer yes or no carefully. Reply \'ANSWER: <word>\'."',
)


def responder(messages: list[dict[str, Any]], rng: random.Random) -> str:
    """Math tasks sometimes get malformed replies (exercising the retry branch)."""
    system = messages[0]["content"]
    if system.startswith("Compute"):
        if rng.random() < 0.3:
            return "I think it is " + str(rng.randrange(5))
        return f"ANSWER: {rng.randrange(5)}"
    return "ANSWER: " + ("yes" if rng.random() < 0.5 else "no")


class ToyBenchmark:
    """Benchmark stub: the gold answer is fixed per task."""

    name = "toy"

    def public_view(self, task: dict[str, Any]) -> dict[str, Any]:
        view = dict(task)
        del view["gold"]
        return view

    def make_env(self, task: dict[str, Any]) -> None:
        return None

    def close_env(self, env: Any) -> None:
        return None

    def score(self, task: dict[str, Any], answer: Any) -> tuple[int, float]:
        correct = int(answer == task["gold"])
        return correct, float(correct)


def make_tasks() -> list[dict[str, Any]]:
    tasks = []
    for index in range(6):
        tasks.append({"task_id": f"m{index}", "kind": "math", "question": f"What is {index} mod 5?", "gold": str(index % 5)})
        tasks.append({"task_id": f"t{index}", "kind": "text", "question": f"Is {index} even?", "gold": "yes" if index % 2 == 0 else "no"})
    return tasks


def test_lemma1_untouched_runs_are_identical_and_free(tmp_path: Path) -> None:
    store_dir = tmp_path / "harnesses"
    parent = HarnessVersion.from_source(PARENT, store_dir)
    child = HarnessVersion.from_source(CHILD, store_dir)
    impact = compute_impact(parent.source, child.source)
    assert not impact.is_global
    assert impact.functions == {"solve_text"}

    client = LLMClient(backend=FakeBackend(responder), model="fake-model", cache=ResponseCache(tmp_path / "cache"))
    store = RecordStore(tmp_path / "records.jsonl")
    tasks = make_tasks()
    replicates = range(8)
    parent_requests = [RunRequest(task, parent, r, True) for task in tasks for r in replicates]
    child_requests = [RunRequest(task, child, r, True) for task in tasks for r in replicates]
    parent_records = run_many(ToyBenchmark(), parent_requests, client, {"temperature": 0.7}, store, 4, 10)
    child_records = run_many(ToyBenchmark(), child_requests, client, {"temperature": 0.7}, store, 4, 10)

    untouched = 0
    touched_and_different = 0
    for parent_record, child_record in zip(parent_records, child_records):
        assert parent_record["task_id"] == child_record["task_id"]
        assert parent_record["replicate_id"] == child_record["replicate_id"]
        assert parent_record["error"] == "" and child_record["error"] == ""
        touched = bool(set(parent_record["covered_functions"]) & impact.functions)
        if not touched:
            untouched += 1
            assert child_record["answer"] == parent_record["answer"]
            assert child_record["outcome"] == parent_record["outcome"]
            assert child_record["covered_functions"] == parent_record["covered_functions"]
            assert child_record["n_llm_calls"] == parent_record["n_llm_calls"]
            assert child_record["n_cache_hits"] == child_record["n_llm_calls"]
            assert child_record["tokens_out"] == 0
        elif child_record["answer"] != parent_record["answer"]:
            touched_and_different += 1
    assert untouched == 6 * 8
    assert touched_and_different > 0


def test_independent_streams_do_not_share_responses(tmp_path: Path) -> None:
    store_dir = tmp_path / "harnesses"
    parent = HarnessVersion.from_source(PARENT, store_dir)
    child = HarnessVersion.from_source(CHILD, store_dir)
    client = LLMClient(backend=FakeBackend(responder), model="fake-model", cache=ResponseCache(tmp_path / "cache"))
    store = RecordStore(tmp_path / "records.jsonl")
    tasks = make_tasks()[:2]
    run_many(ToyBenchmark(), [RunRequest(task, parent, 0, False) for task in tasks], client, {}, store, 2, 10)
    child_records = run_many(ToyBenchmark(), [RunRequest(task, child, 0, False) for task in tasks], client, {}, store, 2, 10)
    for record in child_records:
        assert record["n_cache_hits"] == 0


def test_runner_is_idempotent(tmp_path: Path) -> None:
    store_dir = tmp_path / "harnesses"
    parent = HarnessVersion.from_source(PARENT, store_dir)
    backend = FakeBackend(responder)
    client = LLMClient(backend=backend, model="fake-model", cache=ResponseCache(tmp_path / "cache"))
    requests = [RunRequest(task, parent, 0, True) for task in make_tasks()]
    store = RecordStore(tmp_path / "records.jsonl")
    run_many(ToyBenchmark(), requests, client, {}, store, 3, 10)
    calls_after_first = backend.calls
    reopened = RecordStore(tmp_path / "records.jsonl")
    run_many(ToyBenchmark(), requests, client, {}, reopened, 3, 10)
    assert backend.calls == calls_after_first
    assert len(reopened.records) == len(requests)


def test_same_harness_in_two_stores_keeps_coverage(tmp_path: Path) -> None:
    """Regression test: compiled-code cache must not leak co_filename across stores."""
    client = LLMClient(backend=FakeBackend(responder), model="fake-model", cache=ResponseCache(tmp_path / "cache"))
    task = make_tasks()[1]
    for store_name in ("store_a", "store_b"):
        version = HarnessVersion.from_source(PARENT, tmp_path / store_name)
        records = run_many(ToyBenchmark(), [RunRequest(task, version, 0, True)], client, {}, RecordStore(tmp_path / f"{store_name}.jsonl"), 1, 10)
        assert "solve_text" in records[0]["covered_functions"], store_name


def test_record_store_skips_a_partial_last_line(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text('{"task_id": "a", "harness_hash": "h", "replicate_id": 0, "crn": true, "outcome": 1, "score": 1.0, "answer": "x"}\n{"task_id": "b", "harn')
    store = RecordStore(path)
    assert len(store.records) == 1 and store.skipped_partial_lines == 1
