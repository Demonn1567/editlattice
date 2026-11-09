"""Backend failures are infrastructure faults, never recorded as failed task attempts (D-025)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from editlattice.harness_runtime import runner
from editlattice.harness_runtime.runner import HarnessVersion, InfrastructureFailure, RecordStore, RunRequest, run_many, run_one
from editlattice.llm.cache import ResponseCache
from editlattice.llm.client import GenerationResult, LLMClient, TransientBackendError
from tests.test_lemma1 import ToyBenchmark, make_tasks

SWALLOWING = '''
def solve(task, llm, env):
    try:
        reply = llm.chat([{"role": "user", "content": task["question"]}])
    except Exception:
        return {"answer": None}
    return {"answer": reply.split("ANSWER:")[-1].strip()}
'''


class DownThenUp:
    """Fails the first ``failures`` calls, then answers."""

    name = "down-then-up"

    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0

    def generate(self, model: str, messages: list[dict[str, Any]], params: dict[str, Any], seed: int) -> GenerationResult:
        self.calls += 1
        if self.calls <= self.failures:
            raise TransientBackendError("timeout")
        return GenerationResult(content="ANSWER: yes", tokens_in=3, tokens_out=3, finish_reason="stop")


def make_version(tmp_path: Path) -> HarnessVersion:
    path = tmp_path / "harness.py"
    path.write_text(SWALLOWING)
    return HarnessVersion.from_file(path, tmp_path / "store")


def test_swallowed_backend_failure_is_not_a_task_outcome(tmp_path: Path) -> None:
    client = LLMClient(DownThenUp(10**6), "fake", ResponseCache(tmp_path / "cache"), max_retries=0, base_backoff_seconds=0.0)
    task = make_tasks()[1]
    with pytest.raises(InfrastructureFailure):
        run_one(ToyBenchmark(), task, make_version(tmp_path), 0, client, {}, True, 5)


def test_run_many_retries_until_the_backend_recovers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "INFRA_PAUSE_SECONDS", 0.0)
    backend = DownThenUp(2)
    client = LLMClient(backend, "fake", ResponseCache(tmp_path / "cache"), max_retries=0, base_backoff_seconds=0.0)
    store = RecordStore(tmp_path / "records.jsonl")
    request = RunRequest(make_tasks()[1], make_version(tmp_path), 0, True)
    records = run_many(ToyBenchmark(), [request], client, {}, store, 1, 5)
    assert len(records) == 1 and records[0]["error"] == "" and records[0]["answer"] == "yes"
    assert len(store.records) == 1


def test_resumed_store_reruns_backend_failures(tmp_path: Path) -> None:
    """A backend-failure record does not count as a finished run when a job resumes."""
    path = tmp_path / "records.jsonl"
    base = {"task_id": "t1", "harness_hash": "h", "replicate_id": 0, "crn": True, "namespace": "gt:h", "outcome": 0, "score": 0.0}
    path.write_text(json.dumps(dict(base, error="TransientBackendError: timed out")) + "\n" + json.dumps(dict(base, task_id="t2", error="")) + "\n")
    store = RecordStore(path)
    assert not store.has(runner.record_key("t1", "h", 0, True, "gt:h"))
    assert store.has(runner.record_key("t2", "h", 0, True, "gt:h"))


def test_rejected_request_is_an_infrastructure_failure(tmp_path) -> None:
    """DEC-11: a request the server rejects (HTTP 4xx, e.g. too long for its context) stops the run
    instead of being scored as a failed answer."""
    import httpx
    import pytest
    from editlattice.harness_runtime.runner import HarnessVersion, InfrastructureFailure, run_one
    from editlattice.llm.cache import ResponseCache
    from editlattice.llm.client import LLMClient

    class Rejecting:
        name = "rejecting"

        def generate(self, model, messages, params, seed):
            request = httpx.Request("POST", "http://localhost/v1/chat/completions")
            raise httpx.HTTPStatusError("400 Bad Request", request=request, response=httpx.Response(400, request=request))

    class Bench:
        name = "b"

        def public_view(self, task):
            return task

        def make_env(self, task):
            return None

        def close_env(self, env):
            return None

        def score(self, task, answer):
            return 0, 0.0

    source = 'def solve(task, llm, env):\n    try:\n        return {"answer": llm.chat([{"role": "user", "content": "q"}])}\n    except Exception:\n        return {"answer": None}\n'
    client = LLMClient(backend=Rejecting(), cache=ResponseCache(tmp_path / "c"), model="m", max_retries=0)
    with pytest.raises(InfrastructureFailure):
        run_one(Bench(), {"task_id": "t"}, HarnessVersion.from_source(source, tmp_path / "s"), 0, client, {}, True, 3)


def test_scores_are_shared_across_processes(tmp_path, monkeypatch) -> None:
    """DEC-13: the first score of an answer is stored on disk; a later process (fresh memory cache)
    reuses it even if the scorer would now answer differently (a program near its time limit)."""
    import editlattice.harness_runtime.runner as runner

    monkeypatch.setenv("EDITLATTICE_SCORE_CACHE", str(tmp_path / "scores"))
    verdicts = iter([(1, 1.0), (0, 0.0)])

    class Flaky:
        name = "flaky"

        def score(self, task, answer):
            return next(verdicts)

    first = runner.cached_score(Flaky(), {"task_id": "t"}, "program")
    runner._SCORE_CACHE.clear()  # a new process
    second = runner.cached_score(Flaky(), {"task_id": "t"}, "program")
    assert first == second == (1, 1.0)
