"""Position-seeded (coupled) sampling: correlated responses under a prompt edit, exact marginals."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from editlattice.llm.cache import ResponseCache
from editlattice.llm.client import LLMClient, RunLLM
from editlattice.llm.fake import FakeBackend


def responder(messages: list[dict[str, Any]], rng: random.Random) -> str:
    """Inverse-CDF sampling: the prompt shifts the answer distribution slightly."""
    shift = 0.05 if "carefully" in messages[0]["content"] else 0.0
    u = rng.random()
    return "ANSWER: yes" if u < 0.5 + shift else "ANSWER: no"


def agreement(tmp_path: Path, mode: str, n: int = 400) -> tuple[float, float, float]:
    client = LLMClient(FakeBackend(responder, noise_from_seed_only=True), "fake", ResponseCache(tmp_path / mode), seed_mode=mode)
    same = 0
    parent_yes = 0
    child_yes = 0
    for stream in range(n):
        parent = RunLLM(client, f"crn:t:{stream}", {}).chat([{"role": "system", "content": "Answer."}, {"role": "user", "content": "q"}])
        child = RunLLM(client, f"crn:t:{stream}", {}).chat([{"role": "system", "content": "Answer carefully."}, {"role": "user", "content": "q"}])
        same += 1 if parent == child else 0
        parent_yes += 1 if parent.endswith("yes") else 0
        child_yes += 1 if child.endswith("yes") else 0
    return same / n, parent_yes / n, child_yes / n


def test_position_seeding_couples_responses_across_a_prompt_edit(tmp_path: Path) -> None:
    coupled, parent_rate, child_rate = agreement(tmp_path, "position")
    uncoupled, _, _ = agreement(tmp_path, "request")
    assert coupled >= 0.9
    assert uncoupled < 0.7
    assert abs(parent_rate - 0.5) < 0.08 and abs(child_rate - 0.55) < 0.08


def test_identical_requests_still_replay_in_position_mode(tmp_path: Path) -> None:
    backend = FakeBackend(responder, noise_from_seed_only=True)
    client = LLMClient(backend, "fake", ResponseCache(tmp_path / "c"), seed_mode="position")
    messages = [{"role": "system", "content": "Answer."}, {"role": "user", "content": "q"}]
    first = RunLLM(client, "crn:t:0", {}).chat(messages)
    calls = backend.calls
    second = RunLLM(client, "crn:t:0", {}).chat(messages)
    assert first == second and backend.calls == calls


def test_empty_serving_fault_is_retried_not_cached(tmp_path: Path) -> None:
    """Regression (RQ2 finding D-024): immediate-EOS empty replies are retried, never cached."""
    from editlattice.llm.client import GenerationResult

    class Flaky:
        name = "flaky"

        def __init__(self) -> None:
            self.calls = 0

        def generate(self, model, messages, params, seed):
            self.calls += 1
            if self.calls <= 2:
                return GenerationResult("", 5, 1, "stop")
            return GenerationResult("ANSWER: yes", 5, 3, "stop")

    backend = Flaky()
    client = LLMClient(backend, "fake", ResponseCache(tmp_path / "c"), base_backoff_seconds=0.0)
    client.empty_retries = 4
    reply = RunLLM(client, "crn:t:0", {}).chat([{"role": "user", "content": "q"}])
    assert reply == "ANSWER: yes" and backend.calls == 3 and client.empty_retries_total == 2


def test_concurrent_generation_of_one_key_yields_one_response(tmp_path) -> None:
    """DEC-13: two clients that miss the cache for the same key at the same time both end up using the
    response stored first, so replay coupling holds even if duplicate runs race."""
    import threading
    from editlattice.llm.cache import ResponseCache
    from editlattice.llm.client import GenerationResult, LLMClient

    gate = threading.Barrier(2)

    class Slow:
        name = "slow"

        def __init__(self, text):
            self.text = text

        def generate(self, model, messages, params, seed):
            gate.wait()  # both clients are past their cache lookup before either stores
            return GenerationResult(content=self.text, tokens_in=1, tokens_out=1, finish_reason="stop")

    cache = ResponseCache(tmp_path / "cache")
    first = LLMClient(backend=Slow("first"), cache=cache, model="m")
    second = LLMClient(backend=Slow("second"), cache=cache, model="m")
    results = {}

    def ask(name, client):
        response, _ = client.complete([{"role": "user", "content": "q"}], {}, "stream", 0)
        results[name] = response.content

    threads = [threading.Thread(target=ask, args=(n, c)) for n, c in (("a", first), ("b", second))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    stored, _ = first.complete([{"role": "user", "content": "q"}], {}, "stream", 0)
    assert results["a"] == results["b"] == stored.content
