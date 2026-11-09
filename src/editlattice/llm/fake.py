"""A deterministic stand-in for an LLM, used by unit tests and simulations.

The reply is a pseudo-random function of the request and the per-request seed, so it
behaves like temperature sampling (different seeds give different replies) while being
fully reproducible. A configurable ``responder`` maps (messages, rng) to text.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any, Callable

from editlattice.llm.cache import canonical_json
from editlattice.llm.client import GenerationResult

Responder = Callable[[list[dict[str, Any]], random.Random], str]


def default_responder(messages: list[dict[str, Any]], rng: random.Random) -> str:
    """Return one of a few canned answers, chosen at random."""
    options = ["ANSWER: yes", "ANSWER: no", "ANSWER: maybe"]
    return options[rng.randrange(len(options))]


class FakeBackend:
    """Implements the ``Backend`` protocol without any model."""

    def __init__(self, responder: Responder = default_responder, name: str = "fake", noise_from_seed_only: bool = False) -> None:
        self.responder = responder
        self.name = name
        self.calls = 0
        # Like a real sampler with a per-request seed: the random noise depends only on the
        # seed, and the request only shapes the distribution the noise is pushed through.
        self.noise_from_seed_only = noise_from_seed_only

    def generate(self, model: str, messages: list[dict[str, Any]], params: dict[str, Any], seed: int) -> GenerationResult:
        """Produce a reproducible pseudo-random reply."""
        self.calls += 1
        if self.noise_from_seed_only:
            rng = random.Random(seed)
        else:
            material = f"{seed}|{canonical_json(messages)}|{canonical_json(params)}"
            rng = random.Random(int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:16], 16))
        content = self.responder(messages, rng)
        tokens_in = sum(len(str(message.get("content", "")).split()) for message in messages)
        return GenerationResult(content=content, tokens_in=tokens_in, tokens_out=len(content.split()), finish_reason="stop")


def react_sql_responder(messages: list[dict[str, Any]], rng: random.Random) -> str:
    """Speaks the BIRD seed harness protocol: a few random tool calls, then a submission.

    Used only for plumbing tests of the ground-truth pipeline (no model quality implied).
    """
    if str(messages[0].get("content", "")).startswith("You are an expert engineer"):
        if "failing runs" in str(messages[-1].get("content", "")):
            return "The agent submits before checking values."
        return "<<<<<<< SEARCH\nMAX_ROWS_SHOWN = 10\n=======\nMAX_ROWS_SHOWN = 12\n>>>>>>> REPLACE"
    observations = 0
    for message in messages:
        if message.get("role") == "user" and str(message.get("content", "")).startswith("Observation"):
            observations += 1
    roll = rng.random()
    if roll < 0.1:
        return "I am not sure what to do."
    if observations < 1 + int(roll * 3):
        if roll < 0.5:
            return "Thought: inspect.\nAction: run_sql\nAction Input: SELECT name FROM sqlite_master WHERE type='table' LIMIT 3"
        return "Thought: look for empties.\nAction: run_sql\nAction Input: SELECT name FROM sqlite_master WHERE name = 'nothing_here'"
    if roll < 0.6:
        return "Thought: done.\nAction: submit\nAction Input: SELECT COUNT(*) FROM sqlite_master"
    return "Thought: done.\nAction: submit\nAction Input: SELECT 1"
