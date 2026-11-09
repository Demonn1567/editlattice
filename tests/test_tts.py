"""TTS harness: builds from the seed, takes k independent samples, votes."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from editlattice.harness_runtime.runner import HarnessVersion, RecordStore, RunRequest, run_many
from editlattice.llm.cache import ResponseCache
from editlattice.llm.client import LLMClient
from editlattice.llm.fake import FakeBackend
from editlattice.proposers.edit_format import validate_harness
from editlattice.seed_harnesses.tts import make_tts_harness
from tests.test_lemma1 import PARENT, ToyBenchmark, make_tasks

SEEDS = Path(__file__).resolve().parents[1] / "src" / "editlattice" / "seed_harnesses"


def test_tts_sources_are_valid_harnesses() -> None:
    for name, benchmark in (("bird_react.py", "bird"), ("lcb_repair.py", "lcb")):
        validate_harness(make_tts_harness((SEEDS / name).read_text(), 4, benchmark))


def test_tts_samples_are_independent_draws(tmp_path: Path) -> None:
    replies: list[str] = []

    def responder(messages: list[dict[str, Any]], rng: random.Random) -> str:
        reply = f"ANSWER: {rng.randrange(1000)}"
        replies.append(reply)
        return reply

    toy_tts = PARENT.replace("\ndef solve(task, llm, env):", "\ndef solve_once(task, llm, env):") + "\n\ndef solve(task, llm, env):\n    answers = [solve_once(task, llm, env)['answer'] for _ in range(4)]\n    return {'answer': '|'.join(str(a) for a in answers)}\n"
    version = HarnessVersion.from_source(toy_tts, tmp_path / "h")
    client = LLMClient(backend=FakeBackend(responder), model="fake", cache=ResponseCache(tmp_path / "c"))
    record = run_many(ToyBenchmark(), [RunRequest(make_tasks()[1], version, 0, True)], client, {}, RecordStore(tmp_path / "r.jsonl"), 1, 20)[0]
    assert len(set(record["answer"].split("|"))) > 1
