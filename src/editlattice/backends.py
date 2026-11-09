"""One place to build the LLM client of an experiment script from its config.

Backend names starting with ``fake`` use a deterministic fake model (integration tests of the
scripts without a GPU); everything else is an OpenAI-compatible server (vLLM, mlx-lm).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from editlattice.llm.cache import ResponseCache
from editlattice.llm.client import LLMClient, OpenAICompatibleBackend
from editlattice.llm.fake import FakeBackend


def fake_responder(messages: list[dict[str, Any]], rng: random.Random) -> str:
    """Plausible-looking replies for any of our benchmarks (no quality implied)."""
    text = " ".join(str(message.get("content", "")) for message in messages)
    if "SQLite" in text or "sql" in text.lower():
        return rng.choice(["```sql\nSELECT 1\n```", "```sql\nSELECT COUNT(*) FROM sqlite_master\n```", "no idea"])
    if "competitive programmer" in text:
        return rng.choice(["```python\nprint(input())\n```", "```python\nprint(0)\n```"])
    return rng.choice(["Answer: 42", "Answer: 7", "I am not sure", "```python\nprint(6*7)\n```\nAnswer: 42"])


def make_client(config: dict[str, Any], cache_dir: Path) -> LLMClient:
    """Solver client from ``config['backend']``."""
    backend_config = config["backend"]
    if backend_config["name"].startswith("fake"):
        backend = FakeBackend(fake_responder, name=backend_config["name"])
    else:
        backend = OpenAICompatibleBackend(base_url=backend_config["base_url"], name=backend_config["name"])
    return LLMClient(backend=backend, cache=ResponseCache(cache_dir), model=backend_config["model"])
