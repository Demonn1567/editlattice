"""Common-random-numbers (CRN) replay cache for LLM responses.

A response is stored under a key derived from the full request (model, sampling
parameters, messages), the replicate id of the run, and the occurrence index of that
exact request within the run.

Why the occurrence index: a harness may legitimately send the very same request twice
in one run (for example a retry after a failed parse). Under fresh sampling the two
calls are independent draws. If the cache key ignored the occurrence index, the second
call would replay the first response, which changes the harness's outcome distribution
(a violation of assumption A2). With the index, the k-th identical request in a run is
its own independent draw, and a parent and a child harness that issue the same request
sequence still receive identical responses until their executions diverge.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    """Serialize a value to JSON with sorted keys and no insignificant whitespace."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def request_fingerprint(model: str, params: dict[str, Any], messages: list[dict[str, Any]]) -> str:
    """Return a SHA-256 hex digest identifying one LLM request.

    The replicate id and occurrence index are deliberately not part of the fingerprint;
    they are combined with it in :func:`cache_key`.
    """
    payload = {"model": model, "params": params, "messages": messages}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def cache_key(fingerprint: str, replicate_id: str, occurrence: int) -> str:
    """Combine a request fingerprint with the replicate id and occurrence index."""
    material = f"{fingerprint}|{replicate_id}|{occurrence}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def seed_from_key(key: str) -> int:
    """Derive a deterministic 31-bit sampling seed from a cache key.

    Backends that honour a per-request seed (vLLM) then make cache misses reproducible.
    """
    return int(key[:8], 16) & 0x7FFFFFFF


@dataclass
class CachedResponse:
    """One stored LLM response plus the accounting recorded when it was generated."""

    content: str
    tokens_in: int
    tokens_out: int
    backend_seconds: float
    backend: str
    model: str
    created_utc: str
    finish_reason: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return {
            "content": self.content,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "backend_seconds": self.backend_seconds,
            "backend": self.backend,
            "model": self.model,
            "created_utc": self.created_utc,
            "finish_reason": self.finish_reason,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "CachedResponse":
        """Build an instance from a dictionary produced by :meth:`to_dict`."""
        return CachedResponse(
            content=data["content"],
            tokens_in=int(data["tokens_in"]),
            tokens_out=int(data["tokens_out"]),
            backend_seconds=float(data["backend_seconds"]),
            backend=data["backend"],
            model=data["model"],
            created_utc=data["created_utc"],
            finish_reason=data.get("finish_reason", ""),
        )


class ResponseCache:
    """Disk-backed key/value store; one JSON file per response, sharded by key prefix.

    Writes are atomic (temporary file + rename), so a crash never leaves a partial entry
    and concurrent writers of the same key are harmless (last rename wins, and both
    writers hold a valid response for that key).
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path_for(self, key: str) -> Path:
        """Return the file path that stores ``key``."""
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> CachedResponse | None:
        """Return the stored response for ``key`` or ``None`` if absent."""
        path = self._path_for(key)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            return CachedResponse.from_dict(json.load(handle))

    def put_if_absent(self, key: str, response: CachedResponse) -> CachedResponse:
        """Store ``response`` unless ``key`` is already stored; return the stored response (DEC-13).

        First writer wins, atomically (a hard link fails if the file exists), so every consumer of a
        key sees the same response even if two processes generated it concurrently: replay coupling
        requires exactly one response per (request, stream, occurrence).
        """
        path = self._path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(response.to_dict(), handle, ensure_ascii=False)
        try:
            os.link(temp_name, path)
            return response
        except FileExistsError:
            with path.open("r", encoding="utf-8") as handle:
                return CachedResponse.from_dict(json.load(handle))
        finally:
            os.unlink(temp_name)

    def put(self, key: str, response: CachedResponse) -> None:
        """Store ``response`` under ``key`` atomically."""
        path = self._path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(response.to_dict(), handle, ensure_ascii=False)
        os.replace(temp_name, path)
