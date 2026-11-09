"""OpenAI-compatible LLM client with a CRN replay cache and cost accounting.

The same client serves every backend (mlx-lm server on the laptop, vLLM on a cloud GPU,
or a fake backend in unit tests) because all of them speak the OpenAI chat-completions
protocol or implement :class:`Backend`.

Accounting rule: tokens and backend seconds are charged only for cache misses. A cache
hit is a replayed response and costs nothing, which is exactly the saving that CRN replay
is meant to deliver.
"""

from __future__ import annotations

import hashlib
import os
import random
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

import httpx

from editlattice.llm.cache import CachedResponse, ResponseCache, cache_key, request_fingerprint, seed_from_key

TRACE_CHARS = 500
ALLOWED_PARAMS = ("temperature", "top_p", "max_tokens", "stop", "chat_template_kwargs")


@dataclass
class GenerationResult:
    """What a backend returns for one request."""

    content: str
    tokens_in: int
    tokens_out: int
    finish_reason: str


class Backend(Protocol):
    """Anything that can turn a chat request into a :class:`GenerationResult`."""

    name: str

    def generate(self, model: str, messages: list[dict[str, Any]], params: dict[str, Any], seed: int) -> GenerationResult:
        """Generate one completion."""
        ...


class TransientBackendError(RuntimeError):
    """A failure worth retrying (connection error, HTTP 429 or 5xx)."""


class OpenAICompatibleBackend:
    """Talks to ``{base_url}/chat/completions`` (mlx-lm server, vLLM, Ollama ...).

    ``base_url`` may list several comma-separated replicas of the same model (data
    parallelism); requests are spread round-robin. Responses do not depend on the
    replica because each request carries its own seed.

    ``max_in_flight`` (default: environment variable EDITLATTICE_MAX_IN_FLIGHT, 0 = no limit)
    caps concurrent requests. Without it, parallel loops queued ~640 conversations on 192
    server slots and requests hit the HTTP timeout (D-025); excess requests now wait here,
    where no timeout runs.
    """

    def __init__(self, base_url: str, name: str, api_key: str = "none", timeout_seconds: float = 1800.0, max_in_flight: int | None = None) -> None:
        self.base_urls: list[str] = []
        for part in base_url.split(","):
            if part.strip():
                self.base_urls.append(part.strip().rstrip("/"))
        self.base_url = self.base_urls[0]
        self.name = name
        self._next = 0
        self._next_lock = threading.Lock()
        if max_in_flight is None:
            max_in_flight = int(os.environ.get("EDITLATTICE_MAX_IN_FLIGHT", "0") or 0)
        self.max_in_flight = max_in_flight
        self._in_flight = threading.BoundedSemaphore(max_in_flight) if max_in_flight > 0 else None
        self._http = httpx.Client(
            timeout=timeout_seconds,
            headers={"Authorization": f"Bearer {api_key}"},
            limits=httpx.Limits(max_connections=256, max_keepalive_connections=128),
        )

    def _pick_url(self) -> str:
        """Next replica, round-robin."""
        with self._next_lock:
            url = self.base_urls[self._next % len(self.base_urls)]
            self._next += 1
        return url

    def generate(self, model: str, messages: list[dict[str, Any]], params: dict[str, Any], seed: int) -> GenerationResult:
        """POST one chat-completions request and parse the reply."""
        payload: dict[str, Any] = {"model": model, "messages": messages, "seed": seed, "stream": False}
        payload.update(params)
        if self._in_flight is not None:
            self._in_flight.acquire()
        try:
            response = self._http.post(f"{self._pick_url()}/chat/completions", json=payload)
        except httpx.TransportError as error:
            raise TransientBackendError(str(error)) from error
        finally:
            if self._in_flight is not None:
                self._in_flight.release()
        if response.status_code == 429 or response.status_code >= 500:
            raise TransientBackendError(f"HTTP {response.status_code}: {response.text[:300]}")
        response.raise_for_status()
        body = response.json()
        choice = body["choices"][0]
        usage = body.get("usage") or {}
        return GenerationResult(
            content=choice["message"].get("content") or "",
            tokens_in=int(usage.get("prompt_tokens", 0)),
            tokens_out=int(usage.get("completion_tokens", 0)),
            finish_reason=str(choice.get("finish_reason", "")),
        )


@dataclass
class LLMClient:
    """Cache-first client. Thread-safe; share one instance across worker threads."""

    backend: Backend
    model: str
    cache: ResponseCache
    max_retries: int = 8
    base_backoff_seconds: float = 2.0
    # "request": seed = H(request, stream, occurrence)  - response-level CRN only.
    # "position": seed = H(stream, call index)            - additionally couples the token
    #   sampling noise of parent and child at the same call position (coupled generation,
    #   Benz et al. arXiv 2502.01754), so prompt edits the model barely reacts to still give
    #   correlated responses. Marginals are unchanged: seeds are independent across streams
    #   and call positions.
    seed_mode: str = "request"
    empty_retries: int = 4
    empty_retries_total: int = 0
    _stats_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def complete(
        self,
        messages: list[dict[str, Any]],
        params: dict[str, Any],
        replicate_id: str,
        occurrence: int,
        call_index: int = 0,
    ) -> tuple[CachedResponse, bool]:
        """Return ``(response, cache_hit)`` for one request.

        ``replicate_id`` selects the random stream: parent and child harness runs that
        share a replicate id share responses for identical requests (CRN); runs with
        different replicate ids get independent samples.
        """
        clean_params = filter_params(params)
        fingerprint = request_fingerprint(self.model, clean_params, messages)
        key = cache_key(fingerprint, replicate_id, occurrence)
        cached = self.cache.get(key)
        if cached is not None:
            return cached, True
        if self.seed_mode == "position":
            seed = seed_from_key(hashlib.sha256(f"position|{replicate_id}|{call_index}".encode("utf-8")).hexdigest())
        else:
            seed = seed_from_key(key)
        started = time.perf_counter()
        result = self._generate_with_retries(messages, clean_params, seed)
        elapsed = time.perf_counter() - started
        response = CachedResponse(
            content=result.content,
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            backend_seconds=elapsed,
            backend=self.backend.name,
            model=self.model,
            created_utc=datetime.now(timezone.utc).isoformat(),
            finish_reason=result.finish_reason,
        )
        # Another process may have stored this key meanwhile: use the stored response (DEC-13).
        stored = self.cache.put_if_absent(key, response)
        return stored, False

    def _generate_with_retries(self, messages: list[dict[str, Any]], params: dict[str, Any], seed: int) -> GenerationResult:
        """Call the backend, retrying transient failures and serving faults.

        Serving fault (RQ2 finding, D-024): under heavy load the server sometimes returned an
        immediate end-of-sequence (empty content, one output token) although the same
        request yields a normal reply under normal load (3 such replies in ~33k calls). Such
        a reply depends on server state, not on the request (an A2 violation), so it is retried
        and never cached; only after ``empty_retries`` attempts is it accepted as genuine.
        """
        attempt = 0
        empty_attempts = 0
        while True:
            try:
                result = self.backend.generate(self.model, messages, params, seed)
            except TransientBackendError:
                attempt += 1
                if attempt > self.max_retries:
                    raise
                delay = self.base_backoff_seconds * (2 ** (attempt - 1))
                time.sleep(min(delay, 120.0) * (0.5 + random.random()))
                continue
            if result.content.strip() == "" and result.tokens_out <= 1 and empty_attempts < self.empty_retries:
                empty_attempts += 1
                with self._stats_lock:
                    self.empty_retries_total += 1
                time.sleep(min(30.0, self.base_backoff_seconds * empty_attempts) * (0.5 + random.random()))
                continue
            return result


def filter_params(params: dict[str, Any]) -> dict[str, Any]:
    """Keep only the sampling parameters that define a request; drop ``None`` values."""
    kept: dict[str, Any] = {}
    for name in ALLOWED_PARAMS:
        if name in params and params[name] is not None:
            kept[name] = params[name]
    return kept


@dataclass
class RunUsage:
    """Per-run accounting. Only cache misses are charged."""

    n_llm_calls: int = 0
    n_cache_hits: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    backend_seconds: float = 0.0
    nominal_tokens_in: int = 0
    nominal_tokens_out: int = 0


@dataclass
class RunLLM:
    """The LLM handle a harness receives for exactly one (task, replicate) run.

    It tracks how many times each exact request has been issued in this run so that
    the occurrence index of the cache key is correct (see ``cache.py``).
    """

    client: LLMClient
    replicate_id: str
    default_params: dict[str, Any]
    usage: RunUsage = field(default_factory=RunUsage)
    keep_trace: bool = False
    trace: list[dict[str, str]] = field(default_factory=list)
    request_fingerprints: list[str] = field(default_factory=list)
    response_fingerprints: list[str] = field(default_factory=list)
    infra_failure: str = ""
    _occurrences: dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def chat(self, messages: list[dict[str, Any]], **overrides: Any) -> str:
        """Send ``messages`` and return the assistant text."""
        params = dict(self.default_params)
        params.update(overrides)
        clean_params = filter_params(params)
        fingerprint = request_fingerprint(self.client.model, clean_params, messages)
        with self._lock:
            occurrence = self._occurrences.get(fingerprint, 0)
            self._occurrences[fingerprint] = occurrence + 1
            self.request_fingerprints.append(fingerprint[:12])
        with self._lock:
            call_index = len(self.request_fingerprints) - 1
        try:
            response, hit = self.client.complete(messages, clean_params, self.replicate_id, occurrence, call_index)
        except TransientBackendError as error:
            # Remembered even if the harness swallows the exception: the run then measured the
            # inference server, not the harness, and must not be recorded (see runner.run_one).
            self.infra_failure = str(error)[:300] or "TransientBackendError"
            raise
        except httpx.HTTPStatusError as error:
            # The server rejected the request (e.g. longer than its context, DEC-11): also a fact
            # about the server, never a harness outcome.
            self.infra_failure = f"rejected by server: {str(error)[:280]}"
            raise
        with self._lock:
            self.response_fingerprints.append(hashlib.sha256(response.content.encode("utf-8")).hexdigest()[:12])
        if self.keep_trace:
            last_input = str(messages[-1].get("content", "")) if messages else ""
            self.trace.append({"input_tail": last_input[-TRACE_CHARS:], "reply": response.content[:TRACE_CHARS]})
        with self._lock:
            self.usage.n_llm_calls += 1
            self.usage.nominal_tokens_in += response.tokens_in
            self.usage.nominal_tokens_out += response.tokens_out
            if hit:
                self.usage.n_cache_hits += 1
            else:
                self.usage.tokens_in += response.tokens_in
                self.usage.tokens_out += response.tokens_out
                self.usage.backend_seconds += response.backend_seconds
        return response.content
