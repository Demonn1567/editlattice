"""Run harness versions on benchmark tasks with coverage, CRN replay and JSONL records.

Harness contract
----------------
A harness is one Python module (``harness.py``) that defines::

    def solve(task: dict, llm, env) -> dict      # returns {"answer": ...}

``llm.chat(messages, **params) -> str`` is the only way to call the model; ``env`` is the
benchmark environment (tool primitives). The harness must not start threads or
processes itself (they would escape the coverage tracer).

Isolation (assumption A3): every run executes a *fresh* instance of the harness module,
so module-level state cannot leak between runs, and every benchmark environment is
created per run.

Random streams: in CRN mode, parent and child runs of the same (task, replicate) share
the stream ``crn:{task_id}:{replicate}``; in independent mode each harness version has
its own stream ``ind:{harness_hash}:{task_id}:{replicate}``.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import traceback
import types
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

from editlattice.harness_runtime.coverage_tracer import CoverageTracer
from editlattice.llm.client import LLMClient, RunLLM

_SCORE_CACHE: dict[str, tuple[int, float]] = {}
_SCORE_LOCK = threading.Lock()


def cached_score(benchmark: "Benchmark", task: dict[str, Any], answer: Any) -> tuple[int, float]:
    """Score each distinct (benchmark, task, answer) once (part of assumption A1).

    Code-execution scorers with timeouts can return different outcomes for the same
    answer on reruns; caching makes the outcome a deterministic function of the answer.
    The cache is persisted next to the run records by :class:`RecordStore`.
    """
    material = json.dumps([benchmark.name, str(task["task_id"]), answer], sort_keys=True, default=str)
    key = hashlib.sha256(material.encode("utf-8")).hexdigest()
    with _SCORE_LOCK:
        hit = _SCORE_CACHE.get(key)
    if hit is not None:
        return hit
    # Shared across processes on disk (DEC-13), first writer wins: a borderline program near a
    # time limit gets the same verdict in every run, not only within one process.
    path = score_cache_dir() / key[:2] / f"{key}.json"
    if path.exists():
        stored = json.loads(path.read_text())
        result = (int(stored[0]), float(stored[1]))
    else:
        result = benchmark.score(task, answer)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.parent / f".{key}.{os.getpid()}.{threading.get_ident()}.tmp"
        temp.write_text(json.dumps([int(result[0]), float(result[1])]))
        try:
            os.link(temp, path)
        except FileExistsError:
            stored = json.loads(path.read_text())
            result = (int(stored[0]), float(stored[1]))
        finally:
            temp.unlink()
    with _SCORE_LOCK:
        _SCORE_CACHE.setdefault(key, result)
        return _SCORE_CACHE[key]


def score_cache_dir() -> Path:
    """On-disk score cache: EDITLATTICE_SCORE_CACHE, else <runs>/score_cache."""
    if os.environ.get("EDITLATTICE_SCORE_CACHE"):
        return Path(os.environ["EDITLATTICE_SCORE_CACHE"])
    runs = os.environ.get("EDITLATTICE_RUNS_DIR")
    return Path(runs) / "score_cache" if runs else Path(__file__).resolve().parents[3] / "runs" / "score_cache"

HARNESS_LABEL = "harness.py"
INFRA_ATTEMPTS = 4
INFRA_PAUSE_SECONDS = 60.0


class LLMCallBudgetExceeded(RuntimeError):
    """Raised when a run exceeds its maximum number of LLM calls."""


class Benchmark(Protocol):
    """What the runner needs from a benchmark."""

    name: str

    def public_view(self, task: dict[str, Any]) -> dict[str, Any]:
        """The part of a task a harness may see (no gold answers or hidden tests)."""
        ...

    def make_env(self, task: dict[str, Any]) -> Any:
        """Create a fresh environment for one run."""
        ...

    def close_env(self, env: Any) -> None:
        """Release the environment."""
        ...

    def score(self, task: dict[str, Any], answer: Any) -> tuple[int, float]:
        """Return (binary outcome, graded score in [0, 1])."""
        ...


@dataclass(frozen=True)
class HarnessVersion:
    """An immutable harness version identified by the hash of its source."""

    source: str
    path: str
    harness_hash: str

    @staticmethod
    def from_source(source: str, store_dir: str | Path) -> "HarnessVersion":
        """Write ``source`` to ``store_dir/<hash>/harness.py`` (if new) and return it."""
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
        directory = Path(store_dir) / digest
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / HARNESS_LABEL
        if not path.exists():
            # Atomic: concurrent processes share the store and may read this file.
            temporary = directory / f".{HARNESS_LABEL}.{os.getpid()}.{threading.get_ident()}.tmp"
            temporary.write_text(source, encoding="utf-8")
            os.replace(temporary, path)
        return HarnessVersion(source=source, path=str(path.resolve()), harness_hash=digest)

    @staticmethod
    def from_file(file_path: str | Path, store_dir: str | Path) -> "HarnessVersion":
        """Load a harness source file into the content-addressed store."""
        return HarnessVersion.from_source(Path(file_path).read_text(encoding="utf-8"), store_dir)


_COMPILED: dict[tuple[str, str], types.CodeType] = {}
_COMPILED_LOCK = threading.Lock()


def compiled_module_code(version: HarnessVersion) -> types.CodeType:
    """Compile a harness once per (hash, path).

    The path is part of the key because code objects carry it as ``co_filename`` and the
    coverage tracer filters on it: the same source stored at two paths must not share
    code objects, or coverage silently comes back empty.
    """
    key = (version.harness_hash, version.path)
    with _COMPILED_LOCK:
        code = _COMPILED.get(key)
        if code is None:
            code = compile(version.source, version.path, "exec")
            _COMPILED[key] = code
        return code


def fresh_module(version: HarnessVersion) -> types.ModuleType:
    """Execute the harness source into a brand-new module object."""
    module = types.ModuleType(f"harness_{version.harness_hash}")
    module.__file__ = version.path
    exec(compiled_module_code(version), module.__dict__)  # noqa: S102 - trusted harness code under study
    return module


class GuardedRunLLM(RunLLM):
    """RunLLM with a hard cap on the number of calls per run."""

    max_calls: int = 30

    def chat(self, messages: list[dict[str, Any]], **overrides: Any) -> str:
        """Refuse to exceed the per-run call budget."""
        if self.usage.n_llm_calls >= self.max_calls:
            raise LLMCallBudgetExceeded(f"more than {self.max_calls} LLM calls")
        return super().chat(messages, **overrides)


def stream_id(task_id: str, replicate: int, crn: bool, harness_hash: str, namespace: str = "") -> str:
    """Random-stream identifier passed to the response cache.

    ``namespace`` separates stream families (e.g. the characterization streams of each
    accepted harness, D-009); the empty namespace keeps the original ids.
    """
    prefix = f"{namespace}:" if namespace else ""
    if crn:
        return f"crn:{prefix}{task_id}:{replicate}"
    return f"ind:{prefix}{harness_hash}:{task_id}:{replicate}"


class InfrastructureFailure(RuntimeError):
    """The inference backend failed during a run; the run says nothing about the harness.

    Such runs were once recorded as failed task attempts (outcome 0). Under server overload
    this scored the corpus loops' seed harness at 0.07 instead of about 0.3 (D-025). They are
    now never recorded: the run is retried, and if the backend stays down the job stops.
    """


def run_one(
    benchmark: Benchmark,
    task: dict[str, Any],
    version: HarnessVersion,
    replicate: int,
    client: LLMClient,
    llm_params: dict[str, Any],
    crn: bool,
    max_llm_calls: int,
    keep_trace: bool = False,
    namespace: str = "",
) -> dict[str, Any]:
    """Execute one (task, harness, replicate) run and return its JSONL record."""
    task_id = str(task["task_id"])
    stream = stream_id(task_id, replicate, crn, version.harness_hash, namespace)
    llm = GuardedRunLLM(client=client, replicate_id=stream, default_params=dict(llm_params))
    llm.max_calls = max_llm_calls
    llm.keep_trace = keep_trace
    tracer = CoverageTracer({version.path: HARNESS_LABEL})
    answer: Any = None
    error = ""
    started = time.perf_counter()
    env = None
    try:
        env = benchmark.make_env(task)
        module = fresh_module(version)
        with tracer:
            result = module.solve(benchmark.public_view(task), llm, env)
        if isinstance(result, dict):
            answer = result.get("answer")
        else:
            error = f"harness returned {type(result).__name__}, expected dict"
    except Exception as exc:  # noqa: BLE001 - a crashing harness is a failed run, not a runner crash
        error = f"{type(exc).__name__}: {exc}"[:500]
        error_trace = traceback.format_exc(limit=3)
        error = f"{error} | {error_trace[-300:]}"
    finally:
        if env is not None:
            benchmark.close_env(env)
    wall = time.perf_counter() - started
    if llm.infra_failure:
        raise InfrastructureFailure(f"{task_id} stream {stream}: {llm.infra_failure}")
    try:
        outcome, score = cached_score(benchmark, task, answer)
    except Exception as exc:  # noqa: BLE001 - recorded so that scorer faults are visible, never silent
        outcome, score = 0, 0.0
        error = (error + " | " if error else "") + f"SCORER {type(exc).__name__}: {str(exc)[:300]}"
    return {
        "task_id": task_id,
        "benchmark": benchmark.name,
        "harness_hash": version.harness_hash,
        "replicate_id": replicate,
        "stream": stream,
        "namespace": namespace,
        "crn": crn,
        "outcome": int(outcome),
        "score": float(score),
        "answer": answer if isinstance(answer, (str, int, float, type(None))) else str(answer),
        "covered_functions": tracer.record.sorted_functions(),
        "covered_lines": tracer.record.sorted_lines(),
        "n_llm_calls": llm.usage.n_llm_calls,
        "n_cache_hits": llm.usage.n_cache_hits,
        "tokens_in": llm.usage.tokens_in,
        "tokens_out": llm.usage.tokens_out,
        "nominal_tokens_in": llm.usage.nominal_tokens_in,
        "nominal_tokens_out": llm.usage.nominal_tokens_out,
        "backend_seconds": round(llm.usage.backend_seconds, 3),
        "gpu_seconds": None,
        "wall_time": round(wall, 3),
        "backend": client.backend.name,
        "model": client.model,
        "error": error,
        "trace": llm.trace if keep_trace else None,
        "request_fps": llm.request_fingerprints,
        "response_fps": llm.response_fingerprints,
    }


def run_one_resilient(
    benchmark: Benchmark,
    task: dict[str, Any],
    version: HarnessVersion,
    replicate: int,
    client: LLMClient,
    llm_params: dict[str, Any],
    crn: bool,
    max_llm_calls: int,
    keep_trace: bool = False,
    namespace: str = "",
    attempts: int = INFRA_ATTEMPTS,
) -> dict[str, Any]:
    """``run_one`` retried after infrastructure failures (cached calls replay for free)."""
    for attempt in range(attempts):
        try:
            return run_one(benchmark, task, version, replicate, client, llm_params, crn, max_llm_calls, keep_trace, namespace)
        except InfrastructureFailure:
            if attempt == attempts - 1:
                raise
            time.sleep(INFRA_PAUSE_SECONDS * (attempt + 1))
    raise AssertionError("unreachable")


def record_key(task_id: str, harness_hash: str, replicate: int, crn: bool, namespace: str = "") -> str:
    """Identity of a run inside a records file (used for idempotent resumption)."""
    mode = "crn" if crn else "ind"
    return f"{task_id}|{harness_hash}|{replicate}|{mode}|{namespace}"


def remember_score(record: dict[str, Any]) -> None:
    """Re-seed the score cache from a stored record (benchmark name is in its stream-free key)."""
    benchmark_name = record.get("benchmark")
    if benchmark_name is None:
        return
    material = json.dumps([benchmark_name, str(record["task_id"]), record["answer"]], sort_keys=True, default=str)
    key = hashlib.sha256(material.encode("utf-8")).hexdigest()
    with _SCORE_LOCK:
        _SCORE_CACHE.setdefault(key, (int(record["outcome"]), float(record["score"])))


class RecordStore:
    """Append-only JSONL store of run records with an in-memory index of finished runs."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.records: dict[str, dict[str, Any]] = {}
        self.skipped_partial_lines = 0
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        # A checkpoint taken while a line was being appended can end with a
                        # partial line; that run is simply executed again.
                        self.skipped_partial_lines += 1
                        continue
                    if "TransientBackendError" in (record.get("error") or ""):
                        # The backend failed, not the harness: the run never happened and is
                        # executed again (build_state.py prefers the successful record).
                        continue
                    key = record_key(record["task_id"], record["harness_hash"], record["replicate_id"], record["crn"], record.get("namespace", ""))
                    self.records[key] = record
                    remember_score(record)

    def has(self, key: str) -> bool:
        """True when the run is already recorded."""
        return key in self.records

    def append(self, record: dict[str, Any]) -> None:
        """Persist one record (flushed and fsynced so a crash never loses it)."""
        key = record_key(record["task_id"], record["harness_hash"], record["replicate_id"], record["crn"], record.get("namespace", ""))
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self.records[key] = record


@dataclass
class RunRequest:
    """One requested run."""

    task: dict[str, Any]
    version: HarnessVersion
    replicate: int
    crn: bool
    keep_trace: bool = False
    namespace: str = ""


def run_many(
    benchmark: Benchmark,
    requests: Iterable[RunRequest],
    client: LLMClient,
    llm_params: dict[str, Any],
    store: RecordStore,
    concurrency: int,
    max_llm_calls: int,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Run every request not already in ``store``; return records for all requests."""
    pending: list[RunRequest] = []
    all_keys: list[str] = []
    for request in requests:
        key = record_key(str(request.task["task_id"]), request.version.harness_hash, request.replicate, request.crn, request.namespace)
        all_keys.append(key)
        if not store.has(key):
            pending.append(request)
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = []
        for request in pending:
            future = pool.submit(
                run_one_resilient,
                benchmark,
                request.task,
                request.version,
                request.replicate,
                client,
                llm_params,
                request.crn,
                max_llm_calls,
                request.keep_trace,
                request.namespace,
            )
            futures.append(future)
        for future in as_completed(futures):
            record = future.result()
            store.append(record)
            if progress is not None:
                progress(record)
    results: list[dict[str, Any]] = []
    for key in all_keys:
        results.append(store.records[key])
    return results
