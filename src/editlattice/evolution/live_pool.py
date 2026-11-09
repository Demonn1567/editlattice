"""A lazily executed :class:`EditPool` so that the offline-validated gates run live.

``pool.parent[task][stream]`` and ``pool.child[task][stream]`` return a :class:`PoolRun`,
executing the run (through the resumable runner) when it is not stored yet. All runs of
one edit use the parent's stream namespace, so parent and child share CRN streams; history
streams 0..k_char−1 are the parent's characterization runs.

Throughput: a gate requests runs one at a time. To keep the inference server busy we
*plan* the gate's requests with a dry run (``plan_gate``) and execute them in parallel
batches. On a miss, the pool runs the missed key together with the next ``batch_size``
planned keys. This cannot change any decision: the dry run uses a copy of the RNG, so the
real gate makes the same random choices; a prefetched run is the same run the gate would
have executed later (same task, version, stream, namespace); and planned runs the gate never
reads are reported as overhead (``unused_prefetched``), never as part of its ledger.
"""

from __future__ import annotations

import random
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable

from editlattice.gates.pool import EditPool, PoolRun, pool_run_from_record
from editlattice.harness_runtime.runner import Benchmark, HarnessVersion, RecordStore, record_key, run_one_resilient
from editlattice.impact.diff import ImpactSet
from editlattice.llm.client import LLMClient

Key = tuple[str, str, int]  # (table role, task id, stream)


@dataclass
class Executor:
    """Everything needed to execute (and persist) one run."""

    benchmark: Benchmark
    client: LLMClient
    store: RecordStore
    llm_params: dict[str, Any]
    max_llm_calls: int

    def run(self, task: dict[str, Any], version: HarnessVersion, stream: int, namespace: str, keep_trace: bool = False) -> dict[str, Any]:
        """Return the record of a CRN run, executing it only if it is not stored yet."""
        key = record_key(str(task["task_id"]), version.harness_hash, stream, True, namespace)
        if self.store.has(key):
            return self.store.records[key]
        record = run_one_resilient(self.benchmark, task, version, stream, self.client, self.llm_params, True, self.max_llm_calls, keep_trace, namespace)
        self.store.append(record)
        return record

    def run_parallel(self, jobs: list[tuple[dict[str, Any], HarnessVersion, int, str, bool]], workers: int) -> list[dict[str, Any]]:
        """Run several (task, version, stream, namespace, keep_trace) jobs concurrently."""
        if not jobs:
            return []
        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(jobs)))) as pool:
            futures = []
            for task, version, stream, namespace, keep_trace in jobs:
                futures.append(pool.submit(self.run, task, version, stream, namespace, keep_trace))
            results = []
            for future in futures:
                results.append(future.result())
        return results


@dataclass
class TableSpec:
    """What a table role runs: version, namespace, tasks."""

    version: HarnessVersion
    namespace: str
    tasks: dict[str, dict[str, Any]]


@dataclass
class Prefetcher:
    """Executes planned keys in parallel batches and remembers what was read."""

    executor: Executor
    specs: dict[str, TableSpec]
    batch_size: int = 24
    plan: list[Key] = field(default_factory=list)
    _position: dict[Key, int] = field(default_factory=dict)
    _results: dict[Key, PoolRun] = field(default_factory=dict)
    _read: set[Key] = field(default_factory=set)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def set_plan(self, plan: list[Key]) -> None:
        """Install the access plan recorded by a dry run."""
        self.plan = list(plan)
        self._position = {}
        for index, key in enumerate(self.plan):
            self._position.setdefault(key, index)

    def get(self, key: Key) -> PoolRun:
        """Return a run, prefetching the next planned keys on a miss."""
        self._read.add(key)
        if key in self._results:
            return self._results[key]
        batch = [key]
        start = self._position.get(key)
        # Geometric growth (4, 8, 16, ... up to batch_size) bounds the unused runs left
        # behind when a sequential gate stops early.
        size = min(self.batch_size, max(4, len(self._read)))
        if start is not None:
            for planned in self.plan[start + 1 : start + size]:
                if planned not in self._results and planned not in batch:
                    batch.append(planned)
        jobs = []
        for role, task_id, stream in batch:
            spec = self.specs[role]
            jobs.append((spec.tasks[task_id], spec.version, stream, spec.namespace, False))
        records = self.executor.run_parallel(jobs, self.batch_size)
        for batch_key, record in zip(batch, records):
            self._results[batch_key] = pool_run_from_record(record)
        return self._results[key]

    @property
    def unused_prefetched(self) -> int:
        """Planned runs executed but never read by the gate (overhead)."""
        return len(set(self._results) - self._read)


class LazyStreams:
    """Mapping stream -> PoolRun for one (role, task)."""

    def __init__(self, fetch: Callable[[Key], PoolRun], role: str, task_id: str, n_streams: int) -> None:
        self.fetch = fetch
        self.role = role
        self.task_id = task_id
        self.n_streams = n_streams

    def __getitem__(self, stream: int) -> PoolRun:
        if stream < 0 or stream >= self.n_streams:
            raise KeyError(stream)
        return self.fetch((self.role, self.task_id, stream))

    def __iter__(self):
        return iter(range(self.n_streams))


class LazyTable:
    """Mapping task_id -> LazyStreams."""

    def __init__(self, fetch: Callable[[Key], PoolRun], role: str, task_ids: list[str], n_streams: int) -> None:
        self._rows: dict[str, LazyStreams] = {}
        for task_id in task_ids:
            self._rows[task_id] = LazyStreams(fetch, role, task_id, n_streams)

    def __getitem__(self, task_id: str) -> LazyStreams:
        return self._rows[task_id]


def assemble_pool(edit_id: str, fetch: Callable[[Key], PoolRun], specs: dict[str, TableSpec], n_streams: int, k_char: int, impact: ImpactSet) -> EditPool:
    """An EditPool whose tables call ``fetch``."""
    search_ids = list(specs["parent"].tasks)
    validation_ids = list(specs["val_parent"].tasks)
    return EditPool(
        edit_id=edit_id,
        tasks=search_ids,
        parent=LazyTable(fetch, "parent", search_ids, n_streams),  # type: ignore[arg-type]
        child=LazyTable(fetch, "child", search_ids, n_streams),  # type: ignore[arg-type]
        n_streams=n_streams,
        k_char=k_char,
        impact_functions=frozenset(impact.functions),
        impact_lines=frozenset(impact.lines),
        impact_is_global=impact.is_global,
        validation_tasks=validation_ids,
        validation_parent=LazyTable(fetch, "val_parent", validation_ids, 4),  # type: ignore[arg-type]
        validation_child=LazyTable(fetch, "val_child", validation_ids, 4),  # type: ignore[arg-type]
    )


def make_specs(tasks: dict[str, dict[str, Any]], validation: dict[str, dict[str, Any]], parent: HarnessVersion, child: HarnessVersion, namespace: str) -> dict[str, TableSpec]:
    """Table roles of one edit."""
    return {
        "parent": TableSpec(parent, namespace, tasks),
        "child": TableSpec(child, namespace, tasks),
        "val_parent": TableSpec(parent, f"{namespace}:val", validation),
        "val_child": TableSpec(child, f"{namespace}:val", validation),
    }


def placeholder_run() -> PoolRun:
    """Stand-in for an unexecuted run during planning (outcome 0, no coverage)."""
    return PoolRun(outcome=0, score=0.0, covered_functions=frozenset(), covered_lines=frozenset(), nominal_tokens=0, generated_tokens=0, answer=None)


def planning_variant(gate: Any) -> Any | None:
    """The gate to dry-run, or None when it cannot be planned (it calls an LLM itself).

    Futility stopping is disabled in the dry run so that the plan covers the whole budget.
    """
    if not isinstance(gate, partial):
        return None
    keywords = dict(gate.keywords)
    if "futility_margin" in keywords:
        keywords["futility_margin"] = 0.0
    return partial(gate.func, *gate.args, **keywords)


def plan_gate(gate: Any, specs: dict[str, TableSpec], executor: Executor, n_streams: int, k_char: int, impact: ImpactSet, rng: random.Random) -> list[Key]:
    """Record the gate's access sequence. Parent history streams are real (already run)."""
    variant = planning_variant(gate)
    if variant is None:
        return []
    accessed: list[Key] = []

    def fetch(key: Key) -> PoolRun:
        accessed.append(key)
        role, task_id, stream = key
        if role == "parent" and stream < k_char:
            spec = specs[role]
            return pool_run_from_record(executor.run(spec.tasks[task_id], spec.version, stream, spec.namespace))
        return placeholder_run()

    planning_rng = random.Random()
    planning_rng.setstate(rng.getstate())
    variant(assemble_pool("planning", fetch, specs, n_streams, k_char, impact), planning_rng)
    ordered: list[Key] = []
    seen: set[Key] = set()
    for key in accessed:
        if key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered


def make_live_pool(
    edit_id: str,
    executor: Executor,
    tasks: dict[str, dict[str, Any]],
    parent: HarnessVersion,
    child: HarnessVersion,
    namespace: str,
    n_streams: int,
    k_char: int,
    impact: ImpactSet,
    validation_tasks: dict[str, dict[str, Any]] | None = None,
    gate: Any = None,
    rng: random.Random | None = None,
    batch_size: int = 24,
) -> tuple[EditPool, Prefetcher]:
    """Live pool for one edit; if ``gate`` and ``rng`` are given, its accesses are prefetched."""
    specs = make_specs(tasks, validation_tasks or {}, parent, child, namespace)
    prefetcher = Prefetcher(executor, specs, batch_size)
    if gate is not None and rng is not None and batch_size > 1:
        prefetcher.set_plan(plan_gate(gate, specs, executor, n_streams, k_char, impact, rng))
    return assemble_pool(edit_id, prefetcher.get, specs, n_streams, k_char, impact), prefetcher
