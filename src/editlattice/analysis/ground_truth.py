"""Build edit pools from run records and label edits by their oracle effect (D-010).

Oracle Δ = mean over (task, stream) of C′ − C on the CRN pool. Its 95% CI is a paired
cluster bootstrap that resamples tasks (with all their streams), because streams within a
task are correlated through the task's difficulty. Labels (brief §6.4): improving if the CI
lower bound > 0, harmful if the CI upper bound < 0, neutral otherwise.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from editlattice.gates.pool import EditPool, PoolRun, pool_run_from_record


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL records file."""
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def index_records(records: list[dict[str, Any]]) -> dict[tuple[str, str, bool], dict[str, dict[int, PoolRun]]]:
    """(harness_hash, namespace, crn) -> task -> stream -> PoolRun."""
    table: dict[tuple[str, str, bool], dict[str, dict[int, PoolRun]]] = {}
    for record in records:
        key = (record["harness_hash"], record.get("namespace", ""), bool(record["crn"]))
        by_task = table.setdefault(key, {})
        by_task.setdefault(record["task_id"], {})[int(record["replicate_id"])] = pool_run_from_record(record)
    return table


def build_pool(
    edit_id: str,
    table: dict,
    parent_hash: str,
    child_hash: str,
    namespace: str,
    tasks: list[str],
    n_streams: int,
    k_char: int,
    impact: dict[str, Any],
    validation_tasks: list[str] | None = None,
) -> EditPool:
    """Assemble the EditPool of one edit; raises KeyError if any run is missing."""
    parent_rows = table[(parent_hash, namespace, True)]
    child_rows = table[(child_hash, namespace, True)]
    parent: dict[str, dict[int, PoolRun]] = {}
    child: dict[str, dict[int, PoolRun]] = {}
    for task in tasks:
        parent[task] = {}
        child[task] = {}
        for stream in range(n_streams):
            parent[task][stream] = parent_rows[task][stream]
            child[task][stream] = child_rows[task][stream]
    validation_parent: dict[str, dict[int, PoolRun]] = {}
    validation_child: dict[str, dict[int, PoolRun]] = {}
    kept_validation: list[str] = []
    validation_key_parent = (parent_hash, namespace + ":val", True)
    validation_key_child = (child_hash, namespace + ":val", True)
    if validation_tasks and validation_key_parent in table and validation_key_child in table:
        for task in validation_tasks:
            if task not in table[validation_key_parent] or task not in table[validation_key_child]:
                continue
            # Keep only streams both versions have (a run may be missing after a purge, D-024/25).
            common = sorted(set(table[validation_key_parent][task]) & set(table[validation_key_child][task]))
            if len(common) < 2:
                continue
            validation_parent[task] = {stream: table[validation_key_parent][task][stream] for stream in common}
            validation_child[task] = {stream: table[validation_key_child][task][stream] for stream in common}
            kept_validation.append(task)
    lines: set[tuple[str, int]] = set()
    for label, number in impact.get("lines", []):
        lines.add((label, int(number)))
    return EditPool(
        edit_id=edit_id,
        tasks=list(tasks),
        parent=parent,
        child=child,
        n_streams=n_streams,
        k_char=k_char,
        impact_functions=frozenset(impact.get("functions", [])),
        impact_lines=frozenset(lines),
        impact_is_global=bool(impact.get("is_global", False)),
        validation_tasks=kept_validation,
        validation_parent=validation_parent,
        validation_child=validation_child,
    )


@dataclass
class OracleLabel:
    """Ground-truth effect of one edit."""

    delta: float
    ci_low: float
    ci_high: float
    label: str


def oracle_label(pool: EditPool, n_boot: int = 2000, seed: int = 0) -> OracleLabel:
    """Oracle Δ with a task-cluster bootstrap CI and the improving/harmful/neutral label."""
    per_task: list[float] = []
    for task in pool.tasks:
        total = 0.0
        for stream in range(pool.n_streams):
            total += pool.child[task][stream].outcome - pool.parent[task][stream].outcome
        per_task.append(total / pool.n_streams)
    delta = sum(per_task) / len(per_task)
    rng = random.Random(seed)
    means: list[float] = []
    size = len(per_task)
    for _ in range(n_boot):
        total = 0.0
        for _ in range(size):
            total += per_task[rng.randrange(size)]
        means.append(total / size)
    means.sort()
    low = means[int(0.025 * n_boot)]
    high = means[int(0.975 * n_boot) - 1]
    if low > 0:
        label = "improving"
    elif high < 0:
        label = "harmful"
    else:
        label = "neutral"
    return OracleLabel(delta, low, high, label)
