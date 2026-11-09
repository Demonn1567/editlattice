"""Run pools for offline gate evaluation (D-010).

An :class:`EditPool` holds, for one edit (parent → child), the parent's and the child's
runs on every search task and every stream id 0..R−1, where stream s is shared by the
parent and child (CRN). Streams are partitioned:

* history streams ``0..k_char−1``: the parent's characterization runs (computed once when
  the parent was accepted). Any gate may use them free of charge: k_sel of them for
  selection, the rest for certification.
* draw streams ``k_char..R−1``: runs a gate "executes"; each one is charged.

A CRN gate pairs parent and child on the same draw stream. A black-box gate pairs a
child draw stream with a *different* parent stream, which makes the two runs
independent.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PoolRun:
    """The parts of a run record the gates need."""

    outcome: int
    score: float
    covered_functions: frozenset[str]
    covered_lines: frozenset[tuple[str, int]]
    nominal_tokens: int
    generated_tokens: int
    answer: Any


def pool_run_from_record(record: dict[str, Any]) -> PoolRun:
    """Convert a JSONL run record."""
    lines: set[tuple[str, int]] = set()
    for label, number in record.get("covered_lines", []):
        lines.add((label, int(number)))
    nominal = int(record.get("nominal_tokens_in", record["tokens_in"])) + int(record.get("nominal_tokens_out", record["tokens_out"]))
    return PoolRun(
        outcome=int(record["outcome"]),
        score=float(record["score"]),
        covered_functions=frozenset(record.get("covered_functions", [])),
        covered_lines=frozenset(lines),
        nominal_tokens=nominal,
        generated_tokens=int(record["tokens_in"]) + int(record["tokens_out"]),
        answer=record.get("answer"),
    )


@dataclass
class EditPool:
    """Parent and child runs for one edit, indexed [task_id][stream]."""

    edit_id: str
    tasks: list[str]
    parent: dict[str, dict[int, PoolRun]]
    child: dict[str, dict[int, PoolRun]]
    n_streams: int
    k_char: int
    impact_functions: frozenset[str]
    impact_lines: frozenset[tuple[str, int]]
    impact_is_global: bool
    validation_tasks: list[str] = field(default_factory=list)
    validation_parent: dict[str, dict[int, PoolRun]] = field(default_factory=dict)
    validation_child: dict[str, dict[int, PoolRun]] = field(default_factory=dict)

    @property
    def draw_streams(self) -> list[int]:
        """Stream ids gates may draw (and pay for)."""
        return list(range(self.k_char, self.n_streams))

    def oracle_delta(self) -> float:
        """Mean paired difference over all tasks and all streams (D-010 ground truth)."""
        total = 0.0
        count = 0
        for task in self.tasks:
            for stream in range(self.n_streams):
                total += self.child[task][stream].outcome - self.parent[task][stream].outcome
                count += 1
        return total / count

    def touched(self, task: str, stream: int, granularity: str) -> bool:
        """Did the parent run on (task, stream) execute the impact set?"""
        if self.impact_is_global:
            return True
        run = self.parent[task][stream]
        if granularity == "line":
            return bool(run.covered_lines & self.impact_lines)
        return bool(run.covered_functions & self.impact_functions)


@dataclass
class Ledger:
    """Charges for the runs a gate executes."""

    task_runs: int = 0
    nominal_tokens: int = 0
    generated_tokens: int = 0

    def charge_full(self, run: PoolRun) -> None:
        """A black-box run: pays the full nominal cost."""
        self.task_runs += 1
        self.nominal_tokens += run.nominal_tokens
        self.generated_tokens += run.nominal_tokens

    def charge_crn(self, run: PoolRun) -> None:
        """A CRN-replayed run: pays only for generated (cache-miss) tokens."""
        self.task_runs += 1
        self.nominal_tokens += run.nominal_tokens
        self.generated_tokens += run.generated_tokens


@dataclass
class GateResult:
    """Decision plus cost."""

    gate: str
    accept: bool
    ledger: Ledger
    details: dict[str, Any] = field(default_factory=dict)


def independent_parent_stream(rng: random.Random, pool: EditPool, child_stream: int) -> int:
    """A parent draw stream different from ``child_stream`` (independent pairing)."""
    candidates: list[int] = []
    for stream in pool.draw_streams:
        if stream != child_stream:
            candidates.append(stream)
    return rng.choice(candidates)
