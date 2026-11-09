"""RQ1/RQ2: touch, divergence and soundness of impact analysis on CRN stream pools.

Per (edit, task, stream), comparing the parent and child runs on the same stream:
* touched_function / touched_line: the parent run executed the impact set;
* diverged: the child's sequence of request fingerprints differs from the parent's
  (it sent a different request at some call), or the answers differ;
* divergence_step: index of the first differing request (None if not diverged);
* A4 violation: diverged although not touched (the impact analysis missed a path).
Bias-bound coverage: with k_sel selection and k_cert certification streams, the realized
effect on the unaffected tasks |R| = |(1/N) Σ_{x∈U} mean_s (C′ − C)| is compared with the
certified bound b̂ (Lemma 2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from editlattice.stats.bias_bound import suite_bias_bound


def first_difference(parent: list[str], child: list[str]) -> int | None:
    """Index of the first differing element (length mismatch counts), or None."""
    for index in range(min(len(parent), len(child))):
        if parent[index] != child[index]:
            return index
    if len(parent) != len(child):
        return min(len(parent), len(child))
    return None


@dataclass
class EditDivergence:
    """Aggregates for one edit."""

    edit: str
    is_global: bool
    n_impact_functions: int
    n_runs: int = 0
    touched_function: int = 0
    touched_line: int = 0
    diverged: int = 0
    outcome_changed: int = 0
    a4_violations_function: int = 0
    a4_violations_line: int = 0
    divergence_steps: list[int] = field(default_factory=list)
    violation_examples: list[dict[str, Any]] = field(default_factory=list)
    affected_tasks_function: int = 0
    n_tasks: int = 0
    bias_bound: float | None = None
    realized_unaffected_effect: float | None = None

    def as_dict(self) -> dict[str, Any]:
        """Summary with rates."""
        runs = max(1, self.n_runs)
        return {
            "edit": self.edit,
            "is_global": self.is_global,
            "n_impact_functions": self.n_impact_functions,
            "n_runs": self.n_runs,
            "touched_function_rate": self.touched_function / runs,
            "touched_line_rate": self.touched_line / runs,
            "diverged_rate": self.diverged / runs,
            "outcome_changed_rate": self.outcome_changed / runs,
            "a4_violations_function": self.a4_violations_function,
            "a4_violations_line": self.a4_violations_line,
            "affected_task_fraction": self.affected_tasks_function / max(1, self.n_tasks),
            "behaviourally_inert": self.diverged == 0,
            "mean_divergence_step": (sum(self.divergence_steps) / len(self.divergence_steps)) if self.divergence_steps else None,
            "bias_bound": self.bias_bound,
            "realized_unaffected_effect": self.realized_unaffected_effect,
            "bound_holds": None if self.bias_bound is None else abs(self.realized_unaffected_effect or 0.0) <= self.bias_bound + 1e-12,
            "violation_examples": self.violation_examples[:5],
        }


def lines_of(record: dict[str, Any]) -> set[tuple[str, int]]:
    """Covered lines of a record as a set."""
    covered: set[tuple[str, int]] = set()
    for label, number in record.get("covered_lines", []):
        covered.add((label, int(number)))
    return covered


def analyse_edit(
    edit_name: str,
    impact: dict[str, Any],
    parent_runs: dict[str, dict[int, dict[str, Any]]],
    child_runs: dict[str, dict[int, dict[str, Any]]],
    tasks: list[str],
    n_streams: int,
    k_sel: int,
    k_cert: int,
    alpha_b: float = 0.05,
) -> EditDivergence:
    """All per-run comparisons for one edit (records are raw JSONL dicts)."""
    functions = set(impact.get("functions", []))
    impact_lines: set[tuple[str, int]] = set()
    for label, number in impact.get("lines", []):
        impact_lines.add((label, int(number)))
    is_global = bool(impact.get("is_global"))
    result = EditDivergence(edit=edit_name, is_global=is_global, n_impact_functions=len(functions), n_tasks=len(tasks))
    unaffected: list[str] = []
    cert_touches = 0
    for task in tasks:
        selected = False
        for stream in range(n_streams):
            parent = parent_runs[task][stream]
            child = child_runs[task][stream]
            touched_function = is_global or bool(set(parent.get("covered_functions", [])) & functions)
            touched_line = is_global or bool(lines_of(parent) & impact_lines)
            step = first_difference(parent.get("request_fps") or [], child.get("request_fps") or [])
            diverged = step is not None or parent.get("answer") != child.get("answer")
            result.n_runs += 1
            result.touched_function += 1 if touched_function else 0
            result.touched_line += 1 if touched_line else 0
            if diverged:
                result.diverged += 1
                result.divergence_steps.append(step if step is not None else len(parent.get("request_fps") or []))
            if parent["outcome"] != child["outcome"]:
                result.outcome_changed += 1
            if diverged and not touched_function:
                result.a4_violations_function += 1
                result.violation_examples.append({"task": task, "stream": stream, "step": step, "parent_answer": str(parent.get("answer"))[:200], "child_answer": str(child.get("answer"))[:200]})
            if diverged and not touched_line:
                result.a4_violations_line += 1
            if stream < k_sel and touched_function:
                selected = True
        if selected:
            result.affected_tasks_function += 1
        else:
            unaffected.append(task)
            for stream in range(k_sel, k_sel + k_cert):
                parent = parent_runs[task][stream]
                if is_global or set(parent.get("covered_functions", [])) & functions:
                    cert_touches += 1
    if k_cert > 0:
        result.bias_bound = suite_bias_bound(len(unaffected), len(tasks), cert_touches, k_cert, alpha_b)
        realized = 0.0
        for task in unaffected:
            total = 0.0
            for stream in range(n_streams):
                total += child_runs[task][stream]["outcome"] - parent_runs[task][stream]["outcome"]
            realized += total / n_streams
        result.realized_unaffected_effect = realized / len(tasks)
    return result
