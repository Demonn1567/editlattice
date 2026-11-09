"""Synthetic edit pools with known ground truth (for gate validation and the paper's
controlled simulation).

Model: task x has parent success probability p_x and touch probability t_x (0 for a
chosen fraction of tasks, the "truly local" structure). On each stream the parent
succeeds with probability p_x; the run is touched with probability t_x; on a touched run
the child re-draws its outcome with probability clip(p_x + effect_x); on an untouched
run the child copies the parent (Lemma 1).
"""

from __future__ import annotations

import random

from editlattice.gates.pool import EditPool, PoolRun

FUNCTION = "edited_function"


def make_run(outcome: int, touched: bool, tokens: int, generated: int) -> PoolRun:
    """Build one synthetic run."""
    functions = frozenset({FUNCTION}) if touched else frozenset({"other"})
    lines = frozenset({("harness.py", 1)}) if touched else frozenset({("harness.py", 2)})
    return PoolRun(outcome=outcome, score=float(outcome), covered_functions=functions, covered_lines=lines, nominal_tokens=tokens, generated_tokens=generated, answer=None)


def synthetic_pool(
    seed: int,
    n_tasks: int = 100,
    n_streams: int = 12,
    k_char: int = 4,
    local_fraction: float = 0.7,
    touch_probability: float = 0.8,
    effect: float = 0.0,
    n_validation: int = 0,
) -> EditPool:
    """Generate a pool; ``effect`` is the success-probability change on touched runs."""
    rng = random.Random(seed)
    tasks: list[str] = []
    parent: dict[str, dict[int, PoolRun]] = {}
    child: dict[str, dict[int, PoolRun]] = {}
    for index in range(n_tasks):
        task = f"t{index}"
        tasks.append(task)
        success = rng.uniform(0.05, 0.95)
        touch = 0.0 if rng.random() < local_fraction else touch_probability
        parent[task] = {}
        child[task] = {}
        for stream in range(n_streams):
            parent_outcome = 1 if rng.random() < success else 0
            touched = rng.random() < touch
            if touched:
                child_probability = min(1.0, max(0.0, success + effect))
                child_outcome = 1 if rng.random() < child_probability else 0
                generated = 1000
            else:
                child_outcome = parent_outcome
                generated = 0
            parent[task][stream] = make_run(parent_outcome, touched, 1000, 1000)
            child[task][stream] = make_run(child_outcome, touched, 1000, generated)
    validation_tasks: list[str] = []
    validation_parent: dict[str, dict[int, PoolRun]] = {}
    validation_child: dict[str, dict[int, PoolRun]] = {}
    for index in range(n_validation):
        task = f"v{index}"
        validation_tasks.append(task)
        success = rng.uniform(0.05, 0.95)
        touch = 0.0 if rng.random() < local_fraction else touch_probability
        validation_parent[task] = {}
        validation_child[task] = {}
        for stream in range(4):
            parent_outcome = 1 if rng.random() < success else 0
            touched = rng.random() < touch
            child_probability = min(1.0, max(0.0, success + effect)) if touched else success
            child_outcome = 1 if rng.random() < child_probability else 0
            validation_parent[task][stream] = make_run(parent_outcome, touched, 1000, 1000)
            validation_child[task][stream] = make_run(child_outcome, touched, 1000, 1000)
    return EditPool(
        edit_id=f"synthetic-{seed}",
        tasks=tasks,
        parent=parent,
        child=child,
        n_streams=n_streams,
        k_char=k_char,
        impact_functions=frozenset({FUNCTION}),
        impact_lines=frozenset({("harness.py", 1)}),
        impact_is_global=False,
        validation_tasks=validation_tasks,
        validation_parent=validation_parent,
        validation_child=validation_child,
    )
