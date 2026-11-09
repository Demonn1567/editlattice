"""Exact attribution over the full lattice of products (design D6).

Input: the outcome of every product on every cell (task, replicate), all on shared streams.
The value of product S is its mean outcome over the cells. Everything below is exact on the
sample; uncertainty about the task population comes from a task-cluster bootstrap that
resamples tasks (with all their replicates) once and recomputes every quantity on that
resample, so differences between products stay paired.
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass

Subset = frozenset[int]


def all_subsets(players: list[int]) -> list[Subset]:
    """Every subset of the players, smallest first."""
    return [frozenset(combo) for size in range(len(players) + 1) for combo in itertools.combinations(players, size)]


def mobius(values: dict[Subset, float], players: list[int]) -> dict[Subset, float]:
    """Harsanyi dividends: m(T) = sum over S in T of (-1)^{|T|-|S|} v(S)."""
    return {target: sum((-1) ** (len(target) - len(subset)) * values[subset] for subset in all_subsets(sorted(target)))
            for target in all_subsets(players)}


def shapley(values: dict[Subset, float], players: list[int]) -> dict[int, float]:
    """Exact Shapley values."""
    n = len(players)
    result = {}
    for player in players:
        others = [other for other in players if other != player]
        total = 0.0
        for subset in all_subsets(others):
            weight = math.factorial(len(subset)) * math.factorial(n - len(subset) - 1) / math.factorial(n)
            total += weight * (values[subset | {player}] - values[subset])
        result[player] = total
    return result


def shapley_interaction(values: dict[Subset, float], players: list[int]) -> dict[tuple[int, int], float]:
    """Pairwise Shapley interaction index (Grabisch and Roubens)."""
    n = len(players)
    result = {}
    for first, second in itertools.combinations(players, 2):
        others = [other for other in players if other not in (first, second)]
        total = 0.0
        for subset in all_subsets(others):
            weight = math.factorial(len(subset)) * math.factorial(n - len(subset) - 2) / math.factorial(n - 1)
            total += weight * (values[subset | {first, second}] - values[subset | {first}] - values[subset | {second}] + values[subset])
        result[(first, second)] = total
    return result


def leave_one_out(values: dict[Subset, float], players: list[int]) -> dict[int, float]:
    """v(all) - v(all without i)."""
    full = frozenset(players)
    return {player: values[full] - values[full - {player}] for player in players}


def leave_one_in(values: dict[Subset, float], players: list[int]) -> dict[int, float]:
    """v({i}) - v(empty)."""
    return {player: values[frozenset({player})] - values[frozenset()] for player in players}


@dataclass
class Attribution:
    """All quantities for one value function."""

    values: dict[Subset, float]
    shapley: dict[int, float]
    interaction: dict[tuple[int, int], float]
    mobius: dict[Subset, float]
    loo: dict[int, float]
    loi: dict[int, float]


def attribute(values: dict[Subset, float], players: list[int]) -> Attribution:
    """Every attribution quantity of one value function."""
    return Attribution(values=values, shapley=shapley(values, players), interaction=shapley_interaction(values, players),
                       mobius=mobius(values, players), loo=leave_one_out(values, players), loi=leave_one_in(values, players))


def lattice_values(outcomes: dict[tuple[str, int], dict[Subset, float]], tasks: list[str] | None = None) -> dict[Subset, float]:
    """Mean outcome of every product over the cells of the given tasks (all tasks if None)."""
    cells = [cell for cell in outcomes if tasks is None or cell[0] in set(tasks)]
    if tasks is not None:
        # Keep bootstrap multiplicities: a task drawn twice counts twice.
        counts: dict[str, int] = {}
        for task in tasks:
            counts[task] = counts.get(task, 0) + 1
        cells = [cell for cell in outcomes if cell[0] in counts]
        weights = {cell: counts[cell[0]] for cell in cells}
    else:
        weights = {cell: 1 for cell in cells}
    total = sum(weights.values())
    subsets = next(iter(outcomes.values())).keys()
    return {subset: sum(weights[cell] * outcomes[cell][subset] for cell in cells) / total for subset in subsets}


def bootstrap(outcomes: dict[tuple[str, int], dict[Subset, float]], players: list[int], n_boot: int = 2000, seed: int = 0) -> dict[str, dict]:
    """95% task-cluster bootstrap intervals for Shapley values, pairwise interactions and product values."""
    rng = random.Random(seed)
    tasks = sorted({cell[0] for cell in outcomes})
    draws: dict[str, list[float]] = {}
    for _ in range(n_boot):
        sample = [rng.choice(tasks) for _ in tasks]
        result = attribute(lattice_values(outcomes, sample), players)
        for player, value in result.shapley.items():
            draws.setdefault(f"shapley:{player}", []).append(value)
        for pair, value in result.interaction.items():
            draws.setdefault(f"interaction:{pair[0]},{pair[1]}", []).append(value)
        for subset, value in result.values.items():
            draws.setdefault(f"value:{','.join(map(str, sorted(subset)))}", []).append(value)
    intervals = {}
    for key, values in draws.items():
        values.sort()
        # Two-sided bootstrap p-value for "the quantity is 0" (percentile method).
        below = sum(1 for value in values if value <= 0) / n_boot
        above = sum(1 for value in values if value >= 0) / n_boot
        intervals[key] = {"low": values[int(0.025 * n_boot)], "high": values[int(0.975 * n_boot) - 1],
                          "p": min(1.0, 2 * min(below, above))}
    return intervals


def holm(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm's step-down procedure: which hypotheses are rejected at family-wise level alpha."""
    order = sorted(p_values, key=p_values.get)
    rejected, m = {}, len(order)
    still = True
    for rank, key in enumerate(order):
        still = still and p_values[key] <= alpha / (m - rank)
        rejected[key] = still
    return rejected
