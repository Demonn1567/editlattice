"""Distillation: the smallest or cheapest product that keeps the gain (design D7).

All methods query the same exact value oracle (the lattice), so they are compared on equal
footing; each records how many distinct products it had to evaluate. The chosen product is
then certified on held-out cells with an anytime-valid non-inferiority test against the full
harness on shared streams.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from editlattice.attribution import all_subsets, leave_one_out, shapley
from editlattice.stats.eprocess import paired_difference_test, to_unit_interval

Subset = frozenset[int]


@dataclass
class Oracle:
    """Counts the distinct products a method evaluates."""

    values: dict[Subset, float]
    asked: set[Subset] = field(default_factory=set)

    def __call__(self, subset: Subset) -> float:
        self.asked.add(subset)
        return self.values[subset]


@dataclass
class Choice:
    """A distilled product and what it cost to find it."""

    method: str
    product: Subset
    value: float
    queries: int


def keeps_gain(value: float, full: float, delta: float) -> bool:
    """Within delta of the full harness."""
    return value >= full - delta


def exact_minimum(values: dict[Subset, float], players: list[int], delta: float, cost: Callable[[Subset], float] = len) -> Choice:
    """Cheapest product within delta of the full harness, searching the whole lattice."""
    oracle = Oracle(values)
    full = oracle(frozenset(players))
    admissible = [subset for subset in all_subsets(players) if keeps_gain(oracle(subset), full, delta)]
    best = min(admissible, key=lambda subset: (cost(subset), -values[subset], sorted(subset)))
    return Choice("exact", best, values[best], len(oracle.asked))


def greedy_backward(values: dict[Subset, float], players: list[int], delta: float) -> Choice:
    """Repeatedly drop the unit whose removal hurts least, while the gain is kept."""
    oracle = Oracle(values)
    full = oracle(frozenset(players))
    current = frozenset(players)
    while current:
        candidates = [(oracle(current - {unit}), unit) for unit in sorted(current)]
        value, unit = max(candidates)
        if not keeps_gain(value, full, delta):
            break
        current = current - {unit}
    return Choice("greedy_backward", current, values[current], len(oracle.asked))


def ddmin(values: dict[Subset, float], players: list[int], delta: float) -> Choice:
    """Zeller's ddmin with "keeps the gain" as the predicate (1-minimal result)."""
    oracle = Oracle(values)
    full = oracle(frozenset(players))
    current = sorted(players)
    granularity = 2
    while len(current) >= 2:
        size = max(1, len(current) // granularity)
        chunks = [current[i:i + size] for i in range(0, len(current), size)]
        reduced = False
        for chunk in chunks:
            if keeps_gain(oracle(frozenset(chunk)), full, delta):
                current, granularity, reduced = chunk, 2, True
                break
        if not reduced:
            for chunk in chunks:
                complement = [unit for unit in current if unit not in chunk]
                if keeps_gain(oracle(frozenset(complement)), full, delta):
                    current, granularity, reduced = complement, max(granularity - 1, 2), True
                    break
        if not reduced:
            if granularity >= len(current):
                break
            granularity = min(len(current), 2 * granularity)
    if len(current) == 1 and keeps_gain(oracle(frozenset()), full, delta):
        current = []
    product = frozenset(current)
    return Choice("ddmin", product, values[product], len(oracle.asked))


def threshold(values: dict[Subset, float], players: list[int], delta: float, scores: dict[int, float], method: str) -> Choice:
    """Keep units in order of decreasing score until the gain is kept."""
    oracle = Oracle(values)
    full = oracle(frozenset(players))
    order = sorted(players, key=lambda unit: -scores[unit])
    current: Subset = frozenset()
    for unit in [None] + order:
        if unit is not None:
            current = current | {unit}
        if keeps_gain(oracle(current), full, delta):
            break
    return Choice(method, current, values[current], len(oracle.asked))


def all_methods(values: dict[Subset, float], players: list[int], delta: float) -> list[Choice]:
    """Every distillation method on one lattice. Score-based methods also pay for their scores."""
    choices = [exact_minimum(values, players, delta), greedy_backward(values, players, delta), ddmin(values, players, delta)]
    loo = threshold(values, players, delta, leave_one_out(values, players), "loo_threshold")
    loo.queries += len(players) + 1  # v(all) and v(all - i)
    sv = threshold(values, players, delta, shapley(values, players), "shapley_threshold")
    sv.queries = 2 ** len(players)  # exact Shapley needs the whole lattice
    return choices + [loo, sv]


def certify_non_inferior(differences: list[float], delta: float, alpha: float = 0.05) -> tuple[bool, int]:
    """Anytime-valid test of H0: E[D] <= -delta for paired D = outcome(distilled) - outcome(full) in [-1, 1].

    Returns (certified, pairs used). Rejecting H0 certifies that the distilled harness is
    worse than the full one by less than delta.
    """
    test = paired_difference_test(-delta, alpha)
    for count, difference in enumerate(differences, start=1):
        test.update(to_unit_interval(difference))
        if test.rejects_null:
            return True, count
    return False, len(differences)
