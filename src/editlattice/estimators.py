"""Sampling estimators of Shapley values at a fixed budget of task-runs (RQ2).

Each estimator needs values v(S) of some products. It estimates them from cells (task, replicate)
drawn at random, paying one task-run per (product, cell) it uses. Two evaluation regimes:

* coupled: every product is evaluated on the same drawn cells, with outcomes from the shared-stream
  lattice (``coupled[S][cell]``), so contrasts between products are paired;
* independent: each (product, cell) use is an independent run (``independent[S][cell]`` holds runs
  on streams private to the product), as when every evaluation re-runs the agent.

Ground truth is the exact Shapley value on the full lattice. Errors are reported per estimator,
regime and budget.
"""

from __future__ import annotations

import itertools
import math
import random
from typing import Callable

import numpy as np

from editlattice.attribution import all_subsets, leave_one_in, leave_one_out, shapley

Subset = frozenset[int]
Outcomes = dict[Subset, list[float]]  # product -> outcome per cell index


class ValueSampler:
    """Estimates v(S) from ``cells_per_value`` drawn cells and counts task-runs."""

    def __init__(self, outcomes: Outcomes, cells_per_value: int, rng: random.Random, coupled: bool) -> None:
        self.outcomes = outcomes
        self.cells_per_value = cells_per_value
        self.rng = rng
        self.coupled = coupled
        self.n_cells = len(next(iter(outcomes.values())))
        self.shared_cells = [rng.randrange(self.n_cells) for _ in range(cells_per_value)]
        self.runs = 0
        self._memo: dict[Subset, float] = {}

    def value(self, subset: Subset) -> float:
        """Coupled: same cells for every product (memoised, paid once); independent: fresh cells per call."""
        if self.coupled:
            if subset not in self._memo:
                self.runs += self.cells_per_value
                self._memo[subset] = float(np.mean([self.outcomes[subset][c] for c in self.shared_cells]))
            return self._memo[subset]
        self.runs += self.cells_per_value
        cells = [self.rng.randrange(self.n_cells) for _ in range(self.cells_per_value)]
        return float(np.mean([self.outcomes[subset][c] for c in cells]))


MAX_SAMPLES = 5000  # coupled samplers memoise values, so the budget alone may never run out


def permutation_shapley(sampler: ValueSampler, players: list[int], budget: int) -> dict[int, float]:
    """Monte Carlo over random orders until the budget (task-runs) or MAX_SAMPLES orders are used."""
    totals = {player: 0.0 for player in players}
    count = 0
    while count < MAX_SAMPLES:
        order = players[:]
        sampler.rng.shuffle(order)
        previous_set: Subset = frozenset()
        cost_before = sampler.runs
        previous = sampler.value(previous_set)
        contributions = {}
        for player in order:
            current_set = previous_set | {player}
            current = sampler.value(current_set)
            contributions[player] = current - previous
            previous_set, previous = current_set, current
        if sampler.runs > budget and count > 0:
            sampler.runs = cost_before  # discard the incomplete last permutation's cost overrun
            break
        for player, value in contributions.items():
            totals[player] += value
        count += 1
        if sampler.runs >= budget:
            break
    return {player: totals[player] / count for player in players}


def kernel_shap(sampler: ValueSampler, players: list[int], budget: int) -> dict[int, float]:
    """KernelSHAP: weighted least squares over sampled coalitions with the efficiency constraint."""
    m = len(players)
    index = {player: position for position, player in enumerate(players)}
    empty, full = sampler.value(frozenset()), sampler.value(frozenset(players))
    sizes = list(range(1, m))
    weights = np.array([(m - 1) / (math.comb(m, s) * s * (m - s)) * math.comb(m, s) for s in sizes])
    weights /= weights.sum()
    rows, targets = [], []
    while sampler.runs < budget and len(rows) < MAX_SAMPLES:
        size = sampler.rng.choices(sizes, weights=weights)[0]
        coalition = frozenset(sampler.rng.sample(players, size))
        # Never exceed the budget (DEC-12): a value costs runs unless the coupled sampler has it already.
        cost = 0 if sampler.coupled and coalition in sampler._memo else sampler.cells_per_value
        if sampler.runs + cost > budget:
            break
        z = np.zeros(m)
        for player in coalition:
            z[index[player]] = 1.0
        rows.append(z)
        targets.append(sampler.value(coalition) - empty)
    if not rows:
        return {player: (full - empty) / m for player in players}
    z, y = np.array(rows), np.array(targets)
    # Constrained least squares: sum(phi) = full - empty (eliminate the last coordinate).
    total = full - empty
    a = z[:, :-1] - z[:, [-1]]
    b = y - z[:, -1] * total
    phi_head, *_ = np.linalg.lstsq(a, b, rcond=None)
    phi = np.append(phi_head, total - phi_head.sum())
    return {player: float(phi[index[player]]) for player in players}


def exact_from_samples(sampler: ValueSampler, players: list[int], method: Callable) -> dict[int, float]:
    """Plug-in estimate: method (exact Shapley, LOO, LOI) on sampled values of the products it needs."""
    needed = all_subsets(players) if method is shapley else (
        [frozenset(players)] + [frozenset(players) - {p} for p in players] if method is leave_one_out
        else [frozenset()] + [frozenset({p}) for p in players])
    values = {subset: sampler.value(subset) for subset in needed}
    complete = {subset: values.get(subset, 0.0) for subset in all_subsets(players)}
    return method(complete, players)


def estimate_once(name: str, outcomes: Outcomes, is_coupled: bool, players: list[int], budget: int,
                  rng: random.Random) -> tuple[dict[int, float], int]:
    """One estimate of the Shapley values (or LOO/LOI) at ``budget`` task-runs, and the runs it used."""
    # Each estimator gets the same total budget; plug-ins spread it over the products they need.
    if name in ("plugin_exact", "loo", "loi"):
        needed = 2 ** len(players) if name == "plugin_exact" else len(players) + 1
        cells = max(1, budget // needed)
        sampler = ValueSampler(outcomes, cells, rng, is_coupled)
        method = {"plugin_exact": shapley, "loo": leave_one_out, "loi": leave_one_in}[name]
        return exact_from_samples(sampler, players, method), sampler.runs
    # Coupled samplers pay once per distinct product, so they would stop spending once
    # every product is known; size their cells so the full lattice uses the budget.
    products = 2 ** len(players)
    cells = budget // products if is_coupled and products <= budget else budget // (8 * len(players))
    sampler = ValueSampler(outcomes, max(1, cells), rng, is_coupled)
    estimate = (permutation_shapley if name == "permutation" else kernel_shap)(sampler, players, budget)
    return estimate, sampler.runs


def compare(coupled: Outcomes, independent: Outcomes, players: list[int], truth: dict[int, float], budgets: list[int],
            repetitions: int = 20, seed: int = 0) -> list[dict]:
    """Mean absolute error of every estimator, regime and budget against the exact Shapley values."""
    rows = []
    for budget in budgets:
        for regime, outcomes, is_coupled in (("coupled", coupled, True), ("independent", independent, False)):
            for name in ("permutation", "kernelshap", "plugin_exact", "loo", "loi"):
                errors, spent = [], []
                for repetition in range(repetitions):
                    rng = random.Random(seed * 1000003 + repetition)
                    estimate, runs = estimate_once(name, outcomes, is_coupled, players, budget, rng)
                    errors.append(float(np.mean([abs(estimate[p] - truth[p]) for p in players])))
                    spent.append(runs)
                rows.append({"budget": budget, "regime": regime, "estimator": name, "mae": float(np.mean(errors)),
                             "mae_sd": float(np.std(errors)), "runs": float(np.mean(spent))})
    return rows
