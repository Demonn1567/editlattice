"""Tree-structured variational replay and exact attribution from the trees (design D11).

On one cell (task, stream) every product's run is determined by the values of the units it
reads, in the order it first reads them (Lemma 2 with ordered reads). Exploring the cell depth
first therefore builds a binary decision tree over units: an internal node is the next unit a
run reads, its two branches are "unit absent" and "unit present", and a leaf is one executed
run, standing for every product that agrees with the leaf's path. The leaves partition the
2^m products, so

* a cell costs one execution per leaf, at most 2^w for a cell whose runs read at most w units,
  whatever m is, and no step enumerates the 2^m products;
* the lattice value v(S) is the mean over cells of the outcome of the leaf containing S: an
  ensemble of decision trees over the 0/1 indicators of the units;
* a leaf with present units A, absent units B and outcome y contributes the game
  y * 1[A in S, B disjoint from S], whose Moebius dividends are (-1)^|C| on A u C for C in B.
  Shapley values, pairwise Shapley interaction indices, Banzhaf values and any dividend follow in
  closed form (Beta integrals), so exact attribution costs O(leaves x depth^2), not O(2^m).

The closed forms are those of interventional TreeSHAP for one background point (the seed) and
one explicand (the full harness); here the trees are execution trees, not fitted models.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from editlattice.harness_runtime.runner import Benchmark
from editlattice.llm.client import LLMClient
from editlattice.variational import Instrumented, run_product

Subset = frozenset[int]


class DeterminismViolation(RuntimeError):
    """A run did not read the units of its branch prefix first, in order (assumption A1 fails)."""


@dataclass(frozen=True)
class Leaf:
    """One executed run: the products with every unit of ``ones`` and none of ``zeros``."""

    ones: Subset
    zeros: Subset
    outcome: float
    record: dict[str, Any] = field(compare=False, hash=False, repr=False, default_factory=dict)
    # Root-to-leaf decisions (unit, present) in read order; used for the node formulation of the MILP.
    path: tuple[tuple[int, int], ...] = field(compare=False, default=())

    def contains(self, subset: Subset) -> bool:
        return self.ones <= subset and not (self.zeros & subset)


def explore_tree(benchmark: Benchmark, task: dict[str, Any], instrumented: Instrumented, replicate: int, client: LLMClient,
                 llm_params: dict[str, Any], max_llm_calls: int, namespace: str = "",
                 conflicts: frozenset[frozenset[int]] = frozenset()) -> list[Leaf]:
    """Every feasible product's exact run on one (task, replicate), as the leaves of the decision tree.

    A branch is a prefix of (unit, present) decisions. Its run is the product with the prefix's
    present units and nothing else; every unit the run reads after the prefix opens a sibling
    branch with that unit flipped. Branches whose present units contain a conflicting pair hold
    no feasible product and are not run.
    """
    leaves: list[Leaf] = []
    stack: list[tuple[tuple[int, int], ...]] = [()]
    while stack:
        prefix = stack.pop()
        present = frozenset(unit for unit, value in prefix if value)
        if any(pair <= present for pair in conflicts):
            continue
        record = run_product(benchmark, task, instrumented, present, replicate, client, llm_params, max_llm_calls, namespace)
        order = list(record["read_order"])
        if order[:len(prefix)] != [unit for unit, _ in prefix]:
            raise DeterminismViolation(f"{task['task_id']} replicate {replicate}: branch {prefix} read {order[:len(prefix)]}")
        for position in range(len(prefix), len(order)):
            # Units read after the prefix are absent in this run; the sibling makes the last one present.
            branch = prefix + tuple((unit, 0) for unit in order[len(prefix):position]) + ((order[position], 1),)
            stack.append(branch)
        leaves.append(Leaf(ones=frozenset(u for u in order if u in present), zeros=frozenset(u for u in order if u not in present),
                           outcome=float(record["outcome"]), record=record,
                           path=tuple((unit, int(unit in present)) for unit in order)))
    return leaves


# --------------------------------------------------------------------------- closed forms per leaf

def _beta(p: int, q: int) -> float:
    """B(p, q) = (p-1)!(q-1)!/(p+q-1)! for positive integers."""
    return math.exp(math.lgamma(p) + math.lgamma(q) - math.lgamma(p + q))


def leaf_shapley(leaf: Leaf) -> dict[int, float]:
    """Shapley values of the game y * 1[A in S, B disjoint from S]."""
    a, b, y = len(leaf.ones), len(leaf.zeros), leaf.outcome
    if y == 0 or a + b == 0:
        return {}
    result = {}
    if a:
        share = y * _beta(a, b + 1)
        result.update({unit: share for unit in leaf.ones})
    if b:
        share = -y * _beta(a + 1, b)
        result.update({unit: share for unit in leaf.zeros})
    return result


def leaf_interaction(leaf: Leaf) -> dict[tuple[int, int], float]:
    """Pairwise Shapley interaction indices (Grabisch-Roubens) of one leaf's game."""
    a, b, y = len(leaf.ones), len(leaf.zeros), leaf.outcome
    if y == 0:
        return {}
    ones, zeros = sorted(leaf.ones), sorted(leaf.zeros)
    result: dict[tuple[int, int], float] = {}
    if a >= 2:
        value = y * _beta(a - 1, b + 1)
        for i, first in enumerate(ones):
            for second in ones[i + 1:]:
                result[(first, second)] = value
    if a >= 1 and b >= 1:
        value = -y * _beta(a, b)
        for first in ones:
            for second in zeros:
                result[(min(first, second), max(first, second))] = value
    if b >= 2:
        value = y * _beta(a + 1, b - 1)
        for i, first in enumerate(zeros):
            for second in zeros[i + 1:]:
                result[(first, second)] = value
    return result


def leaf_banzhaf(leaf: Leaf) -> dict[int, float]:
    """Banzhaf values of one leaf's game."""
    a, b, y = len(leaf.ones), len(leaf.zeros), leaf.outcome
    if y == 0 or a + b == 0:
        return {}
    share = y * 2.0 ** (-(a + b - 1))
    return {**{unit: share for unit in leaf.ones}, **{unit: -share for unit in leaf.zeros}}


def leaf_mobius(leaf: Leaf, target: Subset) -> float:
    """Moebius dividend of one leaf's game at ``target``."""
    if not (leaf.ones <= target <= leaf.ones | leaf.zeros):
        return 0.0
    return leaf.outcome * (-1) ** len(target - leaf.ones)


# --------------------------------------------------------------------------- the lattice as a tree ensemble

@dataclass
class TreeLattice:
    """Leaves of every cell; cells are (task_id, replicate)."""

    players: list[int]
    cells: dict[tuple[str, int], list[Leaf]]

    def check_partition(self, sample: int = 2000, seed: int = 0) -> None:
        """Every sampled product lies in exactly one leaf of every cell (all products if 2^m <= sample)."""
        rng = random.Random(seed)
        m = len(self.players)
        if 2 ** m <= sample:
            subsets = [frozenset(p for bit, p in enumerate(self.players) if mask >> bit & 1) for mask in range(2 ** m)]
        else:
            subsets = [frozenset(p for p in self.players if rng.random() < 0.5) for _ in range(sample)]
        for cell, leaves in self.cells.items():
            for subset in subsets:
                hits = sum(1 for leaf in leaves if leaf.contains(subset))
                if hits != 1:
                    raise AssertionError(f"cell {cell}: product {sorted(subset)} lies in {hits} leaves")

    def _weights(self, tasks: list[str] | None) -> dict[tuple[str, int], float]:
        if tasks is None:
            return {cell: 1.0 / len(self.cells) for cell in self.cells}
        counts: dict[str, int] = {}
        for task in tasks:
            counts[task] = counts.get(task, 0) + 1
        cells = [cell for cell in self.cells if cell[0] in counts]
        total = sum(counts[cell[0]] for cell in cells)
        return {cell: counts[cell[0]] / total for cell in cells}

    def value(self, subset: Subset, tasks: list[str] | None = None) -> float:
        """v(S): mean outcome of the leaf containing S, over cells."""
        total = 0.0
        for cell, weight in self._weights(tasks).items():
            leaf = next(leaf for leaf in self.cells[cell] if leaf.contains(subset))
            total += weight * leaf.outcome
        return total

    def _sum(self, per_leaf, tasks: list[str] | None = None) -> dict:
        totals: dict = {}
        for cell, weight in self._weights(tasks).items():
            for leaf in self.cells[cell]:
                for key, value in per_leaf(leaf).items():
                    totals[key] = totals.get(key, 0.0) + weight * value
        return totals

    def shapley(self, tasks: list[str] | None = None) -> dict[int, float]:
        totals = self._sum(leaf_shapley, tasks)
        return {player: totals.get(player, 0.0) for player in self.players}

    def interaction(self, tasks: list[str] | None = None) -> dict[tuple[int, int], float]:
        totals = self._sum(leaf_interaction, tasks)
        ordered = sorted(self.players)
        return {(first, second): totals.get((first, second), 0.0)
                for i, first in enumerate(ordered) for second in ordered[i + 1:]}

    def banzhaf(self, tasks: list[str] | None = None) -> dict[int, float]:
        totals = self._sum(leaf_banzhaf, tasks)
        return {player: totals.get(player, 0.0) for player in self.players}

    def mobius(self, target: Subset, tasks: list[str] | None = None) -> float:
        return sum(weight * sum(leaf_mobius(leaf, target) for leaf in self.cells[cell])
                   for cell, weight in self._weights(tasks).items())

    def loo(self) -> dict[int, float]:
        full = frozenset(self.players)
        top = self.value(full)
        return {player: top - self.value(full - {player}) for player in self.players}

    def loi(self) -> dict[int, float]:
        base = self.value(frozenset())
        return {player: self.value(frozenset({player})) - base for player in self.players}

    def executions(self) -> int:
        return sum(len(leaves) for leaves in self.cells.values())

    def depth(self) -> int:
        """Largest number of units read by one run (the read width w)."""
        return max(len(leaf.ones) + len(leaf.zeros) for leaves in self.cells.values() for leaf in leaves)

    def bootstrap(self, n_boot: int = 2000, seed: int = 0) -> dict[str, dict]:
        """Task-cluster bootstrap of Shapley values and interactions (same scheme as attribution.bootstrap).

        Both are linear in the cells, so each draw reweights per-task totals instead of re-running anything.
        """
        tasks = sorted({cell[0] for cell in self.cells})
        per_task_shapley = {task: self.shapley([task]) for task in tasks}
        per_task_interaction = {task: self.interaction([task]) for task in tasks}
        rng = random.Random(seed)
        draws: dict[str, list[float]] = {}
        for _ in range(n_boot):
            sample = [rng.choice(tasks) for _ in tasks]
            for player in self.players:
                draws.setdefault(f"shapley:{player}", []).append(sum(per_task_shapley[t][player] for t in sample) / len(sample))
            for pair in per_task_interaction[tasks[0]]:
                draws.setdefault(f"interaction:{pair[0]},{pair[1]}", []).append(
                    sum(per_task_interaction[t][pair] for t in sample) / len(sample))
        intervals = {}
        for key, values in draws.items():
            values.sort()
            below = sum(1 for value in values if value <= 0) / n_boot
            above = sum(1 for value in values if value >= 0) / n_boot
            intervals[key] = {"low": values[int(0.025 * n_boot)], "high": values[int(0.975 * n_boot) - 1],
                              "p": min(1.0, 2 * min(below, above))}
        return intervals

    # ----------------------------------------------------------------------- optimisation over products

    def all_values(self, tasks: list[str] | None = None):
        """v of every product as an array indexed by bitmask over ``players`` (bit k = players[k]).

        The leaves of a cell partition the cube, so filling each leaf's subcube costs 2^m per cell in
        total: worthwhile when the trees are dense (2^m not much larger than the number of leaves).
        """
        import numpy as np

        m = len(self.players)
        bit = {player: k for k, player in enumerate(self.players)}
        values = np.zeros(2 ** m)
        for cell, weight in self._weights(tasks).items():
            for leaf in self.cells[cell]:
                if leaf.outcome == 0:
                    continue
                fixed = leaf.ones | leaf.zeros
                base = sum(1 << bit[unit] for unit in leaf.ones)
                offsets = np.zeros(1, dtype=np.int64)
                for player in self.players:
                    if player not in fixed:
                        offsets = np.concatenate([offsets, offsets + (1 << bit[player])])
                values[base + offsets] += weight * leaf.outcome
        return values

    def optimise(self, minimise_size_with_floor: float | None = None, conflicts: frozenset[frozenset[int]] = frozenset(),
                 tasks: list[str] | None = None, method: str = "auto") -> tuple[Subset, float]:
        """Exact optimum over all feasible products: enumeration for dense trees, else a MILP.

        ``auto`` enumerates when 2^m <= 4 x (number of leaves) and m <= 24 (the trees already hold about
        as many runs as there are products), otherwise solves the MILP below.
        """
        m = len(self.players)
        leaves_total = sum(len(self.cells[cell]) for cell in self._weights(tasks))
        if method == "enumerate" or (method == "auto" and m <= 24 and 2 ** m <= 4 * leaves_total):
            return self._optimise_by_enumeration(minimise_size_with_floor, conflicts, tasks)
        return self._optimise_by_milp(minimise_size_with_floor, conflicts, tasks)

    def _optimise_by_enumeration(self, floor: float | None, conflicts: frozenset[frozenset[int]],
                                 tasks: list[str] | None) -> tuple[Subset, float]:
        import numpy as np

        m = len(self.players)
        values = self.all_values(tasks)
        masks = np.arange(2 ** m, dtype=np.int64)
        sizes = np.zeros(2 ** m, dtype=np.int64)
        for k in range(m):
            sizes += (masks >> k) & 1
        feasible = np.ones(2 ** m, dtype=bool)
        bit = {player: k for k, player in enumerate(self.players)}
        for pair in conflicts:
            first, second = sorted(pair)
            both = (1 << bit[first]) | (1 << bit[second])
            feasible &= (masks & both) != both
        # Values are means of 0/1 outcomes over cells: compare on a grid finer than any difference.
        rounded = np.round(values, 12)
        if floor is None:
            candidates = np.flatnonzero(feasible)
            best = rounded[candidates].max()
            chosen = candidates[rounded[candidates] == best]
            choice = chosen[np.lexsort((chosen, sizes[chosen]))[0]]
        else:
            candidates = np.flatnonzero(feasible & (values >= floor - 1e-12))
            smallest = sizes[candidates].min()
            chosen = candidates[sizes[candidates] == smallest]
            choice = chosen[np.lexsort((chosen, -rounded[chosen]))[0]]
        subset = frozenset(player for k, player in enumerate(self.players) if choice >> k & 1)
        return subset, self.value(subset, tasks)

    def _optimise_by_milp(self, minimise_size_with_floor: float | None, conflicts: frozenset[frozenset[int]],
                          tasks: list[str] | None) -> tuple[Subset, float]:
        """Exact optimum over all feasible products by a mixed-integer program on the leaves.

        Variables: x_u (unit present) and z_l (leaf l is the active leaf of its cell). Constraints
        z_l <= x_a (a in A_l), z_l <= 1 - x_b (b in B_l) and sum of z over a cell's leaves = 1: since
        the leaves of a cell partition the cube, integral x forces z to the leaf containing x.
        Without a floor: maximise v(x) (ties: fewer units). With a floor f: minimise |x| subject to
        v(x) >= f (ties: larger v).
        """
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import coo_matrix

        players = list(self.players)
        index = {player: k for k, player in enumerate(players)}
        weights = self._weights(tasks)
        leaves = [(cell, leaf) for cell in weights for leaf in self.cells[cell]]
        n_x, n_z = len(players), len(leaves)
        rows: list[tuple[dict[int, float], float, float]] = []
        if all(leaf.path for _, leaf in leaves if leaf.ones or leaf.zeros):
            # Node formulation (as for tree ensembles, Misic 2020): for every internal node of a
            # cell's tree splitting on u, the leaves below its "present" branch sum to at most x_u
            # and those below its "absent" branch to at most 1 - x_u. Exact for integral x and
            # much tighter than one constraint per (leaf, unit).
            branches: dict[tuple, tuple[int, list[int]]] = {}
            for k, (cell, leaf) in enumerate(leaves):
                for depth, (unit, present) in enumerate(leaf.path):
                    entry = branches.setdefault((cell, leaf.path[:depth], present), (unit, []))
                    entry[1].append(n_x + k)
            for (cell, prefix, present), (unit, columns) in branches.items():
                coefficients = {column: 1.0 for column in columns}
                if present:
                    coefficients[index[unit]] = -1.0
                    rows.append((coefficients, -np.inf, 0.0))
                else:
                    coefficients[index[unit]] = 1.0
                    rows.append((coefficients, -np.inf, 1.0))
        else:
            for k, (cell, leaf) in enumerate(leaves):
                for unit in leaf.ones:
                    rows.append(({n_x + k: 1.0, index[unit]: -1.0}, -np.inf, 0.0))
                for unit in leaf.zeros:
                    rows.append(({n_x + k: 1.0, index[unit]: 1.0}, -np.inf, 1.0))
        for cell in weights:
            rows.append(({n_x + k: 1.0 for k, (c, _) in enumerate(leaves) if c == cell}, 1.0, 1.0))
        for pair in conflicts:
            first, second = sorted(pair)
            rows.append(({index[first]: 1.0, index[second]: 1.0}, -np.inf, 1.0))
        value_row = {n_x + k: weights[cell] * leaf.outcome for k, (cell, leaf) in enumerate(leaves)}
        # Tie-breaks are scaled below the smallest possible difference between two distinct values.
        resolution = min(weights.values()) / (n_x + 2)
        objective = np.zeros(n_x + n_z)
        if minimise_size_with_floor is None:
            for column, coefficient in value_row.items():
                objective[column] = -coefficient
            objective[:n_x] = resolution / (n_x + 1)
        else:
            rows.append((value_row, minimise_size_with_floor - 1e-12, np.inf))
            objective[:n_x] = 1.0
            for column, coefficient in value_row.items():
                objective[column] = -coefficient / (n_x + 1)
        row_index, col_index, data = [], [], []
        lower, upper = np.empty(len(rows)), np.empty(len(rows))
        for r, (coefficients, low, high) in enumerate(rows):
            for column, coefficient in coefficients.items():
                row_index.append(r)
                col_index.append(column)
                data.append(coefficient)
            lower[r], upper[r] = low, high
        matrix = coo_matrix((data, (row_index, col_index)), shape=(len(rows), n_x + n_z))
        integrality = np.concatenate([np.ones(n_x), np.zeros(n_z)])
        result = milp(objective, constraints=LinearConstraint(matrix.tocsr(), lower, upper), integrality=integrality,
                      bounds=Bounds(0, 1), options={"mip_rel_gap": 0.0})
        if not result.success:
            raise RuntimeError(f"MILP failed: {result.message}")
        chosen = frozenset(player for player in players if result.x[index[player]] > 0.5)
        return chosen, self.value(chosen, tasks)
