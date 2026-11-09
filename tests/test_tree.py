"""Tree exploration and closed-form tree attribution agree exactly with the exhaustive lattice.

Fuzz harnesses from test_fuzz_exactness: on every cell, every product's leaf record equals the
plain run of that product; the leaves partition the products; executions never exceed the
exhaustive explorer's; and Shapley values, interaction indices, Banzhaf values, Moebius
dividends, LOO/LOI and the MILP optima equal the exhaustive computations on the full lattice.
"""

from __future__ import annotations

import itertools
import random
from pathlib import Path

import pytest

from editlattice.attribution import mobius, shapley, shapley_interaction
from editlattice.harness_runtime.runner import HarnessVersion, run_one
from editlattice.llm.cache import ResponseCache
from editlattice.llm.client import LLMClient
from editlattice.llm.fake import FakeBackend
from editlattice.tree import Leaf, TreeLattice, explore_tree, leaf_interaction, leaf_shapley
from editlattice.units import build_units
from editlattice.variational import Instrumented, explore_cell
from tests.test_fuzz_exactness import COMPARED, MUTATIONS, SEED_TEMPLATE, TASKS, Bench, make_case, responder


def exhaustive_values(lattice: TreeLattice) -> dict[frozenset[int], float]:
    players = lattice.players
    return {frozenset(c): lattice.value(frozenset(c)) for r in range(len(players) + 1) for c in itertools.combinations(players, r)}


def banzhaf(values: dict[frozenset[int], float], players: list[int]) -> dict[int, float]:
    others = {p: [q for q in players if q != p] for p in players}
    return {p: sum(values[frozenset(c) | {p}] - values[frozenset(c)] for r in range(len(players)) for c in itertools.combinations(others[p], r))
            / 2 ** (len(players) - 1) for p in players}


@pytest.mark.parametrize("statement_level", [False, True])
def test_tree_exploration_is_exact_and_never_costlier(tmp_path: Path, statement_level: bool) -> None:
    rng = random.Random(7)
    client = LLMClient(backend=FakeBackend(responder), cache=ResponseCache(tmp_path / "cache"), model="fake")
    bench = Bench()
    for case in range(15):
        seed, final = make_case(rng)
        units = build_units(seed, final)
        lazy = Instrumented.build(units, statement_level=statement_level, lazy=True)
        eager = Instrumented.build(units, statement_level=statement_level)
        players = [unit.index for unit in units.units]
        cells = {}
        for task in TASKS:
            for replicate in range(2):
                leaves = explore_tree(bench, task, lazy, replicate, client, {}, 6)
                _, executed = explore_cell(bench, task, eager, replicate, client, {}, 6)
                assert len(leaves) == len(executed)  # the same read-class partition, found without enumerating products
                cells[(task["task_id"], replicate)] = leaves
                for subset in units.all_subsets():
                    leaf = [leaf for leaf in leaves if leaf.contains(subset)]
                    assert len(leaf) == 1, (case, sorted(subset))
                    reference = run_one(bench, task, HarnessVersion.from_source(units.product(subset), tmp_path / "store"),
                                        replicate, client, {}, crn=True, max_llm_calls=6)
                    for field in COMPARED:
                        assert leaf[0].record[field] == reference[field], (case, sorted(subset), field)
        lattice = TreeLattice(players=players, cells=cells)
        lattice.check_partition()
        values = exhaustive_values(lattice)
        for player, value in shapley(values, players).items():
            assert lattice.shapley()[player] == pytest.approx(value, abs=1e-12)
        for pair, value in shapley_interaction(values, players).items():
            assert lattice.interaction()[pair] == pytest.approx(value, abs=1e-12)
        for player, value in banzhaf(values, players).items():
            assert lattice.banzhaf()[player] == pytest.approx(value, abs=1e-12)
        for target, value in mobius(values, players).items():
            assert lattice.mobius(target) == pytest.approx(value, abs=1e-12)
        best = max(values.values())
        floor = values[frozenset(players)] - 0.05
        admissible = [s for s, v in values.items() if v >= floor - 1e-12]
        for method in ("milp", "enumerate"):
            chosen, value = lattice.optimise(method=method)
            assert value == pytest.approx(best) and len(chosen) == min(len(s) for s, v in values.items() if v == pytest.approx(best))
            chosen, value = lattice.optimise(minimise_size_with_floor=floor, method=method)
            assert len(chosen) == min(len(s) for s in admissible) and value >= floor - 1e-12
        enumerated = lattice.all_values()
        for subset, value in values.items():
            mask = sum(1 << k for k, p in enumerate(players) if p in subset)
            assert enumerated[mask] == pytest.approx(value, abs=1e-12)


def test_closed_forms_on_random_leaves() -> None:
    """Single-leaf games against brute force, for every (a, b) with a + b <= 7, including dummies."""
    for a in range(5):
        for b in range(5):
            if a + b > 7:
                continue
            players = list(range(a + b + 2))  # two dummy players
            leaf = Leaf(ones=frozenset(range(a)), zeros=frozenset(range(a, a + b)), outcome=0.7)
            values = {frozenset(c): (0.7 if leaf.contains(frozenset(c)) else 0.0)
                      for r in range(len(players) + 1) for c in itertools.combinations(players, r)}
            exact = shapley(values, players)
            closed = leaf_shapley(leaf)
            for p in players:
                assert closed.get(p, 0.0) == pytest.approx(exact[p], abs=1e-12), (a, b, p)
            exact_pairs = shapley_interaction(values, players)
            closed_pairs = leaf_interaction(leaf)
            for pair, value in exact_pairs.items():
                assert closed_pairs.get(pair, 0.0) == pytest.approx(value, abs=1e-12), (a, b, pair)


def test_determinism_violation_is_detected(tmp_path: Path) -> None:
    """Negative control: a run whose reads do not follow its branch prefix raises."""
    from editlattice.tree import DeterminismViolation
    import editlattice.tree as tree_module

    client = LLMClient(backend=FakeBackend(responder), cache=ResponseCache(tmp_path / "cache"), model="fake")
    final = SEED_TEMPLATE
    for old, new in MUTATIONS:
        final = final.replace(old, new)
    units = build_units(SEED_TEMPLATE, final)
    lazy = Instrumented.build(units, lazy=True)
    real = tree_module.run_product
    calls = {"n": 0}

    def shuffled(*args, **kwargs):
        record = real(*args, **kwargs)
        calls["n"] += 1
        if calls["n"] > 1 and len(record["read_order"]) > 1:
            record["read_order"] = list(reversed(record["read_order"]))
        return record

    tree_module.run_product = shuffled
    try:
        with pytest.raises(DeterminismViolation):
            for task in TASKS:
                explore_tree(Bench(), task, lazy, 0, client, {}, 6)
    finally:
        tree_module.run_product = real


def test_compositional_analysis_equals_exhaustive() -> None:
    """Group-by-group static analysis gives exactly the exhaustive read model and markers."""
    from editlattice.units import add_markers, independent_groups
    import editlattice.units as units_module
    import editlattice.variational as variational_module
    from editlattice.variational import ReadModel

    rng = random.Random(11)
    multi = 0
    finals = [make_case(rng)[1] for _ in range(30)]
    everything = SEED_TEMPLATE
    for old, new in MUTATIONS:
        everything = everything.replace(old, new)
    for final in finals + [everything]:
        units = build_units(SEED_TEMPLATE, final)
        groups = independent_groups(units)
        assert groups is not None
        multi += len(groups) > 1
        compositional = ReadModel.build(units)
        marked_c = add_markers(units)
        exhaustive_all = units_module.analysis_subsets
        try:
            units_module.analysis_subsets = lambda unit_set: unit_set.all_subsets()
            variational_module.analysis_subsets = units_module.analysis_subsets
            exhaustive = ReadModel.build(units)
            marked_e = add_markers(units)
        finally:
            units_module.analysis_subsets = exhaustive_all
            variational_module.analysis_subsets = exhaustive_all
        assert compositional.depends == exhaustive.depends
        assert compositional.global_units == exhaustive.global_units and compositional.dynamic == exhaustive.dynamic
        assert marked_c[1] == marked_e[1] and marked_c[0].seed == marked_e[0].seed
    assert multi >= 20  # the comparison is not vacuous


def test_invalid_lazy_product_raises(tmp_path: Path) -> None:
    """Negative control: a product that is not a valid harness stops exploration instead of counting as a failure."""
    from editlattice.units import Hunk, Unit, UnitSet
    from editlattice.variational import InvalidProduct

    seed = 'def solve(task, llm, env):\n    return {"answer": "alpha"}\n'
    broken = UnitSet(seed=seed, units=[Unit(0, (Hunk(0, 1, ("def solved(task, llm, env):\n",)),), ("solve",))])
    client = LLMClient(backend=FakeBackend(responder), cache=ResponseCache(tmp_path / "cache"), model="fake")
    instrumented = Instrumented(model=__import__("editlattice.variational", fromlist=["ReadModel"]).ReadModel(unit_set=broken, global_units=frozenset({0})),
                                sources={}, unit_set=broken, names=frozenset())
    with pytest.raises(InvalidProduct):
        explore_tree(Bench(), TASKS[0], instrumented, 0, client, {}, 6)
