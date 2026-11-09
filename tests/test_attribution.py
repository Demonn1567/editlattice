"""Attribution on games with known answers."""

import itertools
import math

from editlattice.attribution import all_subsets, attribute, bootstrap, lattice_values


def game(function, players):
    return {subset: function(subset) for subset in all_subsets(players)}


def test_additive_game_has_no_interactions() -> None:
    weights = {0: 0.1, 1: -0.05, 2: 0.3}
    result = attribute(game(lambda s: sum(weights[i] for i in s), [0, 1, 2]), [0, 1, 2])
    for player, weight in weights.items():
        assert math.isclose(result.shapley[player], weight, abs_tol=1e-12)
        assert math.isclose(result.loo[player], weight, abs_tol=1e-12)
    assert all(abs(value) < 1e-12 for value in result.interaction.values())
    assert all(abs(value) < 1e-12 for subset, value in result.mobius.items() if len(subset) > 1)


def test_pure_synergy_splits_credit_and_shows_interaction() -> None:
    result = attribute(game(lambda s: 1.0 if {0, 1} <= s else 0.0, [0, 1, 2]), [0, 1, 2])
    assert math.isclose(result.shapley[0], 0.5) and math.isclose(result.shapley[1], 0.5) and abs(result.shapley[2]) < 1e-12
    assert math.isclose(result.interaction[(0, 1)], 1.0)
    assert math.isclose(result.mobius[frozenset({0, 1})], 1.0)
    assert result.loi[0] == 0.0  # leave-one-in misses a pure synergy


def test_efficiency_on_a_random_game() -> None:
    import random
    rng = random.Random(1)
    players = [0, 1, 2, 3]
    values = {subset: rng.random() for subset in all_subsets(players)}
    result = attribute(values, players)
    assert math.isclose(sum(result.shapley.values()), values[frozenset(players)] - values[frozenset()], abs_tol=1e-12)
    # Möbius dividends rebuild every value.
    for subset in all_subsets(players):
        assert math.isclose(sum(result.mobius[t] for t in all_subsets(sorted(subset))), values[subset], abs_tol=1e-12)


def test_lattice_values_and_bootstrap_shapes() -> None:
    players = [0, 1]
    outcomes = {}
    for task, replicate in itertools.product(["a", "b", "c"], [0, 1]):
        outcomes[(task, replicate)] = {s: float(0 in s) for s in all_subsets(players)}
    values = lattice_values(outcomes)
    assert values[frozenset({0})] == 1.0 and values[frozenset()] == 0.0
    intervals = bootstrap(outcomes, players, n_boot=50)
    assert intervals["shapley:0"]["low"] == intervals["shapley:0"]["high"] == 1.0


def test_holm_step_down() -> None:
    from editlattice.attribution import holm
    rejected = holm({"a": 0.001, "b": 0.02, "c": 0.03, "d": 0.5})
    # thresholds 0.0125, 0.0167, 0.025, 0.05: a passes, b fails, so b, c, d are not rejected
    assert rejected == {"a": True, "b": False, "c": False, "d": False}
    assert holm({"a": 0.01, "b": 0.02}) == {"a": True, "b": True}
