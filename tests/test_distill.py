"""Distillation methods on lattices with known answers."""

import random

from editlattice.attribution import all_subsets
from editlattice.distill import all_methods, certify_non_inferior, ddmin, exact_minimum, greedy_backward


def lattice(function, players):
    return {subset: function(subset) for subset in all_subsets(players)}


def test_only_one_unit_matters() -> None:
    players = [0, 1, 2, 3]
    values = lattice(lambda s: 0.6 if 2 in s else 0.3, players)
    for choice in all_methods(values, players, delta=0.01):
        assert choice.product == frozenset({2}), choice


def test_synergy_defeats_greedy_but_not_exact() -> None:
    # Units 0 and 1 only help together; unit 2 alone gives most of the gain and is cheaper.
    players = [0, 1, 2]
    values = lattice(lambda s: 0.3 + 0.25 * ({0, 1} <= s) + 0.2 * (2 in s) - 0.01 * len(s), players)
    exact = exact_minimum(values, players, delta=0.06)
    assert values[exact.product] >= values[frozenset(players)] - 0.06
    assert len(exact.product) <= len(greedy_backward(values, players, delta=0.06).product)
    assert ddmin(values, players, delta=0.06).product  # never empty here


def test_certification_power_under_replay() -> None:
    rng = random.Random(0)
    # Under replay a distilled harness and the full one differ in few runs: most pairs are 0.
    same = [rng.choice([0] * 18 + [1, -1]) for _ in range(1200)]
    certified, used = certify_non_inferior(same, delta=0.05)
    assert certified and used < 800
    worse = [rng.choice([-1, -1, 0, 0, 0]) for _ in range(1200)]  # mean -0.4
    assert not certify_non_inferior(worse, delta=0.1)[0]


def test_certification_type_one_error_at_the_margin() -> None:
    """A harness exactly delta worse is certified at most alpha of the time."""
    rng = random.Random(1)
    delta, certified = 0.05, 0
    for _ in range(300):
        # P(-1) - P(+1) = delta, mostly zeros, as under replay.
        pairs = [-1 if (u := rng.random()) < 0.075 else 1 if u < 0.10 else 0 for _ in range(800)]
        certified += certify_non_inferior(pairs, delta=delta)[0]
    assert certified / 300 <= 0.05 + 0.03
