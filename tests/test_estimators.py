"""Estimators on a synthetic lattice with known Shapley values."""

import random

from editlattice.attribution import all_subsets, shapley
from editlattice.estimators import ValueSampler, compare, kernel_shap, permutation_shapley


def synthetic(players, n_cells=400, seed=0, coupled_noise=0.02):
    """Outcome pools: per-cell Bernoulli outcomes around a known value function with a synergy."""
    rng = random.Random(seed)
    def value(s):
        return 0.3 + 0.1 * (0 in s) + 0.15 * ({1, 2} <= s) - 0.05 * (3 in s)
    base = [rng.random() for _ in range(n_cells)]
    coupled, independent = {}, {}
    for s in all_subsets(players):
        p = value(s)
        coupled[s] = [float(u < p) for u in base]  # shared uniform draw per cell = coupling
        independent[s] = [float(rng.random() < p) for _ in range(n_cells)]
    truth = shapley({s: sum(coupled[s]) / n_cells for s in coupled}, players)
    return coupled, independent, truth


def test_estimators_converge_with_full_information() -> None:
    players = [0, 1, 2, 3]
    coupled, _, truth = synthetic(players)
    sampler = ValueSampler(coupled, 400, random.Random(1), coupled=True)
    sampler.shared_cells = list(range(400))  # all cells: plug-in values are exact
    estimate = permutation_shapley(sampler, players, budget=10**7)
    assert all(abs(estimate[p] - truth[p]) < 0.01 for p in players)  # permutation sampling error only
    sampler2 = ValueSampler(coupled, 400, random.Random(2), coupled=True)
    sampler2.shared_cells = list(range(400))
    estimate2 = kernel_shap(sampler2, players, budget=400 * 200)
    assert all(abs(estimate2[p] - truth[p]) < 0.01 for p in players)


def test_coupling_lowers_error_at_equal_budget() -> None:
    players = [0, 1, 2, 3]
    coupled, independent, truth = synthetic(players)
    rows = compare(coupled, independent, players, truth, budgets=[2000], repetitions=20)
    by = {(r["regime"], r["estimator"]): r["mae"] for r in rows}
    assert by[("coupled", "permutation")] < by[("independent", "permutation")]
    assert by[("coupled", "plugin_exact")] < by[("independent", "plugin_exact")]
    assert all(r["runs"] <= 2000 * 1.01 + 400 for r in rows)


def test_no_estimator_exceeds_its_budget() -> None:
    """DEC-12: equal budgets means no estimator, coupled or independent, uses more task-runs than allowed."""
    import random
    from editlattice.attribution import all_subsets, leave_one_in, leave_one_out, shapley
    from editlattice.estimators import ValueSampler, exact_from_samples, kernel_shap, permutation_shapley

    rng = random.Random(3)
    players = [0, 1, 2]
    outcomes = {s: [rng.random() < 0.3 + 0.1 * len(s) for _ in range(200)] for s in all_subsets(players)}
    for coupled in (True, False):
        for budget in (40, 80, 160, 400, 800, 1600):
            for estimator in (permutation_shapley, kernel_shap):
                sampler = ValueSampler(outcomes, cells_per_value=7, rng=random.Random(budget), coupled=coupled)  # 7 does not divide the budgets
                estimator(sampler, players, budget)
                assert sampler.runs <= budget, (estimator.__name__, coupled, budget, sampler.runs)
