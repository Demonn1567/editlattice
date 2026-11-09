"""Numerical checks of Lemma 2 (certified bound on the mean touch probability)."""

from __future__ import annotations

import numpy as np
from scipy.stats import binom

from editlattice.stats.bias_bound import certified_touch_upper, clopper_pearson_upper, proviso_holds


def poisson_binomial_cdf(probabilities: np.ndarray) -> np.ndarray:
    """Exact CDF of a sum of independent Bernoulli(p_i) by dynamic programming."""
    distribution = np.zeros(len(probabilities) + 1)
    distribution[0] = 1.0
    for p in probabilities:
        shifted = np.zeros_like(distribution)
        shifted[1:] = distribution[:-1] * p
        distribution = distribution * (1.0 - p) + shifted
    return np.cumsum(distribution)


def test_proviso_fails_only_at_m_equals_n_minus_one_and_fix_restores_it() -> None:
    for alpha in (0.01, 0.05, 0.1):
        for n in (5, 20, 60, 200, 1000):
            for m in range(0, n):
                if m < n - 1:
                    assert proviso_holds(m, n, alpha), (alpha, n, m)
                assert certified_touch_upper(m, n, alpha) >= (m + 1) / n


def test_hoeffding_extremal_inequality_numerically() -> None:
    rng = np.random.default_rng(20261017)
    for trial in range(400):
        n = int(rng.integers(5, 120))
        kind = trial % 3
        if kind == 0:
            p = rng.uniform(0.0, 1.0, size=n) ** 3
        elif kind == 1:
            p = np.zeros(n)
            heavy = rng.choice(n, size=max(1, n // 10), replace=False)
            p[heavy] = rng.uniform(0.3, 1.0, size=len(heavy))
        else:
            p = np.full(n, rng.uniform(0.0, 0.2))
            p[: int(rng.integers(1, n))] = 1.0
        mean = float(p.mean())
        cdf = poisson_binomial_cdf(p)
        for m in range(0, n + 1):
            if m <= n * mean - 1:
                assert cdf[m] <= binom.cdf(m, n, mean) + 1e-12, (trial, n, m)


def test_certification_coverage_monte_carlo() -> None:
    rng = np.random.default_rng(7)
    alpha = 0.05
    for scenario in range(6):
        n_unaffected = int(rng.integers(20, 150))
        k_cert = int(rng.integers(1, 5))
        p = np.zeros(n_unaffected)
        heavy = rng.choice(n_unaffected, size=max(1, n_unaffected // 8), replace=False)
        p[heavy] = rng.uniform(0.0, 0.25, size=len(heavy))
        mean = float(p.mean())
        violations = 0
        repetitions = 4000
        for _ in range(repetitions):
            touches = int(rng.binomial(k_cert, p).sum())
            if mean > certified_touch_upper(touches, k_cert * n_unaffected, alpha):
                violations += 1
        rate = violations / repetitions
        assert rate <= alpha + 3.29 * np.sqrt(alpha * (1 - alpha) / repetitions), (scenario, rate)
