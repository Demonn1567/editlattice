"""Type-I error of the e-processes by simulation (must stay ≤ α under optional stopping)."""

from __future__ import annotations

import math
import random

from editlattice.stats.eprocess import PaceSignEProcess, paired_difference_test, run_sequential_gate, to_unit_interval

ALPHA = 0.05


def binomial_upper(rate: float, n: int, z: float = 3.29) -> float:
    """Upper end of a normal-approximation 99.9% interval for a Monte Carlo rate."""
    return rate + z * math.sqrt(max(rate * (1.0 - rate), 1e-12) / n)


def draw_difference(rng: random.Random, p_plus: float, p_minus: float) -> int:
    """Sample D ∈ {−1, 0, 1}."""
    u = rng.random()
    if u < p_plus:
        return 1
    if u < p_plus + p_minus:
        return -1
    return 0


def crossing_rate(p_plus: float, p_minus: float, null_shift: float, paths: int, steps: int, seed: int) -> float:
    """Fraction of null paths whose e-value ever reaches 1/α."""
    rng = random.Random(seed)
    crossings = 0
    for _ in range(paths):
        test = paired_difference_test(null_shift, ALPHA)
        for _ in range(steps):
            test.update(to_unit_interval(draw_difference(rng, p_plus, p_minus)))
            if test.rejects_null:
                crossings += 1
                break
    return crossings / paths


def test_bounded_mean_type_one_error_at_the_null_boundary() -> None:
    scenarios = [
        (0.05, 0.05, 0.0),   # CRN-like: mostly ties, mean exactly 0
        (0.30, 0.30, 0.0),   # noisy, mean 0
        (0.10, 0.05, 0.05),  # bias-shifted null: mean exactly 0.05
        (0.50, 0.50, 0.0),   # dense ±1
    ]
    for index, (p_plus, p_minus, shift) in enumerate(scenarios):
        assert abs((p_plus - p_minus) - shift) < 1e-12
        rate = crossing_rate(p_plus, p_minus, shift, paths=600, steps=600, seed=11 + index)
        assert rate <= ALPHA or binomial_upper(ALPHA, 600) >= rate, (p_plus, p_minus, shift, rate)


def test_pace_type_one_error() -> None:
    rng = random.Random(7)
    paths = 2000
    crossings = 0
    for _ in range(paths):
        test = PaceSignEProcess(alpha=ALPHA, bet=0.5)
        for _ in range(500):
            win = 1 if rng.random() < 0.5 else 0
            test.update(win, 1 - win)
            if test.rejects_null:
                crossings += 1
                break
    assert crossings / paths <= binomial_upper(ALPHA, paths)


def test_gate_has_power_against_a_real_improvement() -> None:
    rng = random.Random(3)
    differences = [draw_difference(rng, 0.25, 0.05) for _ in range(400)]
    result = run_sequential_gate(differences, null_shift=0.0, alpha_accept=ALPHA, alpha_futility=ALPHA)
    assert result.decision == "ACCEPT"
    assert result.n_observations < 200


def test_gate_stops_for_futility_on_a_harmful_edit() -> None:
    rng = random.Random(4)
    differences = [draw_difference(rng, 0.05, 0.25) for _ in range(400)]
    result = run_sequential_gate(differences, null_shift=0.0, alpha_accept=ALPHA, alpha_futility=ALPHA)
    assert result.decision == "REJECT_FUTILITY"
