"""Certified upper bound on the mean touch probability of unaffected tasks (Lemma 2).

Given n = k_cert·|U| certification runs of the parent on the unaffected tasks U and m
of them touching the impact set, the one-sided Clopper-Pearson upper limit p⁺(m) bounds
the mean touch probability p̄_U with probability ≥ 1 − α_b, even though the per-task
probabilities differ (Jensen for m = 0; Hoeffding 1956 for m ≥ 1; see notes/theory.md).
"""

from __future__ import annotations

from scipy.stats import beta


def clopper_pearson_upper(successes: int, trials: int, alpha: float) -> float:
    """One-sided (1 − alpha) Clopper-Pearson upper limit for a binomial proportion."""
    if trials <= 0:
        return 1.0
    if successes >= trials:
        return 1.0
    if successes == 0:
        return 1.0 - alpha ** (1.0 / trials)
    return float(beta.ppf(1.0 - alpha, successes + 1, trials - successes))


def proviso_holds(successes: int, trials: int, alpha: float) -> bool:
    """Hoeffding's condition used in Lemma 2: p⁺(m) ≥ (m + 1) / n."""
    if successes >= trials:
        return True
    return clopper_pearson_upper(successes, trials, alpha) >= (successes + 1) / trials


def certified_touch_upper(successes: int, trials: int, alpha: float) -> float:
    """p⁺′(m) = max(p⁺(m), (m + 1)/n), valid for every m (notes/theory.md, Lemma 2).

    The Clopper-Pearson limit alone violates Hoeffding's proviso only at m = n − 1;
    taking the maximum with (m + 1)/n restores it everywhere.
    """
    if trials <= 0:
        return 1.0
    return min(1.0, max(clopper_pearson_upper(successes, trials, alpha), (successes + 1) / trials))


def suite_bias_bound(n_unaffected: int, n_tasks: int, cert_touches: int, k_cert: int, alpha_b: float) -> float:
    """b̂ = (|U|/N) · p⁺′(M): bound on |R| holding with probability ≥ 1 − α_b."""
    if n_unaffected == 0:
        return 0.0
    upper = certified_touch_upper(cert_touches, k_cert * n_unaffected, alpha_b)
    return (n_unaffected / n_tasks) * upper
