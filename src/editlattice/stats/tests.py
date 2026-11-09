"""Classical paired tests used by baseline gates."""

from __future__ import annotations

import math
import random

from scipy.stats import binomtest, ttest_1samp


def mcnemar_exact_one_sided(child_wins: int, parent_wins: int) -> float:
    """p-value of the exact McNemar test for 'child better' (binomial on discordant pairs)."""
    discordant = child_wins + parent_wins
    if discordant == 0:
        return 1.0
    return float(binomtest(child_wins, discordant, 0.5, alternative="greater").pvalue)


def paired_t_one_sided(differences: list[float]) -> float:
    """p-value of a one-sided one-sample t-test of mean(differences) > 0."""
    if len(differences) < 2:
        return 1.0
    spread = max(differences) - min(differences)
    if spread == 0.0:
        return 0.0 if differences[0] > 0 else 1.0
    result = ttest_1samp(differences, 0.0, alternative="greater")
    if math.isnan(result.pvalue):
        return 1.0
    return float(result.pvalue)


def bootstrap_lower_bound(values: list[float], alpha: float, n_boot: int, rng: random.Random) -> float:
    """One-sided percentile-bootstrap lower bound of the mean at level 1 − alpha."""
    if not values:
        return -math.inf
    size = len(values)
    means: list[float] = []
    for _ in range(n_boot):
        total = 0.0
        for _ in range(size):
            total += values[rng.randrange(size)]
        means.append(total / size)
    means.sort()
    index = max(0, int(math.floor(alpha * n_boot)) - 1)
    return means[index]
