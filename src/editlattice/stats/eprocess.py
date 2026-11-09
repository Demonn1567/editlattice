"""Anytime-valid sequential tests (testing by betting).

Two e-processes:

* :class:`PaceSignEProcess` - PACE (arXiv 2606.08106, Alg. 1, Eq. 2): bet a constant
  fraction λ on each McNemar-discordant pair being a win; ties are discarded.
  H0: P(win | discordant) ≤ 1/2.
* :class:`BoundedMeanEProcess` - one-sided test of H0: E[Y] ≤ μ0 for Y ∈ [0, 1] with a
  predictable betting fraction (Waudby-Smith & Ramdas, JRSS-B 2024). Capital
  K_t = Π_i (1 + λ_i (Y_i − μ0)) with λ_i ∈ [0, c/μ0] chosen from data before Y_i.
  Under H0 each factor has conditional mean ≤ 1 and is ≥ 0 (because λ_i ≤ 1/μ0), so
  K is a non-negative supermartingale and Ville's inequality gives
  P(sup_t K_t ≥ 1/α) ≤ α.

:func:`paired_difference_test` maps paired differences D ∈ [−1, 1] to Y = (D + 1)/2 and
tests H0: E[D] ≤ m0 via μ0 = (m0 + 1)/2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class PaceSignEProcess:
    """PACE's discordant-pair betting e-process with a constant bet."""

    alpha: float = 0.05
    bet: float = 0.5
    wealth: float = 1.0
    n_discordant: int = 0
    n_wins: int = 0

    def update(self, child_correct: int, parent_correct: int) -> None:
        """Feed one paired outcome; ties leave the wealth unchanged."""
        if child_correct == parent_correct:
            return
        win = 1 if child_correct > parent_correct else 0
        self.n_discordant += 1
        self.n_wins += win
        self.wealth *= 1.0 + self.bet * (2 * win - 1)

    @property
    def rejects_null(self) -> bool:
        """True once the evidence crosses 1/α (commit)."""
        return self.wealth >= 1.0 / self.alpha


@dataclass
class BoundedMeanEProcess:
    """Betting e-process for H0: E[Y] ≤ μ0, Y ∈ [0, 1] (aGRAPA bets by default).

    aGRAPA (approximate growth-rate adaptive to the particular alternative) bets
    λ_t = (μ̂_{t−1} − μ0) / (σ̂²_{t−1} + (μ̂_{t−1} − μ0)²), clipped to [0, c/μ0], where
    μ̂ and σ̂² are running estimates with a prior pseudo-observation (μ̂_0 = 1/2,
    σ̂²_0 = 1/4), as in Waudby-Smith & Ramdas (2024, Sec. 5).
    """

    mu0: float
    alpha: float = 0.05
    clip: float = 0.5
    log_wealth: float = 0.0
    n: int = 0
    _sum: float = 0.5
    _sum_squares_dev: float = 0.25
    _count: float = 1.0
    history: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0.0 <= self.mu0 <= 1.0:
            raise ValueError(f"mu0 must be in [0, 1], got {self.mu0}")

    def next_bet(self) -> float:
        """Predictable betting fraction for the next observation."""
        if self.mu0 >= 1.0:
            return 0.0
        mean = self._sum / self._count
        variance = self._sum_squares_dev / self._count
        edge = mean - self.mu0
        if edge <= 0.0:
            return 0.0
        raw = edge / (variance + edge * edge)
        upper = self.clip / self.mu0 if self.mu0 > 0.0 else raw
        return max(0.0, min(raw, upper))

    def update(self, value: float) -> None:
        """Feed one observation Y ∈ [0, 1]."""
        if value < 0.0 or value > 1.0:
            raise ValueError(f"observation {value} outside [0, 1]")
        bet = self.next_bet()
        factor = 1.0 + bet * (value - self.mu0)
        self.log_wealth += math.log(factor)
        self.n += 1
        mean_before = self._sum / self._count
        self._sum += value
        self._count += 1.0
        mean_after = self._sum / self._count
        self._sum_squares_dev += (value - mean_before) * (value - mean_after)

    @property
    def wealth(self) -> float:
        """Current e-value."""
        return math.exp(self.log_wealth)

    @property
    def rejects_null(self) -> bool:
        """True once the e-value crosses 1/α."""
        return self.log_wealth >= math.log(1.0 / self.alpha)


def paired_difference_test(null_shift: float, alpha: float, clip: float = 0.5) -> BoundedMeanEProcess:
    """E-process for H0: E[D] ≤ null_shift with D ∈ [−1, 1] (feed (D + 1) / 2)."""
    shift = max(-1.0, min(1.0, null_shift))
    return BoundedMeanEProcess(mu0=(shift + 1.0) / 2.0, alpha=alpha, clip=clip)


def to_unit_interval(difference: float) -> float:
    """Map D ∈ [−1, 1] to Y ∈ [0, 1]."""
    return (difference + 1.0) / 2.0


@dataclass
class SequentialGateResult:
    """Outcome of running a two-sided sequential gate."""

    decision: str
    n_observations: int
    accept_wealth: float
    futility_wealth: float


def run_sequential_gate(
    differences: list[float],
    null_shift: float,
    alpha_accept: float,
    alpha_futility: float,
    futility_shift: float = 0.0,
    clip: float = 0.5,
) -> SequentialGateResult:
    """Consume paired differences until acceptance, futility, or exhaustion.

    Accept when the e-process for H0: E[D] ≤ null_shift crosses 1/α_accept.
    Stop for futility when the mirrored e-process for H0': E[D] ≥ futility_shift
    crosses 1/α_futility (evidence that the child is not better). Exhaustion rejects.
    """
    accept_test = paired_difference_test(null_shift, alpha_accept, clip)
    futility_test = BoundedMeanEProcess(mu0=(1.0 - max(-1.0, min(1.0, futility_shift))) / 2.0, alpha=alpha_futility, clip=clip)
    count = 0
    for difference in differences:
        count += 1
        accept_test.update(to_unit_interval(difference))
        futility_test.update((1.0 - difference) / 2.0)
        if accept_test.rejects_null:
            return SequentialGateResult("ACCEPT", count, accept_test.wealth, futility_test.wealth)
        if futility_test.rejects_null:
            return SequentialGateResult("REJECT_FUTILITY", count, accept_test.wealth, futility_test.wealth)
    return SequentialGateResult("REJECT_BUDGET", count, accept_test.wealth, futility_test.wealth)
