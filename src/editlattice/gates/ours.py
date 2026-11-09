"""Coupled acceptance gate and its ablation ladder (O1-O5), evaluated on an :class:`EditPool`.

Stream roles for the parent's history (k_char = k_sel + k_cert streams per task):
* streams 0..k_sel−1 define the affected set A (touched in any of them);
* streams k_sel..k_char−1 certify the touch rate on the unaffected set U (Lemma 2).
  On *affected* tasks these streams are independent of A and of the certificate, so
  they are free paired draws for the gate before any new stream is spent.

Draws for the paired tests are taken without replacement from
A × (certification streams ∪ draw streams), so every draw is a new random stream.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from editlattice.gates.pool import EditPool, GateResult, Ledger, independent_parent_stream
from editlattice.stats.bias_bound import suite_bias_bound
from editlattice.stats.eprocess import BoundedMeanEProcess, PaceSignEProcess, paired_difference_test, to_unit_interval
from editlattice.stats.tests import mcnemar_exact_one_sided


@dataclass
class Selection:
    """Affected/unaffected split and the certified bias bound."""

    affected: list[str]
    unaffected: list[str]
    cert_touches: int
    bias_bound: float


def select_and_certify(pool: EditPool, k_sel: int, granularity: str, alpha_b: float) -> Selection:
    """Compute A from selection streams and the bias bound b̂ from certification streams."""
    affected: list[str] = []
    unaffected: list[str] = []
    for task in pool.tasks:
        hit = False
        for stream in range(k_sel):
            if pool.touched(task, stream, granularity):
                hit = True
        if hit:
            affected.append(task)
        else:
            unaffected.append(task)
    k_cert = pool.k_char - k_sel
    touches = 0
    for task in unaffected:
        for stream in range(k_sel, pool.k_char):
            if pool.touched(task, stream, granularity):
                touches += 1
    bound = suite_bias_bound(len(unaffected), len(pool.tasks), touches, k_cert, alpha_b) if k_cert > 0 else len(unaffected) / len(pool.tasks)
    return Selection(affected, unaffected, touches, bound)


def candidate_draws(pool: EditPool, affected: list[str], k_sel: int, rng: random.Random) -> list[tuple[str, int]]:
    """All (task, stream) pairs usable for paired draws, in random order."""
    pairs: list[tuple[str, int]] = []
    for task in affected:
        for stream in range(k_sel, pool.n_streams):
            pairs.append((task, stream))
    rng.shuffle(pairs)
    return pairs


def charge_crn_pair(pool: EditPool, task: str, stream: int, ledger: Ledger) -> int:
    """Run parent and child on the same stream; return D = C′ − C.

    The parent run is free on its history streams; the child always pays CRN cost.
    """
    parent_run = pool.parent[task][stream]
    child_run = pool.child[task][stream]
    if stream >= pool.k_char:
        ledger.charge_full(parent_run)
    ledger.charge_crn(child_run)
    return child_run.outcome - parent_run.outcome


def charge_independent_pair(pool: EditPool, task: str, stream: int, rng: random.Random, ledger: Ledger) -> int:
    """Child on ``stream`` (full cost), parent on an independent stream; return D."""
    child_run = pool.child[task][stream]
    ledger.charge_full(child_run)
    parent_stream = independent_parent_stream(rng, pool, stream)
    parent_run = pool.parent[task][parent_stream]
    ledger.charge_full(parent_run)
    return child_run.outcome - parent_run.outcome


def gate_impact_mcnemar(pool: EditPool, rng: random.Random, k_sel: int = 2, alpha: float = 0.05, crn: bool = False, granularity: str = "function") -> GateResult:
    """O1 (crn=False) / O2 (crn=True): impact selection + fixed-n exact McNemar on A.

    One paired draw per affected task. No bias correction (ablation rungs).
    """
    name = "O2_impact_crn_mcnemar" if crn else "O1_impact_mcnemar"
    ledger = Ledger()
    selection = select_and_certify(pool, k_sel, granularity, 0.05)
    if not selection.affected:
        return GateResult(name, False, ledger, {"affected": 0})
    child_wins = 0
    parent_wins = 0
    for task in selection.affected:
        stream = rng.choice(range(k_sel, pool.n_streams))
        if crn:
            difference = charge_crn_pair(pool, task, stream, ledger)
        else:
            difference = charge_independent_pair(pool, task, stream, rng, ledger)
        if difference > 0:
            child_wins += 1
        elif difference < 0:
            parent_wins += 1
    p_value = mcnemar_exact_one_sided(child_wins, parent_wins)
    return GateResult(name, p_value < alpha, ledger, {"affected": len(selection.affected), "p": p_value})


def gate_impact_sequential(
    pool: EditPool,
    rng: random.Random,
    budget_runs: int,
    k_sel: int = 2,
    alpha_g: float = 0.05,
    alpha_b: float = 0.05,
    alpha_f: float = 0.05,
    crn: bool = True,
    granularity: str = "function",
    use_bias_bound: bool = True,
    test: str = "bounded",
    futility_margin: float = 0.0,
) -> GateResult:
    """O3 (crn=False), O4 (full), O5 (granularity="line"); ``test="pace"`` is the PACE ablation.

    Accept when the e-process for H0: E[D] ≤ m0 = b̂·N/|A| crosses 1/α_g; stop for
    futility when the mirrored e-process for H0′: E[D] ≥ ε_f (``futility_margin``)
    crosses 1/α_f, i.e. when the data show the affected-set effect is below ε_f; reject
    when the draw budget or the candidate draws run out. Futility only ever rejects, so
    it cannot affect the false-acceptance guarantee (Theorem 1).
    """
    if test == "pace":
        name = "O4_pace_test"
    elif granularity == "line":
        name = "O5_coupled_gate_line"
    elif crn:
        name = "O4_coupled_gate"
    else:
        name = "O3_impact_eprocess"
    ledger = Ledger()
    selection = select_and_certify(pool, k_sel, granularity, alpha_b)
    n_affected = len(selection.affected)
    details = {"affected": n_affected, "unaffected": len(selection.unaffected), "bias_bound": selection.bias_bound, "cert_touches": selection.cert_touches}
    if n_affected == 0:
        details["reason"] = "empty affected set"
        return GateResult(name, False, ledger, details)
    shift = selection.bias_bound * len(pool.tasks) / n_affected if use_bias_bound else 0.0
    details["null_shift"] = shift
    if shift >= 1.0:
        details["reason"] = "bias bound too large"
        return GateResult(name, False, ledger, details)
    accept_test = paired_difference_test(shift, alpha_g)
    futility_test = BoundedMeanEProcess(mu0=(1.0 - futility_margin) / 2.0, alpha=alpha_f)
    sign_test = PaceSignEProcess(alpha=alpha_g, bet=0.5)
    draws = 0
    for task, stream in candidate_draws(pool, selection.affected, k_sel, rng):
        if ledger.task_runs >= budget_runs:
            details["reason"] = "budget"
            break
        if crn:
            difference = charge_crn_pair(pool, task, stream, ledger)
        else:
            difference = charge_independent_pair(pool, task, stream, rng, ledger)
        draws += 1
        if test == "pace":
            sign_test.update(1 if difference > 0 else 0, 1 if difference < 0 else 0)
            if sign_test.rejects_null:
                details["draws"] = draws
                return GateResult(name, True, ledger, details)
            continue
        accept_test.update(to_unit_interval(difference))
        futility_test.update((1.0 - difference) / 2.0)
        if accept_test.rejects_null:
            details["draws"] = draws
            return GateResult(name, True, ledger, details)
        if futility_test.rejects_null:
            details["draws"] = draws
            details["reason"] = "futility"
            return GateResult(name, False, ledger, details)
    details["draws"] = draws
    details.setdefault("reason", "exhausted")
    return GateResult(name, False, ledger, details)
