"""Baseline acceptance rules B1-B11, evaluated on an :class:`EditPool`.

Conventions (D-010, D-011): parent history streams (0..k_char−1) are free; every other
run a gate executes is charged. Black-box gates pair a child run with an *independent*
parent run and pay the full nominal cost. B12 (LLM judge) needs live model calls and is
implemented separately.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from editlattice.gates.pool import EditPool, GateResult, Ledger, independent_parent_stream
from editlattice.stats.eprocess import PaceSignEProcess
from editlattice.stats.tests import bootstrap_lower_bound, mcnemar_exact_one_sided, paired_t_one_sided


def child_draw(pool: EditPool, rng: random.Random, task: str, ledger: Ledger) -> int:
    """Execute the child once on a random draw stream (black box, full cost)."""
    stream = rng.choice(pool.draw_streams)
    run = pool.child[task][stream]
    ledger.charge_full(run)
    return run.outcome


def child_draws(pool: EditPool, rng: random.Random, task: str, count: int, ledger: Ledger) -> list[int]:
    """Execute the child ``count`` times on distinct random draw streams."""
    streams = rng.sample(pool.draw_streams, min(count, len(pool.draw_streams)))
    outcomes: list[int] = []
    for stream in streams:
        run = pool.child[task][stream]
        ledger.charge_full(run)
        outcomes.append(run.outcome)
    return outcomes


def parent_history(pool: EditPool, task: str, count: int) -> list[int]:
    """The parent's free characterization outcomes (first ``count`` history streams)."""
    outcomes: list[int] = []
    for stream in range(min(count, pool.k_char)):
        outcomes.append(pool.parent[task][stream].outcome)
    return outcomes


def gate_greedy(pool: EditPool, rng: random.Random) -> GateResult:
    """B1: accept iff the child's single-run search score beats the parent's stored score."""
    ledger = Ledger()
    parent_total = 0
    child_total = 0
    for task in pool.tasks:
        parent_total += pool.parent[task][0].outcome
        child_total += child_draw(pool, rng, task, ledger)
    return GateResult("B1_greedy", child_total > parent_total, ledger, {"parent": parent_total, "child": child_total})


def gate_best_so_far_tokens(pool: EditPool, rng: random.Random) -> GateResult:
    """B2: like B1, ties broken in favour of the child if it used fewer tokens."""
    ledger = Ledger()
    parent_total = 0
    child_total = 0
    parent_tokens = 0
    child_tokens = 0
    for task in pool.tasks:
        parent_run = pool.parent[task][0]
        parent_total += parent_run.outcome
        parent_tokens += parent_run.nominal_tokens
        stream = rng.choice(pool.draw_streams)
        child_run = pool.child[task][stream]
        ledger.charge_full(child_run)
        child_total += child_run.outcome
        child_tokens += child_run.nominal_tokens
    accept = child_total > parent_total or (child_total == parent_total and child_tokens < parent_tokens)
    return GateResult("B2_best_tokens", accept, ledger, {"parent": parent_total, "child": child_total})


def gate_self_harness(pool: EditPool, rng: random.Random, repeats: int = 1) -> GateResult:
    """B3: Self-Harness §3.4 - Δ_in ≥ 0, Δ_ho ≥ 0, max(Δ_in, Δ_ho) > 0 (pass counts summed over repeats).

    D_in / D_ho are the first / second half of the (shuffled, frozen) search tasks.
    """
    ledger = Ledger()
    half = len(pool.tasks) // 2
    splits = {"in": pool.tasks[:half], "ho": pool.tasks[half:]}
    deltas: dict[str, int] = {}
    for name, tasks in splits.items():
        parent_count = 0
        child_count = 0
        for task in tasks:
            parent_count += sum(parent_history(pool, task, repeats))
            child_count += sum(child_draws(pool, rng, task, repeats, ledger))
        deltas[name] = child_count - parent_count
    accept = deltas["in"] >= 0 and deltas["ho"] >= 0 and max(deltas["in"], deltas["ho"]) > 0
    return GateResult(f"B3_self_harness_r{repeats}", accept, ledger, deltas)


def gate_strict_non_regression(pool: EditPool, rng: random.Random) -> GateResult:
    """B3': no task that passed in the parent's stored run fails, and score ≥ parent."""
    ledger = Ledger()
    regressions = 0
    parent_total = 0
    child_total = 0
    for task in pool.tasks:
        parent_outcome = pool.parent[task][0].outcome
        child_outcome = child_draw(pool, rng, task, ledger)
        parent_total += parent_outcome
        child_total += child_outcome
        if parent_outcome == 1 and child_outcome == 0:
            regressions += 1
    accept = regressions == 0 and child_total >= parent_total
    return GateResult("B3p_strict_nonregression", accept, ledger, {"regressions": regressions})


def gate_avg_at_k(pool: EditPool, rng: random.Random, k: int = 3, max_regression: float = 1.0) -> GateResult:
    """B4: DarwinX - g = Σ_t Δ_t > 0 and R = Σ_t (−Δ_t)_+ ≤ δ on avg@k solve rates.

    Parent rates use up to k_char free history runs; any extra parent runs are charged.
    """
    ledger = Ledger()
    gain = 0.0
    regression = 0.0
    for task in pool.tasks:
        parent_outcomes = parent_history(pool, task, k)
        if len(parent_outcomes) < k:
            extra_streams = rng.sample(pool.draw_streams, k - len(parent_outcomes))
            for stream in extra_streams:
                run = pool.parent[task][stream]
                ledger.charge_full(run)
                parent_outcomes.append(run.outcome)
        child_outcomes = child_draws(pool, rng, task, k, ledger)
        delta = sum(child_outcomes) / len(child_outcomes) - sum(parent_outcomes) / len(parent_outcomes)
        gain += delta
        regression += max(0.0, -delta)
    accept = gain > 0 and regression <= max_regression
    return GateResult(f"B4_avg@{k}", accept, ledger, {"gain": gain, "regression": regression})


def gate_mcnemar_full(pool: EditPool, rng: random.Random, alpha: float = 0.05) -> GateResult:
    """B5: exact one-sided McNemar on the full suite, one run per task."""
    ledger = Ledger()
    child_wins = 0
    parent_wins = 0
    for task in pool.tasks:
        parent_outcome = pool.parent[task][0].outcome
        child_outcome = child_draw(pool, rng, task, ledger)
        if child_outcome > parent_outcome:
            child_wins += 1
        elif parent_outcome > child_outcome:
            parent_wins += 1
    p_value = mcnemar_exact_one_sided(child_wins, parent_wins)
    return GateResult("B5_mcnemar", p_value < alpha, ledger, {"p": p_value, "b": child_wins, "c": parent_wins})


def gate_paired_t(pool: EditPool, rng: random.Random, k: int = 3, alpha: float = 0.05, bootstrap: bool = False) -> GateResult:
    """B6: fixed-n paired t-test (or bootstrap CI) on per-task mean differences over k runs."""
    ledger = Ledger()
    differences: list[float] = []
    for task in pool.tasks:
        parent_outcomes = parent_history(pool, task, k)
        child_outcomes = child_draws(pool, rng, task, k, ledger)
        differences.append(sum(child_outcomes) / len(child_outcomes) - sum(parent_outcomes) / len(parent_outcomes))
    if bootstrap:
        lower = bootstrap_lower_bound(differences, alpha, 2000, rng)
        return GateResult(f"B6_bootstrap_k{k}", lower > 0, ledger, {"lower": lower})
    p_value = paired_t_one_sided(differences)
    return GateResult(f"B6_t_k{k}", p_value < alpha, ledger, {"p": p_value})


def gate_fresh_validation(pool: EditPool, rng: random.Random, n_comparisons: int = 20, delta: float = 0.05) -> GateResult:
    """B7: 2610.18813 Eq. 7-8 - accept iff Δ̂(loss) + √(2 ln(KT/δ)/m) < 0 on fresh validation tasks.

    Loss = 1 − outcome. Both agents run once per validation task on independent streams
    (the parent's history covers only the search set, so its validation runs are charged).
    """
    ledger = Ledger()
    if not pool.validation_tasks:
        return GateResult("B7_fresh_validation", False, ledger, {"reason": "no validation pool"})
    total = 0.0
    for task in pool.validation_tasks:
        streams = rng.sample(sorted(pool.validation_child[task]), 2)
        child_run = pool.validation_child[task][streams[0]]
        parent_run = pool.validation_parent[task][streams[1]]
        ledger.charge_full(child_run)
        ledger.charge_full(parent_run)
        total += (1 - child_run.outcome) - (1 - parent_run.outcome)
    m = len(pool.validation_tasks)
    upper = total / m + math.sqrt(2.0 * math.log(n_comparisons / delta) / m)
    return GateResult("B7_fresh_validation", upper < 0, ledger, {"upper": upper})


@dataclass
class TwoGateParameters:
    """Cai et al. Rule 1 thresholds and risk allocation (closest faithful version)."""

    tau: float = 0.02
    delta: float = 0.05
    m: int = 2
    beta_z: float = 0.01
    beta_f: float = 0.01
    beta_r: float = 0.01
    beta_v_total: float = 0.02


def gate_two_gate(pool: EditPool, rng: random.Random, params: TwoGateParameters | None = None) -> GateResult:
    """B8: Cai et al. (2609.08175) Rule 1 with the Appendix C.1 estimators and radii.

    Finite-benchmark adaptation: the stored sample is the search set with failure marks
    from history stream 0; the failure batch F is the set of tasks failed in history
    stream 1 (a separate draw, standing in for Phase E). Both agents get m fresh
    rollouts per evaluated task on disjoint stream sets (independent).
    """
    params = params or TwoGateParameters()
    ledger = Ledger()
    stored_marks: dict[str, int] = {}
    for task in pool.tasks:
        stored_marks[task] = 1 - pool.parent[task][0].outcome
    failure_batch: list[str] = []
    for task in pool.tasks:
        if pool.parent[task][1].outcome == 0:
            failure_batch.append(task)
    retained: list[str] = []
    for task in pool.tasks:
        if stored_marks[task] == 0:
            retained.append(task)
    n_r = len(pool.tasks)
    z_hat = sum(stored_marks.values()) / n_r
    if not failure_batch or not retained or z_hat >= 1.0:
        return GateResult("B8_two_gate", False, ledger, {"reason": "abstain"})

    # m is capped by the number of disjoint draw-stream pairs a finite pool offers (e.g. 3
    # draw streams on LCB allow m = 1); the radii below use the m actually used.
    m = min(params.m, len(pool.draw_streams) // 2)
    if m < 1:
        return GateResult("B8_two_gate", False, ledger, {"reason": "not enough draw streams"})

    def value_estimates(task: str) -> tuple[float, float]:
        """m fresh rollouts of each agent on disjoint draw streams."""
        streams = rng.sample(pool.draw_streams, 2 * m)
        child_values: list[int] = []
        parent_values: list[int] = []
        for stream in streams[:m]:
            run = pool.child[task][stream]
            ledger.charge_full(run)
            child_values.append(run.outcome)
        for stream in streams[m:]:
            run = pool.parent[task][stream]
            ledger.charge_full(run)
            parent_values.append(run.outcome)
        return sum(child_values) / m, sum(parent_values) / m

    failure_advantages: list[float] = []
    for task in failure_batch:
        child_value, parent_value = value_estimates(task)
        failure_advantages.append(child_value - parent_value)
    retained_changes: list[float] = []
    for task in retained:
        child_value, parent_value = value_estimates(task)
        retained_changes.append(abs(child_value - parent_value))
    n_f = len(failure_batch)
    l_hat = z_hat * sum(failure_advantages) / n_f
    d_hat = sum(retained_changes) / len(retained_changes)
    beta_v = params.beta_v_total / (2 * (n_f + len(retained)))
    eps_v = math.sqrt(math.log(2.0 / beta_v) / (2.0 * m))
    eps_z = math.sqrt(math.log(2.0 / params.beta_z) / (2.0 * n_r))
    eps_f = math.sqrt(2.0 * math.log(2.0 / params.beta_f) / n_f)
    eps_mu = 2.0 * eps_v + eps_f
    eps_l = eps_z + (z_hat + eps_z) * eps_mu
    eps_r = math.sqrt(math.log(2.0 / params.beta_r) / (2.0 * n_r))
    eps_d = (2.0 * eps_v + eps_r + eps_z) / (1.0 - z_hat)
    margin = params.tau - eps_l - (1.0 - z_hat + eps_z) * (params.delta + eps_d)
    accept = l_hat >= params.tau and d_hat <= params.delta and margin > 0
    details = {"L_hat": l_hat, "D_hat": d_hat, "margin": margin, "eps_L": eps_l, "eps_D": eps_d, "n_F": n_f}
    return GateResult("B8_two_gate", accept, ledger, details)


def gate_random_subsample(pool: EditPool, rng: random.Random, n_tasks: int, alpha: float = 0.05) -> GateResult:
    """B9: McNemar on a random subset of tasks whose size matches a given budget."""
    ledger = Ledger()
    chosen = rng.sample(pool.tasks, min(n_tasks, len(pool.tasks)))
    child_wins = 0
    parent_wins = 0
    for task in chosen:
        parent_outcome = pool.parent[task][0].outcome
        child_outcome = child_draw(pool, rng, task, ledger)
        if child_outcome > parent_outcome:
            child_wins += 1
        elif parent_outcome > child_outcome:
            parent_wins += 1
    p_value = mcnemar_exact_one_sided(child_wins, parent_wins)
    return GateResult(f"B9_random_{n_tasks}", p_value < alpha, ledger, {"p": p_value})


def gate_historical_caching(pool: EditPool, rng: random.Random, epsilon: float = 0.01) -> GateResult:
    """B10a: 2609.21267 §3.2 - reuse the majority verdict of historically stable tasks.

    A task is cache-eligible when min(ȳ, 1 − ȳ) ≤ ε over the parent's history runs; the
    child's score is estimated from executed + cached verdicts (Eq. 9) and compared with
    the parent's historical mean score.
    """
    ledger = Ledger()
    child_estimate = 0.0
    parent_mean = 0.0
    executed = 0
    for task in pool.tasks:
        history = parent_history(pool, task, pool.k_char)
        rate = sum(history) / len(history)
        parent_mean += rate
        if min(rate, 1.0 - rate) <= epsilon:
            child_estimate += 1.0 if rate >= 0.5 else 0.0
        else:
            child_estimate += child_draw(pool, rng, task, ledger)
            executed += 1
    n = len(pool.tasks)
    accept = child_estimate / n > parent_mean / n
    return GateResult("B10a_hist_caching", accept, ledger, {"executed": executed})


def gate_stratified_subset(pool: EditPool, rng: random.Random, n_strata: int) -> GateResult:
    """B10b: 2609.21267 §3.3 - difficulty-stratified fixed subset with weighted estimate.

    Difficulty = parent's historical pass rate (deviation: the paper fits a Rasch model on
    many historical runs; a single harness lineage gives one difficulty per task).
    """
    ledger = Ledger()
    rates: list[tuple[float, str]] = []
    for task in pool.tasks:
        history = parent_history(pool, task, pool.k_char)
        rates.append((sum(history) / len(history), task))
    rates.sort()
    n = len(rates)
    strata = max(1, min(n_strata, n))
    child_estimate = 0.0
    parent_estimate = 0.0
    for index in range(strata):
        start = (index * n) // strata
        stop = ((index + 1) * n) // strata
        members = rates[start:stop]
        if not members:
            continue
        median_rate, median_task = members[len(members) // 2]
        weight = len(members) / n
        child_estimate += weight * child_draw(pool, rng, median_task, ledger)
        parent_estimate += weight * median_rate
    return GateResult(f"B10b_stratified_{strata}", child_estimate > parent_estimate, ledger, {})


def gate_pace(pool: EditPool, rng: random.Random, budget_runs: int, alpha: float = 0.05, bet: float = 0.5) -> GateResult:
    """B11: PACE Alg. 1 - draw instances, evaluate both agents (independent runs), bet on
    discordant pairs, commit at E ≥ 1/α; reject when the budget is spent."""
    ledger = Ledger()
    test = PaceSignEProcess(alpha=alpha, bet=bet)
    while ledger.task_runs + 2 <= budget_runs:
        task = rng.choice(pool.tasks)
        child_stream = rng.choice(pool.draw_streams)
        parent_stream = independent_parent_stream(rng, pool, child_stream)
        child_run = pool.child[task][child_stream]
        parent_run = pool.parent[task][parent_stream]
        ledger.charge_full(child_run)
        ledger.charge_full(parent_run)
        test.update(child_run.outcome, parent_run.outcome)
        if test.rejects_null:
            return GateResult("B11_pace", True, ledger, {"wealth": test.wealth})
    return GateResult("B11_pace", False, ledger, {"wealth": test.wealth})
