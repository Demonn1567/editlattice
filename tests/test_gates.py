"""Gate sanity checks on synthetic pools with known ground truth."""

from __future__ import annotations

import random

from editlattice.analysis.synthetic import synthetic_pool
from editlattice.gates import baselines, ours


def acceptance_rate(gate_function, effect: float, repetitions: int, **kwargs) -> tuple[float, float]:
    """Fraction of accepts and mean task-runs over independent synthetic edits."""
    accepts = 0
    runs = 0
    for repetition in range(repetitions):
        pool = synthetic_pool(seed=1000 + repetition, effect=effect, n_validation=40)
        result = gate_function(pool, random.Random(repetition), **kwargs)
        accepts += 1 if result.accept else 0
        runs += result.ledger.task_runs
    return accepts / repetitions, runs / repetitions


def test_coupled_gate_controls_false_acceptance_on_null_edits() -> None:
    rate, _ = acceptance_rate(ours.gate_impact_sequential, 0.0, 150, budget_runs=400)
    assert rate <= 0.10, rate


def test_coupled_gate_beats_pace_at_the_same_budget() -> None:
    rate, mean_runs = acceptance_rate(ours.gate_impact_sequential, 0.35, 60, budget_runs=400)
    pace_rate, pace_runs = acceptance_rate(baselines.gate_pace, 0.35, 60, budget_runs=400)
    assert rate >= 0.8, rate
    assert rate > pace_rate
    assert mean_runs < pace_runs


def test_futility_margin_stops_null_edits_early() -> None:
    _, without_margin = acceptance_rate(ours.gate_impact_sequential, 0.0, 40, budget_runs=400)
    rate, with_margin = acceptance_rate(ours.gate_impact_sequential, 0.0, 40, budget_runs=400, futility_margin=0.2)
    assert rate <= 0.1
    assert with_margin < without_margin


def test_greedy_accepts_many_null_edits() -> None:
    rate, _ = acceptance_rate(baselines.gate_greedy, 0.0, 150)
    assert rate >= 0.2, rate


def test_every_gate_runs_on_a_pool() -> None:
    pool = synthetic_pool(seed=5, effect=0.2, n_validation=40)
    rng = random.Random(0)
    results = [
        baselines.gate_greedy(pool, rng),
        baselines.gate_best_so_far_tokens(pool, rng),
        baselines.gate_self_harness(pool, rng, repeats=2),
        baselines.gate_strict_non_regression(pool, rng),
        baselines.gate_avg_at_k(pool, rng, k=3),
        baselines.gate_avg_at_k(pool, rng, k=5),
        baselines.gate_mcnemar_full(pool, rng),
        baselines.gate_paired_t(pool, rng, k=3),
        baselines.gate_paired_t(pool, rng, k=3, bootstrap=True),
        baselines.gate_fresh_validation(pool, rng),
        baselines.gate_two_gate(pool, rng),
        baselines.gate_random_subsample(pool, rng, n_tasks=30),
        baselines.gate_historical_caching(pool, rng),
        baselines.gate_stratified_subset(pool, rng, n_strata=20),
        baselines.gate_pace(pool, rng, budget_runs=400),
        ours.gate_impact_mcnemar(pool, rng),
        ours.gate_impact_mcnemar(pool, rng, crn=True),
        ours.gate_impact_sequential(pool, rng, budget_runs=400, crn=False),
        ours.gate_impact_sequential(pool, rng, budget_runs=400),
        ours.gate_impact_sequential(pool, rng, budget_runs=400, granularity="line"),
        ours.gate_impact_sequential(pool, rng, budget_runs=400, test="pace"),
    ]
    names = set()
    for result in results:
        assert result.ledger.task_runs >= 0
        names.add(result.gate)
    assert len(names) == len(results)


def _distinct_answers(pool):
    """Give every run its own answer (synthetic pools carry answer=None)."""
    from dataclasses import replace

    for side, runs in (("p", pool.parent), ("c", pool.child)):
        for task, streams in runs.items():
            for stream, run in list(streams.items()):
                streams[stream] = replace(run, answer=f"{side}-{task}-{stream}")
    return pool


def test_judge_gate_runs_with_a_fake_judge(tmp_path) -> None:
    from editlattice.gates.judge import JudgeGate
    from editlattice.llm.cache import ResponseCache
    from editlattice.llm.client import LLMClient
    from editlattice.llm.fake import FakeBackend

    def always_a(messages, rng):
        return "A"

    pool = _distinct_answers(synthetic_pool(seed=9, effect=0.0))
    gate = JudgeGate(LLMClient(FakeBackend(always_a), "fake", ResponseCache(tmp_path / "c")), {}, n_tasks=30, margin=2)
    result = gate(pool, random.Random(0))
    assert result.gate == "B12_llm_judge"
    assert result.details["child_wins"] + result.details["parent_wins"] == 30
    assert result.ledger.task_runs == 60


def test_judge_gate_identical_answers_are_ties_without_calls() -> None:
    from editlattice.gates.judge import JudgeGate

    pool = synthetic_pool(seed=9, effect=0.0)  # every answer is None
    gate = JudgeGate(None, {}, n_tasks=30, margin=2)  # no client: any call would fail
    result = gate(pool, random.Random(0))
    assert result.details == {"child_wins": 0, "parent_wins": 0, "judge_tokens": 0}
    assert not result.accept


def test_judge_gate_verdict_table_matches_live_judge(tmp_path) -> None:
    from editlattice.gates.judge import JudgeGate, comparison_key, parse_verdict
    from editlattice.llm.cache import ResponseCache
    from editlattice.llm.client import LLMClient
    from editlattice.llm.fake import FakeBackend

    def by_length(messages, rng):
        text = messages[-1]["content"]
        first = text.split("Answer A:\n")[1].split("\n\nAnswer B:")[0]
        second = text.split("Answer B:\n")[1].split("\n\nWhich")[0]
        return "A" if len(first) >= len(second) else "B"

    pool = _distinct_answers(synthetic_pool(seed=11, effect=0.0))
    live = JudgeGate(LLMClient(FakeBackend(by_length), "fake", ResponseCache(tmp_path / "c")), {}, n_tasks=20, margin=1)
    table = {}
    for task in pool.tasks:
        for child_run in pool.child[task].values():
            for parent_run in pool.parent[task].values():
                for first, second in ((child_run.answer, parent_run.answer), (parent_run.answer, child_run.answer)):
                    verdict = "A" if len(str(first)) >= len(str(second)) else "B"
                    table[comparison_key(task, first, second)] = (parse_verdict(verdict), 7)
    offline = JudgeGate(None, {}, n_tasks=20, margin=1, verdicts=table)
    for seed in range(5):
        a, b = live(pool, random.Random(seed)), offline(pool, random.Random(seed))
        assert (a.accept, a.details["child_wins"], a.details["parent_wins"]) == (b.accept, b.details["child_wins"], b.details["parent_wins"])
        assert a.ledger.task_runs == b.ledger.task_runs