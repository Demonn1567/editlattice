"""Scorer tests for BIRD (execution accuracy) and LiveCodeBench (hidden tests)."""

from __future__ import annotations

import pytest

from editlattice.benchmarks.bird import DEFAULT_ROOT, BirdBenchmark, result_set
from editlattice.benchmarks.livecodebench import DEFAULT_FILES, LiveCodeBenchmark, outputs_match, run_program


def test_outputs_match_rules() -> None:
    assert outputs_match("1 2\n3\n", "1 2\n3")
    assert outputs_match("Yes\n", "Yes   \n\n")
    assert outputs_match("0.3333333333", "0.33333333")
    assert not outputs_match("0.33", "0.34")
    assert not outputs_match("Yes", "yes")
    assert not outputs_match("1 2 3", "1 2")


def test_sandbox_runs_reads_stdin_and_enforces_timeout() -> None:
    echo = run_program("print(int(input()) * 2)", "21\n", 5.0)
    assert echo.stdout.strip() == "42" and echo.returncode == 0
    crash = run_program("raise SystemExit(3)", "", 5.0)
    assert crash.returncode == 3
    slow = run_program("while True:\n    pass\n", "", 1.5)
    assert slow.timed_out or slow.returncode != 0


def test_result_set_ignores_order_and_duplicates() -> None:
    assert result_set([[1, "a"], [2, "b"]]) == result_set([[2, "b"], [1, "a"], [1, "a"]])


@pytest.mark.skipif(not (DEFAULT_ROOT / "mini_dev_sqlite.json").exists(), reason="BIRD data not downloaded")
def test_bird_gold_scores_one_and_wrong_scores_zero() -> None:
    benchmark = BirdBenchmark()
    tasks = benchmark.load_all_tasks()[:5]
    benchmark.ensure_gold(tasks)
    for task in tasks:
        if not benchmark.gold(task["task_id"])["ok"]:
            continue
        assert benchmark.score(task, task["gold_sql"]) == (1, 1.0)
        assert benchmark.score(task, "SELECT 123456789") == (0, 0.0)
        assert benchmark.score(task, "SELEC broken") == (0, 0.0)
        assert benchmark.score(task, None) == (0, 0.0)
        assert "gold_sql" not in benchmark.public_view(task)


@pytest.mark.skipif(not DEFAULT_FILES[1].exists(), reason="LiveCodeBench data not downloaded")
def test_lcb_private_tests_decode_and_wrong_program_fails() -> None:
    benchmark = LiveCodeBenchmark(files=[DEFAULT_FILES[1]], max_hidden_tests=5)
    task = benchmark.load_all_tasks()[0]
    tests = benchmark.all_tests(task)
    assert len(tests) > len(task["public_tests"]) or len(tests) == 5
    assert benchmark.score(task, "print('definitely wrong')")[0] == 0
    assert "private_tests_blob" not in benchmark.public_view(task)
    expected = task["public_tests"][0]["output"]
    echo_expected = f"import sys\nsys.stdin.read()\nprint({expected!r})\n"
    public_only = LiveCodeBenchmark(files=[DEFAULT_FILES[1]], max_hidden_tests=1)
    assert public_only.score(task, echo_expected) == (1, 1.0)


def test_sandbox_output_is_deterministic() -> None:
    """Regression (RQ2 finding): tracebacks must not leak the random sandbox path."""
    first = run_program("x = [1]\nprint(x[5])\n", "", 3.0)
    second = run_program("x = [1]\nprint(x[5])\n", "", 3.0)
    assert first.stderr == second.stderr and "/sandbox" in first.stderr
    assert "0xADDR" in run_program("print(object())", "", 3.0).stdout


def test_sql_time_is_frozen() -> None:
    """Regression (RQ2 finding): queries using "now" must not depend on the wall-clock date."""
    from editlattice.benchmarks.bird import FROZEN_NOW, freeze_time

    rewritten = freeze_time("SELECT JULIANDAY(CURRENT_DATE), date('NOW'), CURRENT_TIMESTAMP, 'nowhere', now_col FROM t")
    assert "CURRENT_DATE" not in rewritten and "'NOW'" not in rewritten
    assert f"'{FROZEN_NOW[:10]}'" in rewritten and "'nowhere'" in rewritten and "now_col" in rewritten
