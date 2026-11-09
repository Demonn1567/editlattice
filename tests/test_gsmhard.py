"""GSM-Hard answer parsing and scoring."""

from editlattice.benchmarks.gsmhard import GsmHardBenchmark, parse_number, plausible_target


def test_parse_number_takes_the_last_number() -> None:
    assert parse_number("so 3 apples, total 1,234.5 dollars") == 1234.5
    assert parse_number("Answer: -7") == -7.0
    assert parse_number("no idea") is None
    assert parse_number(None) is None
    assert parse_number(12) == 12.0


def test_score_uses_absolute_tolerance() -> None:
    benchmark = GsmHardBenchmark()
    task = {"target": 9867630.0}
    assert benchmark.score(task, "Answer: 9,867,630") == (1, 1.0)
    assert benchmark.score(task, "9867631") == (0, 0.0)
    assert benchmark.score({"target": 0.5}, "0.50004") == (1, 1.0)
    assert benchmark.score({"target": 0.5}, "0.502") == (0, 0.0)


def test_plausibility_flag() -> None:
    assert plausible_target(18.0) and plausible_target(12.75)
    assert not plausible_target(-3.0) and not plausible_target(3.3333333)
