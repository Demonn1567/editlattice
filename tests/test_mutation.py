"""Every mutation operator applies to its seed harness and yields the expected impact."""

from __future__ import annotations

from pathlib import Path

from editlattice.impact.diff import compute_impact
from editlattice.mutation.operators import apply_operator, operators_for

SEED = Path(__file__).resolve().parents[1] / "src" / "editlattice" / "seed_harnesses" / "bird_react.py"


def test_bird_operators_apply_and_have_sensible_impact() -> None:
    parent = SEED.read_text()
    operators = operators_for("bird")
    assert len(operators) >= 10
    for operator in operators:
        child = apply_operator(operator, parent)
        impact = compute_impact(parent, child)
        if operator.name == "noop_comment":
            assert impact.is_empty, impact.to_dict()
        elif operator.name == "noop_refactor":
            assert impact.functions == {"format_cell"} and not impact.is_global
        else:
            assert not impact.is_empty, operator.name


LCB_SEED = Path(__file__).resolve().parents[1] / "src" / "editlattice" / "seed_harnesses" / "lcb_repair.py"


def test_lcb_operators_apply_and_have_sensible_impact() -> None:
    parent = LCB_SEED.read_text()
    operators = operators_for("lcb")
    assert len(operators) >= 10
    for operator in operators:
        child = apply_operator(operator, parent)
        impact = compute_impact(parent, child)
        if operator.name == "noop_comment":
            assert impact.is_empty, impact.to_dict()
        elif operator.name == "noop_refactor":
            assert impact.functions == {"truncate"} and not impact.is_global
        else:
            assert not impact.is_empty, operator.name
