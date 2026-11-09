"""Edit-category rubric on mutants whose category is known by construction."""

from __future__ import annotations

from pathlib import Path

from editlattice.analysis.edit_category import classify_edit
from editlattice.mutation.operators import apply_operator, operators_for

SEEDS = Path(__file__).resolve().parents[1] / "src" / "editlattice" / "seed_harnesses"
EXPECTED = {
    ("bird", "remove_verification"): "verification", ("bird", "halve_step_budget"): "control-flow",
    ("bird", "drop_schema_probe"): "tool", ("bird", "truncate_tool_output"): "control-flow",
    ("bird", "corrupt_prompt_rule"): "prompt", ("bird", "lenient_answer_format"): "parsing",
    ("bird", "break_empty_recovery"): "recovery", ("bird", "noop_comment"): "no-op",
    ("bird", "format_reminder_wording"): "prompt", ("lcb", "remove_verification"): "verification",
    ("lcb", "drop_expected_output"): "context", ("lcb", "fragile_code_extraction"): "parsing",
    ("lcb", "swap_feedback_order"): "prompt", ("lcb", "edge_case_hint"): "prompt", ("lcb", "noop_comment"): "no-op",
}


def test_primary_labels_of_mutants() -> None:
    sources = {"bird": (SEEDS / "bird_react.py").read_text(), "lcb": (SEEDS / "lcb_repair.py").read_text()}
    for benchmark, source in sources.items():
        for operator in operators_for(benchmark):
            key = (benchmark, operator.name)
            if key in EXPECTED:
                assert classify_edit(source, apply_operator(operator, source)).primary == EXPECTED[key], key
