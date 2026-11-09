"""Mutation operators for the LiveCodeBench seed harness (``lcb_repair.py``)."""

from __future__ import annotations

from editlattice.mutation.operators import MutationOperator

WRONG_ANSWER_LINE = (
    "    return f\"Input:\\n{truncate(test['input'])}\\nExpected output:\\n"
    "{truncate(test['output'])}\\nYour output:\\n{truncate(stdout)}\""
)
TIMEOUT_LINE = (
    "    return f\"Input:\\n{truncate(test['input'])}\\nThe program exceeded the time limit. "
    "Use a faster algorithm.\""
)

LCB_OPERATORS: list[MutationOperator] = [
    MutationOperator(
        name="remove_verification",
        benchmark="lcb",
        scope="local",
        intended="harmful",
        description="Public tests are never run; the first program is returned.",
        old="        passed, failure = run_public_tests(env, code)",
        new="        passed, failure = 0, None",
    ),
    MutationOperator(
        name="halve_repair_budget",
        benchmark="lcb",
        scope="local",
        intended="harmful",
        description="Repair rounds 3 -> 1.",
        old="MAX_REPAIRS = 3",
        new="MAX_REPAIRS = 1",
    ),
    MutationOperator(
        name="double_repair_budget",
        benchmark="lcb",
        scope="local",
        intended="improving",
        description="Repair rounds 3 -> 6.",
        old="MAX_REPAIRS = 3",
        new="MAX_REPAIRS = 6",
    ),
    MutationOperator(
        name="drop_expected_output",
        benchmark="lcb",
        scope="local",
        intended="harmful",
        description="Wrong-answer feedback no longer shows the expected output.",
        old=WRONG_ANSWER_LINE,
        new="    return f\"Input:\\n{truncate(test['input'])}\\nYour output was wrong.\"",
    ),
    MutationOperator(
        name="truncate_feedback",
        benchmark="lcb",
        scope="local",
        intended="harmful",
        description="Feedback cut to 40 characters.",
        old="MAX_FEEDBACK_CHARS = 600",
        new="MAX_FEEDBACK_CHARS = 40",
    ),
    MutationOperator(
        name="corrupt_prompt_rule",
        benchmark="lcb",
        scope="global",
        intended="harmful",
        description="System-prompt rule replaced by a wrong one (print debug output).",
        old="- Print exactly the required output and nothing else.",
        new="- Print helpful debugging information before the answer.",
    ),
    MutationOperator(
        name="fragile_code_extraction",
        benchmark="lcb",
        scope="local",
        intended="harmful",
        description="Only ```python blocks are accepted (generic ``` fences ignored).",
        old=(
            '    if "```" in reply:\n'
            '        parts = reply.split("```")\n'
            "        if len(parts) >= 3:\n"
            "            return parts[-2].strip()\n"
            "    return None"
        ),
        new="    return None",
    ),
    MutationOperator(
        name="break_timeout_feedback",
        benchmark="lcb",
        scope="local",
        intended="harmful",
        description="Time-out feedback no longer says the program was too slow.",
        old=TIMEOUT_LINE,
        new="    return f\"Input:\\n{truncate(test['input'])}\\nThe program failed.\"",
    ),
    MutationOperator(
        name="inject_irrelevant_instruction",
        benchmark="lcb",
        scope="global",
        intended="harmful",
        description="Irrelevant instruction appended to the system prompt.",
        old="- Put the whole program in one ```python code block.",
        new="- Put the whole program in one ```python code block.\n- Name every variable after a fruit.",
    ),
    MutationOperator(
        name="swap_feedback_order",
        benchmark="lcb",
        scope="local",
        intended="neutral",
        description="Repair message puts the failure before the instruction.",
        old='        messages.append({"role": "user", "content": REPAIR_INSTRUCTION + "\\n\\n" + failure})',
        new='        messages.append({"role": "user", "content": failure + "\\n\\n" + REPAIR_INSTRUCTION})',
    ),
    MutationOperator(
        name="noop_comment",
        benchmark="lcb",
        scope="local",
        intended="neutral",
        description="Equivalent mutant: comment added (empty impact set).",
        old='def truncate(text):\n    """Keep feedback short."""',
        new='def truncate(text):\n    """Keep feedback short."""\n    # Long inputs and outputs are cut so the repair prompt stays small.',
    ),
    MutationOperator(
        name="noop_refactor",
        benchmark="lcb",
        scope="local",
        intended="neutral",
        description="Equivalent mutant: semantics-preserving rewrite of truncate (non-empty impact, true delta 0).",
        old=(
            "    if len(text) > MAX_FEEDBACK_CHARS:\n"
            '        return text[:MAX_FEEDBACK_CHARS] + "..."\n'
            "    return text"
        ),
        new=(
            "    if len(text) <= MAX_FEEDBACK_CHARS:\n"
            "        return text\n"
            '    return text[:MAX_FEEDBACK_CHARS] + "..."'
        ),
    ),
    MutationOperator(
        name="edge_case_hint",
        benchmark="lcb",
        scope="local",
        intended="improving",
        description="Repair instruction asks the model to check edge cases and constraints.",
        old='REPAIR_INSTRUCTION = "Your program is wrong on this example. Find the bug,',
        new='REPAIR_INSTRUCTION = "Your program is wrong on this example. Re-read the constraints, check edge cases (minimum and maximum sizes), find the bug,',
    ),
]
