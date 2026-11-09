"""Harness mutation operators (ground-truth edits with a known intended direction).

Each operator is an exact source replacement on a specific seed harness. ``apply``
fails loudly unless the target text occurs exactly once and the result parses, so a
mutant can never silently equal its parent.

Tags (assigned by construction, before any run):
* ``scope``: "local" if the edited code runs only on some code paths, "global" if it is
  read by code every run executes (e.g. the system prompt). The impact analysis
  computes the actual impact set independently; RQ1 compares the two.
* ``intended``: "harmful", "neutral" (equivalent mutant: true Δ = 0 by construction),
  or "improving". The measured ground truth (Sec. 6.4) may disagree; we report both.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass(frozen=True)
class MutationOperator:
    """One exact-replacement mutation."""

    name: str
    benchmark: str
    scope: str
    intended: str
    description: str
    old: str
    new: str

    def apply(self, source: str) -> str:
        """Return the mutated source; raise if the edit is not well defined."""
        count = source.count(self.old)
        if count != 1:
            raise ValueError(f"operator {self.name}: target occurs {count} times (expected 1)")
        mutated = source.replace(self.old, self.new)
        ast.parse(mutated)
        if mutated == source:
            raise ValueError(f"operator {self.name}: no change")
        return mutated


BIRD_OPERATORS: list[MutationOperator] = [
    MutationOperator(
        name="remove_verification",
        benchmark="bird",
        scope="local",
        intended="harmful",
        description="Submitted queries are no longer checked for execution errors.",
        old='''    try:
        env.execute(clean_sql(sql), max_rows=1)
    except Exception as error:
        return f"Your submitted query failed: {error}. Fix it and submit again."
    return None''',
        new='''    return None''',
    ),
    MutationOperator(
        name="halve_step_budget",
        benchmark="bird",
        scope="global",
        intended="harmful",
        description="Maximum ReAct steps 8 -> 4.",
        old="MAX_STEPS = 8",
        new="MAX_STEPS = 4",
    ),
    MutationOperator(
        name="double_step_budget",
        benchmark="bird",
        scope="global",
        intended="improving",
        description="Maximum ReAct steps 8 -> 16.",
        old="MAX_STEPS = 8",
        new="MAX_STEPS = 16",
    ),
    MutationOperator(
        name="drop_schema_probe",
        benchmark="bird",
        scope="local",
        intended="harmful",
        description="describe_table returns no column information.",
        old='''    parts = []
    for row in rows:
        parts.append(f"{row[1]} ({row[2]})")
    return f"Table {table}: " + ", ".join(parts)''',
        new='''    return f"Table {table}: (column information unavailable)"''',
    ),
    MutationOperator(
        name="truncate_tool_output",
        benchmark="bird",
        scope="local",
        intended="harmful",
        description="Query results show 2 rows instead of 10.",
        old="MAX_ROWS_SHOWN = 10",
        new="MAX_ROWS_SHOWN = 2",
    ),
    MutationOperator(
        name="truncate_cells",
        benchmark="bird",
        scope="local",
        intended="harmful",
        description="Result cells cut to 8 characters.",
        old="MAX_CELL_CHARS = 60",
        new="MAX_CELL_CHARS = 8",
    ),
    MutationOperator(
        name="corrupt_prompt_rule",
        benchmark="bird",
        scope="global",
        intended="harmful",
        description="System-prompt rule replaced by a wrong one (SELECT *).",
        old="- Select only the columns the question asks for.",
        new="- Always use SELECT * so that no information is lost.",
    ),
    MutationOperator(
        name="lenient_answer_format",
        benchmark="bird",
        scope="local",
        intended="harmful",
        description="Submitted SQL is no longer cleaned of code fences/semicolons.",
        old='''            if problem is None:
                return {"answer": clean_sql(argument)}''',
        new='''            if problem is None:
                return {"answer": argument}''',
    ),
    MutationOperator(
        name="swap_column_type_order",
        benchmark="bird",
        scope="local",
        intended="neutral",
        description="describe_table prints 'type name' instead of 'name (type)'.",
        old='''        parts.append(f"{row[1]} ({row[2]})")''',
        new='''        parts.append(f"{row[2]} {row[1]}")''',
    ),
    MutationOperator(
        name="break_empty_recovery",
        benchmark="bird",
        scope="local",
        intended="harmful",
        description="Empty-result recovery hint removed.",
        old='''    if observation == "The query returned 0 rows.":
        return observation + " Check the filter values: look at real values in the column with run_sql before submitting."
    return observation''',
        new='''    return observation''',
    ),
    MutationOperator(
        name="inject_irrelevant_instruction",
        benchmark="bird",
        scope="global",
        intended="harmful",
        description="Irrelevant instruction appended to the system prompt.",
        old="- Test your query with run_sql before you submit it.",
        new="- Test your query with run_sql before you submit it.\n- Mention the weather in Paris in every thought.",
    ),
    MutationOperator(
        name="noop_comment",
        benchmark="bird",
        scope="local",
        intended="neutral",
        description="Equivalent mutant: comment added (empty impact set).",
        old='''def format_cell(value):
    """Shorten one result cell."""''',
        new='''def format_cell(value):
    """Shorten one result cell."""
    # Cells longer than the limit are cut and marked with an ellipsis.''',
    ),
    MutationOperator(
        name="noop_refactor",
        benchmark="bird",
        scope="local",
        intended="neutral",
        description="Equivalent mutant: semantics-preserving rewrite of format_cell (non-empty impact, true delta 0).",
        old='''    text = str(value)
    if len(text) > MAX_CELL_CHARS:
        return text[:MAX_CELL_CHARS] + "..."
    return text''',
        new='''    text = str(value)
    too_long = len(text) > MAX_CELL_CHARS
    if not too_long:
        return text
    return text[:MAX_CELL_CHARS] + "..."''',
    ),
    MutationOperator(
        name="value_grounding",
        benchmark="bird",
        scope="global",
        intended="improving",
        description="Adds a sample_values tool (distinct values of a column) and documents it in the prompt.",
        old='''    if tool == "list_tables":''',
        new='''    if tool == "sample_values":
        return sample_values(env, argument)
    if tool == "list_tables":''',
    ),
    MutationOperator(
        name="format_reminder_wording",
        benchmark="bird",
        scope="local",
        intended="neutral",
        description="Different wording of the malformed-reply reminder (runs only after a malformed reply).",
        old='FORMAT_REMINDER = "Your reply had no valid action. Reply with a thought and then exactly one \'Action:\' line and one \'Action Input:\' line."',
        new='FORMAT_REMINDER = "No action found. Write one \'Action:\' line and one \'Action Input:\' line."',
    ),
]

VALUE_GROUNDING_SUPPORT = '''

def sample_values(env, argument):
    """Up to 10 distinct values of table.column."""
    if "." not in argument:
        return "Error: use the form table.column"
    table, column = argument.strip().split(".", 1)
    sql = f'SELECT DISTINCT "{column.strip()}" FROM "{table.strip()}" LIMIT 10'
    return run_sql(env, sql)
'''

VALUE_GROUNDING_PROMPT_OLD = "- submit: give your final SQLite query. Input: the query."
VALUE_GROUNDING_PROMPT_NEW = "- sample_values: show up to 10 distinct values of a column. Input: table.column\n- submit: give your final SQLite query. Input: the query."


def apply_operator(operator: MutationOperator, source: str) -> str:
    """Apply an operator, including multi-part operators that need support code."""
    mutated = operator.apply(source)
    if operator.name == "value_grounding":
        prompt_operator = MutationOperator(
            name="value_grounding_prompt",
            benchmark=operator.benchmark,
            scope=operator.scope,
            intended=operator.intended,
            description="prompt part of value_grounding",
            old=VALUE_GROUNDING_PROMPT_OLD,
            new=VALUE_GROUNDING_PROMPT_NEW,
        )
        mutated = prompt_operator.apply(mutated) + VALUE_GROUNDING_SUPPORT
        ast.parse(mutated)
    return mutated


def operators_for(benchmark: str) -> list[MutationOperator]:
    """All operators defined for a benchmark."""
    from editlattice.mutation.lcb_operators import LCB_OPERATORS

    selected: list[MutationOperator] = []
    for operator in BIRD_OPERATORS + LCB_OPERATORS:
        if operator.benchmark == benchmark:
            selected.append(operator)
    return selected
