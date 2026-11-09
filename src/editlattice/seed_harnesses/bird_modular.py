"""Modular seed harness for BIRD (RQ7): capabilities are primitives activated per task.

Same tools and protocol as bird_react.py, but every capability is a primitive that owns
its prompt fragment and logic inside its own function. A router activates primitives from
features of the question, so an edit to one primitive only changes runs of tasks that
activate it.
"""

MAX_STEPS = 8


def base_instructions():
    """Core protocol (always active)."""
    return """You are an expert SQLite analyst. Answer the user's question by writing one SQLite query.
In each reply, write a short thought and then exactly one action:

Action: <tool name>
Action Input: <argument>

Tools:
- describe_table: show the columns of a table. Input: the table name.
- run_sql: run a read-only SQLite query and see up to 10 rows. Input: the query.
- submit: give your final SQLite query. Input: the query."""


def wants_value_grounding(task):
    """Activation: the question mentions literal values (quotes or capitalised names)."""
    text = task["question"]
    has_quote = "'" in text or '"' in text
    has_name = False
    for word in text.split()[1:]:
        if word[:1].isupper():
            has_name = True
    return has_quote or has_name


def value_grounding_rules():
    """Primitive: ground filter literals in the data."""
    return "- Before filtering on a text value, check how it is spelled in the data with run_sql (SELECT DISTINCT ... LIKE ...)."


def wants_aggregation(task):
    """Activation: ratio, percentage, average and count questions."""
    lowered = task["question"].lower()
    for keyword in ("percent", "ratio", "average", "how many", "count", "proportion", "rate"):
        if keyword in lowered:
            return True
    return False


def aggregation_rules():
    """Primitive: arithmetic conventions for aggregates."""
    return "- For ratios and percentages, CAST the numerator to REAL and multiply by 100 only if a percentage is asked."


def wants_join_planning(task):
    """Activation: the hint references several table.column pairs, or the question says join."""
    hint = task.get("evidence", "")
    return hint.count(".") >= 2 or " join " in task["question"].lower()


def join_planning_rules():
    """Primitive: plan joins from the schema."""
    return "- Identify the join keys with describe_table before writing a JOIN."


def wants_output_shape(task):
    """Activation: list/name questions."""
    return task["question"].lower().startswith(("list", "name", "which", "what are", "give"))


def output_shape_rules():
    """Primitive: return exactly the asked columns."""
    return "- Select only the columns the question asks for, in the order asked."


def route(task):
    """Rules of the primitives activated for this task (inactive primitives never run)."""
    fragments = []
    if wants_value_grounding(task):
        fragments.append(value_grounding_rules())
    if wants_aggregation(task):
        fragments.append(aggregation_rules())
    if wants_join_planning(task):
        fragments.append(join_planning_rules())
    if wants_output_shape(task):
        fragments.append(output_shape_rules())
    return fragments


def build_system_prompt(task):
    """Base protocol plus the activated primitives' rules."""
    fragments = route(task)
    if not fragments:
        return base_instructions()
    return base_instructions() + "\n\nRules:\n" + "\n".join(fragments)


def list_tables(env):
    """Names of all user tables."""
    columns, rows, truncated = env.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name", max_rows=200)
    names = []
    for row in rows:
        names.append(str(row[0]))
    return names


def tool_describe_table(env, table):
    """Column names and types of a table."""
    table = table.strip().strip('"').strip("`")
    try:
        columns, rows, truncated = env.execute(f'PRAGMA table_info("{table}")', max_rows=200)
    except Exception as error:
        return f"Error: {error}"
    if not rows:
        return f"Error: no table named {table}. Tables: {', '.join(list_tables(env))}"
    parts = []
    for row in rows:
        parts.append(f"{row[1]} ({row[2]})")
    return f"Table {table}: " + ", ".join(parts)


def format_rows(columns, rows, truncated):
    """Compact rendering of query results."""
    if not rows:
        return "The query returned 0 rows."
    lines = [" | ".join(columns)]
    for row in rows[:10]:
        cells = []
        for value in row:
            text = str(value)
            cells.append(text if len(text) <= 60 else text[:60] + "...")
        lines.append(" | ".join(cells))
    if truncated or len(rows) > 10:
        lines.append("(more rows not shown)")
    return "\n".join(lines)


def clean_sql(text):
    """Strip code fences and trailing semicolons."""
    sql = text.strip()
    if sql.startswith("```"):
        sql = sql.strip("`")
        if sql.lower().startswith("sql"):
            sql = sql[3:]
    return sql.strip().rstrip(";").strip()


def tool_run_sql(env, sql):
    """Execute a query and return a text observation."""
    try:
        columns, rows, truncated = env.execute(clean_sql(sql), max_rows=10)
    except Exception as error:
        return recover_from_error(str(error))
    observation = format_rows(columns, rows, truncated)
    if observation == "The query returned 0 rows.":
        return recover_from_empty_result(observation)
    return observation


def recover_from_error(message):
    """Recovery primitive: SQL errors."""
    return f"Error: {message}. Check table and column names with describe_table."


def recover_from_empty_result(observation):
    """Recovery primitive: empty results."""
    return observation + " Check the filter values against real values in the column before submitting."


def verify_submission(env, sql):
    """Verification primitive: the submitted query must run."""
    try:
        env.execute(clean_sql(sql), max_rows=1)
    except Exception as error:
        return f"Your submitted query failed: {error}. Fix it and submit again."
    return None


def parse_action(reply):
    """(tool, argument) or (None, None)."""
    if "Action:" not in reply:
        return None, None
    after = reply.rsplit("Action:", 1)[1]
    if "Action Input:" not in after:
        return None, None
    tool_part, argument = after.split("Action Input:", 1)
    tool = tool_part.strip().split()[0].lower() if tool_part.strip() else ""
    return tool, argument.strip()


def format_reminder():
    """Message after a malformed reply."""
    return "Your reply had no valid action. Reply with a thought and then exactly one 'Action:' line and one 'Action Input:' line."


def solve(task, llm, env):
    """ReAct loop over the activated primitives."""
    messages = [
        {"role": "system", "content": build_system_prompt(task)},
        {"role": "user", "content": f"Database tables: {', '.join(list_tables(env))}\n\nQuestion: {task['question']}" + (f"\nHint: {task['evidence']}" if task.get("evidence") else "")},
    ]
    last_sql = None
    for step in range(MAX_STEPS):
        reply = llm.chat(messages)
        messages.append({"role": "assistant", "content": reply})
        tool, argument = parse_action(reply)
        if tool is None:
            messages.append({"role": "user", "content": format_reminder()})
            continue
        if tool == "submit":
            problem = verify_submission(env, argument)
            if problem is None:
                return {"answer": clean_sql(argument)}
            messages.append({"role": "user", "content": problem})
            continue
        if tool == "describe_table":
            observation = tool_describe_table(env, argument)
        elif tool == "run_sql":
            last_sql = clean_sql(argument)
            observation = tool_run_sql(env, argument)
        else:
            observation = f"Unknown tool '{tool}'. Use describe_table, run_sql or submit."
        messages.append({"role": "user", "content": "Observation: " + observation})
    return {"answer": last_sql}
