"""Seed harness for BIRD text-to-SQL: a ReAct agent with schema tools.

Protocol: every model reply must contain one action, written as

    Action: <tool name>
    Action Input: <argument>

Tools: list_tables, describe_table, run_sql, submit.
"""

MAX_STEPS = 8
MAX_ROWS_SHOWN = 10
MAX_CELL_CHARS = 60

SYSTEM_PROMPT = """You are an expert SQLite analyst. Answer the user's question by writing one SQLite query.
Work step by step. In each reply, write a short thought and then exactly one action:

Action: <tool name>
Action Input: <argument>

Tools:
- describe_table: show the columns of a table. Input: the table name.
- run_sql: run a read-only SQLite query and see up to 10 rows. Input: the query.
- submit: give your final SQLite query. Input: the query.

Rules:
- Use only tables and columns that exist.
- Quote identifiers that contain spaces or special characters with double quotes.
- Select only the columns the question asks for.
- Test your query with run_sql before you submit it."""

FORMAT_REMINDER = "Your reply had no valid action. Reply with a thought and then exactly one 'Action:' line and one 'Action Input:' line."


def list_tables(env):
    """Names of all user tables."""
    columns, rows, truncated = env.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name", max_rows=200)
    names = []
    for row in rows:
        names.append(str(row[0]))
    return names


def describe_table(env, table):
    """Column names and types of a table, as text."""
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


def format_cell(value):
    """Shorten one result cell."""
    text = str(value)
    if len(text) > MAX_CELL_CHARS:
        return text[:MAX_CELL_CHARS] + "..."
    return text


def format_rows(columns, rows, truncated):
    """Render query results compactly for the model."""
    if not rows:
        return "The query returned 0 rows."
    lines = [" | ".join(columns)]
    for row in rows[:MAX_ROWS_SHOWN]:
        cells = []
        for value in row:
            cells.append(format_cell(value))
        lines.append(" | ".join(cells))
    if truncated or len(rows) > MAX_ROWS_SHOWN:
        lines.append("(more rows not shown)")
    return "\n".join(lines)


def run_sql(env, sql):
    """Execute a query and return a text observation."""
    try:
        columns, rows, truncated = env.execute(clean_sql(sql), max_rows=MAX_ROWS_SHOWN)
    except Exception as error:
        return f"Error: {error}"
    return format_rows(columns, rows, truncated)


def clean_sql(text):
    """Strip code fences and trailing semicolons from model-written SQL."""
    sql = text.strip()
    if sql.startswith("```"):
        sql = sql.strip("`")
        if sql.lower().startswith("sql"):
            sql = sql[3:]
    sql = sql.strip().rstrip(";").strip()
    return sql


def parse_action(reply):
    """Return (tool, argument) from a reply, or (None, None) if malformed."""
    if "Action:" not in reply:
        return None, None
    after = reply.rsplit("Action:", 1)[1]
    if "Action Input:" not in after:
        return None, None
    tool_part, argument = after.split("Action Input:", 1)
    tool = tool_part.strip().split()[0].lower() if tool_part.strip() else ""
    argument = argument.strip()
    if argument.startswith("```"):
        argument = clean_sql(argument)
    return tool, argument


def build_task_prompt(task, env):
    """First user message: question, evidence and the table list."""
    tables = ", ".join(list_tables(env))
    prompt = f"Database tables: {tables}\n\nQuestion: {task['question']}"
    if task.get("evidence"):
        prompt += f"\nHint: {task['evidence']}"
    return prompt


def recover_empty_result(observation):
    """Add a hint when a query returns nothing."""
    if observation == "The query returned 0 rows.":
        return observation + " Check the filter values: look at real values in the column with run_sql before submitting."
    return observation


def verify_submission(env, sql):
    """Return None if the query runs, otherwise an error message."""
    try:
        env.execute(clean_sql(sql), max_rows=1)
    except Exception as error:
        return f"Your submitted query failed: {error}. Fix it and submit again."
    return None


def execute_tool(env, tool, argument):
    """Dispatch one tool call and return the observation text."""
    if tool == "describe_table":
        return describe_table(env, argument)
    if tool == "run_sql":
        return recover_empty_result(run_sql(env, argument))
    if tool == "list_tables":
        return "Tables: " + ", ".join(list_tables(env))
    return f"Unknown tool '{tool}'. Use describe_table, run_sql or submit."


def solve(task, llm, env):
    """ReAct loop; returns the submitted (or last tested) SQL as the answer."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_task_prompt(task, env)},
    ]
    last_sql = None
    for step in range(MAX_STEPS):
        reply = llm.chat(messages)
        messages.append({"role": "assistant", "content": reply})
        tool, argument = parse_action(reply)
        if tool is None:
            messages.append({"role": "user", "content": FORMAT_REMINDER})
            continue
        if tool == "submit":
            problem = verify_submission(env, argument)
            if problem is None:
                return {"answer": clean_sql(argument)}
            messages.append({"role": "user", "content": problem})
            continue
        if tool == "run_sql":
            last_sql = clean_sql(argument)
        observation = execute_tool(env, tool, argument)
        messages.append({"role": "user", "content": "Observation: " + observation})
    return {"answer": last_sql}
