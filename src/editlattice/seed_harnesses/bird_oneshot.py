"""Seed harness for BIRD text-to-SQL: one model call with the full schema, no execution.

The model sees the question, the hint and every table with its columns and types, and
answers with one SQLite query in a code block. Nothing is executed before submission, so
execution feedback, value inspection and repair are left for evolution to discover.

Available to a harness: env.execute(sql, max_rows) -> (columns, rows, truncated) on a
read-only copy of the task's database.
"""

SYSTEM_PROMPT = """You are an expert SQLite analyst.
Write one SQLite query that answers the user's question.
Put the query in one ```sql code block."""


def schema_text(env):
    """Every user table with its columns and types."""
    columns, rows, truncated = env.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name", max_rows=200
    )
    lines = []
    for row in rows:
        table = str(row[0])
        info_columns, info_rows, info_truncated = env.execute(f'PRAGMA table_info("{table}")', max_rows=200)
        described = ", ".join(f"{info[1]} ({info[2]})" for info in info_rows)
        lines.append(f"Table {table}: {described}")
    return "\n".join(lines)


def extract_sql(reply):
    """The last ```sql block, else the last code block, else the whole reply."""
    if "```sql" in reply:
        return reply.rsplit("```sql", 1)[1].split("```", 1)[0].strip()
    if "```" in reply:
        parts = reply.split("```")
        if len(parts) >= 3:
            return parts[-2].strip()
    return reply.strip()


def solve(task, llm, env):
    prompt = f"Database schema:\n{schema_text(env)}\n\nQuestion: {task['question']}\nHint: {task['evidence']}"
    reply = llm.chat([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}])
    return {"answer": extract_sql(reply)}
