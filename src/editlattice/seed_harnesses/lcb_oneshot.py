"""Seed harness for LiveCodeBench: one generation, no tests, no repair.

The model writes a complete Python 3 program in one code block; the harness returns it
as is. Running the public examples, repairing on failure and choosing among candidates are
left for evolution to discover.

Available to a harness: task["public_tests"] (a list of {"input": ..., "output": ...} examples)
and env.run_python(code, stdin, timeout_seconds) -> result with .stdout, .stderr, .returncode
and .timed_out (sandboxed).
"""

SYSTEM_PROMPT = """You are an expert competitive programmer.
Write a complete Python 3 program that reads the input from standard input and prints the answer to standard output.
Put the whole program in one ```python code block."""


def extract_code(reply):
    """The last ```python block, else the last code block, else None."""
    if "```python" in reply:
        return reply.rsplit("```python", 1)[1].split("```", 1)[0].strip()
    if "```" in reply:
        parts = reply.split("```")
        if len(parts) >= 3:
            return parts[-2].strip()
    return None


def solve(task, llm, env):
    prompt = f"Problem: {task['title']}\n\n{task['question']}"
    reply = llm.chat([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}])
    return {"answer": extract_code(reply)}
