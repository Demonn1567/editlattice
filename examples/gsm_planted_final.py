"""Seed harness for word problems: one model call, answer read from the last line.

The model is asked to reason briefly and end with 'Answer: <number>'. Running code, checking
the result, retrying and voting are left for evolution to discover; the environment offers
``env.run_python(code)`` for harnesses that want it.
"""

SYSTEM_PROMPT = """Solve the math word problem by writing a short Python program that prints only the final number.
Put the program in one ```python code block, then end your reply with a final line of the form:
Answer: <number>"""


def extract_answer(reply):
    """Text after the last 'Answer:', else the whole reply."""
    if "Answer:" in reply:
        return reply.rsplit("Answer:", 1)[1].strip().split(".")[0]
    return reply.strip()


def solve(task, llm, env):
    reply = llm.chat([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": task["question"]}])
    if "```python" in reply:
        code = reply.split("```python", 1)[1].split("```", 1)[0]
        result = env.run_python(code)
        if result.returncode == 0 and result.stdout.strip():
            return {"answer": result.stdout.strip().splitlines()[-1]}
    return {"answer": extract_answer(reply)}


def describe_units(value):
    """Format a quantity with its unit (not used by solve)."""
    return f"{value} units"
