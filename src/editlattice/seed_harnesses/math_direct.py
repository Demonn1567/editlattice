"""Seed harness for word problems: one model call, answer read from the last line.

The model is asked to reason briefly and end with 'Answer: <number>'. Running code, checking
the result, retrying and voting are left for evolution to discover; the environment offers
``env.run_python(code)`` for harnesses that want it.
"""

SYSTEM_PROMPT = """Solve the math word problem.
Think step by step, then end your reply with a final line of the form:
Answer: <number>"""


def extract_answer(reply):
    """Text after the last 'Answer:', else the whole reply."""
    if "Answer:" in reply:
        return reply.rsplit("Answer:", 1)[1].strip()
    return reply.strip()


def solve(task, llm, env):
    reply = llm.chat([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": task["question"]}])
    return {"answer": extract_answer(reply)}
