"""Seed harness for word problems: one model call, answer read from the last line.

The model is asked to reason briefly and end with 'Answer: <number>'. Running code, checking
the result, retrying and voting are left for evolution to discover; the environment offers
``env.run_python(code)`` for harnesses that want it.
"""

SYSTEM_PROMPT = """Solve the math word problem. The numbers can be very large: compute exactly.
Think step by step, then end your reply with a final line of the form:
Answer: <number>"""


def extract_answer(reply):
    """Text after the last 'Answer:', else the whole reply."""
    if "Answer:" in reply:
        text = reply.rsplit("Answer:", 1)[1].strip()
    else:
        text = reply.strip()
    return text.replace(",", "").replace("$", "").strip()


def solve(task, llm, env):
    reply = llm.chat([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": task["question"]}])
    # Program-aided answer (PAL): write a program, run it, and prefer its printed result.
    program_prompt = ("Write a Python 3 program that computes the answer to the math word problem and prints only "
                      "the final number. Put the program in one ```python code block.")
    for attempt in range(2):
        code_reply = llm.chat([{"role": "system", "content": program_prompt}, {"role": "user", "content": task["question"]}])
        if "```python" in code_reply:
            code = code_reply.rsplit("```python", 1)[1].split("```", 1)[0]
            result = env.run_python(code)
            lines = result.stdout.strip().splitlines()
            if result.returncode == 0 and not result.timed_out and lines:
                reply = "Answer: " + lines[-1]
                break
    return {"answer": extract_answer(reply)}
