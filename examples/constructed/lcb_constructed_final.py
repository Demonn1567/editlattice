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
Read all input at once with sys.stdin, mind the constraints, and print exactly the required output format.
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
    prompt = f"Problem: {task['title']}\n\n{task['question']}\n\nThink about the time limit before choosing an algorithm."
    reply = llm.chat([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}])
    # Self-repair on the public examples: run the program, and on the first failing example ask for a fix (two rounds).
    for attempt in range(2):
        code = extract_code(reply)
        failure = None
        for test in task.get("public_tests", []):
            result = env.run_python(code or "", test["input"], 6.0)
            if result.timed_out or result.returncode != 0 or result.stdout.strip() != test["output"].strip():
                failure = (test, result)
                break
        if code is None or failure is None:
            break
        test, result = failure
        feedback = (f"Your program fails on this example.\nInput:\n{test['input']}\nExpected output:\n{test['output']}\n"
                    f"Your output:\n{result.stdout[-1000:]}\nErrors:\n{result.stderr[-1000:]}\n"
                    "Fix the program. Put the whole corrected program in one ```python code block.")
        reply = llm.chat([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt},
                          {"role": "assistant", "content": reply}, {"role": "user", "content": feedback}])
    return {"answer": extract_code(reply)}
