"""LiveCodeBench harness with self-repair whose feedback depends on the failure class.

The branched LiveCodeBench lineage starts from the constructed LiveCodeBench harness
(examples/constructed/lcb_constructed_final.py) with its repair message split by failure class. Its seed
(lcb_branched_seed.py) reports each class plainly. Its final harness (lcb_branched_final.py) adds advice
specific to each class and a rule in the system prompt, so most of its units sit on paths that only some
runs take.

Available to a harness: task["public_tests"] (a list of {"input": ..., "output": ...} examples)
and env.run_python(code, stdin, timeout_seconds) -> result with .stdout, .stderr, .returncode
and .timed_out (sandboxed).
"""

SYSTEM_PROMPT = """You are an expert competitive programmer.
Write a complete Python 3 program that reads the input from standard input and prints the answer to standard output.
Read all input at once with sys.stdin, mind the constraints, and print exactly the required output format.
Avoid recursion deeper than a few thousand calls; use loops or an explicit stack instead.
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


def failure_feedback(test, result):
    """Feedback for a program that failed a public example, by class of failure."""
    shown = (f"Input:\n{test['input']}\nExpected output:\n{test['output']}\n"
             f"Your output:\n{result.stdout[-1000:]}\nErrors:\n{result.stderr[-1000:]}\n")
    if result.timed_out:
        return ("Your program exceeded the time limit on this example. Estimate the operation count from the constraints "
                "and choose an algorithm that fits; read input with sys.stdin.\n" + shown)
    if "SyntaxError" in result.stderr or "IndentationError" in result.stderr:
        return "Your program does not compile. Check indentation, brackets and Python 3 syntax.\n" + shown
    if "RecursionError" in result.stderr:
        return "Your program exceeds the recursion limit. Rewrite the recursion as a loop, or raise the limit with sys.setrecursionlimit and run the work in a thread with a larger stack.\n" + shown
    if "IndexError" in result.stderr:
        return "Your program reads past the end of a list. Check loop bounds, empty inputs and the number of input values.\n" + shown
    if "ValueError" in result.stderr:
        return "Your program fails to convert a value. Split the input on whitespace and convert each token separately.\n" + shown
    if "KeyError" in result.stderr:
        return "Your program looks up a missing key. Use dict.get or collections.defaultdict for keys that may be absent.\n" + shown
    if "ZeroDivisionError" in result.stderr:
        return "Your program divides by zero. Handle the case where a divisor or a count is zero.\n" + shown
    if result.returncode != 0:
        return "Your program crashes on this example. Read the traceback and fix the line it points to.\n" + shown
    if not result.stdout.strip():
        return "Your program printed nothing on this example. Make sure it calls its main function and prints the answer.\n" + shown
    # Wrong output.
    return ("Your program prints a wrong answer on this example. Trace the example by hand, and check edge cases and "
            "the output format.\n" + shown)


def solve(task, llm, env):
    prompt = f"Problem: {task['title']}\n\n{task['question']}\n\nThink about the time limit before choosing an algorithm."
    reply = llm.chat([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}])
    # Self-repair on the public examples: run the program, and on the first failing example ask for a fix (two rounds).
    for attempt in range(2):
        code = extract_code(reply)
        if code is None:
            feedback = "Your reply contains no program. Write the complete program in one ```python code block."
        else:
            feedback = None
            for test in task.get("public_tests", []):
                result = env.run_python(code, test["input"], 6.0)
                if result.timed_out or result.returncode != 0 or result.stdout.strip() != test["output"].strip():
                    feedback = failure_feedback(test, result)
                    break
        if feedback is None:
            break
        reply = llm.chat([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt},
                          {"role": "assistant", "content": reply},
                          {"role": "user", "content": feedback + " Fix the program. Put the whole corrected program in one ```python code block."}])
    return {"answer": extract_code(reply)}
