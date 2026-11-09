"""Seed harness for LiveCodeBench: generate, run public tests, repair.

The model writes a complete Python 3 program that reads stdin and writes stdout. The
harness runs it on the public examples and, on failure, asks for a fix with the failing
example as feedback. It returns the candidate that passes the most public tests.
"""

MAX_REPAIRS = 3
MAX_FEEDBACK_CHARS = 600
TEST_TIMEOUT_SECONDS = 6.0

SYSTEM_PROMPT = """You are an expert competitive programmer.
Write a complete Python 3 program that reads the input from standard input and prints the answer to standard output.
Rules:
- Read all input with sys.stdin, not with input() in a slow loop, when the input is large.
- Use efficient algorithms; the time limit is a few seconds.
- Print exactly the required output and nothing else.
- Put the whole program in one ```python code block."""

REPAIR_INSTRUCTION = "Your program is wrong on this example. Find the bug, then write the corrected complete program in one ```python code block."


def build_prompt(task):
    """First user message: the problem statement."""
    return f"Problem: {task['title']}\n\n{task['question']}"


def extract_code(reply):
    """Return the last ```python block of a reply, or None."""
    marker = "```python"
    if marker in reply:
        after = reply.rsplit(marker, 1)[1]
        return after.split("```", 1)[0].strip()
    if "```" in reply:
        parts = reply.split("```")
        if len(parts) >= 3:
            return parts[-2].strip()
    return None


def truncate(text):
    """Keep feedback short."""
    if len(text) > MAX_FEEDBACK_CHARS:
        return text[:MAX_FEEDBACK_CHARS] + "..."
    return text


def outputs_equal(expected, actual):
    """Whitespace-insensitive comparison of outputs."""
    return expected.split() == actual.split()


def run_public_tests(env, code):
    """Run all public tests; return (number passed, first failure description or None)."""
    passed = 0
    first_failure = None
    for test in env.public_tests:
        result = env.run_python(code, test["input"], TEST_TIMEOUT_SECONDS)
        if result.timed_out:
            failure = describe_timeout(test)
        elif result.returncode != 0:
            failure = describe_crash(test, result.stderr)
        elif not outputs_equal(test["output"], result.stdout):
            failure = describe_wrong_answer(test, result.stdout)
        else:
            passed += 1
            continue
        if first_failure is None:
            first_failure = failure
    return passed, first_failure


def describe_timeout(test):
    """Feedback for a time-out."""
    return f"Input:\n{truncate(test['input'])}\nThe program exceeded the time limit. Use a faster algorithm."


def describe_crash(test, stderr):
    """Feedback for a runtime error."""
    return f"Input:\n{truncate(test['input'])}\nThe program crashed:\n{truncate(stderr.strip()[-400:])}"


def describe_wrong_answer(test, stdout):
    """Feedback for a wrong answer."""
    return f"Input:\n{truncate(test['input'])}\nExpected output:\n{truncate(test['output'])}\nYour output:\n{truncate(stdout)}"


def missing_code_message():
    """Feedback when no code block was found."""
    return "I could not find a ```python code block in your reply. Write the complete program in one ```python code block."


def solve(task, llm, env):
    """Generate a program, then repair it on failing public tests."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(task)},
    ]
    best_code = None
    best_passed = -1
    for attempt in range(MAX_REPAIRS + 1):
        reply = llm.chat(messages)
        messages.append({"role": "assistant", "content": reply})
        code = extract_code(reply)
        if code is None:
            messages.append({"role": "user", "content": missing_code_message()})
            continue
        passed, failure = run_public_tests(env, code)
        if passed > best_passed:
            best_code = code
            best_passed = passed
        if failure is None:
            return {"answer": code}
        messages.append({"role": "user", "content": REPAIR_INSTRUCTION + "\n\n" + failure})
    return {"answer": best_code}
