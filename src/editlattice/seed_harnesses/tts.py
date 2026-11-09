"""Matched-budget test-time-scaling baselines (Rethinking, 2607.12227), built from a seed.

``make_tts_harness(seed_source, k, benchmark)`` renames the seed's ``solve`` to
``solve_once`` and appends a ``solve`` that calls it k times and selects an answer without
labels:
* BIRD: majority vote over the *execution results* of the k submitted queries
  (self-consistency); ties go to the earliest sample.
* LCB: the program passing the most public tests; ties broken by the most common
  public-test output signature, then the earliest sample.
Repeated identical requests inside one run get fresh samples because the CRN cache key
includes the occurrence index (D-003), so the k attempts are independent draws.
"""

from __future__ import annotations

BIRD_SELECTOR = '''

def result_signature(env, sql):
    """Execution result of a query as a hashable value, or None if it fails."""
    if not sql:
        return None
    try:
        columns, rows, truncated = env.execute(sql, max_rows=200)
    except Exception:
        return None
    return tuple(sorted(repr(row) for row in rows))


def solve(task, llm, env):
    """Run the seed TTS_K times and return the answer with the most common result."""
    answers = []
    for attempt in range(TTS_K):
        answers.append(solve_once(task, llm, env).get("answer"))
    counts = {}
    first_index = {}
    for index, answer in enumerate(answers):
        signature = result_signature(env, answer)
        if signature is None:
            continue
        counts[signature] = counts.get(signature, 0) + 1
        first_index.setdefault(signature, index)
    if not counts:
        return {"answer": answers[0]}
    best = None
    for signature, count in counts.items():
        if best is None or count > counts[best] or (count == counts[best] and first_index[signature] < first_index[best]):
            best = signature
    return {"answer": answers[first_index[best]]}
'''

LCB_SELECTOR = '''

def public_signature(env, code):
    """(number of public tests passed, outputs on the public inputs) of a program."""
    if not code:
        return -1, ()
    passed = 0
    outputs = []
    for test in env.public_tests:
        result = env.run_python(code, test["input"], TEST_TIMEOUT_SECONDS)
        outputs.append(result.stdout.strip())
        if not result.timed_out and result.returncode == 0 and outputs_equal(test["output"], result.stdout):
            passed += 1
    return passed, tuple(outputs)


def solve(task, llm, env):
    """Run the seed TTS_K times; keep the program passing most public tests (ties: majority output)."""
    programs = []
    for attempt in range(TTS_K):
        programs.append(solve_once(task, llm, env).get("answer"))
    signatures = []
    output_counts = {}
    for program in programs:
        signature = public_signature(env, program)
        signatures.append(signature)
        output_counts[signature[1]] = output_counts.get(signature[1], 0) + 1
    best_index = 0
    for index in range(1, len(programs)):
        current = (signatures[index][0], output_counts[signatures[index][1]])
        incumbent = (signatures[best_index][0], output_counts[signatures[best_index][1]])
        if current > incumbent:
            best_index = index
    return {"answer": programs[best_index]}
'''


def make_tts_harness(seed_source: str, k: int, benchmark: str) -> str:
    """Source of a best-of-k harness built from ``seed_source``."""
    marker = "\ndef solve(task, llm, env):"
    if seed_source.count(marker) != 1:
        raise ValueError("seed must define exactly one top-level solve(task, llm, env)")
    renamed = seed_source.replace(marker, "\ndef solve_once(task, llm, env):")
    selector = BIRD_SELECTOR if benchmark == "bird" else LCB_SELECTOR
    return renamed + f"\n\nTTS_K = {int(k)}\n" + selector
