"""Exactness of variational replay on a toy harness with a fake LLM.

Three net units: a text-route prompt (read only by text tasks), a math-route retry (read only
by math tasks) and a shared parser (read by every run). For every (task, replicate) cell:
(1) an instrumented product run is identical to the plain product run (same requests, answer,
outcome, call count); (2) exploring the lattice with shared executions gives every product
exactly the record of running it on its own; (3) sharing really happens.
"""

from __future__ import annotations

import random
import textwrap
from pathlib import Path
from typing import Any

import pytest

from editlattice.harness_runtime.runner import HarnessVersion, run_one
from editlattice.llm.cache import ResponseCache
from editlattice.llm.client import LLMClient
from editlattice.llm.fake import FakeBackend
from editlattice.units import build_units
from editlattice.variational import Instrumented, ReadModel, explore_cell, instrument, run_product

SEED = textwrap.dedent(
    '''
    """Toy two-route harness."""

    MATH_PROMPT = "Compute the value. Reply 'ANSWER: <n>'."
    TEXT_PROMPT = "Answer yes or no. Reply 'ANSWER: <word>'."


    def parse(reply):
        if "ANSWER:" not in reply:
            return None
        return reply.split("ANSWER:", 1)[1].strip()


    def solve_math(task, llm):
        reply = llm.chat([{"role": "system", "content": MATH_PROMPT}, {"role": "user", "content": task["question"]}])
        return parse(reply)


    def solve_text(task, llm):
        reply = llm.chat([{"role": "system", "content": TEXT_PROMPT}, {"role": "user", "content": task["question"]}])
        return parse(reply)


    def solve(task, llm, env):
        if task["kind"] == "math":
            return {"answer": solve_math(task, llm)}
        return {"answer": solve_text(task, llm)}
    '''
).lstrip("\n")

FINAL = (
    SEED.replace('TEXT_PROMPT = "Answer yes or no. Reply \'ANSWER: <word>\'."', 'TEXT_PROMPT = "Answer with one word. Reply \'ANSWER: <word>\'."')
    .replace(
        'def solve_math(task, llm):\n    reply = llm.chat([{"role": "system", "content": MATH_PROMPT}, {"role": "user", "content": task["question"]}])\n    return parse(reply)',
        'def solve_math(task, llm):\n    reply = llm.chat([{"role": "system", "content": MATH_PROMPT}, {"role": "user", "content": task["question"]}])\n'
        '    answer = parse(reply)\n'
        '    if answer is None:\n'
        '        reply = llm.chat([{"role": "system", "content": MATH_PROMPT}, {"role": "user", "content": "Again: " + task["question"]}])\n'
        '        answer = parse(reply)\n'
        '    return answer',
    )
    .replace('    return reply.split("ANSWER:", 1)[1].strip()', '    return reply.split("ANSWER:", 1)[1].strip().lower()')
)


assert FINAL.count("lower()") == 1 and "Again: " in FINAL, "fixture edits did not apply"


def responder(messages: list[dict[str, Any]], rng: random.Random) -> str:
    """Sometimes malformed, sometimes upper case, so every unit can matter."""
    roll = rng.random()
    if roll < 0.3:
        return "I think the answer is unclear"
    return "ANSWER: " + rng.choice(["Yes", "no", "42", "41"])


class ToyBenchmark:
    """Two kinds of task; correct answers depend on case and format."""

    name = "toy"

    def public_view(self, task: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in task.items() if key != "gold"}

    def make_env(self, task: dict[str, Any]) -> None:
        return None

    def close_env(self, env: Any) -> None:
        return None

    def score(self, task: dict[str, Any], answer: Any) -> tuple[int, float]:
        correct = int(answer == task["gold"])
        return correct, float(correct)


TASKS = [{"task_id": f"m{i}", "kind": "math", "question": f"What is {i} + 40?", "gold": "42"} for i in range(4)] + [
    {"task_id": f"t{i}", "kind": "text", "question": f"Is {i} even?", "gold": "yes"} for i in range(4)
]


def client(tmp_path: Path) -> LLMClient:
    return LLMClient(backend=FakeBackend(responder), cache=ResponseCache(tmp_path / "cache"), model="fake")


def test_units_and_read_model() -> None:
    units = build_units(SEED, FINAL)
    assert units.m == 3
    assert units.product(frozenset(range(3))) == FINAL
    model = ReadModel.build(units)
    by_item = {unit.items[0]: unit.index for unit in units.units}
    assert model.depends["TEXT_PROMPT"] == {by_item["TEXT_PROMPT"]}
    assert model.depends["solve_math"] == {by_item["solve_math"]}
    assert model.depends["parse"] == {by_item["parse"]}
    assert "solve" not in model.depends and not model.global_units


def test_instrumented_runs_equal_plain_runs(tmp_path: Path) -> None:
    units = build_units(SEED, FINAL)
    instrumented = Instrumented.build(units)
    llm_client = client(tmp_path)
    benchmark = ToyBenchmark()
    for subset in units.all_subsets():
        version = HarnessVersion.from_source(units.product(subset), tmp_path / "store")
        for task in TASKS:
            for replicate in range(3):
                plain = run_one(benchmark, task, version, replicate, llm_client, {}, crn=True, max_llm_calls=5)
                variant = run_product(benchmark, task, instrumented, subset, replicate, llm_client, {}, 5)
                for key in ("answer", "outcome", "n_llm_calls", "request_fps"):
                    assert plain[key] == variant[key], (subset, task["task_id"], replicate, key)


@pytest.mark.parametrize("statement_level", [False, True])
def test_exploration_is_exact_and_shares_runs(tmp_path: Path, statement_level: bool) -> None:
    units = build_units(SEED, FINAL)
    instrumented = Instrumented.build(units, statement_level=statement_level)
    llm_client = client(tmp_path)
    benchmark = ToyBenchmark()
    executed_total = 0
    for task in TASKS:
        for replicate in range(3):
            results, executed = explore_cell(benchmark, task, instrumented, replicate, llm_client, {}, 5)
            executed_total += len(executed)
            for subset in units.all_subsets():
                alone = run_product(benchmark, task, instrumented, subset, replicate, llm_client, {}, 5)
                for key in ("answer", "outcome", "n_llm_calls", "request_fps"):
                    assert results[subset][key] == alone[key], (subset, task["task_id"], replicate, key)
    cells = len(TASKS) * 3
    assert executed_total < cells * 2 ** units.m  # sharing happened
    assert executed_total >= cells  # at least one run per cell


def test_accessor_keeps_local_names_untouched() -> None:
    source = textwrap.dedent(
        '''
        X = 1

        def f(X):
            return X + 1

        def g():
            return X

        def solve(task, llm, env):
            return {"answer": [X for _ in range(1)]}
        '''
    )
    rewritten = instrument(source, frozenset({"X"}))
    assert "def f(X):\n    return X + 1" in rewritten
    assert "return __el_get__('X')" in rewritten
    assert "[__el_get__('X') for _ in range(1)]" in rewritten


def test_exactness_check_detects_missing_reads(tmp_path: Path) -> None:
    """Negative control: if reads were not logged, sharing would give wrong outcomes."""
    units = build_units(SEED, FINAL)
    instrumented = Instrumented.build(units)
    instrumented.model.depends.clear()  # pretend no name depends on any unit
    llm_client = client(tmp_path)
    benchmark = ToyBenchmark()
    wrong = 0
    for task in TASKS:
        for replicate in range(3):
            results, _ = explore_cell(benchmark, task, instrumented, replicate, llm_client, {}, 5)
            for subset in units.all_subsets():
                alone = run_product(benchmark, task, instrumented, subset, replicate, llm_client, {}, 5)
                wrong += results[subset]["request_fps"] != alone["request_fps"] or results[subset]["answer"] != alone["answer"]
    assert wrong > 0


BRANCHY_SEED = textwrap.dedent(
    '''
    def solve(task, llm, env):
        reply = llm.chat([{"role": "user", "content": task["question"]}])
        if task["kind"] == "rare":
            reply = reply + " (checked)"
        return {"answer": reply.split()[-1] if reply.split() else ""}
    '''
).lstrip("\n")
BRANCHY_FINAL = BRANCHY_SEED.replace('reply = reply + " (checked)"', 'reply = reply.lower() + " (checked)"')


def test_statement_level_reads_share_more(tmp_path: Path) -> None:
    units = build_units(BRANCHY_SEED, BRANCHY_FINAL)
    tasks = [{"task_id": f"r{i}", "kind": "rare" if i == 0 else "common", "question": "q", "gold": "x"} for i in range(5)]
    llm_client = client(tmp_path)
    executed = {}
    for level in (False, True):
        instrumented = Instrumented.build(units, statement_level=level)
        assert bool(instrumented.marked) == level
        total = 0
        for task in tasks:
            results, runs = explore_cell(ToyBenchmark(), task, instrumented, 0, llm_client, {}, 5)
            total += len(runs)
            for subset in units.all_subsets():
                alone = run_product(ToyBenchmark(), task, instrumented, subset, 0, llm_client, {}, 5)
                assert results[subset]["answer"] == alone["answer"]
        executed[level] = total
    # Name-level: solve is read by every run (2 runs per task). Statement-level: only the rare task needs 2.
    assert executed == {False: 10, True: 6}


INTROSPECTIVE_SEED = textwrap.dedent(
    '''
    import traceback


    def helper(text):
        return text.split()[5]


    def solve(task, llm, env):
        reply = llm.chat([{"role": "user", "content": task["question"]}])
        try:
            return {"answer": helper(reply)}
        except Exception:
            again = llm.chat([{"role": "user", "content": traceback.format_exc()}])
            return {"answer": again}
    '''
).lstrip("\\n")
INTROSPECTIVE_FINAL = INTROSPECTIVE_SEED.replace("return text.split()[5]", "return text.split()[6]")


@pytest.mark.parametrize("statement_level", [False, True])
def test_introspective_harness_runs_plain_products(tmp_path: Path, statement_level: bool) -> None:
    """A harness that sends its own traceback (with line numbers) to the model must be run unchanged."""
    units = build_units(INTROSPECTIVE_SEED, INTROSPECTIVE_FINAL)
    instrumented = Instrumented.build(units, statement_level=statement_level, store_dir=tmp_path / "store")
    assert instrumented.model.dynamic and not instrumented.marked
    for subset in units.all_subsets():
        assert instrumented.sources[subset] == units.product(subset)
    llm_client = client(tmp_path)
    for task in TASKS[:3]:
        results, runs = explore_cell(ToyBenchmark(), task, instrumented, 0, llm_client, {}, 5)
        assert len(runs) == 2 ** units.m  # brute force
        for subset in units.all_subsets():
            version = HarnessVersion.from_source(units.product(subset), tmp_path / "store")
            plain = run_one(ToyBenchmark(), task, version, 0, llm_client, {}, crn=True, max_llm_calls=5)
            assert results[subset]["request_fps"] == plain["request_fps"]


def test_comment_only_unit_is_never_read_in_either_mode(tmp_path: Path) -> None:
    final = BRANCHY_SEED.replace('    reply = llm.chat(', '    # ask once\n    reply = llm.chat(')
    units = build_units(BRANCHY_SEED, final)
    assert units.m == 1
    for level in (False, True):
        instrumented = Instrumented.build(units, statement_level=level)
        assert not instrumented.marked
        _, runs = explore_cell(ToyBenchmark(), {"task_id": "c", "kind": "common", "question": "q", "gold": "x"}, instrumented, 0, client(tmp_path), {}, 5)
        assert len(runs) == 1 and runs[0]["reads"] == []


def test_marker_must_dominate_a_hunk_that_leaves_its_block(tmp_path) -> None:
    """Regression (found by the Phase B audit of an LCB lineage): a unit whose hunk replaces a line inside
    an inner block and adds statements one level up must not be tracked by a marker placed inside the
    inner block, because execution can reach the added statements without passing the marker."""
    import random as _random
    from editlattice.harness_runtime.runner import HarnessVersion, run_one
    from editlattice.llm.cache import ResponseCache
    from editlattice.llm.client import LLMClient
    from editlattice.llm.fake import FakeBackend
    from editlattice.tree import explore_tree
    from editlattice.units import add_markers, build_units
    from editlattice.variational import Instrumented

    seed = '''"""Extraction harness."""


def extract(reply):
    if "```python" in reply:
        return reply.split("```python", 1)[1].split("```", 1)[0].strip()
    if "```" in reply:
        parts = reply.split("```")
        if len(parts) >= 3:
            return parts[-2].strip()
    return None


def solve(task, llm, env):
    return {"answer": extract(llm.chat([{"role": "user", "content": task["question"]}]))}
'''
    final = '''"""Extraction harness."""


def extract(reply):
    blocks = []
    if "```python" in reply:
        blocks.append(reply.split("```python", 1)[1].split("```", 1)[0].strip())
    if "```" in reply:
        parts = reply.split("```")
        if len(parts) >= 3:
            blocks.append(parts[-2].strip())
    if blocks:
        return blocks[0]
    return None


def solve(task, llm, env):
    return {"answer": extract(llm.chat([{"role": "user", "content": task["question"]}]))}
'''
    units = build_units(seed, final)
    _, marked = add_markers(units)
    spanning = [u.index for u in units.units if any("if blocks:" in "".join(h.replacement) for h in u.hunks)]
    assert spanning and not set(spanning) & marked, (spanning, marked)

    class Bench:
        name = "extract"

        def public_view(self, task):
            return {"task_id": task["task_id"], "question": task["question"]}

        def make_env(self, task):
            return None

        def close_env(self, env):
            return None

        def score(self, task, answer):
            return int(answer == "x = 1"), float(answer == "x = 1")

    replies = ["```python\nx = 1", "```python\nx = 1\n```", "text ```x = 1``` more"]  # first: no closing fence
    client = LLMClient(backend=FakeBackend(lambda messages, rng: replies[int(messages[-1]["content"][-1])]),
                       cache=ResponseCache(tmp_path / "cache"), model="fake")
    instrumented = Instrumented.build(units, statement_level=True, lazy=True)
    for number in range(3):
        task = {"task_id": f"t{number}", "question": f"q{number}"}
        leaves = explore_tree(Bench(), task, instrumented, 0, client, {}, 4)
        for subset in units.all_subsets():
            leaf = next(leaf for leaf in leaves if leaf.contains(subset))
            plain = run_one(Bench(), task, HarnessVersion.from_source(units.product(subset), tmp_path / "store"), 0, client, {}, crn=True, max_llm_calls=4)
            assert leaf.record["answer"] == plain["answer"], (number, sorted(subset))
