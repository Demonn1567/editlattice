"""Fuzz test of variational replay on random harnesses and random edits.

Each case builds a random seed harness from building blocks that stress the read model
(prompt constants, constants derived at import time, default arguments bound to constants,
a class with methods, comprehensions, lambdas, helpers that edits add or delete, extra LLM
calls), applies several random non-overlapping mutations to get a "final" harness, and checks
on every (task, replicate) cell that exploring the lattice with shared executions gives every
product exactly what running the plain, uninstrumented product gives.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from editlattice.harness_runtime.runner import HarnessVersion, run_one
from editlattice.llm.cache import ResponseCache
from editlattice.llm.client import LLMClient
from editlattice.llm.fake import FakeBackend
from editlattice.units import build_units
from editlattice.variational import Instrumented, explore_cell

COMPARED = ("answer", "outcome", "n_llm_calls", "request_fps")


def responder(messages: list[dict[str, Any]], rng: random.Random) -> str:
    return rng.choice(["alpha", "beta", "ALPHA beta", "gamma 7", "delta"])


class Bench:
    name = "fuzz"

    def public_view(self, task):
        return {k: v for k, v in task.items() if k != "gold"}

    def make_env(self, task):
        return None

    def close_env(self, env):
        return None

    def score(self, task, answer):
        correct = int(str(answer).lower().startswith(task["gold"]))
        return correct, float(correct)


TASKS = [{"task_id": f"t{i}", "kind": ["a", "b", "c"][i % 3], "question": f"q{i}", "gold": ["alpha", "beta", "gamma"][i % 3]} for i in range(6)]

SEED_TEMPLATE = '''"""Random harness."""

PROMPT_A = "Answer about A."
PROMPT_B = "Answer about B."
SUFFIX = " Be brief."
FULL_B = PROMPT_B + SUFFIX
LIMIT = 2


class Formatter:
    def clean(self, text):
        return text.strip()

    def tag(self, text):
        return "[" + text + "]"


def ask(llm, prompt, question, extra=SUFFIX):
    return llm.chat([{"role": "system", "content": prompt + extra}, {"role": "user", "content": question}])


def route_a(task, llm):
    reply = ask(llm, PROMPT_A, task["question"])
    return Formatter().clean(reply)


def route_b(task, llm):
    reply = ask(llm, FULL_B, task["question"])
    words = [w for w in reply.split() if len(w) <= LIMIT * 4]
    return " ".join(words)


def route_c(task, llm):
    pick = lambda text: text.split()[0] if text.split() else text
    return pick(ask(llm, PROMPT_A, task["question"] + "?"))


def solve(task, llm, env):
    if task["kind"] == "a":
        return {"answer": route_a(task, llm)}
    if task["kind"] == "b":
        return {"answer": route_b(task, llm)}
    return {"answer": route_c(task, llm)}
'''

# Each mutation: (old text, new text). Applied mutations must not overlap.
MUTATIONS = [
    ('PROMPT_A = "Answer about A."', 'PROMPT_A = "Answer about A, carefully."'),
    ('SUFFIX = " Be brief."', 'SUFFIX = " Be very brief."'),  # affects FULL_B and ask's default
    ('LIMIT = 2', 'LIMIT = 1'),
    ('        return text.strip()', '        return text.strip().lower()'),
    ('    return Formatter().clean(reply)', '    return Formatter().tag(Formatter().clean(reply))'),
    ('    words = [w for w in reply.split() if len(w) <= LIMIT * 4]', '    words = [w for w in reply.split() if len(w) <= LIMIT * 4 and w.isalpha()]'),
    ('    pick = lambda text: text.split()[0] if text.split() else text', '    pick = lambda text: text.split()[-1] if text.split() else text'),
    ('    reply = ask(llm, PROMPT_A, task["question"])\n    return', '    reply = ask(llm, PROMPT_A, task["question"])\n    if len(reply) < 6:\n        reply = ask(llm, PROMPT_A, "Again: " + task["question"])\n    return'),
    ('def solve(task, llm, env):', 'def shout(text):\n    return text.upper()\n\n\ndef solve(task, llm, env):'),
    ('        return {"answer": route_b(task, llm)}', '        return {"answer": route_b(task, llm).lower()}'),
]


def make_case(rng: random.Random) -> tuple[str, str]:
    """Seed and a final harness with 2-4 random, non-overlapping mutations."""
    chosen = rng.sample(MUTATIONS, rng.randint(2, 4))
    final = SEED_TEMPLATE
    for old, new in chosen:
        assert final.count(old) == 1, old
        final = final.replace(old, new)
    return SEED_TEMPLATE, final


import pytest


@pytest.mark.parametrize("statement_level", [False, True])
def test_fuzz_exploration_matches_plain_products(tmp_path: Path, statement_level: bool) -> None:
    rng = random.Random(2026)
    client = LLMClient(backend=FakeBackend(responder), cache=ResponseCache(tmp_path / "cache"), model="fake")
    bench = Bench()
    cases = 0
    for case in range(25):
        seed, final = make_case(rng)
        units = build_units(seed, final)
        instrumented = Instrumented.build(units, statement_level=statement_level)
        plain = {subset: HarnessVersion.from_source(units.product(subset), tmp_path / "store") for subset in units.all_subsets()}
        for task in TASKS:
            for replicate in range(2):
                results, _ = explore_cell(bench, task, instrumented, replicate, client, {}, 6)
                for subset, version in plain.items():
                    reference = run_one(bench, task, version, replicate, client, {}, crn=True, max_llm_calls=6)
                    for field in COMPARED:
                        assert results[subset][field] == reference[field], (case, sorted(subset), task["task_id"], replicate, field)
        cases += 1
    assert cases == 25


def test_fuzz_detects_each_unread_unit(tmp_path: Path) -> None:
    """Negative control: making any one unit unread (removed from every name) is caught,
    except a unit that can never affect behaviour (a helper that nothing calls)."""
    client = LLMClient(backend=FakeBackend(responder), cache=ResponseCache(tmp_path / "cache"), model="fake")
    bench = Bench()
    final = SEED_TEMPLATE
    for old, new in MUTATIONS:
        final = final.replace(old, new)
    units = build_units(SEED_TEMPLATE, final)
    plain = {subset: HarnessVersion.from_source(units.product(subset), tmp_path / "store") for subset in units.all_subsets()}
    references = {(task["task_id"], replicate, subset): run_one(bench, task, version, replicate, client, {}, crn=True, max_llm_calls=6)
                  for task in TASKS for replicate in range(2) for subset, version in plain.items()}
    undetected = []
    for unit in units.units:
        instrumented = Instrumented.build(units)
        model = instrumented.model
        model.depends = {name: frozenset(units_ - {unit.index}) for name, units_ in model.depends.items()}
        caught = False
        for task in TASKS:
            for replicate in range(2):
                results, _ = explore_cell(bench, task, instrumented, replicate, client, {}, 6)
                caught |= any(results[subset][field] != references[(task["task_id"], replicate, subset)][field]
                              for subset in plain for field in COMPARED)
        if not caught:
            undetected.append(unit.items)
    assert undetected == [("<new>",)], undetected  # only the never-called helper shout


def test_fuzz_detects_silent_markers(tmp_path: Path) -> None:
    """Negative control for statement-level reads: if markers log nothing, marked units go unread
    and exploration must disagree with the plain products somewhere."""
    client = LLMClient(backend=FakeBackend(responder), cache=ResponseCache(tmp_path / "cache"), model="fake")
    bench = Bench()
    final = SEED_TEMPLATE
    for old, new in MUTATIONS:
        final = final.replace(old, new)
    units = build_units(SEED_TEMPLATE, final)
    instrumented = Instrumented.build(units, statement_level=True)
    assert instrumented.marked, "nothing was marked; the control would be vacuous"
    instrumented.sources = {subset: source.replace("__el_mark__(", "(lambda unit: None)(") for subset, source in instrumented.sources.items()}
    plain = {subset: HarnessVersion.from_source(units.product(subset), tmp_path / "store") for subset in units.all_subsets()}
    mismatches = 0
    for task in TASKS:
        for replicate in range(2):
            results, _ = explore_cell(bench, task, instrumented, replicate, client, {}, 6)
            for subset, version in plain.items():
                reference = run_one(bench, task, version, replicate, client, {}, crn=True, max_llm_calls=6)
                mismatches += any(results[subset][field] != reference[field] for field in COMPARED)
    assert mismatches > 0
