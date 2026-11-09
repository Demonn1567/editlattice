"""Recombination: two lineages from one seed, one conflicting region, exact exploration."""

from pathlib import Path

import pytest

from editlattice.harness_runtime.runner import HarnessVersion, run_one
from editlattice.units import combine_lineages
from editlattice.variational import Instrumented, explore_cell
import sys
sys.path.insert(0, str(Path(__file__).parent))
from test_variational import SEED, TASKS, ToyBenchmark, client

PROMPT = 'TEXT_PROMPT = "Answer yes or no. Reply \'ANSWER: <word>\'."'
LINEAGE_A = SEED.replace(PROMPT, 'TEXT_PROMPT = "Answer with yes or no only. Reply \'ANSWER: <word>\'."').replace(
    '    return reply.split("ANSWER:", 1)[1].strip()', '    return reply.split("ANSWER:", 1)[1].strip().lower()')
LINEAGE_B = SEED.replace(PROMPT, 'TEXT_PROMPT = "Reply \'ANSWER: <word>\' with yes or no."').replace(
    'MATH_PROMPT = "Compute the value. Reply \'ANSWER: <n>\'."', 'MATH_PROMPT = "Compute the value exactly. Reply \'ANSWER: <n>\'."')


def test_conflicts_and_products() -> None:
    units, origin = combine_lineages(SEED, [LINEAGE_A, LINEAGE_B])
    prompt_units = [u.index for u in units.units if "TEXT_PROMPT" in u.items]
    assert len(prompt_units) == 2 and frozenset(prompt_units) in units.conflicts
    products = units.all_subsets()
    assert all(not frozenset(prompt_units) <= p for p in products)
    assert len(products) == 2 ** units.m - 2 ** (units.m - 2)  # all subsets minus those with both alternatives
    assert sorted(len(v) for v in origin.values()) == [1] * units.m


@pytest.mark.parametrize("statement_level", [False, True])
def test_recombined_exploration_is_exact(tmp_path: Path, statement_level: bool) -> None:
    units, _ = combine_lineages(SEED, [LINEAGE_A, LINEAGE_B])
    instrumented = Instrumented.build(units, statement_level=statement_level)
    llm_client = client(tmp_path)
    for task in TASKS:
        for replicate in range(2):
            results, _ = explore_cell(ToyBenchmark(), task, instrumented, replicate, llm_client, {}, 5)
            assert set(results) == set(units.all_subsets())
            for subset in units.all_subsets():
                version = HarnessVersion.from_source(units.product(subset), tmp_path / "store")
                alone = run_one(ToyBenchmark(), task, version, replicate, llm_client, {}, crn=True, max_llm_calls=5)
                for field in ("answer", "outcome", "n_llm_calls", "request_fps"):
                    assert results[subset][field] == alone[field]


def test_insertion_at_start_of_replaced_range_conflicts() -> None:
    """DEC-19: one lineage inserts a line where another replaces a line; both orders are ambiguous."""
    from editlattice.units import Hunk, hunks_conflict
    insert = Hunk(13, 13, ("inserted\n",))
    replace = Hunk(13, 14, ("replaced\n",))
    assert hunks_conflict(insert, replace) and hunks_conflict(replace, insert)
    # Insertion right after the replaced range, or strictly before it, is unambiguous.
    assert not hunks_conflict(Hunk(14, 14, ("after\n",)), replace)
    assert not hunks_conflict(Hunk(12, 12, ("before\n",)), replace)
    # The ambiguity itself: the two application orders give different products.
    from editlattice.units import apply_hunks
    seed = "".join(f"line {i}\n" for i in range(16))
    assert apply_hunks(seed, [insert, replace]) != apply_hunks(seed, [replace, insert])
