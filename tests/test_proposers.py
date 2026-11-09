"""Edit-format parsing/application and proposers with a fake LLM."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pytest

from editlattice.llm.cache import ResponseCache
from editlattice.llm.client import LLMClient
from editlattice.llm.fake import FakeBackend
from editlattice.proposers.edit_format import EditError, apply_edit_blocks, parse_edit_blocks
from editlattice.proposers.proposers import MetaHarnessStyleProposer, SelfHarnessStyleProposer

SEED = (Path(__file__).resolve().parents[1] / "src" / "editlattice" / "seed_harnesses" / "bird_react.py").read_text()

GOOD_EDIT = """I will show more rows.
<<<<<<< SEARCH
MAX_ROWS_SHOWN = 10
=======
MAX_ROWS_SHOWN = 20
>>>>>>> REPLACE
"""


def test_parse_and_apply() -> None:
    blocks = parse_edit_blocks(GOOD_EDIT)
    assert len(blocks) == 1
    child = apply_edit_blocks(SEED, blocks)
    assert "MAX_ROWS_SHOWN = 20" in child


def test_rejects_missing_ambiguous_and_forbidden() -> None:
    with pytest.raises(EditError):
        apply_edit_blocks(SEED, parse_edit_blocks(GOOD_EDIT.replace("MAX_ROWS_SHOWN = 10", "NOT_PRESENT = 1", 1)))
    with pytest.raises(EditError):
        apply_edit_blocks(SEED, parse_edit_blocks("<<<<<<< SEARCH\n    return text\n=======\n    return text\n>>>>>>> REPLACE"))
    forbidden = "<<<<<<< SEARCH\nMAX_STEPS = 8\n=======\nimport subprocess\nMAX_STEPS = 8\n>>>>>>> REPLACE"
    with pytest.raises(EditError):
        apply_edit_blocks(SEED, parse_edit_blocks(forbidden))
    with pytest.raises(EditError):
        apply_edit_blocks(SEED, [])


def fake_client(tmp_path: Path, reply: str) -> LLMClient:
    def responder(messages: list[dict[str, Any]], rng: random.Random) -> str:
        return reply
    return LLMClient(backend=FakeBackend(responder), model="fake", cache=ResponseCache(tmp_path / "cache"))


def test_proposers_produce_valid_children(tmp_path: Path) -> None:
    records = [{"task_id": "a", "outcome": 0, "answer": "SELECT 1", "error": "", "trace": [{"input_tail": "x", "reply": "y"}]}]
    questions = {"a": "How many?"}
    for proposer_class in (MetaHarnessStyleProposer, SelfHarnessStyleProposer):
        proposer = proposer_class(client=fake_client(tmp_path, GOOD_EDIT), params={}, stream_prefix="p")
        result = proposer.propose(0, SEED, records, questions, [], random.Random(0))
        assert result.valid and "MAX_ROWS_SHOWN = 20" in result.child_source


def test_invalid_proposal_is_recorded(tmp_path: Path) -> None:
    proposer = MetaHarnessStyleProposer(client=fake_client(tmp_path, "no blocks here"), params={}, stream_prefix="p")
    result = proposer.propose(0, SEED, [], {}, [], random.Random(0))
    assert not result.valid and result.llm_calls == 2


def test_uniformly_over_indented_blocks_are_shifted_back() -> None:
    from editlattice.proposers.edit_format import EditBlock, EditError, apply_edit_blocks
    source = 'def solve(task, llm, env):\n    x = 1\n    return {"answer": x}\n'
    block = EditBlock(search='    def solve(task, llm, env):\n        x = 1', replace='    def solve(task, llm, env):\n        x = 2')
    assert apply_edit_blocks(source, [block]) == source.replace("x = 1", "x = 2")
    # Not uniform or still not found: still an error.
    import pytest
    with pytest.raises(EditError):
        apply_edit_blocks(source, [EditBlock(search='    def solve(task, llm, env):\n      y = 1', replace='')])
    # Ambiguous after shifting: an error, never a guess.
    twice = source + '\ndef other():\n    x = 1\n'
    with pytest.raises(EditError):
        apply_edit_blocks(twice, [EditBlock(search='        x = 1', replace='        x = 3')])


def test_first_line_over_indented_string_assignment_is_shifted_back() -> None:
    import pytest
    from editlattice.proposers.edit_format import EditBlock, EditError, apply_edit_blocks
    source = 'PROMPT = """Line one.\nLine two."""\n\n\ndef solve(task, llm, env):\n    return {"answer": PROMPT}\n'
    block = EditBlock(search='    PROMPT = """Line one.\nLine two."""', replace='    PROMPT = """Line one.\nBe careful.\nLine two."""')
    assert apply_edit_blocks(source, [block]) == source.replace("Line one.\n", "Line one.\nBe careful.\n")
    # Only a match at the start of a line counts.
    inline = 'X = 1; PROMPT = """a\nb"""\n\n\ndef solve(task, llm, env):\n    return {"answer": X}\n'
    with pytest.raises(EditError):
        apply_edit_blocks(inline, [EditBlock(search='    PROMPT = """a\nb"""', replace='    PROMPT = """a\nc"""')])
    # REPLACE without the same first-line indentation: an error, never a guess.
    with pytest.raises(EditError):
        apply_edit_blocks(source, [EditBlock(search='    PROMPT = """Line one.\nLine two."""', replace='PROMPT = 1')])


def test_diff_blocks_only_when_enabled() -> None:
    from editlattice.proposers.edit_format import apply_edit_blocks, parse_edit_blocks
    source = 'X = 1\n\n\ndef solve(task, llm, env):\n    return {"answer": X}\n'
    reply = "<<<<<<< SEARCH\n-X = 1\n+X = 2\n>>>>>>> REPLACE"
    assert parse_edit_blocks(reply) == []
    blocks = parse_edit_blocks(reply, accept_diff=True)
    assert apply_edit_blocks(source, blocks) == source.replace("X = 1", "X = 2")
    # Ordinary blocks parse the same way in both modes.
    normal = "<<<<<<< SEARCH\nX = 1\n=======\nX = 3\n>>>>>>> REPLACE"
    assert parse_edit_blocks(normal) == parse_edit_blocks(normal, accept_diff=True)
    # Lines without a diff prefix: not a diff block.
    assert parse_edit_blocks("<<<<<<< SEARCH\nX = 1\n+X = 2\n>>>>>>> REPLACE", accept_diff=True) == []


def test_thinking_is_stripped_only_when_enabled() -> None:
    from editlattice.proposers.proposers import MetaHarnessStyleProposer
    reply = "<think>draft <<<<<<< SEARCH\nX\n=======\nY\n>>>>>>> REPLACE</think>\nfinal answer"
    assert MetaHarnessStyleProposer(client=None, params={}, stream_prefix="p").visible(reply) == reply
    assert MetaHarnessStyleProposer(client=None, params={}, stream_prefix="p", strip_thinking=True).visible(reply) == "final answer"
