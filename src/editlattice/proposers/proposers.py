"""Proposers that turn evaluation feedback into harness edits.

* :class:`MetaHarnessStyleProposer` - re-implementation in the spirit of Meta-Harness
  (2603.28052): the proposer sees the current harness, the history of earlier candidates
  (diff summaries, scores, accept/reject), and failing and passing traces, then proposes
  one edit. The original gives a coding agent a filesystem with the full history; we pass
  a compressed history in the prompt (an 8B model has a short context), which is a
  documented simplification.
* :class:`SelfHarnessStyleProposer` - in the spirit of Self-Harness (2606.09498): first
  mine one model-specific failure pattern from failing traces (weakness mining), then
  propose a minimal edit tied to that pattern.

Both answer with SEARCH/REPLACE blocks (``edit_format.py``). An invalid proposal gets one
retry with the error message; a second failure is recorded as an invalid proposal.
"""

from __future__ import annotations

import difflib
import random
from dataclasses import dataclass, field
from typing import Any

from editlattice.llm.client import LLMClient, RunLLM
from editlattice.proposers.edit_format import EditError, apply_edit_blocks, parse_edit_blocks

EDIT_FORMAT_HELP = """Answer with one or more edit blocks in exactly this format:

<<<<<<< SEARCH
(lines copied exactly from the current harness)
=======
(the new lines)
>>>>>>> REPLACE

Rules:
- The SEARCH text must be copied exactly (same spaces) and must occur only once in the harness.
- Keep the function solve(task, llm, env) and its return value {"answer": ...}.
- Do not import threading, subprocess, os, asyncio or multiprocessing.
- Make one focused change that you expect to fix the observed failures."""

PROPOSER_SYSTEM = "You are an expert engineer who improves the Python harness around a language-model agent. You edit code precisely."


@dataclass
class ProposalRecord:
    """What a proposer produced in one iteration."""

    valid: bool
    child_source: str | None
    error: str
    rationale: str
    llm_calls: int
    tokens_in: int
    tokens_out: int


@dataclass
class CandidateSummary:
    """Compressed history entry for Meta-Harness-style context."""

    diff: str
    parent_score: float
    child_score: float | None
    accepted: bool


def short_diff(parent_source: str, child_source: str, max_lines: int = 40) -> str:
    """Unified diff limited to ``max_lines`` lines."""
    lines = list(difflib.unified_diff(parent_source.splitlines(), child_source.splitlines(), lineterm="", n=1))
    if len(lines) > max_lines:
        lines = lines[:max_lines] + ["... (diff truncated)"]
    return "\n".join(lines)


def describe_run(record: dict[str, Any], question: str, max_chars: int = 900) -> str:
    """One run as text: question, final answer, outcome, error, and the last trace step."""
    parts = [f"Task: {question[:400]}", f"Final answer: {str(record.get('answer'))[:300]}", f"Correct: {'yes' if record['outcome'] else 'no'}"]
    if record.get("error"):
        parts.append(f"Harness error: {record['error'][:200]}")
    trace = record.get("trace") or []
    if trace:
        last = trace[-1]
        parts.append(f"Last model input (end): {last['input_tail'][-250:]}")
        parts.append(f"Last model reply (start): {last['reply'][:250]}")
    text = "\n".join(parts)
    return text[:max_chars]


def pick_examples(records: list[dict[str, Any]], questions: dict[str, str], rng: random.Random, n_fail: int, n_pass: int) -> tuple[list[str], list[str]]:
    """Sample failing and passing run descriptions."""
    failing: list[dict[str, Any]] = []
    passing: list[dict[str, Any]] = []
    for record in records:
        if record["outcome"]:
            passing.append(record)
        else:
            failing.append(record)
    rng.shuffle(failing)
    rng.shuffle(passing)
    fail_texts: list[str] = []
    for record in failing[:n_fail]:
        fail_texts.append(describe_run(record, questions.get(record["task_id"], "")))
    pass_texts: list[str] = []
    for record in passing[:n_pass]:
        pass_texts.append(describe_run(record, questions.get(record["task_id"], "")))
    return fail_texts, pass_texts


COPY_RULE_V2 = "- Copy SEARCH lines from the CURRENT harness shown above, never from earlier candidates."


def fence_for(source: str) -> str:
    """A Markdown fence longer than any run of backticks inside ``source``."""
    longest = max((len(run) for run in source.split("\n") for run in _backtick_runs(run)), default=0)
    return "`" * max(3, longest + 1)


def _backtick_runs(line: str) -> list[str]:
    runs, current = [], ""
    for char in line:
        if char == "`":
            current += char
        elif current:
            runs.append(current)
            current = ""
    if current:
        runs.append(current)
    return runs


def history_without_old_code(diff: str) -> str:
    """A candidate's diff as the new lines only (Phase A2): old lines are not shown, so they cannot be copied."""
    added = [line[1:] for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++")]
    removed = sum(1 for line in diff.splitlines() if line.startswith("-") and not line.startswith("---"))
    head = f"(changed {removed} line(s); new lines:)" if added else f"(removed {removed} line(s))"
    return head + ("\n" + "\n".join(added) if added else "")


def describe_environment(env: Any, public_task: dict[str, Any]) -> str:
    """The same procedure for every benchmark: public methods of the environment object (signature and
    first docstring line) and the field names of a task."""
    import inspect

    lines = ["The harness can use:", "- llm.chat(messages, **params) -> str: one language-model call (calls per task are limited)."]
    import dataclasses
    import typing

    for name, member in inspect.getmembers(type(env), predicate=inspect.isfunction):
        if name.startswith("_"):
            continue
        doc = (inspect.getdoc(member) or "").split("\n")[0]
        signature = inspect.signature(member)
        parameters = []
        for key, parameter in signature.parameters.items():
            if key == "self":
                continue
            text = key
            if parameter.annotation is not inspect.Parameter.empty:
                text += ": " + (parameter.annotation if isinstance(parameter.annotation, str) else getattr(parameter.annotation, "__name__", str(parameter.annotation)))
            if parameter.default is not inspect.Parameter.empty:
                text += f" = {parameter.default!r}"
            parameters.append(text)
        try:
            returned = typing.get_type_hints(member).get("return")
        except Exception:  # noqa: BLE001 - unresolvable annotations: show the text only
            returned = None
        result = str(signature.return_annotation).replace("'", "")
        if dataclasses.is_dataclass(returned):
            result += " with fields " + ", ".join(f".{field.name}" for field in dataclasses.fields(returned))
        lines.append(f"- env.{name}({', '.join(parameters)}) -> {result}" + (f": {doc}" if doc else ""))
    lines.append("- task fields: " + ", ".join(sorted(public_task)))
    return "\n".join(lines)


@dataclass
class ProposerBase:
    """Shared LLM plumbing for proposers.

    ``prompt_version`` 2 (Phase A2, pre-registered addendum) changes only the prompt: a fence longer
    than any backtick run in the harness, history without old code, an environment description
    (``env_doc``) and a copy rule; ``accept_diff`` also reads unified-diff blocks.
    """

    client: LLMClient
    params: dict[str, Any]
    stream_prefix: str
    calls_made: int = 0
    prompt_version: int = 1
    env_doc: str = ""
    accept_diff: bool = False
    # Phase A3: the proposer reasons first (thinking mode); only the text after </think> is the answer.
    strip_thinking: bool = False

    def visible(self, reply: str) -> str:
        """The answer part of a reply (everything after the last </think> when thinking is on)."""
        return reply.rsplit("</think>", 1)[-1].strip() if self.strip_thinking else reply

    def format_help(self) -> str:
        return EDIT_FORMAT_HELP if self.prompt_version == 1 else EDIT_FORMAT_HELP + "\n" + COPY_RULE_V2

    def show_harness(self, source: str) -> str:
        fence = "```" if self.prompt_version == 1 else fence_for(source)
        return f"{fence}python\n{source}\n{fence}"

    def environment(self) -> str:
        return f"{self.env_doc}\n\n" if self.prompt_version >= 2 and self.env_doc else ""

    def new_llm(self, iteration: int, attempt: int) -> RunLLM:
        """A fresh LLM handle on a proposer-specific random stream."""
        return RunLLM(client=self.client, replicate_id=f"{self.stream_prefix}:{iteration}:{attempt}", default_params=dict(self.params))

    def ask_for_edit(self, llm: RunLLM, messages: list[dict[str, str]], parent_source: str) -> tuple[str | None, str, str]:
        """Request an edit, retrying once on an invalid proposal. Returns (child, error, reply)."""
        reply = self.visible(llm.chat(messages))
        try:
            return apply_edit_blocks(parent_source, parse_edit_blocks(reply, accept_diff=self.accept_diff)), "", reply
        except EditError as error:
            messages = messages + [
                {"role": "assistant", "content": reply},
                {"role": "user", "content": f"That edit could not be applied: {error}. Try again, following the format exactly."},
            ]
        reply = self.visible(llm.chat(messages))
        try:
            return apply_edit_blocks(parent_source, parse_edit_blocks(reply, accept_diff=self.accept_diff)), "", reply
        except EditError as error:
            return None, str(error), reply


class MetaHarnessStyleProposer(ProposerBase):
    """Full-history proposer (compressed history in the prompt)."""

    def propose(self, iteration: int, parent_source: str, records: list[dict[str, Any]], questions: dict[str, str], history: list[CandidateSummary], rng: random.Random) -> ProposalRecord:
        """One proposal from the current harness, its runs and the candidate history."""
        fail_texts, pass_texts = pick_examples(records, questions, rng, n_fail=4, n_pass=2)
        score = sum(record["outcome"] for record in records) / max(1, len(records))
        history_lines: list[str] = []
        for number, item in enumerate(history[-6:], start=1):
            verdict = "accepted" if item.accepted else "rejected"
            child = "n/a" if item.child_score is None else f"{item.child_score:.3f}"
            shown = item.diff if self.prompt_version == 1 else history_without_old_code(item.diff)
            history_lines.append(f"Candidate {number} ({verdict}; parent score {item.parent_score:.3f}, candidate score {child}):\n{shown}")
        history_text = "\n\n".join(history_lines) if history_lines else "(no earlier candidates)"
        user = (
            f"{self.environment()}"
            f"Current harness (search-set accuracy {score:.3f}):\n{self.show_harness(parent_source)}\n\n"
            f"Earlier candidate edits and their results:\n{history_text}\n\n"
            f"Failing runs:\n" + "\n---\n".join(fail_texts) + "\n\n"
            f"Passing runs:\n" + "\n---\n".join(pass_texts) + "\n\n"
            f"Propose one edit to the harness that improves accuracy. Briefly explain your reasoning first.\n\n{self.format_help()}"
        )
        llm = self.new_llm(iteration, 0)
        messages = [{"role": "system", "content": PROPOSER_SYSTEM}, {"role": "user", "content": user}]
        child, error, reply = self.ask_for_edit(llm, messages, parent_source)
        usage = llm.usage
        return ProposalRecord(child is not None, child, error, reply[:1500], usage.n_llm_calls, usage.tokens_in, usage.tokens_out)


class SelfHarnessStyleProposer(ProposerBase):
    """Weakness mining, then a minimal edit tied to one failure mechanism."""

    def propose(self, iteration: int, parent_source: str, records: list[dict[str, Any]], questions: dict[str, str], history: list[CandidateSummary], rng: random.Random) -> ProposalRecord:
        """One proposal: diagnose one failure pattern, then fix it minimally."""
        fail_texts, _ = pick_examples(records, questions, rng, n_fail=6, n_pass=0)
        llm = self.new_llm(iteration, 0)
        mining_prompt = (
            "Here are failing runs of a language-model agent:\n\n" + "\n---\n".join(fail_texts) +
            "\n\nIdentify ONE recurring failure pattern that the harness (prompts, tools, control flow, checks) could prevent. "
            "Describe the pattern and the mechanism in at most 5 sentences."
        )
        diagnosis = self.visible(llm.chat([{"role": "system", "content": PROPOSER_SYSTEM}, {"role": "user", "content": mining_prompt}]))
        edit_prompt = (
            f"{self.environment()}"
            f"Current harness:\n{self.show_harness(parent_source)}\n\nObserved failure pattern:\n{diagnosis}\n\n"
            f"Propose a MINIMAL edit to the harness that addresses exactly this failure mechanism and nothing else.\n\n{self.format_help()}"
        )
        messages = [{"role": "system", "content": PROPOSER_SYSTEM}, {"role": "user", "content": edit_prompt}]
        child, error, reply = self.ask_for_edit(llm, messages, parent_source)
        usage = llm.usage
        rationale = f"DIAGNOSIS: {diagnosis[:700]}\nEDIT: {reply[:800]}"
        return ProposalRecord(child is not None, child, error, rationale, usage.n_llm_calls, usage.tokens_in, usage.tokens_out)
