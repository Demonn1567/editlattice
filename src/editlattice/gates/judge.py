"""B12: label-free LLM-judge gate (in the spirit of TTHE, 2607.08124).

For ``n_tasks`` sampled tasks, the judge model sees the question and the final answers of
the parent and the child (one independent run each), in random order to cancel position
bias, and answers A, B or TIE. The gate never sees outcomes. It accepts when the child
wins at least ``margin`` more comparisons than the parent. Judge calls are charged as
generated tokens; the two runs per task are charged as task-runs.

Offline evaluation passes ``verdicts`` (precomputed by scripts/b12_judgments.py, keyed by
the ordered comparison): every distinct comparison is judged once at temperature 0, so the
lookup returns exactly what a live call would. Identical answers are a TIE without a call
(a verdict between identical answers can only reflect position bias).
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Any

from editlattice.gates.pool import EditPool, GateResult, Ledger, independent_parent_stream
from editlattice.llm.client import LLMClient, RunLLM

JUDGE_SYSTEM = "You compare two answers to the same task and decide which is more likely to be correct. Reply with exactly one word: A, B, or TIE."


def judge_prompt(question: str, first: Any, second: Any) -> str:
    """The user message of one comparison."""
    return (
        f"Task:\n{question[:2000]}\n\n"
        f"Answer A:\n{str(first)[:2000]}\n\nAnswer B:\n{str(second)[:2000]}\n\n"
        "Which answer is more likely to be correct? Reply A, B, or TIE."
    )


def parse_verdict(reply: str) -> str:
    """A, B or TIE (anything unparseable counts as TIE)."""
    word = reply.strip().upper()
    if word.startswith("TIE"):
        return "TIE"
    if word.startswith("A"):
        return "A"
    if word.startswith("B"):
        return "B"
    return "TIE"


def comparison_key(task: str, first: Any, second: Any) -> str:
    """Stable identifier of one ordered comparison (shared with scripts/b12_judgments.py)."""
    return hashlib.sha256(json.dumps([task, str(first), str(second)]).encode()).hexdigest()[:24]


@dataclass
class JudgeGate:
    """Callable gate; holds the judge client and task questions."""

    client: LLMClient | None
    questions: dict[str, str]
    n_tasks: int = 20
    margin: int = 2
    stream_prefix: str = "judge"
    verdicts: dict[str, tuple[str, int]] | None = field(default=None, repr=False)

    def __call__(self, pool: EditPool, rng: random.Random) -> GateResult:
        """Run the judge on sampled tasks and decide."""
        ledger = Ledger()
        chosen = rng.sample(pool.tasks, min(self.n_tasks, len(pool.tasks)))
        child_wins = 0
        parent_wins = 0
        judge_tokens = 0
        for index, task in enumerate(chosen):
            child_stream = rng.choice(pool.draw_streams)
            parent_stream = independent_parent_stream(rng, pool, child_stream)
            child_run = pool.child[task][child_stream]
            parent_run = pool.parent[task][parent_stream]
            ledger.charge_full(child_run)
            ledger.charge_full(parent_run)
            child_first = rng.random() < 0.5
            first, second = (child_run.answer, parent_run.answer) if child_first else (parent_run.answer, child_run.answer)
            verdict, tokens = self.ask(pool.edit_id, index, task, first, second)
            judge_tokens += tokens
            if verdict == "TIE":
                continue
            child_won = (verdict == "A") == child_first
            if child_won:
                child_wins += 1
            else:
                parent_wins += 1
        ledger.generated_tokens += judge_tokens
        accept = child_wins - parent_wins >= self.margin
        return GateResult("B12_llm_judge", accept, ledger, {"child_wins": child_wins, "parent_wins": parent_wins, "judge_tokens": judge_tokens})

    def ask(self, edit_id: str, index: int, task: str, first: Any, second: Any) -> tuple[str, int]:
        """One judge decision; returns (verdict, judge tokens)."""
        if str(first) == str(second):
            return "TIE", 0
        if self.verdicts is not None:
            return self.verdicts[comparison_key(task, first, second)]
        llm = RunLLM(client=self.client, replicate_id=f"{self.stream_prefix}:{edit_id}:{index}", default_params={"temperature": 0.0, "max_tokens": 5})
        reply = llm.chat([{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": judge_prompt(self.questions.get(task, ""), first, second)}])
        return parse_verdict(reply), llm.usage.tokens_in + llm.usage.tokens_out


def gate_llm_judge(pool: EditPool, rng: random.Random, n_tasks: int = 20, margin: int = 2, verdicts: dict | None = None) -> GateResult:
    """B12 from a precomputed verdict table (offline gate evaluation)."""
    return JudgeGate(None, {}, n_tasks=n_tasks, margin=margin, verdicts=verdicts)(pool, rng)
