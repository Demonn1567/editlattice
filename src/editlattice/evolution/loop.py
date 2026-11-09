"""Generic harness-evolution loop: proposer × gate under a total evaluation budget.

Every iteration: the proposer edits the current harness; the impact set is computed;
the gate decides on a live pool; on acceptance the child becomes the parent and is
characterized on fresh streams (D-009). Every candidate is logged (diff, impact set,
decision, cost) - these logs are the real-edit corpus for RQ1.

Budget accounting (D-011): the loop stops when gate runs + characterization runs reach
``total_budget_runs``. Proposer LLM calls are logged separately.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from editlattice.evolution.live_pool import Executor, make_live_pool
from editlattice.gates.pool import EditPool, GateResult
from editlattice.harness_runtime.runner import HarnessVersion
from editlattice.impact.diff import compute_impact
from editlattice.llm.client import TransientBackendError
from editlattice.proposers.proposers import CandidateSummary, ProposalRecord, short_diff

GateFunction = Callable[[EditPool, random.Random], GateResult]

PROPOSER_OUTAGE_WAIT_SECONDS = 60.0
PROPOSER_OUTAGE_RETRIES = 720


@dataclass
class LoopConfig:
    """Parameters of one evolution run."""

    run_id: str
    total_budget_runs: int
    n_streams: int = 1000  # live streams are executed lazily; many streams keep every draw fresh
    k_char: int = 4
    max_iterations: int = 30
    seed: int = 0
    prefetch_batch: int = 24
    characterize_workers: int = 64


def characterize(executor: Executor, tasks: dict[str, dict[str, Any]], version: HarnessVersion, namespace: str, k_char: int, workers: int = 64) -> list[dict[str, Any]]:
    """Run a harness on every task × history stream in parallel (with traces for the proposer)."""
    jobs = []
    for task in tasks.values():
        for stream in range(k_char):
            jobs.append((task, version, stream, namespace, True))
    return executor.run_parallel(jobs, workers)


def evolve(
    config: LoopConfig,
    executor: Executor,
    search_tasks: dict[str, dict[str, Any]],
    validation_tasks: dict[str, dict[str, Any]],
    seed_version: HarnessVersion,
    harness_store: Path,
    proposer: Any,
    gate: GateFunction,
    log_path: Path,
) -> HarnessVersion:
    """Run the loop; append one JSON line per candidate to ``log_path``; return the final harness."""
    rng = random.Random(config.seed)
    parent = seed_version
    namespace = f"{config.run_id}:{parent.harness_hash}"
    parent_records = characterize(executor, search_tasks, parent, namespace, config.k_char, config.characterize_workers)
    spent = len(parent_records)
    questions: dict[str, str] = {}
    for task_id, task in search_tasks.items():
        questions[task_id] = task.get("question", "")
    history: list[CandidateSummary] = []
    log_path.parent.mkdir(parents=True, exist_ok=True)
    for iteration in range(config.max_iterations):
        if spent >= config.total_budget_runs:
            break
        started = time.time()
        parent_score = sum(record["outcome"] for record in parent_records) / len(parent_records)
        outages = 0
        rng_state = rng.getstate()
        while True:
            rng.setstate(rng_state)  # a retried proposal sees the same examples as an uninterrupted one
            try:
                proposal = proposer.propose(iteration, parent.source, parent_records, questions, history, rng)
            except TransientBackendError:
                # The proposer server is unreachable (DEC-6): an outage is infrastructure, not an
                # invalid proposal. Wait and ask again with the same stream (same request, same seed),
                # for up to 12 hours; then the job fails loudly.
                outages += 1
                if outages > PROPOSER_OUTAGE_RETRIES:
                    raise
                time.sleep(PROPOSER_OUTAGE_WAIT_SECONDS)
                continue
            except Exception as error:  # noqa: BLE001 - e.g. context overflow; logged as an invalid proposal
                proposal = ProposalRecord(False, None, f"proposer error {type(error).__name__}: {str(error)[:300]}", "", 0, 0, 0)
            break
        entry: dict[str, Any] = {
            "run_id": config.run_id,
            "iteration": iteration,
            "parent_hash": parent.harness_hash,
            "parent_score": parent_score,
            "proposal_valid": proposal.valid,
            "proposal_error": proposal.error,
            "proposer_calls": proposal.llm_calls,
            "proposer_tokens_in": proposal.tokens_in,
            "proposer_tokens_out": proposal.tokens_out,
            "rationale": proposal.rationale,
        }
        if not proposal.valid:
            entry["accepted"] = False
            append_jsonl(log_path, entry)
            continue
        child = HarnessVersion.from_source(proposal.child_source, harness_store)
        impact = compute_impact(parent.source, child.source)
        pool, prefetcher = make_live_pool(
            f"{config.run_id}:{iteration}", executor, search_tasks, parent, child, namespace, config.n_streams, config.k_char,
            impact, validation_tasks, gate=gate, rng=rng, batch_size=config.prefetch_batch,
        )
        try:
            result = gate(pool, rng)
        except Exception as error:  # noqa: BLE001 - logged; the candidate is rejected, never silently dropped
            entry.update({"child_hash": child.harness_hash, "impact": impact.to_dict(), "accepted": False, "gate_error": f"{type(error).__name__}: {str(error)[:300]}"})
            append_jsonl(log_path, entry)
            continue
        spent += result.ledger.task_runs
        entry.update(
            {
                "child_hash": child.harness_hash,
                "diff": short_diff(parent.source, child.source, max_lines=400),
                "impact": impact.to_dict(),
                "gate": result.gate,
                "accepted": result.accept,
                "gate_task_runs": result.ledger.task_runs,
                "gate_generated_tokens": result.ledger.generated_tokens,
                "gate_nominal_tokens": result.ledger.nominal_tokens,
                "gate_details": result.details,
                "prefetch_unused_runs": prefetcher.unused_prefetched,
                "spent_after": spent,
                "seconds": time.time() - started,
            }
        )
        append_jsonl(log_path, entry)
        history.append(CandidateSummary(short_diff(parent.source, child.source), parent_score, None, result.accept))
        if result.accept:
            parent = child
            namespace = f"{config.run_id}:{parent.harness_hash}"
            parent_records = characterize(executor, search_tasks, parent, namespace, config.k_char, config.characterize_workers)
            spent += len(parent_records)
    return parent


def append_jsonl(path: Path, entry: dict[str, Any]) -> None:
    """Append one JSON line."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, default=str) + "\n")
