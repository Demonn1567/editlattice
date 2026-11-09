"""BIRD Mini-Dev (SQLite) text-to-SQL benchmark with execution-accuracy scoring.

Environment primitives offered to a harness (tools are built on top of these *inside*
the harness, so that they are editable harness code):

* ``env.execute(sql, max_rows)`` -> ``(columns, rows, truncated)``; raises ``SQLError``.
* ``env.description(table)`` -> the BIRD column-description CSV text for a table, or "".
* ``env.db_id`` -> database name.

Isolation: every run opens its own read-only, immutable SQLite connection, so runs can
neither see nor leave state behind (assumption A3).

Scoring follows BIRD's execution accuracy: the predicted query's result *set* must equal
the gold query's result set. Gold results are computed once and cached.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "data" / "bird" / "minidev" / "MINIDEV"
FROZEN_GOLD = Path(__file__).resolve().parents[3] / "configs" / "bird_gold_results.json"


class SQLError(RuntimeError):
    """A query failed or timed out."""


HANDLER_PERIOD = 10_000  # SQLite VM instructions between progress-handler calls
# Instruction budgets (D-015), calibrated on the 494 gold queries (max 8,865 calls):
HARNESS_BUDGET_CALLS = 18_000   # 2x the most expensive gold query
SCORER_BUDGET_CALLS = 90_000    # 10x
HARNESS_WALL_BACKSTOP = 60.0
SCORER_WALL_BACKSTOP = 300.0


class QueryBudget:
    """Progress handler that aborts a query after a fixed number of VM instructions.

    An instruction budget gives the same verdict for the same query on any machine and
    under any load (assumption A1); a wall-clock deadline does not. The wall-clock
    backstop only guards against hangs (e.g. I/O stalls) and should never bind.
    """

    def __init__(self, max_calls: int, wall_seconds: float) -> None:
        self.max_calls = max_calls
        self.calls = 0
        self.deadline = time.monotonic() + wall_seconds
        self.reason = ""

    def __call__(self) -> int:
        """Return non-zero to make SQLite interrupt the running statement."""
        self.calls += 1
        if self.calls > self.max_calls:
            self.reason = "instruction budget"
            return 1
        if time.monotonic() > self.deadline:
            self.reason = "wall-clock backstop"
            return 1
        return 0


def open_read_only(db_path: Path) -> sqlite3.Connection:
    """Open a database read-only and immutable (no locks, no writes, no journal)."""
    uri = f"file:{db_path}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True, check_same_thread=False)
    connection.text_factory = decode_bytes_leniently
    return connection


def decode_bytes_leniently(raw: bytes) -> str:
    """BIRD databases contain a few non-UTF-8 strings; never crash on them."""
    return raw.decode("utf-8", errors="replace")


FROZEN_NOW = "2026-10-17 00:00:00"
TIME_KEYWORDS = re.compile(r"\bCURRENT_TIMESTAMP\b|\bCURRENT_DATE\b|\bCURRENT_TIME\b", re.IGNORECASE)
NOW_LITERAL = re.compile(r"(['\"])now\1", re.IGNORECASE)


def freeze_time(sql: str) -> str:
    """Replace SQLite's notions of "now" with a fixed instant (assumption A1).

    BIRD questions about ages use CURRENT_TIMESTAMP / date('now'); their results change
    with the wall-clock date, so the same query could give different observations to a
    parent and a child run executed on different days (observed in the BIRD GT pool).
    """
    def keyword_value(match: re.Match) -> str:
        word = match.group(0).upper()
        if word == "CURRENT_DATE":
            return "'" + FROZEN_NOW[:10] + "'"
        if word == "CURRENT_TIME":
            return "'" + FROZEN_NOW[11:] + "'"
        return "'" + FROZEN_NOW + "'"

    return NOW_LITERAL.sub("'" + FROZEN_NOW + "'", TIME_KEYWORDS.sub(keyword_value, sql))


def run_query(connection: sqlite3.Connection, sql: str, budget_calls: int, wall_seconds: float, max_rows: int | None) -> tuple[list[str], list[tuple[Any, ...]], bool]:
    """Execute ``sql`` (time frozen) under an instruction budget; return (columns, rows, truncated)."""
    sql = freeze_time(sql)
    budget = QueryBudget(budget_calls, wall_seconds)
    connection.set_progress_handler(budget, HANDLER_PERIOD)
    try:
        cursor = connection.execute(sql)
        columns: list[str] = []
        if cursor.description is not None:
            for column in cursor.description:
                columns.append(str(column[0]))
        if max_rows is None:
            rows = cursor.fetchall()
            truncated = False
        else:
            rows = cursor.fetchmany(max_rows + 1)
            truncated = len(rows) > max_rows
            rows = rows[:max_rows]
        return columns, rows, truncated
    except sqlite3.Error as error:
        message = str(error)
        if "interrupted" in message:
            message = "query was stopped: it exceeded the execution budget"
            if budget.reason == "wall-clock backstop":
                message += " (wall-clock backstop)"
        raise SQLError(message) from error
    finally:
        connection.set_progress_handler(None, 0)


@dataclass
class BirdEnv:
    """Per-run environment handed to the harness."""

    db_id: str
    connection: sqlite3.Connection
    description_dir: Path

    def execute(self, sql: str, max_rows: int = 50) -> tuple[list[str], list[tuple[Any, ...]], bool]:
        """Run a read-only query under the harness instruction budget."""
        return run_query(self.connection, sql, HARNESS_BUDGET_CALLS, HARNESS_WALL_BACKSTOP, max_rows)

    def description(self, table: str) -> str:
        """BIRD's column description CSV for ``table`` (empty if unavailable)."""
        path = self.description_dir / f"{table}.csv"
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8", errors="replace")


class BirdBenchmark:
    """BIRD Mini-Dev SQLite benchmark."""

    name = "bird"

    def __init__(self, root: str | Path = DEFAULT_ROOT, gold_timeout_seconds: float = 60.0) -> None:
        self.root = Path(root)
        self.gold_timeout_seconds = gold_timeout_seconds
        # Gold results are frozen with the splits (computed once on the development machine,
        # 2026-10-17) so that a slower or faster scoring machine cannot change which gold
        # queries "time out" and silently zero a task.
        self.gold_cache_path = FROZEN_GOLD if FROZEN_GOLD.exists() else self.root / "gold_results.json"
        self._gold: dict[str, Any] = {}
        if self.gold_cache_path.exists():
            self._gold = json.loads(self.gold_cache_path.read_text(encoding="utf-8"))

    def db_path(self, db_id: str) -> Path:
        """Path of a database file."""
        return self.root / "dev_databases" / db_id / f"{db_id}.sqlite"

    def load_all_tasks(self) -> list[dict[str, Any]]:
        """All Mini-Dev questions as task dictionaries."""
        raw = json.loads((self.root / "mini_dev_sqlite.json").read_text(encoding="utf-8"))
        tasks: list[dict[str, Any]] = []
        seen_ids: set[int] = set()
        for item in raw:
            if item["question_id"] in seen_ids:
                continue  # Mini-Dev lists questions 137 and 138 twice, verbatim.
            seen_ids.add(item["question_id"])
            tasks.append(
                {
                    "task_id": f"bird-{item['question_id']}",
                    "db_id": item["db_id"],
                    "question": item["question"],
                    "evidence": item.get("evidence", ""),
                    "gold_sql": item["SQL"],
                    "difficulty": item["difficulty"],
                }
            )
        return tasks

    def public_view(self, task: dict[str, Any]) -> dict[str, Any]:
        """Question, hint and database id only; never the gold SQL."""
        return {
            "task_id": task["task_id"],
            "db_id": task["db_id"],
            "question": task["question"],
            "evidence": task["evidence"],
        }

    def make_env(self, task: dict[str, Any]) -> BirdEnv:
        """Fresh read-only environment for one run."""
        return BirdEnv(
            db_id=task["db_id"],
            connection=open_read_only(self.db_path(task["db_id"])),
            description_dir=self.root / "dev_databases" / task["db_id"] / "database_description",
        )

    def close_env(self, env: BirdEnv) -> None:
        """Close the connection."""
        env.connection.close()

    def compute_gold(self, task: dict[str, Any]) -> dict[str, Any]:
        """Execute the gold query (no row limit) and describe the outcome."""
        connection = open_read_only(self.db_path(task["db_id"]))
        started = time.perf_counter()
        try:
            columns, rows, truncated = run_query(connection, task["gold_sql"], SCORER_BUDGET_CALLS, self.gold_timeout_seconds, None)
            return {"ok": True, "rows": rows_to_canonical(rows), "seconds": time.perf_counter() - started, "n_rows": len(rows)}
        except SQLError as error:
            return {"ok": False, "error": str(error), "seconds": time.perf_counter() - started}
        finally:
            connection.close()

    def ensure_gold(self, tasks: list[dict[str, Any]]) -> None:
        """Compute and cache gold results for every task that lacks one."""
        changed = False
        for task in tasks:
            if task["task_id"] in self._gold:
                continue
            self._gold[task["task_id"]] = self.compute_gold(task)
            changed = True
        if changed:
            self._write_gold_cache()

    def _write_gold_cache(self) -> None:
        """Merge with whatever is on disk, then replace the file atomically."""
        merged: dict[str, Any] = {}
        if self.gold_cache_path.exists():
            merged.update(json.loads(self.gold_cache_path.read_text(encoding="utf-8")))
        merged.update(self._gold)
        self._gold = merged
        temporary = self.gold_cache_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(merged), encoding="utf-8")
        os.replace(temporary, self.gold_cache_path)

    def gold(self, task_id: str) -> dict[str, Any]:
        """Cached gold record for a task."""
        return self._gold[task_id]

    def score(self, task: dict[str, Any], answer: Any) -> tuple[int, float]:
        """BIRD execution accuracy: result set of the answer equals the gold result set."""
        if not isinstance(answer, str) or not answer.strip():
            return 0, 0.0
        gold = self._gold.get(task["task_id"])
        if gold is None or not gold.get("ok"):
            return 0, 0.0
        connection = open_read_only(self.db_path(task["db_id"]))
        try:
            correct = int(streamed_set_equals(connection, answer, result_set(gold["rows"])))
        except SQLError:
            return 0, 0.0
        finally:
            connection.close()
        return correct, float(correct)


def streamed_set_equals(connection: sqlite3.Connection, sql: str, gold: set[tuple[Any, ...]]) -> bool:
    """Whether the query's result set equals ``gold``, reading its rows one by one (DEC-18).

    Gives the verdict of comparing the fully fetched result set with ``gold``: a row outside
    ``gold`` makes equality impossible whether the query would later finish or fail, so the
    scan stops there. Memory stays bounded by the gold set, so a query whose result is huge
    (a cross join, say) cannot exhaust the scoring machine's memory.
    """
    sql = freeze_time(sql)
    budget = QueryBudget(SCORER_BUDGET_CALLS, SCORER_WALL_BACKSTOP)
    connection.set_progress_handler(budget, HANDLER_PERIOD)
    try:
        cursor = connection.execute(sql)
        seen: set[tuple[Any, ...]] = set()
        while True:
            batch = cursor.fetchmany(1000)
            if not batch:
                break
            for row in rows_to_canonical(batch):
                key = tuple(row)
                if key not in gold:
                    return False
                seen.add(key)
        return seen == gold
    except sqlite3.Error as error:
        raise SQLError(str(error)) from error
    finally:
        connection.set_progress_handler(None, 0)


def rows_to_canonical(rows: list[tuple[Any, ...]]) -> list[list[Any]]:
    """JSON-friendly rows (bytes decoded, tuples to lists)."""
    canonical: list[list[Any]] = []
    for row in rows:
        values: list[Any] = []
        for value in row:
            if isinstance(value, bytes):
                values.append(value.decode("utf-8", errors="replace"))
            else:
                values.append(value)
        canonical.append(values)
    return canonical


def result_set(rows: list[list[Any]]) -> set[tuple[Any, ...]]:
    """Set of row tuples, as BIRD's evaluation compares ``set(pred) == set(gold)``."""
    rows_as_tuples: set[tuple[Any, ...]] = set()
    for row in rows:
        rows_as_tuples.add(tuple(row))
    return rows_as_tuples
