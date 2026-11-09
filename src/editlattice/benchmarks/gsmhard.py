"""GSM-Hard (Gao et al., PAL): grade-school word problems with large numbers.

Targets were produced by running each GSM8K solution program with its numbers replaced, so a
few targets are implausible (negative or fractional counts); they are kept, flagged by
``plausible_target``, and the analysis reports results on the plausible subset as well.
The environment offers sandboxed Python execution (the LiveCodeBench sandbox: CPU-time
limit, isolated interpreter), which harnesses may or may not use.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from editlattice.benchmarks.livecodebench import ExecutionResult, run_program

DATA_FILE = Path(__file__).resolve().parents[3] / "datasets" / "gsmhard" / "gsmhardv2.jsonl"
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?")
# Absolute tolerance as in PAL; the tiny relative term only absorbs float representation of large values.
ABSOLUTE_TOLERANCE = 1e-3
RELATIVE_TOLERANCE = 1e-12


def parse_number(text: Any) -> float | None:
    """The last number in an answer string (commas allowed), or None."""
    if text is None:
        return None
    if isinstance(text, (int, float)) and not isinstance(text, bool):
        return float(text)
    matches = NUMBER.findall(str(text))
    if not matches:
        return None
    try:
        return float(matches[-1].replace(",", ""))
    except ValueError:
        return None


def plausible_target(target: float) -> bool:
    """Non-negative and whole, or a short decimal (money); flags label noise."""
    if target < 0 or math.isnan(target) or math.isinf(target):
        return False
    return abs(target - round(target)) < 1e-9 or abs(target * 100 - round(target * 100)) < 1e-6


@dataclass
class GsmEnv:
    """Per-run environment: sandboxed Python."""

    def run_python(self, code: str, stdin: str = "", timeout_seconds: float = 5.0) -> ExecutionResult:
        """Run a program and return its output."""
        return run_program(code, stdin, min(timeout_seconds, 10.0))


class GsmHardBenchmark:
    """1,319 problems; task ids gsmh-<line number>."""

    name = "gsmhard"

    def __init__(self, data_file: Path = DATA_FILE) -> None:
        self.data_file = data_file

    def load_all_tasks(self) -> list[dict[str, Any]]:
        """All tasks in file order."""
        tasks = []
        for number, line in enumerate(self.data_file.read_text(encoding="utf-8").splitlines()):
            record = json.loads(line)
            target = float(record["target"])
            tasks.append({"task_id": f"gsmh-{number}", "question": record["input"], "target": target,
                          "plausible": plausible_target(target)})
        return tasks

    def public_view(self, task: dict[str, Any]) -> dict[str, Any]:
        """The question only."""
        return {"task_id": task["task_id"], "question": task["question"]}

    def make_env(self, task: dict[str, Any]) -> GsmEnv:
        """Fresh environment for one run."""
        return GsmEnv()

    def close_env(self, env: GsmEnv) -> None:
        """Nothing to release."""

    def score(self, task: dict[str, Any], answer: Any) -> tuple[int, float]:
        """Correct if the answer's last number matches the target (absolute tolerance 1e-3)."""
        value = parse_number(answer)
        if value is None:
            return 0, 0.0
        target = task["target"]
        correct = abs(value - target) <= max(ABSOLUTE_TOLERANCE, RELATIVE_TOLERANCE * abs(target))
        return int(correct), float(correct)
