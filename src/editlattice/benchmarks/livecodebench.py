"""LiveCodeBench code_generation_lite (stdin/stdout problems), hidden-test scoring.

Environment primitive offered to a harness:

* ``env.run_python(code, stdin, timeout_seconds)`` -> ``ExecutionResult``
* ``env.public_tests`` -> list of {"input", "output"} (also in the task's public view)

Programs run in a subprocess (``python -I``) under CPU-time, file-size and process
limits, in a fresh temporary directory, with a wall-clock timeout. No Docker.

Private tests are stored by LiveCodeBench as base64(zlib(pickle(json-string))). Pickle
can execute code, so we decode with an unpickler that refuses every class lookup: only
primitive types (the expected ``str``) can be produced.

Output comparison: lines are right-stripped and trailing blank lines dropped; outputs
match if they are equal, or if they have the same tokens where numeric tokens agree to
an absolute/relative tolerance of 1e-6 (floating-point answers).
"""

from __future__ import annotations

import base64
import io
import json
import os
import pickle
import re
import signal
import subprocess
import sys
import tempfile
import threading
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_FILES = [
    Path(__file__).resolve().parents[3] / "data" / "lcb_raw" / "test5.jsonl",
    Path(__file__).resolve().parents[3] / "data" / "lcb_raw" / "test6.jsonl",
]
LAUNCHER = Path(__file__).resolve().parent / "sandbox_launcher.py"
MIN_CONTEST_DATE = "2024-10-01"
MAX_OUTPUT_BYTES = 2_000_000


class PrimitiveOnlyUnpickler(pickle.Unpickler):
    """Unpickler that cannot instantiate any class or call any function."""

    def find_class(self, module: str, name: str) -> Any:
        """Refuse every global lookup."""
        raise pickle.UnpicklingError(f"blocked: {module}.{name}")


def decode_private_tests(blob: str) -> list[dict[str, str]]:
    """Decode LiveCodeBench's compressed private test cases safely."""
    raw = zlib.decompress(base64.b64decode(blob))
    payload = PrimitiveOnlyUnpickler(io.BytesIO(raw)).load()
    if not isinstance(payload, str):
        raise ValueError(f"unexpected private-test payload type {type(payload).__name__}")
    return json.loads(payload)


@dataclass
class ExecutionResult:
    """Outcome of running a program once."""

    stdout: str
    stderr: str
    returncode: int
    timed_out: bool


SANDBOX_SLOTS = threading.BoundedSemaphore(max(1, (os.cpu_count() or 2) - 1))
CPU_KILL_SIGNALS = (-signal.SIGXCPU, -signal.SIGKILL)


def run_program(code: str, stdin: str, timeout_seconds: float) -> ExecutionResult:
    """Run ``code`` in a sandboxed subprocess.

    The time limit is on CPU time (RLIMIT_CPU), which does not depend on machine load, so
    the outcome of the same program does not change with how many runs execute in
    parallel (assumption A1). The wall-clock limit is only a generous backstop, and at
    most cpu_count − 1 programs run at once.
    """
    cpu_limit = max(1, int(round(timeout_seconds)))
    wall_limit = 3.0 * cpu_limit + 10.0
    with SANDBOX_SLOTS:
        with tempfile.TemporaryDirectory(prefix="ig_sbx_") as workdir:
            program = Path(workdir) / "solution.py"
            program.write_text(code, encoding="utf-8")
            command = [sys.executable, "-I", str(LAUNCHER), str(cpu_limit), str(MAX_OUTPUT_BYTES), str(program)]
            try:
                completed = subprocess.run(
                    command,
                    input=stdin.encode("utf-8"),
                    capture_output=True,
                    timeout=wall_limit,
                    cwd=workdir,
                    env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"},
                )
            except subprocess.TimeoutExpired as expired:
                stdout = (expired.stdout or b"").decode("utf-8", errors="replace")
                return ExecutionResult(stdout=stdout[:MAX_OUTPUT_BYTES], stderr="wall-clock timeout", returncode=-1, timed_out=True)
    stdout = normalize_output(completed.stdout[:MAX_OUTPUT_BYTES].decode("utf-8", errors="replace"), workdir)
    stderr = normalize_output(completed.stderr[-4000:].decode("utf-8", errors="replace"), workdir)
    timed_out = completed.returncode in CPU_KILL_SIGNALS
    return ExecutionResult(stdout=stdout, stderr=stderr, returncode=completed.returncode, timed_out=timed_out)


HEX_ADDRESS = re.compile(r"0x[0-9a-fA-F]{6,}")


def normalize_output(text: str, workdir: str) -> str:
    """Remove run-specific noise so that the same program gives the same feedback (A1).

    Tracebacks contain the random temporary directory of the sandbox, and some error
    messages contain memory addresses; both differ between otherwise identical runs and
    would make parent and child diverge for reasons unrelated to the edit.
    """
    normalized = text.replace(workdir, "/sandbox")
    return HEX_ADDRESS.sub("0xADDR", normalized)


def normalize_lines(text: str) -> list[str]:
    """Right-strip lines and drop trailing blank lines."""
    lines: list[str] = []
    for line in text.replace("\r\n", "\n").split("\n"):
        lines.append(line.rstrip())
    while lines and lines[-1] == "":
        lines.pop()
    return lines


def tokens_match(expected: str, actual: str) -> bool:
    """Exact token match, or numeric match within 1e-6 (absolute or relative)."""
    if expected == actual:
        return True
    try:
        expected_value = float(expected)
        actual_value = float(actual)
    except ValueError:
        return False
    tolerance = 1e-6 * max(1.0, abs(expected_value))
    return abs(expected_value - actual_value) <= tolerance


def outputs_match(expected: str, actual: str) -> bool:
    """Compare program output with the expected output."""
    expected_lines = normalize_lines(expected)
    actual_lines = normalize_lines(actual)
    if expected_lines == actual_lines:
        return True
    expected_tokens = " ".join(expected_lines).split()
    actual_tokens = " ".join(actual_lines).split()
    if len(expected_tokens) != len(actual_tokens):
        return False
    for expected_token, actual_token in zip(expected_tokens, actual_tokens):
        if not tokens_match(expected_token, actual_token):
            return False
    return True


@dataclass
class LCBEnv:
    """Per-run environment handed to the harness."""

    public_tests: list[dict[str, str]]

    def run_python(self, code: str, stdin: str, timeout_seconds: float = 6.0) -> ExecutionResult:
        """Run a program on one input."""
        return run_program(code, stdin, min(timeout_seconds, 10.0))


class LiveCodeBenchmark:
    """LiveCodeBench stdin problems released after the solvers' training cutoff."""

    name = "lcb"

    def __init__(self, files: list[Path] | None = None, test_timeout_seconds: float = 6.0, max_hidden_tests: int | None = None) -> None:
        self.files = files or DEFAULT_FILES
        self.test_timeout_seconds = test_timeout_seconds
        self.max_hidden_tests = max_hidden_tests

    def load_all_tasks(self) -> list[dict[str, Any]]:
        """Load the pre-registered pool (D-008)."""
        tasks: list[dict[str, Any]] = []
        for path in self.files:
            with Path(path).open("r", encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    task = self._task_from_record(record)
                    if task is not None:
                        tasks.append(task)
        return tasks

    def _task_from_record(self, record: dict[str, Any]) -> dict[str, Any] | None:
        """Convert one raw record, or return None if it is outside the pool."""
        public_tests = json.loads(record["public_test_cases"])
        if not public_tests or public_tests[0].get("testtype") != "stdin":
            return None
        if record["contest_date"][:10] < MIN_CONTEST_DATE:
            return None
        simple_public: list[dict[str, str]] = []
        for test in public_tests:
            simple_public.append({"input": test["input"], "output": test["output"]})
        return {
            "task_id": f"lcb-{record['question_id']}",
            "title": record["question_title"],
            "question": record["question_content"],
            "difficulty": record["difficulty"],
            "platform": record["platform"],
            "contest_date": record["contest_date"][:10],
            "public_tests": simple_public,
            "private_tests_blob": record["private_test_cases"],
        }

    def public_view(self, task: dict[str, Any]) -> dict[str, Any]:
        """Statement and public tests only."""
        return {
            "task_id": task["task_id"],
            "title": task["title"],
            "question": task["question"],
            "public_tests": task["public_tests"],
        }

    def make_env(self, task: dict[str, Any]) -> LCBEnv:
        """Fresh environment for one run."""
        return LCBEnv(public_tests=list(task["public_tests"]))

    def close_env(self, env: LCBEnv) -> None:
        """Nothing to release."""
        return None

    def all_tests(self, task: dict[str, Any]) -> list[dict[str, str]]:
        """Public plus private tests."""
        tests = list(task["public_tests"])
        for test in decode_private_tests(task["private_tests_blob"]):
            tests.append({"input": test["input"], "output": test["output"]})
        if self.max_hidden_tests is not None:
            tests = tests[: self.max_hidden_tests]
        return tests

    def score(self, task: dict[str, Any], answer: Any) -> tuple[int, float]:
        """Binary pass of all tests; graded score = fraction of tests passed.

        Stops at the first failing test for the binary outcome but still reports the
        fraction of tests passed before the failure as the graded score.
        """
        if not isinstance(answer, str) or not answer.strip():
            return 0, 0.0
        tests = self.all_tests(task)
        passed = 0
        for test in tests:
            result = run_program(answer, test["input"], self.test_timeout_seconds)
            if result.timed_out or result.returncode != 0 or not outputs_match(test["output"], result.stdout):
                return 0, passed / len(tests)
            passed += 1
        return 1, 1.0
