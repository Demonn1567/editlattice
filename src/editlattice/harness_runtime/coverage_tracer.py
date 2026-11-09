"""Per-run dynamic coverage of harness code (function and line level).

Python 3.11 has no ``sys.monitoring`` (added in 3.12), so we use ``sys.settrace``.
``sys.settrace`` installs the tracer for the *calling thread only*, which is what we
want: each (task, replicate) run executes in its own worker thread and gets its own
coverage record. Frames whose code does not come from a traced harness file are not
line-traced, which keeps the overhead small.

Function identity is the code object's qualified name, normalised to the enclosing
top-level function or method: ``Agent.step.<locals>.helper`` and
``solve.<locals>.<listcomp>`` are recorded as ``Agent.step`` and ``solve``. Nested code
can only run after its enclosing function has started, so attributing it to the
enclosing function never hides a touch. Module-level code is recorded as ``<module>``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from types import CodeType, FrameType
from typing import Any, Callable

LOCALS_MARKER = ".<locals>."


def normalize_qualname(qualname: str) -> str:
    """Map a code object's qualified name to its enclosing top-level function/method."""
    marker_index = qualname.find(LOCALS_MARKER)
    if marker_index == -1:
        return qualname
    return qualname[:marker_index]


@dataclass
class CoverageRecord:
    """Functions and (file, line) pairs executed during one run."""

    functions: set[str] = field(default_factory=set)
    lines: set[tuple[str, int]] = field(default_factory=set)

    def sorted_functions(self) -> list[str]:
        """Return the covered functions in sorted order (for JSONL output)."""
        return sorted(self.functions)

    def sorted_lines(self) -> list[list[Any]]:
        """Return covered lines as sorted ``[file_label, line]`` pairs."""
        ordered = sorted(self.lines)
        return [[label, number] for label, number in ordered]


class CoverageTracer:
    """Context manager that records coverage of the given harness files in this thread.

    ``traced_files`` maps an absolute file path to a short, stable label (for example
    ``"harness.py"``) so that records from parent and child harness files, which live
    at different paths, are comparable.
    """

    def __init__(self, traced_files: dict[str, str]) -> None:
        self.traced_files = dict(traced_files)
        self.record = CoverageRecord()
        self._previous_tracer: Callable[..., Any] | None = None

    def _label_for(self, code: CodeType) -> str | None:
        """Return the file label for ``code`` or ``None`` if it is not harness code."""
        return self.traced_files.get(code.co_filename)

    def _global_trace(self, frame: FrameType, event: str, arg: Any) -> Callable[..., Any] | None:
        """Called on every new frame; returns the local tracer only for harness frames."""
        if event != "call":
            return None
        label = self._label_for(frame.f_code)
        if label is None:
            return None
        self.record.functions.add(normalize_qualname(frame.f_code.co_qualname))
        self.record.lines.add((label, frame.f_lineno))
        return self._local_trace

    def _local_trace(self, frame: FrameType, event: str, arg: Any) -> Callable[..., Any] | None:
        """Records every executed line of a harness frame."""
        if event == "line":
            label = self._label_for(frame.f_code)
            if label is not None:
                self.record.lines.add((label, frame.f_lineno))
        return self._local_trace

    def __enter__(self) -> "CoverageTracer":
        self._previous_tracer = sys.gettrace()
        sys.settrace(self._global_trace)
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        sys.settrace(self._previous_tracer)
