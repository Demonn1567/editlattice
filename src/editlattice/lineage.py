"""Lineages of accepted harness edits and the products (sub-harnesses) they span.

A lineage is the sequence of accepted harness versions H_0 (seed), H_1, ..., H_k of one
evolution run. Edit i turns H_{i-1} into H_i. We re-derive each edit from the two sources as
minimal SEARCH/REPLACE blocks (the proposer's own blocks may carry context that a later edit
rewrote), and check that the blocks reproduce H_i exactly.

Product S (a subset of edit indices) is H_0 with the edits in S applied in lineage order.
S is feasible iff every block of every edit in S matches exactly once when it is applied and
the result is a valid harness; an edit that needs text only an earlier edit introduces makes
products without that earlier edit infeasible (design D2).
"""

from __future__ import annotations

import difflib
import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path

from editlattice.proposers.edit_format import EditBlock, EditError, apply_edit_blocks, validate_harness


@dataclass(frozen=True)
class Edit:
    """One accepted edit of a lineage (1-based index)."""

    index: int
    blocks: tuple[EditBlock, ...]
    parent_hash: str = ""
    child_hash: str = ""
    rationale: str = ""


@dataclass
class Lineage:
    """Seed source and its accepted edits."""

    name: str
    seed: str
    edits: list[Edit]
    _products: dict[frozenset[int], str | None] = field(default_factory=dict, repr=False)

    @property
    def k(self) -> int:
        """Number of accepted edits."""
        return len(self.edits)

    def product(self, subset: frozenset[int]) -> str | None:
        """Source of product ``subset``, or None if it is infeasible."""
        if subset not in self._products:
            self._products[subset] = apply_subset(self.seed, [edit for edit in self.edits if edit.index in subset])
        return self._products[subset]

    def feasible_products(self) -> dict[frozenset[int], str]:
        """Every feasible product, by brute-force enumeration (fine for k <= ~14)."""
        result: dict[frozenset[int], str] = {}
        indices = [edit.index for edit in self.edits]
        for size in range(len(indices) + 1):
            for combination in itertools.combinations(indices, size):
                subset = frozenset(combination)
                source = self.product(subset)
                if source is not None:
                    result[subset] = source
        return result

    def dependencies(self) -> dict[int, frozenset[int]]:
        """For each edit, the edits it needs: those present in every feasible product containing it."""
        feasible = self.feasible_products()
        needs: dict[int, frozenset[int]] = {}
        for edit in self.edits:
            containing = [subset for subset in feasible if edit.index in subset]
            common = frozenset.intersection(*containing) if containing else frozenset()
            needs[edit.index] = common - {edit.index}
        return needs


def apply_subset(seed: str, edits: list[Edit]) -> str | None:
    """Apply edits in lineage order; None if any block fails or the result is invalid."""
    source = seed
    for edit in sorted(edits, key=lambda item: item.index):
        try:
            source = apply_blocks_exactly(source, edit.blocks)
        except EditError:
            return None
    return source


def apply_blocks_exactly(source: str, blocks: tuple[EditBlock, ...]) -> str:
    """Like the proposers' application, but an edit that changes nothing in this product is
    still applied (it may be a no-op here and not elsewhere); only matching must be exact."""
    edited = source
    for number, block in enumerate(blocks, start=1):
        count = edited.count(block.search)
        if count != 1:
            raise EditError(f"block {number}: SEARCH text occurs {count} times")
        edited = edited.replace(block.search, block.replace)
    validate_harness(edited)
    return edited


def derive_blocks(old: str, new: str) -> tuple[EditBlock, ...]:
    """Minimal SEARCH/REPLACE blocks turning ``old`` into ``new``.

    Each changed line range becomes one block whose SEARCH text is the old lines plus just
    enough surrounding context to occur exactly once in the text it is applied to. Blocks
    are applied top to bottom, each to the text left by the previous ones.
    """
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    opcodes = [op for op in difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False).get_opcodes() if op[0] != "equal"]
    blocks: list[EditBlock] = []
    current = list(old_lines)
    offset = 0  # how far earlier blocks shifted line numbers in ``current``
    for _, i1, i2, j1, j2 in opcodes:
        start, end = i1 + offset, i2 + offset
        replacement = new_lines[j1:j2]
        before, after = 0, 0
        while True:
            search = "".join(current[start - before:end + after])
            text = "".join(current)
            if search and text.count(search) == 1:
                break
            if start - before > 0:
                before += 1
            if end + after < len(current):
                after += 1
            if start - before <= 0 and end + after >= len(current) and text.count("".join(current[start - before:end + after])) != 1:
                raise EditError("no unique context for a hunk")
        replace = "".join(current[start - before:start] + replacement + current[end:end + after])
        blocks.append(EditBlock(search=search, replace=replace))
        current[start - before:end + after] = current[start - before:start] + replacement + current[end:end + after]
        offset += len(replacement) - (i2 - i1)
    if "".join(current) != new:
        raise EditError("derived blocks do not reproduce the child")
    return tuple(blocks)


def lineage_from_sources(name: str, versions: list[str], rationales: list[str] | None = None, hashes: list[str] | None = None) -> Lineage:
    """Build a lineage from the accepted versions H_0..H_k; checks every edit reproduces its child."""
    edits = []
    for index in range(1, len(versions)):
        blocks = derive_blocks(versions[index - 1], versions[index])
        if apply_blocks_exactly(versions[index - 1], blocks) != versions[index]:
            raise EditError(f"edit {index} does not reproduce its child")
        edits.append(Edit(index=index, blocks=blocks,
                          parent_hash=hashes[index - 1] if hashes else "", child_hash=hashes[index] if hashes else "",
                          rationale=rationales[index - 1] if rationales else ""))
    return Lineage(name=name, seed=versions[0], edits=edits)


def load_loop_lineage(log_path: str | Path, harness_dir: str | Path) -> Lineage:
    """Lineage of one evolution loop from its JSONL log and the content-addressed harness store."""
    entries = [json.loads(line) for line in Path(log_path).read_text().splitlines() if line.strip()]
    accepted = [entry for entry in entries if entry.get("accepted")]
    if not entries:
        raise ValueError(f"empty log {log_path}")
    hashes = [entries[0]["parent_hash"]] + [entry["child_hash"] for entry in accepted]
    sources = [read_harness(harness_dir, digest) for digest in hashes]
    return lineage_from_sources(Path(log_path).name.removesuffix(".log.jsonl"), sources,
                                [entry.get("rationale", "") for entry in accepted], hashes)


def read_harness(harness_dir: str | Path, digest: str) -> str:
    """Source of a stored harness version (``<dir>/<hash>.py`` or ``<dir>/<hash>/harness.py``)."""
    directory = Path(harness_dir)
    for candidate in (directory / f"{digest}.py", directory / digest / "harness.py"):
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
    raise FileNotFoundError(f"harness {digest} not in {directory}")


def churn_stats(lineage: Lineage) -> dict[str, float | int | list[float]]:
    """How much of each accepted step survives into the final harness.

    A step's added lines that are absent from the final harness were undone or rewritten by a
    later step. ``survival`` is, per step, the fraction of its added lines present in the final
    harness (by content, line multiset); ``churn_ratio`` is total lines changed by all steps
    divided by lines changed by the net diff seed -> final.
    """
    versions = [lineage.seed]
    for edit in lineage.edits:
        versions.append(apply_blocks_exactly(versions[-1], edit.blocks))
    final_lines: dict[str, int] = {}
    for line in versions[-1].splitlines():
        final_lines[line] = final_lines.get(line, 0) + 1

    def changed(old: str, new: str) -> tuple[int, list[str]]:
        matcher = difflib.SequenceMatcher(a=old.splitlines(), b=new.splitlines(), autojunk=False)
        count, added = 0, []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != "equal":
                count += (i2 - i1) + (j2 - j1)
                added.extend(new.splitlines()[j1:j2])
        return count, added

    survival, step_total = [], 0
    for old, new in zip(versions, versions[1:]):
        count, added = changed(old, new)
        step_total += count
        survival.append(sum(1 for line in added if final_lines.get(line, 0) > 0) / len(added) if added else 1.0)
    net, _ = changed(versions[0], versions[-1])
    return {"steps": lineage.k, "step_lines_changed": step_total, "net_lines_changed": net,
            "churn_ratio": step_total / net if net else float("inf"), "survival": survival}
