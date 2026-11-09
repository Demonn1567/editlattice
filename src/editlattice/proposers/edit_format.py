"""SEARCH/REPLACE edit blocks: parsing, application and validation.

Small models cannot reliably reproduce a whole harness file, so proposers answer with
one or more blocks::

    <<<<<<< SEARCH
    exact lines from the current harness
    =======
    replacement lines
    >>>>>>> REPLACE

Every SEARCH text must occur exactly once in the current harness. The edited harness
must parse, must still define ``solve(task, llm, env)``, and must not import modules
that start threads or processes (they would escape the coverage tracer, D-005).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

SEARCH_MARKER = "<<<<<<< SEARCH"
DIVIDER = "======="
REPLACE_MARKER = ">>>>>>> REPLACE"
FORBIDDEN_MODULES = frozenset({"threading", "multiprocessing", "subprocess", "concurrent", "asyncio", "os", "signal", "socket"})


@dataclass(frozen=True)
class EditBlock:
    """One search/replace pair."""

    search: str
    replace: str


class EditError(ValueError):
    """The proposal could not be turned into a valid harness."""


def diff_block(body: str) -> EditBlock | None:
    """A unified-diff body ('-', '+' and ' ' lines) as a search/replace pair, else None."""
    lines = strip_one_newline(body).split("\n")
    if not lines or not any(line.startswith(("-", "+")) for line in lines):
        return None
    if any(line and line[0] not in "-+ " for line in lines):
        return None
    search = [line[1:] for line in lines if line.startswith(("-", " ")) or line == ""]
    replace = [line[1:] for line in lines if line.startswith(("+", " ")) or line == ""]
    return EditBlock(search="\n".join(search), replace="\n".join(replace))


def parse_edit_blocks(text: str, accept_diff: bool = False) -> list[EditBlock]:
    """Extract all well-formed blocks from a model reply.

    With ``accept_diff`` (Phase A2 only), a block whose body between the SEARCH and REPLACE markers
    has no divider but is a unified diff is read as '-'/' ' lines -> '+'/' ' lines.
    """
    blocks: list[EditBlock] = []
    remaining = text
    while accept_diff and SEARCH_MARKER in remaining:
        after_search = remaining.split(SEARCH_MARKER, 1)[1]
        if REPLACE_MARKER not in after_search:
            break
        body, rest = after_search.split(REPLACE_MARKER, 1)
        if DIVIDER in body:
            search_part, replace_part = body.split(DIVIDER, 1)
            blocks.append(EditBlock(search=strip_one_newline(search_part), replace=strip_one_newline(replace_part)))
        else:
            block = diff_block(body)
            if block is None:
                break
            blocks.append(block)
        remaining = rest
    if accept_diff:
        return blocks
    while SEARCH_MARKER in remaining:
        after_search = remaining.split(SEARCH_MARKER, 1)[1]
        if DIVIDER not in after_search or REPLACE_MARKER not in after_search:
            break
        search_part, after_divider = after_search.split(DIVIDER, 1)
        if REPLACE_MARKER not in after_divider:
            break
        replace_part, remaining = after_divider.split(REPLACE_MARKER, 1)
        blocks.append(EditBlock(search=strip_one_newline(search_part), replace=strip_one_newline(replace_part)))
    return blocks


def strip_one_newline(text: str) -> str:
    """Remove the newline right after a marker and right before the next one."""
    if text.startswith("\n"):
        text = text[1:]
    if text.endswith("\n"):
        text = text[:-1]
    return text


def apply_edit_blocks(source: str, blocks: list[EditBlock]) -> str:
    """Apply blocks in order; raise :class:`EditError` on any ambiguity."""
    if not blocks:
        raise EditError("no edit blocks found")
    edited = source
    for number, block in enumerate(blocks, start=1):
        if not block.search.strip():
            raise EditError(f"block {number}: empty SEARCH text")
        count = edited.count(block.search)
        if count == 0:
            shifted = shift_indentation(block, edited) or shift_first_line(block, edited)
            if shifted is None:
                raise EditError(f"block {number}: SEARCH text not found")
            block = shifted
            count = 1
        if count > 1:
            raise EditError(f"block {number}: SEARCH text occurs {count} times")
        edited = edited.replace(block.search, block.replace)
    if edited == source:
        raise EditError("the edit does not change the harness")
    validate_harness(edited)
    return edited


def shift_indentation(block: EditBlock, source: str) -> EditBlock | None:
    """The block with its common indentation removed, if that makes SEARCH occur exactly once.

    Small proposers sometimes copy a whole block with every line indented by the same extra
    amount. Removing that common amount from SEARCH and REPLACE alike is unambiguous when the
    result occurs exactly once; anything else is still an error.
    """
    lines = block.search.splitlines()
    widths = [len(line) - len(line.lstrip(" ")) for line in lines if line.strip()]
    shift = min(widths) if widths else 0
    if shift == 0:
        return None
    prefix = " " * shift

    def dedent(text: str) -> str:
        return "\n".join(line[shift:] if line.startswith(prefix) else line for line in text.split("\n"))

    search = dedent(block.search)
    if source.count(search) != 1:
        return None
    return EditBlock(search=search, replace=dedent(block.replace))


def shift_first_line(block: EditBlock, source: str) -> EditBlock | None:
    """The block with extra indentation removed from its first line only, if that is unambiguous.

    A module-level assignment of a multi-line string is sometimes copied with only its first
    line indented (the continuation lines of the string have no indentation to copy). The same
    amount is removed from the first line of SEARCH and REPLACE; the edit applies only if the
    result occurs exactly once and starts at the beginning of a line.
    """
    first, _, rest = block.search.partition("\n")
    shift = len(first) - len(first.lstrip(" "))
    replace_first, newline, replace_rest = block.replace.partition("\n")
    if shift == 0 or not replace_first.startswith(" " * shift):
        return None
    search = first[shift:] + ("\n" + rest if rest or block.search.count("\n") else "")
    if source.count(search) != 1:
        return None
    at = source.index(search)
    if at and source[at - 1] != "\n":
        return None
    return EditBlock(search=search, replace=replace_first[shift:] + newline + replace_rest)


def validate_harness(source: str) -> None:
    """Check the harness contract (parses, defines solve, no forbidden imports)."""
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise EditError(f"syntax error: {error}") from error
    has_solve = False
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "solve":
            argument_count = len(node.args.args)
            has_solve = argument_count == 3
    if not has_solve:
        raise EditError("the harness must define solve(task, llm, env)")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in FORBIDDEN_MODULES:
                    raise EditError(f"forbidden import {alias.name}")
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] in FORBIDDEN_MODULES:
                raise EditError(f"forbidden import {node.module}")
