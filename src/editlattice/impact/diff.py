"""Change-impact analysis between a parent and a child harness module.

The impact set I(H, H') is expressed in *parent* coordinates: the parent functions
(function level) or parent source lines (line level) whose execution is a necessary
condition for the child to behave differently on a run (assumption A4). A task is
unaffected when no selection run of the parent on it executed any element of I.

Rules (function level), applied in this order:

1. GLOBAL if either version contains a construct that defeats name-based reasoning
   (``exec``/``eval``/``compile``/``__import__``/``setattr``/``delattr``, import
   machinery, frame hacking, monkey-patching an imported module, star imports).
2. Modified or deleted functions/methods are in I.
3. Changed module-level statements: the statement *sequence* (without docstrings and
   ``if __name__ == "__main__"`` blocks) is diffed; every name bound by a changed
   binding statement is "value-changed". A changed non-binding statement (a top-level
   call, loop, ``with`` ...) is GLOBAL because it may have arbitrary import-time effects.
4. Added/deleted top-level functions and classes make their name value-changed (a new
   ``len`` shadows the builtin for every unchanged caller).
5. Added/deleted methods and changed class-level attributes make their attribute name
   "attribute-changed". Adding or deleting a dunder method is GLOBAL (implicit calls).
   A changed class header (bases, keywords, decorators) is GLOBAL.
6. Import-time closure: module-level statements, class bodies and function headers
   (decorators, defaults) are evaluated at import time, so if they load a value-changed
   name, a modified function, or a class with any changed member, their own results
   change. Iterated to a fixpoint.
7. Readers: every parent function that loads a value-changed global name, reads an
   attribute-changed name, or performs a dynamic lookup that could resolve to one, is
   in I. Functions whose header loads a changed name are in I.
8. Added or deleted functions with a non-pure decorator are GLOBAL (registration side
   effects), as are modified functions whose decorator list changed and is not pure.

A child identical to the parent up to comments and formatting has an empty impact set.
"""

from __future__ import annotations

import ast
import difflib
from dataclasses import dataclass, field

from editlattice.impact.names import (
    ClassInfo,
    FunctionInfo,
    ModuleIndex,
    NameUsage,
    StatementInfo,
    decorator_is_pure,
    index_module,
)

HARNESS_LABEL = "harness.py"


@dataclass
class ImpactSet:
    """Result of the change-impact analysis (parent coordinates)."""

    is_global: bool
    global_reasons: list[str]
    functions: set[str]
    lines: set[tuple[str, int]]
    modified_functions: set[str] = field(default_factory=set)
    added_functions: set[str] = field(default_factory=set)
    deleted_functions: set[str] = field(default_factory=set)
    value_changed_names: set[str] = field(default_factory=set)
    attribute_changed_names: set[str] = field(default_factory=set)

    @property
    def is_empty(self) -> bool:
        """True when the edit cannot change behaviour on any run."""
        return (not self.is_global) and (not self.functions)

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable summary."""
        return {
            "is_global": self.is_global,
            "global_reasons": list(self.global_reasons),
            "functions": sorted(self.functions),
            "lines": [[label, number] for label, number in sorted(self.lines)],
            "modified_functions": sorted(self.modified_functions),
            "added_functions": sorted(self.added_functions),
            "deleted_functions": sorted(self.deleted_functions),
            "value_changed_names": sorted(self.value_changed_names),
            "attribute_changed_names": sorted(self.attribute_changed_names),
        }


def all_usages(index: ModuleIndex) -> list[NameUsage]:
    """Every NameUsage recorded anywhere in a module index."""
    usages: list[NameUsage] = []
    for info in index.functions.values():
        usages.append(info.usage)
    for statement in index.module_statements:
        usages.append(statement.usage)
    for class_info in index.classes.values():
        usages.append(class_info.header_usage)
        for statement in class_info.body_statements:
            usages.append(statement.usage)
    return usages


def collect_global_reasons(index: ModuleIndex, version: str) -> list[str]:
    """Rule 1: constructs anywhere in a module version that force GLOBAL."""
    reasons: list[str] = []
    for usage in all_usages(index):
        for reason in usage.global_reasons:
            reasons.append(f"{version}: {reason}")
    return reasons


def live_statements(statements: list[StatementInfo]) -> list[StatementInfo]:
    """Statements that execute on import (drops docstrings and main guards)."""
    kept: list[StatementInfo] = []
    for statement in statements:
        if not statement.is_inert:
            kept.append(statement)
    return kept


def changed_statements(
    parent_statements: list[StatementInfo],
    child_statements: list[StatementInfo],
) -> tuple[list[StatementInfo], list[StatementInfo]]:
    """Diff two statement sequences; return (changed parent stmts, changed child stmts)."""
    parent_live = live_statements(parent_statements)
    child_live = live_statements(child_statements)
    parent_dumps: list[str] = []
    for statement in parent_live:
        parent_dumps.append(statement.dump)
    child_dumps: list[str] = []
    for statement in child_live:
        child_dumps.append(statement.dump)
    matcher = difflib.SequenceMatcher(a=parent_dumps, b=child_dumps, autojunk=False)
    changed_parent: list[StatementInfo] = []
    changed_child: list[StatementInfo] = []
    for tag, parent_start, parent_end, child_start, child_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        changed_parent.extend(parent_live[parent_start:parent_end])
        changed_child.extend(child_live[child_start:child_end])
    return changed_parent, changed_child


def is_dunder(name: str) -> bool:
    """True for ``__name__``-style special method names."""
    return name.startswith("__") and name.endswith("__") and len(name) > 4


def short_name(qualname: str) -> str:
    """Last component of a qualified name."""
    return qualname.rsplit(".", 1)[-1]


def top_level_name(qualname: str) -> str:
    """First component of a qualified name (the module-level binding)."""
    return qualname.split(".", 1)[0]


def has_impure_decorator(info: FunctionInfo) -> bool:
    """True when any decorator of the function is not a known pure wrapper."""
    for decorator in info.node.decorator_list:
        if not decorator_is_pure(decorator):
            return True
    return False


def decorator_dumps(info: FunctionInfo) -> list[str]:
    """Structural dumps of a function's decorators."""
    dumps: list[str] = []
    for decorator in info.node.decorator_list:
        dumps.append(ast.dump(decorator, include_attributes=False))
    return dumps


@dataclass
class _ChangeState:
    """Mutable working state of one impact computation."""

    global_reasons: list[str] = field(default_factory=list)
    value_changed: set[str] = field(default_factory=set)
    attribute_changed: set[str] = field(default_factory=set)
    import_time_changed: set[str] = field(default_factory=set)
    header_impacted: set[str] = field(default_factory=set)


def compute_impact(parent_source: str, child_source: str) -> ImpactSet:
    """Compute the function- and line-level impact set of editing parent into child."""
    parent = index_module(parent_source)
    child = index_module(child_source)
    state = _ChangeState()
    state.global_reasons.extend(collect_global_reasons(parent, "parent"))
    state.global_reasons.extend(collect_global_reasons(child, "child"))

    parent_names = set(parent.functions)
    child_names = set(child.functions)
    modified: set[str] = set()
    for qualname in parent_names & child_names:
        if parent.functions[qualname].dump != child.functions[qualname].dump:
            modified.add(qualname)
    added = child_names - parent_names
    deleted = parent_names - child_names

    apply_module_statement_changes(parent, child, state)
    apply_function_set_changes(parent, child, modified, added, deleted, state)
    apply_class_changes(parent, child, state)
    apply_import_time_closure(parent, modified, state)

    functions = set(modified) | set(deleted)
    functions.update(find_readers(parent, state))
    functions.update(state.header_impacted)
    is_global = len(state.global_reasons) > 0
    lines = compute_line_impact(parent, child, modified, deleted, functions, state)
    return ImpactSet(
        is_global=is_global,
        global_reasons=sorted(set(state.global_reasons)),
        functions=functions,
        lines=lines,
        modified_functions=modified,
        added_functions=added,
        deleted_functions=deleted,
        value_changed_names=set(state.value_changed),
        attribute_changed_names=set(state.attribute_changed),
    )


def apply_module_statement_changes(parent: ModuleIndex, child: ModuleIndex, state: _ChangeState) -> None:
    """Rule 3."""
    changed_parent, changed_child = changed_statements(parent.module_statements, child.module_statements)
    for statement in changed_parent + changed_child:
        if not statement.is_binding:
            state.global_reasons.append(f"changed module-level non-binding statement at line {statement.first_line}")
        state.value_changed.update(statement.bound_names)
        state.import_time_changed.update(statement.bound_names)


def apply_function_set_changes(
    parent: ModuleIndex,
    child: ModuleIndex,
    modified: set[str],
    added: set[str],
    deleted: set[str],
    state: _ChangeState,
) -> None:
    """Rules 4, 5 (methods) and 8."""
    for qualname in added | deleted:
        info = child.functions.get(qualname) or parent.functions.get(qualname)
        if info is None:
            continue
        if has_impure_decorator(info):
            state.global_reasons.append(f"added/deleted function with side-effecting decorator: {qualname}")
        if info.class_qualname is None:
            state.value_changed.add(qualname)
            state.import_time_changed.add(qualname)
            continue
        method = short_name(qualname)
        if is_dunder(method):
            state.global_reasons.append(f"added/deleted special method {qualname}")
        state.attribute_changed.add(method)
        state.import_time_changed.add(top_level_name(qualname))
    for qualname in modified:
        parent_info = parent.functions[qualname]
        child_info = child.functions[qualname]
        decorators_changed = decorator_dumps(parent_info) != decorator_dumps(child_info)
        impure = has_impure_decorator(parent_info) or has_impure_decorator(child_info)
        if decorators_changed and impure:
            state.global_reasons.append(f"changed side-effecting decorator on {qualname}")
        state.import_time_changed.add(top_level_name(qualname))


def apply_class_changes(parent: ModuleIndex, child: ModuleIndex, state: _ChangeState) -> None:
    """Rule 5 (class headers, class attributes, added/deleted classes)."""
    parent_classes = set(parent.classes)
    child_classes = set(child.classes)
    for qualname in parent_classes ^ child_classes:
        if "." in qualname:
            state.attribute_changed.add(short_name(qualname))
        else:
            state.value_changed.add(qualname)
        state.import_time_changed.add(top_level_name(qualname))
    for qualname in parent_classes & child_classes:
        parent_class = parent.classes[qualname]
        child_class = child.classes[qualname]
        if parent_class.header_dump != child_class.header_dump:
            state.global_reasons.append(f"changed class header of {qualname}")
        changed_parent, changed_child = changed_statements(parent_class.body_statements, child_class.body_statements)
        for statement in changed_parent + changed_child:
            if not statement.is_binding:
                state.global_reasons.append(f"changed non-binding statement in class {qualname}")
            state.attribute_changed.update(statement.bound_names)
            state.import_time_changed.add(top_level_name(qualname))


def usage_hits(usage: NameUsage, names: set[str]) -> bool:
    """True when ``usage`` loads any of ``names`` (or resolves globals dynamically)."""
    if usage.global_names & names:
        return True
    return usage.dynamic_globals and len(names) > 0


def apply_import_time_closure(parent: ModuleIndex, modified: set[str], state: _ChangeState) -> None:
    """Rule 6: propagate changes through code evaluated at import time (fixpoint)."""
    changed = True
    while changed:
        changed = False
        for statement in live_statements(parent.module_statements):
            if usage_hits(statement.usage, state.import_time_changed):
                new_names = statement.bound_names - state.value_changed
                if new_names:
                    state.value_changed.update(new_names)
                    state.import_time_changed.update(new_names)
                    changed = True
        for class_info in parent.classes.values():
            if closure_on_class(class_info, state):
                changed = True
    for info in parent.functions.values():
        if usage_hits(info.header_usage, state.import_time_changed):
            state.header_impacted.add(info.qualname)


def closure_on_class(class_info: ClassInfo, state: _ChangeState) -> bool:
    """Apply the import-time closure to one class; return True if state grew."""
    grew = False
    decorators_usage_hit = False
    for decorator in class_info.node.decorator_list:
        for node in ast.walk(decorator):
            if isinstance(node, ast.Name) and node.id in state.import_time_changed:
                decorators_usage_hit = True
    if decorators_usage_hit:
        reason = f"class decorator of {class_info.qualname} depends on a changed name"
        if reason not in state.global_reasons:
            state.global_reasons.append(reason)
            grew = True
    for statement in live_statements(class_info.body_statements):
        if usage_hits(statement.usage, state.import_time_changed):
            new_names = statement.bound_names - state.attribute_changed
            if new_names:
                state.attribute_changed.update(new_names)
                grew = True
            top = top_level_name(class_info.qualname)
            if top not in state.import_time_changed:
                state.import_time_changed.add(top)
                grew = True
    return grew


def find_readers(parent: ModuleIndex, state: _ChangeState) -> set[str]:
    """Rule 7: parent functions that read a changed global or attribute name."""
    readers: set[str] = set()
    for info in parent.functions.values():
        usage = info.usage
        if usage_hits(usage, state.value_changed):
            readers.add(info.qualname)
            continue
        if usage.attribute_names & state.attribute_changed:
            readers.add(info.qualname)
            continue
        if usage.dynamic_attributes and state.attribute_changed:
            readers.add(info.qualname)
    return readers


# ---------------------------------------------------------------------------
# Line-level impact (ablation O5)
# ---------------------------------------------------------------------------


def statement_line_ranges(function_node: ast.AST) -> list[tuple[int, int, int, bool]]:
    """(start, end, header_end, is_compound) for every statement inside a function."""
    ranges: list[tuple[int, int, int, bool]] = []
    for node in ast.walk(function_node):
        if not isinstance(node, ast.stmt):
            continue
        start = node.lineno
        end = node.end_lineno or node.lineno
        body = getattr(node, "body", None)
        is_compound = isinstance(body, list) and len(body) > 0 and isinstance(body[0], ast.stmt)
        header_end = end
        if is_compound:
            header_end = max(start, body[0].lineno - 1)
        ranges.append((start, end, header_end, is_compound))
    return ranges


def expand_line(line: int, ranges: list[tuple[int, int, int, bool]]) -> set[int]:
    """Expand a source line to the lines of the innermost statement part containing it.

    A line event is not guaranteed for every physical line of a multi-line statement,
    so we take the whole simple statement, or the whole header of a compound statement.
    """
    best: tuple[int, int, int, bool] | None = None
    for candidate in ranges:
        start, end, header_end, is_compound = candidate
        inside = start <= line <= (header_end if is_compound else end)
        if not inside:
            continue
        if best is None or (end - start) < (best[1] - best[0]):
            best = candidate
    if best is None:
        return {line}
    start, end, header_end, is_compound = best
    stop = header_end if is_compound else end
    return set(range(start, stop + 1))


def enclosing_header(line: int, ranges: list[tuple[int, int, int, bool]]) -> set[int]:
    """Header lines of the innermost compound statement whose span contains ``line``."""
    best: tuple[int, int, int, bool] | None = None
    for candidate in ranges:
        start, end, header_end, is_compound = candidate
        if not is_compound or not (start <= line <= end):
            continue
        if best is None or (end - start) < (best[1] - best[0]):
            best = candidate
    if best is None:
        return set()
    return set(range(best[0], best[2] + 1))


def function_source_lines(index: ModuleIndex, info: FunctionInfo) -> list[str]:
    """Source lines of one function (decorators included)."""
    return index.source_lines[info.first_line - 1 : info.last_line]


def changed_parent_lines_in_function(parent: ModuleIndex, child: ModuleIndex, qualname: str) -> set[int]:
    """Parent lines of a modified function whose execution signals the change."""
    parent_info = parent.functions[qualname]
    child_info = child.functions[qualname]
    parent_lines = function_source_lines(parent, parent_info)
    child_lines = function_source_lines(child, child_info)
    ranges = statement_line_ranges(parent_info.node)
    offset = parent_info.first_line
    matcher = difflib.SequenceMatcher(a=parent_lines, b=child_lines, autojunk=False)
    impacted: set[int] = set()
    for tag, parent_start, parent_end, child_start, child_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        if parent_end > parent_start:
            for relative in range(parent_start, parent_end):
                impacted.update(expand_line(offset + relative, ranges))
        else:
            impacted.update(insertion_lines(offset, parent_start, len(parent_lines), ranges))
    header_end = parent_info.node.body[0].lineno - 1
    if any(parent_info.first_line <= line <= header_end for line in impacted):
        impacted.add(parent_info.first_line)
    return impacted


def insertion_lines(offset: int, position: int, length: int, ranges: list[tuple[int, int, int, bool]]) -> set[int]:
    """Lines whose execution is necessary for code inserted before relative ``position``."""
    lines: set[int] = set()
    before = offset + position - 1
    after = offset + position
    if position - 1 >= 0:
        lines.update(expand_line(before, ranges))
        lines.update(enclosing_header(before, ranges))
    if position < length:
        lines.update(expand_line(after, ranges))
        lines.update(enclosing_header(after, ranges))
    if not lines:
        lines.add(offset)
    return lines


def reader_lines(info: FunctionInfo, state: _ChangeState) -> set[int]:
    """Lines of a reader function where a changed name or attribute is read."""
    ranges = statement_line_ranges(info.node)
    lines: set[int] = set()
    for node in ast.walk(info.node):
        hit = False
        if isinstance(node, ast.Name) and node.id in state.value_changed:
            hit = True
        if isinstance(node, ast.Attribute) and node.attr in state.attribute_changed:
            hit = True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            dynamic_attr = node.func.id in {"getattr", "hasattr", "vars"} and bool(state.attribute_changed)
            dynamic_global = node.func.id in {"globals", "locals"} and bool(state.value_changed)
            if dynamic_attr or dynamic_global:
                hit = True
        if hit and hasattr(node, "lineno"):
            lines.update(expand_line(node.lineno, ranges))
    return lines


def compute_line_impact(
    parent: ModuleIndex,
    child: ModuleIndex,
    modified: set[str],
    deleted: set[str],
    functions: set[str],
    state: _ChangeState,
) -> set[tuple[str, int]]:
    """Line-level impact set in parent coordinates."""
    lines: set[int] = set()
    for qualname in modified:
        lines.update(changed_parent_lines_in_function(parent, child, qualname))
    for qualname in deleted:
        info = parent.functions[qualname]
        lines.update(range(info.first_line, info.last_line + 1))
    for qualname in state.header_impacted:
        lines.add(parent.functions[qualname].first_line)
    for qualname in functions - modified - deleted - state.header_impacted:
        lines.update(reader_lines(parent.functions[qualname], state))
    labelled: set[tuple[str, int]] = set()
    for line in lines:
        labelled.add((HARNESS_LABEL, line))
    return labelled


def task_is_affected(impact: ImpactSet, covered_functions_per_run: list[set[str]], granularity: str = "function", covered_lines_per_run: list[set[tuple[str, int]]] | None = None) -> bool:
    """A task is affected iff the edit is GLOBAL or some selection run touched I."""
    if impact.is_global:
        return True
    if granularity == "function":
        for covered in covered_functions_per_run:
            if covered & impact.functions:
                return True
        return False
    if covered_lines_per_run is None:
        raise ValueError("line granularity needs covered_lines_per_run")
    for covered in covered_lines_per_run:
        if covered & impact.lines:
            return True
    return False
