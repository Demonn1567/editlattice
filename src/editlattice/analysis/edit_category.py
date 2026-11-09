"""RQ1: classify a harness edit into categories from its AST diff.

Rubric (deterministic; 50 edits are hand-validated and agreement is reported):
1. Every changed code unit gets a *role* from its descriptive identifier: verification,
   parsing, recovery, prompt, context (what the model is shown), tool, orchestration
   (the main loop), or logic. Module-level bindings with prompt-like names are prompts;
   numeric limits (MAX_*, *_LIMIT, ...) are control-flow; other bindings are config.
2. If a changed line of a *generic* unit (main loop or plain logic) references a unit with a
   specific role (e.g. the main loop stops calling ``run_public_tests``), the edit takes it.
3. Control-structure or numeric-literal changes inside a function add "control-flow".
4. The primary label is the role of the changed unit, or "multi" if units disagree;
   "no-op" means no code unit changed (comments/formatting only).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from editlattice.impact.names import index_module

ROLE_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("verification", ("verify", "check", "test", "validate", "assert")),
    ("parsing", ("parse", "extract", "clean")),
    ("recovery", ("recover", "repair", "retry", "fallback")),
    ("prompt", ("prompt", "instruction", "reminder", "system", "rules", "hint")),
    ("context", ("truncate", "format", "feedback", "describe_wrong", "describe_crash", "describe_timeout", "history")),
    ("tool", ("tool", "describe_table", "run_sql", "list_tables", "sample", "execute", "run_python")),
    ("orchestration", ("solve", "route", "main", "loop", "step")),
]
PROMPT_NAME_WORDS = ("prompt", "instruction", "reminder", "system", "rules", "hint", "message")
LIMIT_WORDS = ("max", "limit", "steps", "repairs", "rows", "chars", "timeout", "budget")
SPECIFIC_ROLES = ("verification", "prompt", "parsing", "recovery", "tool")


@dataclass
class EditCategories:
    """Labels of one edit, its primary label, and the evidence."""

    labels: set[str]
    primary: str
    evidence: list[str]


def role_of(name: str) -> str:
    """Role of a function from its identifier."""
    lowered = name.lower()
    for role, words in ROLE_RULES:
        for word in words:
            if word in lowered:
                return role
    return "logic"


def module_name_label(name: str) -> str:
    """Label of a module-level binding (prompt-like names first: they hold text)."""
    lowered = name.lower()
    for word in PROMPT_NAME_WORDS:
        if word in lowered:
            return "prompt"
    for word in LIMIT_WORDS:
        if word in lowered:
            return "control-flow"
    role = role_of(name)
    if role == "logic":
        return "config"
    return role


def control_signature(node: ast.AST) -> list[str]:
    """Sequence of control-flow node types inside a node."""
    kinds = (ast.If, ast.For, ast.While, ast.Break, ast.Continue, ast.Return, ast.Try, ast.IfExp)
    sequence: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, kinds):
            sequence.append(type(child).__name__)
    return sequence


def numeric_constants(node: ast.AST) -> list[float]:
    """All numeric literals inside a node, in order."""
    found: list[float] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, (int, float)) and not isinstance(child.value, bool):
            found.append(float(child.value))
    return found


def names_on_changed_lines(before: ast.AST, after: ast.AST) -> list[str]:
    """Identifiers on the lines that differ between two versions of a function."""
    before_lines = ast.unparse(before).splitlines()
    after_lines = ast.unparse(after).splitlines()
    changed = set(before_lines) ^ set(after_lines)
    names: list[str] = []
    for line in sorted(changed):
        try:
            tree = ast.parse(line.strip())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id not in names:
                names.append(node.id)
    return names


def referenced_role(name: str) -> str:
    """Role of a name referenced on a changed line (constants by their module label)."""
    if name.isupper():
        return module_name_label(name)
    return role_of(name)


def classify_edit(parent_source: str, child_source: str) -> EditCategories:
    """Categories of the edit parent -> child."""
    parent = index_module(parent_source)
    child = index_module(child_source)
    unit_labels: list[str] = []
    labels: set[str] = set()
    evidence: list[str] = []

    parent_statements: dict[str, object] = {}
    for statement in parent.module_statements:
        if not statement.is_inert:
            parent_statements[statement.dump] = statement
    child_statements: dict[str, object] = {}
    for statement in child.module_statements:
        if not statement.is_inert:
            child_statements[statement.dump] = statement
    changed_names: set[str] = set()
    for dump, statement in list(parent_statements.items()) + list(child_statements.items()):
        if dump not in parent_statements or dump not in child_statements:
            changed_names.update(statement.bound_names)
    for name in sorted(changed_names):
        label = module_name_label(name)
        unit_labels.append(label)
        labels.add(label)
        evidence.append(f"module name {name} -> {label}")

    for qualname in sorted(set(parent.functions) | set(child.functions)):
        before = parent.functions.get(qualname)
        after = child.functions.get(qualname)
        if before is not None and after is not None and before.dump == after.dump:
            continue
        role = role_of(qualname.rsplit(".", 1)[-1])
        if before is None or after is None:
            if role == "logic":
                role = "tool"
            evidence.append(f"{'added' if before is None else 'removed'} {qualname} -> {role}")
        else:
            evidence.append(f"changed {qualname} -> {role}")
            if control_signature(before.node) != control_signature(after.node) or numeric_constants(before.node) != numeric_constants(after.node):
                labels.add("control-flow")
                evidence.append(f"control/limits changed in {qualname}")
            generic_unit = role in ("orchestration", "logic")
            for name in names_on_changed_lines(before.node, after.node) if generic_unit else []:
                candidate = referenced_role(name)
                if candidate in SPECIFIC_ROLES and candidate != role:
                    evidence.append(f"changed line references {name} -> {candidate}")
                    role = candidate
                    break
        unit_labels.append(role)
        labels.add(role)

    if not unit_labels:
        return EditCategories({"no-op"}, "no-op", ["no code unit changed"])
    primary = unit_labels[0] if len(set(unit_labels)) == 1 else "multi"
    return EditCategories(labels, primary, evidence)
