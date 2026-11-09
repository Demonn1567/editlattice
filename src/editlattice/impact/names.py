"""AST helpers: top-level definitions of a harness module and the names they read.

A harness is a single Python module. We index:
* functions: top-level functions and methods of (possibly nested) classes, keyed by the
  same qualified name the coverage tracer records (``solve``, ``Agent.step``);
* classes: keyed by qualified name;
* module-level statements: every top-level statement that is not a function or class
  definition, in source order.

For every function we record which global names it loads, which attribute names it
reads, and whether it performs a *dynamic* lookup (``getattr`` with a non-constant name,
``vars()``, ``__dict__``) that could resolve to any attribute, or a dynamic global
lookup (``globals()``) that could resolve to any name.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

# Calls that make name-based reasoning unsound wherever they appear => GLOBAL.
GLOBAL_CALL_NAMES = frozenset({"exec", "eval", "compile", "__import__", "setattr", "delattr"})
# Module attribute access that indicates import machinery or frame hacking => GLOBAL.
GLOBAL_ATTRIBUTE_BASES = frozenset({"importlib", "inspect", "builtins", "ctypes", "gc"})
# Only these ``sys`` attributes can redirect execution or name resolution.
GLOBAL_SYS_ATTRIBUTES = frozenset({"modules", "_getframe", "settrace", "setprofile", "meta_path", "path_hooks"})
# Calls that make *attribute* resolution dynamic for the calling function.
DYNAMIC_ATTRIBUTE_CALLS = frozenset({"getattr", "hasattr", "vars"})
# Calls that make *global name* resolution dynamic for the calling function.
DYNAMIC_GLOBAL_CALLS = frozenset({"globals", "locals"})
# Decorators known to be pure wrappers with no import-time side effects.
PURE_DECORATORS = frozenset({"staticmethod", "classmethod", "property", "abstractmethod", "dataclass", "cache", "lru_cache", "wraps"})
# Dunder methods are invoked implicitly; adding or deleting one changes behaviour
# without any reader. Modifying one is fine (its execution is traced).
DUNDER_PREFIX = "__"


@dataclass
class NameUsage:
    """Names read by a piece of code."""

    global_names: set[str] = field(default_factory=set)
    attribute_names: set[str] = field(default_factory=set)
    dynamic_attributes: bool = False
    dynamic_globals: bool = False
    global_reasons: list[str] = field(default_factory=list)


@dataclass
class FunctionInfo:
    """One function or method of the harness module."""

    qualname: str
    node: ast.FunctionDef | ast.AsyncFunctionDef
    dump: str
    usage: NameUsage
    header_usage: NameUsage
    class_qualname: str | None
    first_line: int
    last_line: int


@dataclass
class ClassInfo:
    """One class of the harness module (without its methods' bodies)."""

    qualname: str
    node: ast.ClassDef
    header_dump: str
    header_usage: NameUsage
    body_statements: list["StatementInfo"]
    method_names: set[str]


@dataclass
class StatementInfo:
    """A non-definition statement at module or class level."""

    dump: str
    bound_names: set[str]
    usage: NameUsage
    is_inert: bool
    is_binding: bool
    first_line: int
    last_line: int


@dataclass
class ModuleIndex:
    """Everything the impact analysis needs to know about one harness version."""

    source_lines: list[str]
    functions: dict[str, FunctionInfo]
    classes: dict[str, ClassInfo]
    module_statements: list[StatementInfo]
    imported_module_names: set[str]
    global_reasons: list[str]


def dump_node(node: ast.AST) -> str:
    """Structural dump of a node that ignores line numbers and column offsets."""
    return ast.dump(node, annotate_fields=True, include_attributes=False)


def collect_usage(nodes: list[ast.AST], imported_modules: set[str]) -> NameUsage:
    """Collect the names, attribute names and dynamic features used inside ``nodes``."""
    usage = NameUsage()
    for root in nodes:
        for node in ast.walk(root):
            inspect_node_for_usage(node, usage, imported_modules)
    return usage


def inspect_node_for_usage(node: ast.AST, usage: NameUsage, imported_modules: set[str]) -> None:
    """Update ``usage`` with what a single AST node reads or does."""
    if isinstance(node, ast.Name):
        usage.global_names.add(node.id)
        return
    if isinstance(node, ast.Attribute):
        usage.attribute_names.add(node.attr)
        if node.attr == "__dict__":
            usage.dynamic_attributes = True
        base = node.value
        if isinstance(base, ast.Name) and base.id in GLOBAL_ATTRIBUTE_BASES:
            usage.global_reasons.append(f"uses {base.id}.{node.attr}")
        if isinstance(base, ast.Name) and base.id == "sys" and node.attr in GLOBAL_SYS_ATTRIBUTES:
            usage.global_reasons.append(f"uses sys.{node.attr}")
        if isinstance(node.ctx, (ast.Store, ast.Del)) and isinstance(base, ast.Name) and base.id in imported_modules:
            usage.global_reasons.append(f"monkey-patches {base.id}.{node.attr}")
        return
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        called = node.func.id
        if called in GLOBAL_CALL_NAMES:
            usage.global_reasons.append(f"calls {called}()")
        if called in DYNAMIC_GLOBAL_CALLS:
            usage.dynamic_globals = True
        if called in DYNAMIC_ATTRIBUTE_CALLS:
            record_attribute_lookup(node, called, usage)
        return
    if isinstance(node, (ast.Global, ast.Nonlocal)):
        usage.global_names.update(node.names)


def record_attribute_lookup(node: ast.Call, called: str, usage: NameUsage) -> None:
    """``getattr(obj, "x")`` reads attribute ``x``; a non-constant name is dynamic."""
    if called == "vars":
        usage.dynamic_attributes = True
        return
    if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):
        usage.attribute_names.add(node.args[1].value)
        return
    usage.dynamic_attributes = True


def bound_names_of_statement(statement: ast.stmt) -> set[str]:
    """Module-level names a statement (re)binds or mutates."""
    names: set[str] = set()
    targets: list[ast.AST] = []
    if isinstance(statement, ast.Assign):
        targets = list(statement.targets)
    elif isinstance(statement, (ast.AnnAssign, ast.AugAssign)):
        targets = [statement.target]
    elif isinstance(statement, (ast.Import, ast.ImportFrom)):
        for alias in statement.names:
            bound = alias.asname or alias.name.split(".")[0]
            names.add(bound)
        return names
    elif isinstance(statement, ast.Delete):
        targets = list(statement.targets)
    else:
        for node in ast.walk(statement):
            if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                names.add(node.id)
        return names
    for target in targets:
        names.update(base_names_of_target(target))
    return names


def base_names_of_target(target: ast.AST) -> set[str]:
    """``X``, ``X[k]``, ``X.attr`` and tuple targets all mutate the base name ``X``."""
    names: set[str] = set()
    for node in ast.walk(target):
        if isinstance(node, ast.Name):
            names.add(node.id)
    return names


def is_main_guard(statement: ast.stmt) -> bool:
    """True for ``if __name__ == "__main__":`` (never executed when imported)."""
    if not isinstance(statement, ast.If):
        return False
    test = statement.test
    if not isinstance(test, ast.Compare) or len(test.comparators) != 1:
        return False
    left = test.left
    right = test.comparators[0]
    left_is_name = isinstance(left, ast.Name) and left.id == "__name__"
    right_is_main = isinstance(right, ast.Constant) and right.value == "__main__"
    return left_is_name and right_is_main


def is_docstring(statement: ast.stmt) -> bool:
    """True for a bare string-constant expression statement."""
    return isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str)


def is_binding_statement(statement: ast.stmt) -> bool:
    """Statements whose effect is fully described by the names they bind."""
    return isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Import, ast.ImportFrom))


def build_statement_info(statement: ast.stmt, imported_modules: set[str]) -> StatementInfo:
    """Describe one module- or class-level non-definition statement."""
    inert = is_main_guard(statement) or is_docstring(statement)
    usage = collect_usage([statement], imported_modules)
    if isinstance(statement, ast.ImportFrom) and any(alias.name == "*" for alias in statement.names):
        usage.global_reasons.append("star import")
    return StatementInfo(
        dump=dump_node(statement),
        bound_names=bound_names_of_statement(statement),
        usage=usage,
        is_inert=inert,
        is_binding=is_binding_statement(statement),
        first_line=statement.lineno,
        last_line=statement.end_lineno or statement.lineno,
    )


def function_first_line(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """First line of a function including decorators (matches ``co_firstlineno``)."""
    lines = [node.lineno]
    for decorator in node.decorator_list:
        lines.append(decorator.lineno)
    return min(lines)


def decorator_is_pure(decorator: ast.expr) -> bool:
    """True when a decorator is a known pure wrapper."""
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    if isinstance(target, ast.Name):
        return target.id in PURE_DECORATORS
    if isinstance(target, ast.Attribute):
        return target.attr in PURE_DECORATORS or target.attr in {"setter", "getter", "deleter"}
    return False


def header_nodes(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.AST]:
    """Parts of a function evaluated at definition (import) time."""
    parts: list[ast.AST] = list(node.decorator_list)
    parts.extend(node.args.defaults)
    for default in node.args.kw_defaults:
        if default is not None:
            parts.append(default)
    if node.returns is not None:
        parts.append(node.returns)
    return parts


def index_module(source: str) -> ModuleIndex:
    """Parse ``source`` and build a :class:`ModuleIndex`."""
    tree = ast.parse(source)
    imported_modules = collect_imported_modules(tree)
    index = ModuleIndex(
        source_lines=source.splitlines(),
        functions={},
        classes={},
        module_statements=[],
        imported_module_names=imported_modules,
        global_reasons=[],
    )
    for statement in tree.body:
        index_top_level_statement(statement, "", index, imported_modules)
    return index


def collect_imported_modules(tree: ast.Module) -> set[str]:
    """Names bound by ``import x`` / ``import x as y`` anywhere in the module."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
    return names


def index_top_level_statement(statement: ast.stmt, prefix: str, index: ModuleIndex, imported_modules: set[str]) -> None:
    """Add one statement (and, for classes, its members) to ``index``."""
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
        add_function(statement, prefix, None, index, imported_modules)
        return
    if isinstance(statement, ast.ClassDef):
        add_class(statement, prefix, index, imported_modules)
        return
    index.module_statements.append(build_statement_info(statement, imported_modules))


def add_function(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    prefix: str,
    class_qualname: str | None,
    index: ModuleIndex,
    imported_modules: set[str],
) -> None:
    """Register a function or method."""
    qualname = f"{prefix}{node.name}"
    index.functions[qualname] = FunctionInfo(
        qualname=qualname,
        node=node,
        dump=dump_node(node),
        usage=collect_usage([node], imported_modules),
        header_usage=collect_usage(header_nodes(node), imported_modules),
        class_qualname=class_qualname,
        first_line=function_first_line(node),
        last_line=node.end_lineno or node.lineno,
    )


def add_class(node: ast.ClassDef, prefix: str, index: ModuleIndex, imported_modules: set[str]) -> None:
    """Register a class, its methods, nested classes and class-level statements."""
    qualname = f"{prefix}{node.name}"
    header_parts: list[ast.AST] = list(node.bases) + list(node.keywords) + list(node.decorator_list)
    header_dumps: list[str] = []
    for part in header_parts:
        header_dumps.append(dump_node(part))
    header_dump = "|".join(header_dumps)
    body_statements: list[StatementInfo] = []
    method_names: set[str] = set()
    for member in node.body:
        if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
            method_names.add(member.name)
            add_function(member, f"{qualname}.", qualname, index, imported_modules)
        elif isinstance(member, ast.ClassDef):
            method_names.add(member.name)
            add_class(member, f"{qualname}.", index, imported_modules)
        else:
            body_statements.append(build_statement_info(member, imported_modules))
    index.classes[qualname] = ClassInfo(
        qualname=qualname,
        node=node,
        header_dump=header_dump,
        header_usage=collect_usage(header_parts, imported_modules),
        body_statements=body_statements,
        method_names=method_names,
    )
