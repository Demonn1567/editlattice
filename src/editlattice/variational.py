"""Variational replay: run one product, learn every product that must behave identically.

Design D3/D5. A product's run can depend on a unit only through a top-level name whose
definition differs across products (a function, class, constant or prompt). We

1. compute, for each such name, the set of units its definition depends on, including names
   its definition evaluates at import time (decorators, default arguments, class bodies,
   right-hand sides of module-level assignments), transitively;
2. rewrite, inside every function body of every product, each load of such a global name into
   ``__el_get__("name")``, which records a read of those units and returns the binding (or
   raises the same NameError Python would);
3. record the units of ``solve`` itself when the runner calls it.

Lemma (variational replay): under A1-A3, if the run of product S on (task, stream) read the
unit set Q, then every product S' with S' & Q == S & Q executes identically: every name the run
used has the same definition in S', and nothing else can influence the run. Extra reads only
reduce sharing. Constructs that defeat this reasoning (``globals()``, ``exec``, ``eval``,
``vars()``, ``__import__``, ``sys.modules``, ``global`` declarations of a differing name)
make every unit read at the start of every run.
"""

from __future__ import annotations

import ast
import symtable
import threading
import time
import traceback
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from editlattice.harness_runtime.runner import Benchmark, GuardedRunLLM, HarnessVersion, InfrastructureFailure, cached_score, stream_id
from editlattice.llm.client import LLMClient
from editlattice.units import MARKER, UnitSet, add_markers, analysis_subsets, is_valid

ACCESSOR = "__el_get__"
DEFAULT_STORE = Path(__file__).resolve().parents[2] / "runs" / "harnesses"
DYNAMIC_CALLS = frozenset({"globals", "vars", "exec", "eval", "__import__", "locals"})
# Line numbers shift when markers are inserted; code that can see them defeats statement-level reads.
INTROSPECTION_MODULES = frozenset({"traceback", "inspect"})
SCOPE_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


# --------------------------------------------------------------------------- definitions

def top_level_bindings(tree: ast.Module) -> dict[str, list[ast.stmt]]:
    """Top-level statements that bind each name, in order."""
    bindings: dict[str, list[ast.stmt]] = {}
    for node in tree.body:
        names: list[str] = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names = [node.name]
        elif isinstance(node, ast.Assign):
            names = [name.id for target in node.targets for name in ast.walk(target) if isinstance(name, ast.Name)]
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [(alias.asname or alias.name).split(".")[0] for alias in node.names]
        for name in names:
            bindings.setdefault(name, []).append(node)
    return bindings


def import_time_loads(node: ast.stmt) -> set[str]:
    """Names a top-level statement loads when the module is executed (not when called)."""
    loads: set[str] = set()

    def collect(expression: ast.AST | None) -> None:
        if expression is not None:
            loads.update(item.id for item in ast.walk(expression) if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load))

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        for decorator in node.decorator_list:
            collect(decorator)
        for default in node.args.defaults + [d for d in node.args.kw_defaults if d is not None]:
            collect(default)
        collect(node.returns)
        for argument in node.args.args + node.args.kwonlyargs:
            collect(argument.annotation)
    elif isinstance(node, ast.ClassDef):
        for expression in node.bases + [keyword.value for keyword in node.keywords] + node.decorator_list:
            collect(expression)
        for statement in node.body:  # the class body runs at import; method bodies do not
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                loads.update(import_time_loads(statement))
            else:
                collect(statement)
    elif isinstance(node, (ast.Import, ast.ImportFrom)):
        pass
    else:
        collect(node)
    return loads


def has_dynamic_constructs(tree: ast.Module) -> bool:
    """True if the module uses constructs that can reach globals without a visible name load."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in DYNAMIC_CALLS:
            return True
        if isinstance(node, ast.Attribute) and node.attr == "modules" and isinstance(node.value, ast.Name) and node.value.id == "sys":
            return True
        if isinstance(node, ast.Global):
            return True
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            if any(module.split(".")[0] in INTROSPECTION_MODULES for module in modules):
                return True
        if isinstance(node, ast.Attribute) and node.attr == "_getframe":
            return True
    return False


@dataclass
class ReadModel:
    """Per-name unit dependencies over all products of a unit set."""

    unit_set: UnitSet
    depends: dict[str, frozenset[int]] = field(default_factory=dict)  # name -> units (closed)
    global_units: frozenset[int] = frozenset()  # units every run reads (dynamic constructs, top-level code)
    dynamic: bool = False  # some product uses introspection or dynamic globals

    @classmethod
    def build(cls, unit_set: UnitSet, ignore: frozenset[int] = frozenset()) -> "ReadModel":
        """Compare every product's top-level definitions (exhaustive over subsets).

        Units in ``ignore`` are tracked by statement markers instead; they are never applied
        here, so they do not enter any name's dependency set.
        """
        subsets = [subset for subset in analysis_subsets(unit_set) if not subset & ignore]
        trees = {subset: ast.parse(unit_set.product(subset)) for subset in subsets}
        dumps: dict[frozenset[int], dict[str, str]] = {}
        loads: dict[str, set[str]] = {}
        other_code: dict[frozenset[int], str] = {}
        dynamic = False
        for subset, tree in trees.items():
            bindings = top_level_bindings(tree)
            dumps[subset] = {name: "\n".join(ast.dump(node) for node in nodes) for name, nodes in bindings.items()}
            for name, nodes in bindings.items():
                for node in nodes:
                    loads.setdefault(name, set()).update(import_time_loads(node))
            bound = {id(node) for nodes in bindings.values() for node in nodes}
            other_code[subset] = "\n".join(ast.dump(node) for node in tree.body if id(node) not in bound and not is_docstring(node))
            if has_dynamic_constructs(tree):
                dynamic = True
        all_units = frozenset(unit.index for unit in unit_set.units) - ignore
        direct: dict[str, set[int]] = {}
        names = {name for table in dumps.values() for name in table}
        global_units: set[int] = set()
        for unit in all_units:
            for subset in subsets:
                if unit in subset:
                    continue
                other = subset | {unit}
                if other not in dumps:  # infeasible combination (recombination alternatives)
                    continue
                for name in names:
                    if dumps[subset].get(name) != dumps[other].get(name):
                        direct.setdefault(name, set()).add(unit)
                if other_code[subset] != other_code[other]:
                    global_units.add(unit)
        if dynamic:
            global_units = set(all_units) | set(ignore)
        # Close over import-time dependencies (fixpoint).
        closed = {name: set(units) for name, units in direct.items()}
        changed = True
        while changed:
            changed = False
            for name in names:
                for dependency in loads.get(name, set()):
                    extra = closed.get(dependency, set()) - closed.get(name, set())
                    if extra:
                        closed.setdefault(name, set()).update(extra)
                        changed = True
        return cls(unit_set=unit_set, depends={name: frozenset(units) for name, units in closed.items() if units},
                   global_units=frozenset(global_units), dynamic=dynamic)


def is_docstring(node: ast.stmt) -> bool:
    """Module docstring (ignored when comparing top-level code)."""
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)


# --------------------------------------------------------------------------- instrumentation

class _GlobalLoadRewriter(ast.NodeTransformer):
    """Replace loads of differing global names inside function scopes by accessor calls."""

    def __init__(self, table: symtable.SymbolTable, names: frozenset[str]) -> None:
        self.names = names
        self.stack: list[symtable.SymbolTable] = [table]
        # Python 3.12 inlines list/set/dict comprehensions (PEP 709): they have no symbol table of
        # their own, but their loop variables stay private. Track those names explicitly.
        self.inlined_locals: list[set[str]] = []
        self.children: dict[tuple[str, int], list[symtable.SymbolTable]] = {}
        self._index(table)

    def _index(self, table: symtable.SymbolTable) -> None:
        for child in table.get_children():
            self.children.setdefault((child.get_name(), child.get_lineno()), []).append(child)
            self._index(child)

    def _child(self, name: str, lineno: int) -> symtable.SymbolTable | None:
        candidates = self.children.get((name, lineno), [])
        if not candidates and name in ("listcomp", "setcomp", "dictcomp"):
            return None  # inlined comprehension (Python 3.12+)
        if not candidates:
            raise RuntimeError(f"no symbol table for {name} at line {lineno}")
        return candidates.pop(0)

    def _in_function(self) -> bool:
        return self.stack[-1].get_type() == "function"

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if isinstance(node.ctx, ast.Load) and node.id in self.names and self._in_function():
            if any(node.id in scope for scope in self.inlined_locals):
                return node
            try:
                symbol = self.stack[-1].lookup(node.id)
            except KeyError as error:
                # Neither skipping (a missed read) nor rewriting (a local read as a global) is safe.
                raise RuntimeError(f"cannot resolve the scope of {node.id!r} at line {node.lineno}") from error
            if symbol.is_global():
                call = ast.Call(func=ast.Name(id=ACCESSOR, ctx=ast.Load()), args=[ast.Constant(node.id)], keywords=[])
                return ast.copy_location(call, node)
        return node

    def _visit_function(self, node: ast.AST, name: str) -> ast.AST:
        # Decorators, defaults and annotations belong to the enclosing scope.
        for attribute in ("decorator_list",):
            setattr(node, attribute, [self.visit(item) for item in getattr(node, attribute, [])])
        arguments = node.args
        arguments.defaults = [self.visit(item) for item in arguments.defaults]
        arguments.kw_defaults = [self.visit(item) if item is not None else None for item in arguments.kw_defaults]
        self.stack.append(self._child(name, node.lineno))
        if isinstance(node, ast.Lambda):
            node.body = self.visit(node.body)
        else:
            node.body = [self.visit(statement) for statement in node.body]
        self.stack.pop()
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        return self._visit_function(node, node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        return self._visit_function(node, node.name)

    def visit_Lambda(self, node: ast.Lambda) -> ast.AST:
        return self._visit_function(node, "lambda")

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        node.bases = [self.visit(item) for item in node.bases]
        node.keywords = [self.visit(item) for item in node.keywords]
        node.decorator_list = [self.visit(item) for item in node.decorator_list]
        self.stack.append(self._child(node.name, node.lineno))
        node.body = [self.visit(statement) for statement in node.body]
        self.stack.pop()
        return node

    def _visit_comprehension(self, node: ast.AST, name: str) -> ast.AST:
        # The first iterable is evaluated in the enclosing scope.
        generators = node.generators
        generators[0].iter = self.visit(generators[0].iter)
        table = self._child(name, node.lineno)
        inlined = table is None
        if inlined:
            targets = {item.id for generator in generators for item in ast.walk(generator.target) if isinstance(item, ast.Name)}
            self.inlined_locals.append(targets)
        else:
            self.stack.append(table)
        for position, generator in enumerate(generators):
            generator.target = self.visit(generator.target)
            if position > 0:
                generator.iter = self.visit(generator.iter)
            generator.ifs = [self.visit(item) for item in generator.ifs]
        for attribute in ("elt", "key", "value"):
            if hasattr(node, attribute):
                setattr(node, attribute, self.visit(getattr(node, attribute)))
        if inlined:
            self.inlined_locals.pop()
        else:
            self.stack.pop()
        return node

    def visit_ListComp(self, node: ast.ListComp) -> ast.AST:
        return self._visit_comprehension(node, "listcomp")

    def visit_SetComp(self, node: ast.SetComp) -> ast.AST:
        return self._visit_comprehension(node, "setcomp")

    def visit_DictComp(self, node: ast.DictComp) -> ast.AST:
        return self._visit_comprehension(node, "dictcomp")

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> ast.AST:
        return self._visit_comprehension(node, "genexpr")


def instrument(source: str, names: frozenset[str]) -> str:
    """Source with every in-function load of a global name in ``names`` routed through the accessor."""
    tree = ast.parse(source)
    table = symtable.symtable(source, "harness", "exec")
    tree = _GlobalLoadRewriter(table, names).visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


class ReadLog:
    """Units read during one run, and the order in which each was first read."""

    def __init__(self) -> None:
        self.units: set[int] = set()
        self.order: list[int] = []

    def read(self, units: frozenset[int]) -> None:
        """Record a read (units read together are ordered by index)."""
        for unit in sorted(units - self.units):
            self.order.append(unit)
        self.units.update(units)


def make_accessor(namespace: dict[str, Any], model: ReadModel, log: ReadLog):
    """``__el_get__``: log the units a name depends on, then behave like a global load."""
    depends = model.depends

    def accessor(name: str) -> Any:
        log.read(depends.get(name, frozenset()))
        try:
            return namespace[name]
        except KeyError:
            builtins = namespace.get("__builtins__")
            table = builtins if isinstance(builtins, dict) else getattr(builtins, "__dict__", {})
            if name in table:
                return table[name]
            raise NameError(f"name '{name}' is not defined") from None

    return accessor


# --------------------------------------------------------------------------- running products

class InvalidProduct(RuntimeError):
    """A product built on demand does not parse or violates the harness contract."""


@dataclass
class Instrumented:
    """Instrumented sources of every product (compiled once)."""

    model: ReadModel
    sources: dict[frozenset[int], str]
    marked: frozenset[int] = frozenset()
    # Lazy mode: sources are built on first use from these (no enumeration of all subsets).
    unit_set: UnitSet | None = None
    names: frozenset[str] | None = None
    store_dir: Path = field(default_factory=lambda: DEFAULT_STORE)
    _code: dict[frozenset[int], types.CodeType] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    statement_level: bool = False

    @classmethod
    def build(cls, unit_set: UnitSet, statement_level: bool = False, store_dir: str | Path | None = None,
              lazy: bool = False) -> "Instrumented":
        """Read model plus instrumented source for every subset.

        With ``statement_level`` the units that pass the marker checks (``units.add_markers``)
        are read when execution reaches one of their changed statements, not when the enclosing
        function is used; the others keep name-level reads. With ``lazy`` a product's source is
        built when it is first run (tree exploration runs few of the 2^m products).
        """
        marked: frozenset[int] = frozenset()
        if statement_level:
            coarse = ReadModel.build(unit_set)
            alive = frozenset(unit for units in coarse.depends.values() for unit in units) | coarse.global_units
            unit_set, marked = add_markers(unit_set, eligible=alive)
        model = ReadModel.build(unit_set, ignore=marked)
        if model.dynamic:
            # The harness can observe its own code (introspection, dynamic globals): run the plain
            # products unchanged; every unit counts as read, so exploration is brute force.
            plain = UnitSet(seed=unit_set.seed, units=unit_set.units, conflicts=unit_set.conflicts) if not marked else None
            original = plain if plain is not None else _without_markers(unit_set)
            store = Path(store_dir) if store_dir else DEFAULT_STORE
            if lazy:
                return cls(model=model, sources={}, statement_level=statement_level, marked=frozenset(), store_dir=store,
                           unit_set=original, names=None)
            sources = {subset: original.product(subset) for subset in original.all_subsets()}
            return cls(model=model, sources=sources, statement_level=statement_level, marked=frozenset(), store_dir=store)
        names = frozenset(model.depends)
        store = Path(store_dir) if store_dir else DEFAULT_STORE
        if lazy:
            return cls(model=model, sources={}, statement_level=statement_level, marked=marked, store_dir=store,
                       unit_set=unit_set, names=names)
        sources = {subset: instrument(unit_set.product(subset), names) for subset in unit_set.all_subsets()}
        return cls(model=model, sources=sources, statement_level=statement_level, marked=marked, store_dir=store)

    def code(self, subset: frozenset[int]) -> types.CodeType:
        """Compiled instrumented product."""
        with self._lock:
            if subset not in self.sources:
                if self.unit_set is None:
                    raise KeyError(f"no product {sorted(subset)}")
                source = self.unit_set.product(subset)
                # Every product must be a valid harness (precondition of Lemma 2); a lazily built one
                # that is not would otherwise look like an ordinary failed run.
                if not is_valid(source):
                    raise InvalidProduct(f"product {sorted(subset)} is not a valid harness")
                self.sources[subset] = source if self.names is None else instrument(source, self.names)
            if subset not in self._code:
                # The same content-addressed path a plain run of this source would use, so file
                # names in tracebacks match plain runs exactly.
                path = HarnessVersion.from_source(self.sources[subset], self.store_dir).path
                self._code[subset] = compile(self.sources[subset], path, "exec")
            return self._code[subset]


def run_product(benchmark: Benchmark, task: dict[str, Any], instrumented: Instrumented, subset: frozenset[int], replicate: int,
                client: LLMClient, llm_params: dict[str, Any], max_llm_calls: int, namespace: str = "") -> dict[str, Any]:
    """Run one product on (task, replicate) on the shared CRN stream; record the units it read."""
    task_id = str(task["task_id"])
    stream = stream_id(task_id, replicate, True, "", namespace)
    llm = GuardedRunLLM(client=client, replicate_id=stream, default_params=dict(llm_params))
    llm.max_calls = max_llm_calls
    log = ReadLog()
    log.read(instrumented.model.global_units)
    log.read(instrumented.model.depends.get("solve", frozenset()))
    answer: Any = None
    error = ""
    env = None
    started = time.perf_counter()
    try:
        env = benchmark.make_env(task)
        module = types.ModuleType("harness_product")
        module.__dict__[ACCESSOR] = make_accessor(module.__dict__, instrumented.model, log)
        module.__dict__[MARKER] = lambda unit: log.read(frozenset({unit}))
        exec(instrumented.code(subset), module.__dict__)  # noqa: S102 - trusted harness code under study
        result = module.solve(benchmark.public_view(task), llm, env)
        if isinstance(result, dict):
            answer = result.get("answer")
        else:
            error = f"harness returned {type(result).__name__}, expected dict"
    except InvalidProduct:
        raise
    except Exception as exc:  # noqa: BLE001 - a crashing harness is a failed run
        error = f"{type(exc).__name__}: {exc}"[:500] + " | " + traceback.format_exc(limit=2)[-200:]
    finally:
        if env is not None:
            benchmark.close_env(env)
    if llm.infra_failure:
        raise InfrastructureFailure(f"{task_id} stream {stream}: {llm.infra_failure}")
    outcome, score = cached_score(benchmark, task, answer)
    return {
        "task_id": task_id, "replicate_id": replicate, "stream": stream, "product": sorted(subset),
        "reads": sorted(log.units), "read_order": list(log.order), "outcome": int(outcome), "score": float(score),
        "answer": answer if isinstance(answer, (str, int, float, type(None))) else str(answer),
        "n_llm_calls": llm.usage.n_llm_calls, "n_cache_hits": llm.usage.n_cache_hits,
        "tokens_out": llm.usage.tokens_out, "nominal_tokens_out": llm.usage.nominal_tokens_out,
        "request_fps": llm.request_fingerprints, "error": error, "wall_time": round(time.perf_counter() - started, 3),
    }


def explore_cell(benchmark: Benchmark, task: dict[str, Any], instrumented: Instrumented, replicate: int, client: LLMClient,
                 llm_params: dict[str, Any], max_llm_calls: int, namespace: str = "") -> tuple[dict[frozenset[int], dict[str, Any]], list[dict[str, Any]]]:
    """Exact outcomes of every product on one (task, replicate), executing one run per read pattern.

    Returns (outcome record per product, list of executed runs).
    """
    pending = set(instrumented.sources)
    results: dict[frozenset[int], dict[str, Any]] = {}
    executed: list[dict[str, Any]] = []
    while pending:
        subset = min(pending, key=lambda item: (len(item), sorted(item)))
        record = run_product(benchmark, task, instrumented, subset, replicate, client, llm_params, max_llm_calls, namespace)
        executed.append(record)
        reads = frozenset(record["reads"])
        for other in [item for item in pending if item & reads == subset & reads]:
            results[other] = record
            pending.discard(other)
    return results, executed


def _without_markers(unit_set: UnitSet) -> UnitSet:
    """The unit set with marker lines removed from its seed and hunks (line numbers recomputed)."""
    from editlattice.units import build_units

    def strip(text: str) -> str:
        return "".join(line for line in text.splitlines(keepends=True) if f"{MARKER}(" not in line)

    seed = strip(unit_set.seed)
    rebuilt = build_units(seed, strip(unit_set.product(frozenset(unit.index for unit in unit_set.units))))
    if [unit.index for unit in rebuilt.units] != [unit.index for unit in unit_set.units]:
        raise RuntimeError("could not rebuild units without markers")
    return rebuilt
