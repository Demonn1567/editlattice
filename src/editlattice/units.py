"""Net change units of an evolved harness (design D2').

The units are the changed line ranges of the diff seed -> final harness. They never overlap
and are all expressed in seed line coordinates, so product S (a subset of units) is built by
replacing exactly the seed lines of the units in S: no text search, hence no ambiguity.
Units whose presence or absence alone would break the harness (a ``try:`` without its
``except``, a definition and its only use split across hunks that do not parse apart) are
merged until every subset of units is a valid harness.
"""

from __future__ import annotations

import ast
import difflib
import itertools
from dataclasses import dataclass

from editlattice.proposers.edit_format import EditError, validate_harness

# Above this many units the exhaustive validity check is replaced by singleton/co-singleton checks.
EXHAUSTIVE_CHECK_LIMIT = 14


@dataclass(frozen=True)
class Hunk:
    """One changed seed line range [start, end) and its replacement lines."""

    start: int
    end: int
    replacement: tuple[str, ...]


@dataclass(frozen=True)
class Unit:
    """A group of hunks that is switched on or off as a whole (0-based index)."""

    index: int
    hunks: tuple[Hunk, ...]
    items: tuple[str, ...]  # top-level names whose definitions the unit touches

    def describe(self, seed_lines: list[str]) -> str:
        """Short human-readable diff of the unit."""
        parts = []
        for hunk in self.hunks:
            removed = "".join("- " + line for line in seed_lines[hunk.start:hunk.end])
            added = "".join("+ " + line for line in hunk.replacement)
            parts.append(f"@@ seed lines {hunk.start + 1}-{hunk.end}\n{removed}{added}")
        return "\n".join(parts)


@dataclass
class UnitSet:
    """The seed and its net change units.

    ``conflicts`` (recombination only) lists pairs of units that rewrite overlapping seed lines
    and so cannot be combined; products are the conflict-free subsets that form a valid harness.
    """

    seed: str
    units: list[Unit]
    conflicts: frozenset[frozenset[int]] = frozenset()

    @property
    def m(self) -> int:
        """Number of units."""
        return len(self.units)

    def product(self, subset: frozenset[int]) -> str:
        """Seed with the units in ``subset`` applied."""
        return apply_hunks(self.seed, [hunk for unit in self.units if unit.index in subset for hunk in unit.hunks])

    def all_subsets(self) -> list[frozenset[int]]:
        """Every product (conflict-free, valid subset of units), smallest first."""
        indices = [unit.index for unit in self.units]
        subsets = [frozenset(combo) for size in range(len(indices) + 1) for combo in itertools.combinations(indices, size)]
        if not self.conflicts:
            return subsets
        return [subset for subset in subsets
                if not any(pair <= subset for pair in self.conflicts) and is_valid(self.product(subset))]


def apply_hunks(seed: str, hunks: list[Hunk]) -> str:
    """Replace seed line ranges (applied bottom-up so earlier ranges keep their positions)."""
    lines = seed.splitlines(keepends=True)
    for hunk in sorted(hunks, key=lambda item: item.start, reverse=True):
        lines[hunk.start:hunk.end] = list(hunk.replacement)
    return "".join(lines)


def net_hunks(seed: str, final: str) -> list[Hunk]:
    """Changed line ranges of seed -> final."""
    old, new = seed.splitlines(keepends=True), final.splitlines(keepends=True)
    matcher = difflib.SequenceMatcher(a=old, b=new, autojunk=False)
    return [Hunk(i1, i2, tuple(new[j1:j2])) for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag != "equal"]


def top_level_items(source: str) -> list[tuple[int, int, str]]:
    """(first line, last line, name) of each top-level statement, 0-based inclusive."""
    items = []
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = node.name
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            name = "<imports>"
        else:
            name = "<module>"
        first = min([node.lineno] + [decorator.lineno for decorator in getattr(node, "decorator_list", [])]) - 1
        items.append((first, node.end_lineno - 1, name))
    return items


def items_of(hunks: tuple[Hunk, ...], seed_items: list[tuple[int, int, str]]) -> tuple[str, ...]:
    """Top-level items whose seed lines a group of hunks touches (insertions: the item they sit in or next to)."""
    names = []
    for hunk in hunks:
        last = max(hunk.start, hunk.end - 1)
        for first, end, name in seed_items:
            if first <= last and hunk.start <= end and name not in names:
                names.append(name)
        if not any(first <= hunk.start <= end for first, end, _ in seed_items) and "<new>" not in names:
            names.append("<new>")
    return tuple(names)


def independent_groups(unit_set: UnitSet) -> list[frozenset[int]] | None:
    """Groups of units whose effects on the module text are independent (design D11).

    The seed is cut into segments: each top-level statement and each gap between statements
    (plus an empty segment at the end for appended text). A hunk touches the segments its seed
    lines cover (an insertion: the segment it is inserted before); segments touched by one hunk
    form one region, and units touching a common region form one group. A product's text is the
    concatenation, region by region, of text that depends only on the product's units in that
    region. If every variant of every region parses on its own as complete top-level statements,
    the product's module body is the concatenation of the regions' bodies, so any check that
    looks at one top-level statement at a time can be run group by group: over the subsets of
    one group, with every other unit absent (cost: sum of 2^|group| instead of 2^m).
    Returns None when some region variant does not parse on its own; callers then enumerate all
    subsets.
    """
    lines = unit_set.seed.splitlines(keepends=True)
    total = len(lines)
    segment_of = [-1] * (total + 1)
    count = 0
    position = 0
    for first, last, _ in top_level_items(unit_set.seed):
        if first > position:
            for line in range(position, first):
                segment_of[line] = count
            count += 1
        for line in range(first, last + 1):
            segment_of[line] = count
        count += 1
        position = last + 1
    if position < total:
        for line in range(position, total):
            segment_of[line] = count
        count += 1
    segment_of[total] = count
    count += 1
    parent = list(range(count))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    touched: dict[int, set[int]] = {}
    hunk_segment: list[tuple[int, Hunk, int]] = []
    for unit in unit_set.units:
        for hunk in unit.hunks:
            covered = [segment_of[line] for line in range(hunk.start, hunk.end)] or [segment_of[hunk.start]]
            low, high = min(covered), max(covered)
            for segment in range(low, high + 1):
                parent[find(segment)] = find(low)
            touched.setdefault(unit.index, set()).add(low)
            hunk_segment.append((unit.index, hunk, low))
    region_units: dict[int, set[int]] = {}
    for unit, segments in touched.items():
        for segment in segments:
            region_units.setdefault(find(segment), set()).add(unit)
    unit_parent = {unit.index: unit.index for unit in unit_set.units}

    def find_unit(x: int) -> int:
        while unit_parent[x] != x:
            unit_parent[x] = unit_parent[unit_parent[x]]
            x = unit_parent[x]
        return x

    for members in region_units.values():
        first_member = min(members)
        for member in members:
            unit_parent[find_unit(member)] = find_unit(first_member)
    groups: dict[int, set[int]] = {}
    for unit in unit_set.units:
        groups.setdefault(find_unit(unit.index), set()).add(unit.index)
    # Precondition: every variant of every region parses on its own.
    region_lines: dict[int, list[int]] = {}
    for line in range(total + 1):
        region_lines.setdefault(find(segment_of[line]), []).append(line)
    for root, members in region_units.items():
        start, end = min(region_lines[root]), max(region_lines[root]) + 1
        end = min(end, total)
        text_lines = lines[start:end]
        member_list = sorted(members)
        for size in range(len(member_list) + 1):
            for combo in itertools.combinations(member_list, size):
                if any(frozenset(pair) <= frozenset(combo) for pair in unit_set.conflicts):
                    continue
                hunks = [Hunk(h.start - start, h.end - start, h.replacement) for index, h, segment in hunk_segment
                         if index in combo and find(segment) == root]
                text = apply_hunks("".join(text_lines), hunks)
                if text and not text.endswith("\n") and end < total:
                    return None
                try:
                    ast.parse(text)
                except SyntaxError:
                    return None
    return [frozenset(group) for group in groups.values()]


def analysis_subsets(unit_set: UnitSet) -> list[frozenset[int]]:
    """The subsets a per-statement static check must examine: group by group when the groups are
    independent (``independent_groups``), otherwise every product."""
    groups = independent_groups(unit_set)
    if groups is None or len(groups) == 1:
        return unit_set.all_subsets()
    seen: set[frozenset[int]] = set()
    subsets: list[frozenset[int]] = []
    for group in groups:
        members = sorted(group)
        for size in range(len(members) + 1):
            for combo in itertools.combinations(members, size):
                subset = frozenset(combo)
                if subset in seen or any(pair <= subset for pair in unit_set.conflicts):
                    continue
                if unit_set.conflicts and not is_valid(unit_set.product(subset)):
                    continue
                seen.add(subset)
                subsets.append(subset)
    return sorted(subsets, key=lambda subset: (len(subset), sorted(subset)))


def is_valid(source: str) -> bool:
    """Parses and satisfies the harness contract."""
    try:
        validate_harness(source)
    except EditError:
        return False
    return True


def build_units(seed: str, final: str) -> UnitSet:
    """Net units of seed -> final, merged until every subset is a valid harness."""
    hunks = net_hunks(seed, final)
    groups: list[list[Hunk]] = [[hunk] for hunk in hunks]
    seed_items = top_level_items(seed)

    def product(selected: list[list[Hunk]]) -> str:
        return apply_hunks(seed, [hunk for group in selected for hunk in group])

    def merge(first: int, second: int) -> None:
        groups[first].extend(groups[second])
        groups[first].sort(key=lambda hunk: hunk.start)
        del groups[second]

    changed = True
    while changed:
        changed = False
        # Singleton and co-singleton checks; a failing group is merged with its nearest neighbour.
        for position in range(len(groups)):
            alone = product([groups[position]])
            without = product([group for index, group in enumerate(groups) if index != position])
            if not (is_valid(alone) and is_valid(without)) and len(groups) > 1:
                neighbour = position + 1 if position + 1 < len(groups) else position - 1
                merge(min(position, neighbour), max(position, neighbour))
                changed = True
                break
        if changed or len(groups) > EXHAUSTIVE_CHECK_LIMIT:
            continue
        for size in range(2, len(groups)):
            failing = next((combo for combo in itertools.combinations(range(len(groups)), size)
                            if not is_valid(product([groups[index] for index in combo]))), None)
            if failing is not None:
                for offset, index in enumerate(failing[1:]):
                    merge(failing[0], index - offset)
                changed = True
                break
    if not is_valid(product(groups)) or product(groups) != final:
        raise EditError("units do not rebuild the final harness")
    units = [Unit(index=number, hunks=tuple(group), items=items_of(tuple(group), seed_items)) for number, group in enumerate(groups)]
    return UnitSet(seed=seed, units=units)


# --------------------------------------------------------------------------- statement-level markers

MARKER = "__el_mark__"


def _marker_line(lines: list[str], unit_index: int) -> list[str]:
    """A marker statement indented like the first non-blank line of ``lines``."""
    for line in lines:
        if line.strip():
            indent = line[: len(line) - len(line.lstrip())]
            return [f"{indent}{MARKER}({unit_index})\n"]
    return []


def strip_markers(tree: ast.AST) -> ast.AST:
    """Remove marker statements (for comparing marked and unmarked products)."""

    class Stripper(ast.NodeTransformer):
        def generic_visit(self, node: ast.AST) -> ast.AST:
            super().generic_visit(node)
            for name, value in ast.iter_fields(node):
                if isinstance(value, list) and value and all(isinstance(item, ast.stmt) for item in value):
                    kept = [item for item in value if not (isinstance(item, ast.Expr) and isinstance(item.value, ast.Call)
                                                          and isinstance(item.value.func, ast.Name) and item.value.func.id == MARKER)]
                    setattr(node, name, kept or [ast.Pass()])
            return node

    return Stripper().visit(tree)


def _markers_inside_functions(source: str) -> bool:
    """Every marker statement sits inside a function body."""
    tree = ast.parse(source)
    inside: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            inside.update(id(child) for child in ast.walk(node))
    return all(id(node) in inside for node in ast.walk(tree)
               if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == MARKER)


def scope_signature(source: str) -> dict[str, tuple]:
    """Per function: locals, declared globals, free variables, generator/coroutine flags."""
    import symtable

    signature: dict[str, tuple] = {}
    tree = ast.parse(source)
    generators = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            body = [node.body] if isinstance(node, ast.Lambda) else node.body
            flags = set()
            for statement in body:
                for child in ast.walk(statement):
                    if isinstance(child, (ast.Yield, ast.YieldFrom)):
                        flags.add("generator")
                    if isinstance(child, ast.Await):
                        flags.add("await")
            generators[(getattr(node, "name", "lambda"), node.lineno)] = tuple(sorted(flags))

    def walk(table, prefix: str) -> None:
        for child in table.get_children():
            name = f"{prefix}.{child.get_name()}"
            if child.get_type() == "function":
                symbols = child.get_symbols()
                signature[f"{name}#{sum(1 for key in signature if key.startswith(name + '#'))}"] = (
                    tuple(sorted(s.get_name() for s in symbols if s.is_local())),
                    tuple(sorted(s.get_name() for s in symbols if s.is_declared_global())),
                    tuple(sorted(s.get_name() for s in symbols if s.is_free())),
                    generators.get((child.get_name(), child.get_lineno()), ()),
                )
            walk(child, name)

    walk(symtable.symtable(source, "harness", "exec"), "")
    return signature


def add_markers(unit_set: UnitSet, eligible: frozenset[int] | None = None) -> tuple[UnitSet, frozenset[int]]:
    """Insert a marker before every hunk of the units that can be tracked statement by statement.

    A unit is marked only if, for its hunks alone and together with every other marked unit:
    every product parses and validates, every marker lies inside a function body, removing the
    markers gives exactly the unmarked product (same AST), and toggling the unit never changes any
    function's scope signature. Unmarked units keep name-level reads.
    """
    seed_lines = unit_set.seed.splitlines(keepends=True)
    subsets = analysis_subsets(unit_set)
    unmarked_trees = {subset: ast.dump(ast.parse(unit_set.product(subset))) for subset in subsets}
    signatures = {subset: scope_signature(unit_set.product(subset)) for subset in subsets}

    def marked_unit(unit: Unit) -> Unit:
        hunks = []
        for hunk in unit.hunks:
            old = seed_lines[hunk.start:hunk.end]
            marker = _marker_line(old + list(hunk.replacement), unit.index)
            hunks.append(Hunk(hunk.start, hunk.end, tuple(marker + list(hunk.replacement))))
        return Unit(unit.index, tuple(hunks), unit.items)

    def marked_seed(units: list[Unit]) -> str:
        lines = list(seed_lines)
        for unit in units:
            for hunk in sorted(unit.hunks, key=lambda item: item.start, reverse=True):
                marker = _marker_line(seed_lines[hunk.start:hunk.end] + list(hunk.replacement), unit.index)
                lines[hunk.start:hunk.start] = marker
        return "".join(lines)

    def candidate(marked: set[int]) -> UnitSet:
        chosen = [unit for unit in unit_set.units if unit.index in marked]
        seed = marked_seed(sorted(chosen, key=lambda unit: -min(h.start for h in unit.hunks)))
        # Re-express every unit's hunks against the marked seed (markers shift line numbers).
        shifts = sorted((h.start, len(_marker_line(seed_lines[h.start:h.end] + list(h.replacement), u.index)))
                        for u in chosen for h in u.hunks)

        def shift(line: int) -> int:
            return line + sum(size for start, size in shifts if start < line)

        units = []
        for unit in unit_set.units:
            hunks = []
            for hunk in unit.hunks:
                replacement = list(hunk.replacement)
                if unit.index in marked:
                    replacement = _marker_line(seed_lines[hunk.start:hunk.end] + replacement, unit.index) + replacement
                    # The region starts at its own marker; shift(end) already counts that marker
                    # unless the region is empty (a pure insertion).
                    own = len(_marker_line(seed_lines[hunk.start:hunk.end] + list(hunk.replacement), unit.index))
                    hunks.append(Hunk(shift(hunk.start), shift(hunk.end) + (own if hunk.end == hunk.start else 0), tuple(replacement)))
                else:
                    hunks.append(Hunk(shift(hunk.start), shift(hunk.end), tuple(replacement)))
            units.append(Unit(unit.index, tuple(hunks), unit.items))
        return UnitSet(seed=seed, units=units, conflicts=unit_set.conflicts)

    by_index = {unit.index: unit for unit in unit_set.units}

    def dominated(source: str, subset: frozenset[int], marked: set[int]) -> bool:
        """Every line of every marked region belongs to statements that follow the region's marker
        directly, in the marker's own block, so no execution can reach the region without first
        executing the marker. Checked in each product, i.e. with and without the unit."""
        lines = source.splitlines()
        positions: dict[int, tuple[list[ast.stmt], int]] = {}
        for node in ast.walk(ast.parse(source)):
            for field in ("body", "orelse", "finalbody"):
                body = getattr(node, field, None)
                if isinstance(body, list):
                    for position, statement in enumerate(body):
                        if (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
                                and isinstance(statement.value.func, ast.Name) and statement.value.func.id == MARKER):
                            positions[statement.lineno] = (body, position)
        for index in marked:
            hunks = [h for h in by_index[index].hunks if _marker_line(seed_lines[h.start:h.end] + list(h.replacement), index)]
            marker_lines = [number + 1 for number, text in enumerate(lines) if text.strip() == f"{MARKER}({index})"]
            if len(marker_lines) != len(hunks):
                return False
            for line, hunk in zip(marker_lines, hunks):
                if line not in positions:
                    return False
                body, position = positions[line]
                length = len(hunk.replacement) if index in subset else hunk.end - hunk.start
                needed = [number for number in range(line + 1, line + length + 1)
                          if lines[number - 1].strip() and not lines[number - 1].strip().startswith("#")]
                covered: set[int] = set()
                for statement in body[position + 1:]:
                    first = min([statement.lineno] + [d.lineno for d in getattr(statement, "decorator_list", [])])
                    if needed and first > max(needed):
                        break
                    covered.update(range(first, statement.end_lineno + 1))
                if any(number not in covered for number in needed):
                    return False
        return True

    def acceptable(marked: set[int]) -> bool:
        trial = candidate(marked)
        for subset in subsets:
            source = trial.product(subset)
            if not is_valid(source) or not _markers_inside_functions(source):
                return False
            if not dominated(source, subset, marked):
                return False
            if ast.dump(strip_markers(ast.parse(source))) != unmarked_trees[subset]:
                return False
            if "".join(line for line in source.splitlines(keepends=True) if f"{MARKER}(" not in line) != unit_set.product(subset):
                return False
        return True

    def scope_stable(unit: Unit) -> bool:
        return all(signatures[subset] == signatures[subset | {unit.index}] for subset in subsets
                   if unit.index not in subset and subset | {unit.index} in signatures)

    conflicting = {index for pair in unit_set.conflicts for index in pair}
    marked: set[int] = set()
    for unit in unit_set.units:
        # Alternatives rewrite overlapping lines; they keep name-level reads.
        if unit.index in conflicting:
            continue
        # Units that change no definition (comments, formatting) are never read; a marker would
        # only make them look read.
        if eligible is not None and unit.index not in eligible:
            continue
        if scope_stable(unit) and acceptable(marked | {unit.index}):
            marked.add(unit.index)
    return candidate(marked), frozenset(marked)


# --------------------------------------------------------------------------- recombination

def hunks_conflict(first: Hunk, second: Hunk) -> bool:
    """Two hunks of the same seed overlap (or insert at the same point)."""
    if first.start == first.end and second.start == second.end:
        return first.start == second.start
    # An insertion at the first line of a replaced range is ambiguous (DEC-19): applied in one
    # order the replacement overwrites the inserted lines instead of the seed line.
    if first.start == first.end:
        return second.start <= first.start < second.end
    if second.start == second.end:
        return first.start <= second.start < first.end
    return max(first.start, second.start) < min(first.end, second.end)


def combine_lineages(seed: str, finals: list[str]) -> tuple[UnitSet, dict[int, list[int]]]:
    """One product line over the net units of several lineages from the same seed.

    Identical units are merged; units of different lineages that rewrite overlapping seed lines
    become mutually exclusive alternatives. Returns the unit set and, per unit, the lineages it
    came from.
    """
    pooled: dict[tuple[Hunk, ...], list[int]] = {}
    order: list[tuple[tuple[Hunk, ...], tuple[str, ...]]] = []
    for lineage, final in enumerate(finals):
        for unit in build_units(seed, final).units:
            if unit.hunks not in pooled:
                pooled[unit.hunks] = []
                order.append((unit.hunks, unit.items))
            pooled[unit.hunks].append(lineage)
    units = [Unit(index, hunks, items) for index, (hunks, items) in enumerate(order)]
    origin = {unit.index: pooled[unit.hunks] for unit in units}
    conflicts = set()
    for first in units:
        for second in units:
            if first.index < second.index and any(hunks_conflict(a, b) for a in first.hunks for b in second.hunks):
                conflicts.add(frozenset({first.index, second.index}))
    return UnitSet(seed=seed, units=units, conflicts=frozenset(conflicts)), origin


def group_by_items(unit_set: UnitSet) -> UnitSet:
    """Coarser features: merge units that touch the same top-level item (pre-registered for
    lineages with more than 12 units). A unit touching several items joins the group of its first."""
    groups: dict[str, list[Unit]] = {}
    for unit in unit_set.units:
        groups.setdefault(unit.items[0] if unit.items else "<module>", []).append(unit)
    merged = [Unit(index, tuple(sorted((h for u in members for h in u.hunks), key=lambda h: h.start)),
                   tuple(dict.fromkeys(item for u in members for item in u.items)))
              for index, members in enumerate(groups.values())]
    grouped = UnitSet(seed=unit_set.seed, units=merged)
    if grouped.product(frozenset(u.index for u in merged)) != unit_set.product(frozenset(u.index for u in unit_set.units)):
        raise EditError("grouping changed the final harness")
    return grouped
