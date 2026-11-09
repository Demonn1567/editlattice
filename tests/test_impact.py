"""Unit tests for change-impact analysis on synthetic diffs, incl. GLOBAL fallbacks."""

from __future__ import annotations

import textwrap

from editlattice.impact.diff import compute_impact, task_is_affected


def src(text: str) -> str:
    """Dedent a test module."""
    return textwrap.dedent(text).lstrip("\n")


BASE = src(
    '''
    """Toy harness."""
    import json

    PROMPT = "Answer briefly."
    RULES = "Use SQL."
    SYSTEM = PROMPT + " " + RULES
    LIMIT = 3


    def helper_a(x):
        return x + 1


    def helper_b(x):
        if x > 0:
            y = x * 2
        else:
            y = -x
        return y


    def uses_system():
        return SYSTEM


    def uses_limit(n=LIMIT):
        return n


    class Agent:
        retries = 2

        def step(self):
            return self.retries

        def act(self, name):
            return getattr(self, "tool_" + name)()

        def tool_echo(self):
            return "echo"


    def solve(task, llm, env):
        return {"answer": helper_a(1)}


    if __name__ == "__main__":
        print("debug")
    '''
)


def test_identical_and_comment_only_edits_have_empty_impact() -> None:
    assert compute_impact(BASE, BASE).is_empty
    commented = BASE.replace("def helper_a(x):", "def helper_a(x):  # a comment")
    reformatted = commented.replace("return x + 1", "return (x + 1)")
    impact = compute_impact(BASE, reformatted)
    assert impact.is_empty, impact.to_dict()


def test_main_guard_change_has_no_impact() -> None:
    child = BASE.replace('print("debug")', 'print("other")')
    assert compute_impact(BASE, child).is_empty


def test_modified_function_body_impacts_only_that_function() -> None:
    child = BASE.replace("return x + 1", "return x + 2")
    impact = compute_impact(BASE, child)
    assert not impact.is_global
    assert impact.functions == {"helper_a"}


def test_constant_change_impacts_transitive_readers() -> None:
    child = BASE.replace('RULES = "Use SQL."', 'RULES = "Use SQLite."')
    impact = compute_impact(BASE, child)
    assert not impact.is_global
    assert "SYSTEM" in impact.value_changed_names
    assert impact.functions == {"uses_system"}


def test_default_argument_reading_changed_constant() -> None:
    child = BASE.replace("LIMIT = 3", "LIMIT = 5")
    impact = compute_impact(BASE, child)
    assert impact.functions == {"uses_limit"}


def test_import_time_call_of_modified_function_propagates() -> None:
    parent = src(
        """
        def build():
            return "v1"

        PROMPT = build()

        def reader():
            return PROMPT

        def other():
            return 1
        """
    )
    child = parent.replace('return "v1"', 'return "v2"')
    impact = compute_impact(parent, child)
    assert impact.functions == {"build", "reader"}


def test_reordered_module_statements_are_detected() -> None:
    parent = src(
        """
        Y = 1
        X = Y
        Y = 2

        def read_x():
            return X
        """
    )
    child = src(
        """
        Y = 1
        Y = 2
        X = Y

        def read_x():
            return X
        """
    )
    impact = compute_impact(parent, child)
    assert "read_x" in impact.functions


def test_added_function_shadowing_a_builtin_impacts_callers() -> None:
    parent = src(
        """
        def count(items):
            return len(items)

        def unrelated():
            return 0
        """
    )
    child = parent + "\n\ndef len(items):\n    return 0\n"
    impact = compute_impact(parent, child)
    assert impact.functions == {"count"}


def test_added_method_impacts_dynamic_dispatch_and_attribute_readers() -> None:
    child = BASE.replace(
        '    def tool_echo(self):\n        return "echo"\n',
        '    def tool_echo(self):\n        return "echo"\n\n    def tool_new(self):\n        return "new"\n',
    )
    impact = compute_impact(BASE, child)
    assert not impact.is_global
    assert "Agent.act" in impact.functions
    assert "helper_a" not in impact.functions


def test_class_attribute_change_impacts_self_readers() -> None:
    child = BASE.replace("    retries = 2", "    retries = 4")
    impact = compute_impact(BASE, child)
    assert "Agent.step" in impact.functions
    assert "helper_b" not in impact.functions


def test_global_fallbacks() -> None:
    cases = {
        "exec": BASE.replace("return x + 1", 'return exec("x")'),
        "eval": BASE.replace("return x + 1", 'return eval("x")'),
        "setattr": BASE.replace("return x + 1", 'setattr(x, "a", 1)'),
        "monkeypatch": BASE.replace("LIMIT = 3", "LIMIT = 3\njson.dumps = str"),
        "star import": BASE.replace("import json", "import json\nfrom os import *"),
        "module call": BASE.replace("LIMIT = 3", "LIMIT = 3\nprint(LIMIT)"),
        "dunder": BASE.replace("    retries = 2", "    retries = 2\n\n    def __call__(self):\n        return 1"),
        "class header": BASE.replace("class Agent:", "class Agent(dict):"),
        "importlib": BASE.replace("return x + 1", 'return importlib.import_module("os")'),
    }
    for label, child in cases.items():
        impact = compute_impact(BASE, child)
        assert impact.is_global, label


def test_registering_decorator_on_added_function_is_global() -> None:
    parent = src(
        """
        TOOLS = {}

        def tool(fn):
            TOOLS[fn.__name__] = fn
            return fn

        @tool
        def search(q):
            return q

        def dispatch(name, arg):
            return TOOLS[name](arg)
        """
    )
    child = parent + "\n\n@tool\ndef lookup(q):\n    return q.upper()\n"
    impact = compute_impact(parent, child)
    assert impact.is_global


def test_pure_decorator_added_method_is_not_global() -> None:
    child = BASE.replace(
        "    def step(self):",
        "    @staticmethod\n    def helper():\n        return 0\n\n    def step(self):",
    )
    impact = compute_impact(BASE, child)
    assert not impact.is_global


def test_line_level_change_in_one_branch() -> None:
    child = BASE.replace("        y = -x\n", "        y = -x - 1\n")
    impact = compute_impact(BASE, child)
    assert impact.functions == {"helper_b"}
    changed_lines = {line for _, line in impact.lines}
    parent_lines = BASE.splitlines()
    else_branch_line = parent_lines.index("        y = -x") + 1
    if_branch_line = parent_lines.index("        y = x * 2") + 1
    assert else_branch_line in changed_lines
    assert if_branch_line not in changed_lines


def test_task_is_affected_function_and_line_level() -> None:
    child = BASE.replace("        y = -x\n", "        y = -x - 1\n")
    impact = compute_impact(BASE, child)
    assert task_is_affected(impact, [{"solve", "helper_b"}])
    assert not task_is_affected(impact, [{"solve", "helper_a"}])
    parent_lines = BASE.splitlines()
    if_branch_line = parent_lines.index("        y = x * 2") + 1
    only_if_branch = [{("harness.py", if_branch_line)}]
    assert not task_is_affected(impact, [{"helper_b"}], granularity="line", covered_lines_per_run=only_if_branch)
