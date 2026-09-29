import ast
from pathlib import Path


def _signal_handler():
    tree = ast.parse(Path(__file__).parents[1].joinpath("__init__.py").read_text(encoding="utf-8"))
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_signal_handler")


def test_sigterm_previous_handler_is_only_invoked_when_callable():
    handler = _signal_handler()
    parents = {child: parent for parent in ast.walk(handler) for child in ast.iter_child_nodes(parent)}
    calls = [node for node in ast.walk(handler) if isinstance(node, ast.Call)]
    previous_calls = [node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "_prev_sigterm_handler"]
    assert len(previous_calls) == 1
    guard = parents[previous_calls[0]]
    while not isinstance(guard, ast.If):
        guard = parents[guard]
    assert isinstance(guard.test, ast.Call)
    assert isinstance(guard.test.func, ast.Name) and guard.test.func.id == "callable"
    assert isinstance(guard.test.args[0], ast.Name) and guard.test.args[0].id == "_prev_sigterm_handler"


def test_sigterm_explicit_ignore_remains_a_noop():
    handler = _signal_handler()
    comparisons = [node for node in ast.walk(handler) if isinstance(node, ast.Compare)]
    assert any(any(isinstance(part, ast.Attribute) and part.attr == "SIG_IGN" for part in node.comparators) for node in comparisons)
