import ast
import os
from pathlib import Path
import signal
import subprocess
import sys
import textwrap


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


def _run_handler(previous):
    source = ast.get_source_segment(
        Path(__file__).parents[1].joinpath("__init__.py").read_text(encoding="utf-8"), _signal_handler())
    script = textwrap.dedent("""
        import logging, os, signal
        def stop():
            print("monitor-stopped", flush=True)
        namespace = {"logging": logging, "os": os, "signal": signal,
            "_stop_workflow_monitor": stop, "_prev_sigterm_handler": PREVIOUS,
            "_prev_sigint_handler": signal.default_int_handler}
        exec(SOURCE, namespace)
        namespace["_signal_handler"](signal.SIGTERM, None)
        print("handler-returned", flush=True)
    """).replace("PREVIOUS", previous).replace("SOURCE", repr(source))
    return subprocess.run([sys.executable, "-c", script], capture_output=True, timeout=5)


def test_default_sigterm_really_exits_after_monitor_cleanup():
    result = _run_handler("signal.SIG_DFL")
    assert result.returncode == (-signal.SIGTERM if os.name == "posix" else signal.SIGTERM)
    assert b"monitor-stopped" in result.stdout
    assert b"handler-returned" not in result.stdout


def test_missing_previous_handler_has_default_termination_semantics():
    result = _run_handler("None")
    assert result.returncode == (-signal.SIGTERM if os.name == "posix" else signal.SIGTERM)
    assert b"handler-returned" not in result.stdout


def test_explicit_ignore_really_returns_after_cleanup():
    result = _run_handler("signal.SIG_IGN")
    assert result.returncode == 0
    assert b"monitor-stopped" in result.stdout and b"handler-returned" in result.stdout


def test_callable_handler_remains_the_owner_of_shutdown_policy():
    result = _run_handler("lambda signum, frame: print('previous-handler', flush=True)")
    assert result.returncode == 0
    assert b"previous-handler" in result.stdout and b"handler-returned" in result.stdout
