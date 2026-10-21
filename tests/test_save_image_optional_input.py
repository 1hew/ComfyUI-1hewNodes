"""Regression tests for optional v3 input ports that are simply absent.

ComfyUI builds node arguments from the keys present in the prompt
(execution.get_input_data iterates prompt[node]["inputs"]). An unconnected
optional port (e.g. Save Image's image input) is never serialized, so the
keyword does not reach execute() at all. Without a Python default the call
raises "TypeError: ... missing 1 required positional argument" before the
node's own "if image is None" guard can run.
"""

import ast
import asyncio
import inspect
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch

from _node_loader import load_node

ROOT = Path(__file__).parents[1]


def _port(name):
    class _Port:
        def Input(self, *args, **kwargs):
            return ("input", name, args, kwargs)

        def Output(self, *args, **kwargs):
            return ("output", name, args, kwargs)

    return _Port()


class _NodeOutput:
    def __init__(self, *values, ui=None, **kwargs):
        self.values = values
        self.ui = ui


def _load_save_image(directory):
    folders = types.ModuleType("folder_paths")
    folders.get_output_directory = lambda: directory
    folders.get_temp_directory = lambda: directory

    saved_calls = []

    class _Helper:
        @staticmethod
        def save_images(image, *, filename_prefix, folder_type, cls, compress_level):
            saved_calls.append(
                {
                    "shape": tuple(image.shape),
                    "filename_prefix": filename_prefix,
                    "folder_type": folder_type,
                }
            )
            return [types.SimpleNamespace(subfolder="", filename="ComfyUI_00001_.png")]

    io = types.SimpleNamespace(
        ComfyNode=object,
        Schema=lambda **kwargs: kwargs,
        NodeOutput=_NodeOutput,
        FolderType=types.SimpleNamespace(output="output", temp="temp"),
        Image=_port("IMAGE"),
        String=_port("STRING"),
        Boolean=_port("BOOLEAN"),
    )
    ui = types.SimpleNamespace(
        ImageSaveHelper=_Helper,
        SavedImages=lambda results: ("saved", results),
    )
    latest = types.ModuleType("comfy_api.latest")
    latest.io = io
    latest.ui = ui

    previous = {
        name: sys.modules.get(name)
        for name in ("folder_paths", "comfy_api", "comfy_api.latest")
    }
    sys.modules["folder_paths"] = folders
    sys.modules.setdefault("comfy_api", types.ModuleType("comfy_api"))
    sys.modules["comfy_api.latest"] = latest
    try:
        module = load_node("save_image_optional_test", "nodes/io/save_image.py")
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
    return module, saved_calls


class SaveImageOptionalInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.module, self.saved_calls = _load_save_image(self.temp.name)

    def test_execute_reports_default_for_image(self):
        signature = inspect.signature(self.module.SaveImage.execute)
        self.assertIs(signature.parameters["image"].default, None)

    def test_unconnected_image_is_not_forwarded(self):
        # Mirrors the prompt ComfyUI executes when the port is empty: widget
        # values only, no "image" key at all.
        result = asyncio.run(
            self.module.SaveImage.execute(
                filename="image/ComfyUI",
                auto_increment=True,
                save_output=True,
                save_metadata=True,
            )
        )

        self.assertEqual(result.values, ())
        self.assertEqual(self.saved_calls, [])

    def test_bare_execute_call_does_not_raise(self):
        result = asyncio.run(self.module.SaveImage.execute())
        self.assertEqual(result.values, ())

    def test_connected_image_still_saves(self):
        image = torch.zeros((2, 8, 8, 3), dtype=torch.float32)
        result = asyncio.run(
            self.module.SaveImage.execute(
                image=image,
                filename="image/ComfyUI",
                auto_increment=True,
                save_output=False,
                save_metadata=False,
            )
        )

        self.assertEqual(result.values, ("",))
        self.assertEqual(
            result.ui,
            ("saved", [types.SimpleNamespace(subfolder="", filename="ComfyUI_00001_.png")]),
        )
        self.assertEqual(self.saved_calls[0]["shape"], (2, 8, 8, 3))


_FUNCTION_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef)


def _execute_signature(class_node):
    for statement in class_node.body:
        if isinstance(statement, _FUNCTION_DEFS) and statement.name == "execute":
            return statement
    return None


def _optional_ports_without_schema_default(class_node):
    """Port-only optional inputs: optional=True and no widget default."""
    ports = set()
    for statement in class_node.body:
        if not isinstance(statement, _FUNCTION_DEFS) or statement.name != "define_schema":
            continue
        for call in ast.walk(statement):
            if not isinstance(call, ast.Call):
                continue
            if not isinstance(call.func, ast.Attribute) or call.func.attr != "Input":
                continue
            keywords = {keyword.arg for keyword in call.keywords}
            if "optional" not in keywords or "default" in keywords:
                continue
            optional = next(
                keyword.value for keyword in call.keywords if keyword.arg == "optional"
            )
            if not (isinstance(optional, ast.Constant) and optional.value is True):
                continue
            if call.args and isinstance(call.args[0], ast.Constant):
                if isinstance(call.args[0].value, str):
                    ports.add(call.args[0].value)
    return ports


def _parameters_with_defaults(exec_node):
    arguments = exec_node.args
    positional = arguments.posonlyargs + arguments.args
    defaults = [None] * (len(positional) - len(arguments.defaults)) + list(arguments.defaults)
    values = {
        argument.arg: default is not None
        for argument, default in zip(positional, defaults)
        if argument.arg not in ("cls", "self")
    }
    for argument, default in zip(arguments.kwonlyargs, arguments.kw_defaults):
        values[argument.arg] = default is not None
    return values


class OptionalPortDefaultContractTests(unittest.TestCase):
    """Any port-only optional input must be defaulted in execute()."""

    def test_port_only_optional_inputs_have_python_defaults(self):
        offenders = []
        for path in sorted((ROOT / "nodes").rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for class_node in [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]:
                ports = _optional_ports_without_schema_default(class_node)
                if not ports:
                    continue
                exec_node = _execute_signature(class_node)
                if exec_node is None or exec_node.args.kwarg is not None:
                    continue
                defaults = _parameters_with_defaults(exec_node)
                for port in sorted(ports):
                    if port not in defaults:
                        continue
                    if not defaults[port]:
                        offenders.append(f"{path.relative_to(ROOT)}::{class_node.name}.{port}")

        self.assertEqual(offenders, [], "optional ports need a Python default")


if __name__ == "__main__":
    unittest.main()
