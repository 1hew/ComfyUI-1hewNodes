"""Save Video by Image must accept an empty image input.

The image port is optional, so ComfyUI never forwards the keyword when it is
unconnected (see execution.get_input_data). The node must therefore default the
argument and return an empty output instead of raising.
"""

import asyncio
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch

from _node_loader import load_node

ROOT = Path(__file__).parents[1]
_MODULES = (
    "folder_paths",
    "comfy",
    "comfy.cli_args",
    "comfy_api",
    "comfy_api.latest",
    "torchaudio",
)


class _NodeOutput:
    def __init__(self, *values, ui=None, **kwargs):
        self.values = values
        self.ui = ui


class _Port:
    def __init__(self, name):
        self.name = name

    def Input(self, *args, **kwargs):
        return ("input", self.name, args, kwargs)

    def Output(self, *args, **kwargs):
        return ("output", self.name, args, kwargs)


def _load_node(directory):
    folders = types.ModuleType("folder_paths")
    folders.get_output_directory = lambda: directory
    folders.get_temp_directory = lambda: directory
    folders.get_save_image_path = lambda prefix, *args, **kwargs: (
        directory,
        os.path.basename(prefix) or "ComfyUI",
        1,
        "",
        prefix,
    )

    cli = types.ModuleType("comfy.cli_args")
    # Skips the hidden prompt/extra_pnginfo access, which the stub class lacks.
    cli.args = types.SimpleNamespace(disable_metadata=True)

    io = types.SimpleNamespace(
        ComfyNode=object,
        Schema=lambda **kwargs: kwargs,
        NodeOutput=_NodeOutput,
        FolderType=types.SimpleNamespace(output="output", temp="temp"),
        Hidden=types.SimpleNamespace(prompt="prompt", extra_pnginfo="extra_pnginfo"),
        Image=_Port("IMAGE"),
        Audio=_Port("AUDIO"),
        Float=_Port("FLOAT"),
        String=_Port("STRING"),
        Boolean=_Port("BOOLEAN"),
    )
    ui = types.SimpleNamespace(
        PreviewVideo=lambda values: values,
        SavedResult=lambda *values: values,
    )
    latest = types.ModuleType("comfy_api.latest")
    latest.io = io
    latest.ui = ui

    # The node imports torchaudio at module level; audio encoding never runs here.
    torchaudio = types.ModuleType("torchaudio")
    torchaudio.save = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("torchaudio must not be used without an audio input")
    )

    previous = {name: sys.modules.get(name) for name in _MODULES}
    sys.modules["folder_paths"] = folders
    sys.modules.setdefault("comfy", types.ModuleType("comfy"))
    sys.modules["comfy.cli_args"] = cli
    sys.modules.setdefault("comfy_api", types.ModuleType("comfy_api"))
    sys.modules["comfy_api.latest"] = latest
    sys.modules["torchaudio"] = torchaudio
    try:
        module = load_node(
            "save_video_by_image_optional_test",
            "nodes/io/save_video_by_image.py",
        )
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
    return module


class SaveVideoByImageOptionalImageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.module = _load_node(self.temp.name)

    def _execute(self, **kwargs):
        arguments = {
            "fps": 8.0,
            "filename": "video/ComfyUI",
            "auto_increment": True,
            "save_output": True,
            "save_metadata": True,
        }
        arguments.update(kwargs)
        return asyncio.run(self.module.SaveVideoByImage.execute(**arguments))

    def test_schema_marks_image_optional(self):
        schema = self.module.SaveVideoByImage.define_schema()
        image = next(
            item
            for item in schema["inputs"]
            if item[0] == "input" and item[1] == "IMAGE"
        )
        self.assertIs(image[3]["optional"], True)

    def test_unconnected_image_returns_empty_output(self):
        result = self._execute()
        self.assertEqual(result.values, ())
        self.assertEqual(os.listdir(self.temp.name), [])

    def test_zero_frame_batch_returns_empty_output(self):
        result = self._execute(image=torch.zeros((0, 8, 8, 3), dtype=torch.float32))
        self.assertEqual(result.values, ())
        self.assertEqual(os.listdir(self.temp.name), [])

    def test_bare_execute_call_does_not_raise(self):
        result = asyncio.run(self.module.SaveVideoByImage.execute())
        self.assertEqual(result.values, ())


if __name__ == "__main__":
    unittest.main()
