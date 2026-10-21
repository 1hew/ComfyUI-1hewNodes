"""Save Text metadata, defaults, and Load Text stripping.

Save Text appends the UI workflow to text/Markdown files so they can be
dragged back into ComfyUI. These tests pin the current widget order, the
new text/ComfyUI default folder, the metadata round-trip, the csv/json
metadata skip, and the marker agreement with the front-end parser.
"""

import asyncio
import importlib
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]


class _NodeOutput:
    def __init__(self, *values, ui=None, **kwargs):
        self.values = values
        self.ui = ui


class _IOType:
    def __init__(self, type_name):
        self.type_name = type_name

    def Input(self, name, **kwargs):
        return ("input", self.type_name, name, kwargs)

    def Output(self, **kwargs):
        return ("output", self.type_name, kwargs)


class _SavedResult:
    def __init__(self, filename, subfolder, folder_type):
        self.filename = filename
        self.subfolder = subfolder
        self.folder_type = folder_type


def _load_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _build_stubs(directory):
    folders = types.ModuleType("folder_paths")
    folders.get_output_directory = lambda: directory
    folders.get_temp_directory = lambda: directory
    folders.get_input_directory = lambda: directory

    io = types.SimpleNamespace(
        ComfyNode=object,
        Schema=lambda **kwargs: kwargs,
        NodeOutput=_NodeOutput,
        FolderType=types.SimpleNamespace(output="output"),
        Hidden=types.SimpleNamespace(
            prompt="PROMPT", extra_pnginfo="EXTRA_PNGINFO"
        ),
        String=_IOType("STRING"),
        Boolean=_IOType("BOOLEAN"),
        Combo=_IOType("COMBO"),
        Int=_IOType("INT"),
    )
    ui = types.SimpleNamespace(SavedResult=_SavedResult)
    latest = types.ModuleType("comfy_api.latest")
    latest.io = io
    latest.ui = ui
    package = types.ModuleType("comfy_api")
    package.latest = latest
    return folders, package, latest


class SaveTextMetadataTests(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

        self.saved_modules = {
            name: sys.modules.get(name)
            for name in ("folder_paths", "comfy_api", "comfy_api.latest")
        }
        folders, package, latest = _build_stubs(self.temp.name)
        sys.modules["folder_paths"] = folders
        sys.modules["comfy_api"] = package
        sys.modules["comfy_api.latest"] = latest

        # Build a package hierarchy deep enough for the node's 3-dot relative
        # imports (...utils) without running utils/__init__ (workflow monitor).
        self.package_name = "save_text_metadata_test_pkg"
        package_module = types.ModuleType(self.package_name)
        package_module.__path__ = [str(ROOT)]
        sys.modules[self.package_name] = package_module

        nodes_pkg = types.ModuleType(self.package_name + ".nodes")
        nodes_pkg.__path__ = [str(ROOT / "nodes")]
        sys.modules[nodes_pkg.__name__] = nodes_pkg

        io_pkg = types.ModuleType(self.package_name + ".nodes.io")
        io_pkg.__path__ = [str(ROOT / "nodes" / "io")]
        sys.modules[io_pkg.__name__] = io_pkg

        utils_pkg = types.ModuleType(self.package_name + ".utils")
        utils_pkg.__path__ = [str(ROOT / "utils")]
        sys.modules[utils_pkg.__name__] = utils_pkg

        self.text_metadata = _load_module(
            self.package_name + ".utils._text_metadata",
            ROOT / "utils" / "_text_metadata.py",
        )
        _load_module(
            self.package_name + ".utils._file_ops",
            ROOT / "utils" / "_file_ops.py",
        )
        self.save_text_module = _load_module(
            self.package_name + ".nodes.io.save_text",
            ROOT / "nodes" / "io" / "save_text.py",
        )
        self.load_text_module = _load_module(
            self.package_name + ".nodes.io.load_text",
            ROOT / "nodes" / "io" / "load_text.py",
        )

        self.addCleanup(self._cleanup_modules)

    def _cleanup_modules(self):
        for name in list(sys.modules):
            if name == self.package_name or name.startswith(self.package_name + "."):
                sys.modules.pop(name, None)
        for name, value in self.saved_modules.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value

    def _attach_hidden(self, workflow, prompt):
        self.save_text_module.SaveText.hidden = types.SimpleNamespace(
            prompt=prompt,
            extra_pnginfo={"workflow": workflow},
        )

    def _execute(self, text, **kwargs):
        kwargs.setdefault("filename", "text/metadata")
        kwargs.setdefault("auto_increment", False)
        kwargs.setdefault("save_output", True)
        kwargs.setdefault("save_metadata", True)
        return asyncio.run(self.save_text_module.SaveText.execute(text=text, **kwargs))

    def test_schema_order_and_text_default_folder(self):
        schema = self.save_text_module.SaveText.define_schema()
        inputs = [(item[2], item[3]) for item in schema["inputs"]]
        self.assertEqual(
            [name for name, _ in inputs],
            [
                "text",
                "filename",
                "format",
                "auto_increment",
                "save_output",
                "save_metadata",
            ],
        )
        filename_kwargs = dict(inputs)["filename"]
        self.assertEqual(filename_kwargs["default"], "text/ComfyUI")

    def test_default_filename_writes_to_text_folder(self):
        self._attach_hidden({"nodes": [], "links": []}, {})
        result = asyncio.run(
            self.save_text_module.SaveText.execute(
                text="hello",
                auto_increment=False,
                save_output=True,
                save_metadata=False,
            )
        )
        path = Path(result.values[0])
        self.assertEqual(path.parent.name, "text")
        self.assertEqual(path.name, "ComfyUI.txt")
        self.assertEqual(path.read_text(encoding="utf-8"), "hello")

    def test_metadata_round_trip_and_load_text_strips(self):
        workflow = {
            "nodes": [{"id": 1, "type": "SaveText"}],
            "links": [],
        }
        prompt = {"1": {"class_type": "SaveText", "inputs": {}}}
        self._attach_hidden(workflow, prompt)

        text = "第一行\n第二行\n\n"
        result = self._execute(text)
        path = result.values[0]
        content = Path(path).read_text(encoding="utf-8")

        metadata = self.text_metadata.parse_metadata(content)
        self.assertEqual(metadata["workflow"], workflow)
        self.assertEqual(metadata["prompt"], prompt)
        self.assertEqual(self.text_metadata.strip_metadata(content), text)

        loaded = self.load_text_module.LoadText.execute(
            file=path,
            encode="utf-8",
            index=0,
            include_subdir=False,
        )
        self.assertEqual(loaded.values[0], text)

    def test_json_and_csv_skip_metadata(self):
        self._attach_hidden({"nodes": [], "links": []}, {})

        json_result = self._execute('{"a":1}', format="json")
        json_path = Path(json_result.values[0])
        json_text = json_path.read_text(encoding="utf-8")
        self.assertEqual(json_text, '{\n  "a": 1\n}')
        self.assertEqual(self.text_metadata.parse_metadata(json_text), {})

        csv_result = self._execute("a,b\n1,2", format="csv")
        csv_text = Path(csv_result.values[0]).read_text(encoding="utf-8")
        self.assertEqual(csv_text, "a,b\n1,2")
        self.assertEqual(self.text_metadata.parse_metadata(csv_text), {})

    def test_save_without_metadata_keeps_pure_text(self):
        self._attach_hidden({"nodes": [], "links": []}, {})
        result = self._execute("plain text", save_metadata=False)
        content = Path(result.values[0]).read_text(encoding="utf-8")
        self.assertEqual(content, "plain text")

    def test_missing_hidden_keeps_pure_text(self):
        self.save_text_module.SaveText.hidden = None
        result = self._execute("no metadata")
        content = Path(result.values[0]).read_text(encoding="utf-8")
        self.assertEqual(content, "no metadata")

    def test_payload_containing_marker_text_still_parses(self):
        workflow = {
            "nodes": [
                {
                    "id": 1,
                    "type": "SaveText",
                    "title": "1hew_nodes_txt_metadata_end --> and <!-- 1hew_nodes_txt_metadata_begin",
                }
            ],
            "links": [],
        }
        self._attach_hidden(workflow, {})
        result = self._execute("payload")
        content = Path(result.values[0]).read_text(encoding="utf-8")
        self.assertEqual(
            self.text_metadata.parse_metadata(content)["workflow"],
            workflow,
        )
        self.assertEqual(self.text_metadata.strip_metadata(content), "payload")

    def test_front_end_markers_match_python(self):
        js = (ROOT / "web" / "js" / "core" / "text_metadata.js").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'export const TEXT_METADATA_BEGIN = "' + self.text_metadata.METADATA_BEGIN + '";',
            js,
        )
        self.assertIn(
            'export const TEXT_METADATA_END = "' + self.text_metadata.METADATA_END + '";',
            js,
        )


if __name__ == "__main__":
    unittest.main()
