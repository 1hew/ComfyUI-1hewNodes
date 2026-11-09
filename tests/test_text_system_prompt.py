"""Registry rules for the Text System Prompt node.

The node is a thin wrapper around utils/_system_prompt.py, so these tests pin
the contract that matters: a flat dropdown built from the folder tree, keys
without the .txt suffix, verbatim bodies, and a silent empty string instead of
an exception when a key disappears.
"""

import asyncio
import sys
import types

from _node_loader import load_node


def _install_comfy_stub():
    class _PortType:
        def __init__(self, name):
            self.name = name

        def Input(self, id, *args, **kwargs):
            return {"kind": "input", "type": self.name, "id": id, **kwargs}

        def Output(self, *args, **kwargs):
            return {"kind": "output", "type": self.name, **kwargs}

    io = types.SimpleNamespace(
        ComfyNode=object,
        Schema=lambda **kwargs: kwargs,
        NodeOutput=lambda *values, **kwargs: values,
    )
    for name, io_type in {
        "String": "STRING",
        "Combo": "COMBO",
        "Int": "INT",
        "Float": "FLOAT",
        "Image": "IMAGE",
        "Boolean": "BOOLEAN",
    }.items():
        setattr(io, name, _PortType(io_type))
    io.Custom = lambda name: _PortType(name)

    latest = types.ModuleType("comfy_api.latest")
    latest.io = io
    latest.ui = types.SimpleNamespace()
    latest.ComfyExtension = object
    package = types.ModuleType("comfy_api")
    package.latest = latest
    sys.modules["comfy_api"] = package
    sys.modules["comfy_api.latest"] = latest


_install_comfy_stub()

NODE = load_node(
    "text_system_prompt_under_test", "nodes/text/text_system_prompt.py"
).TextSystemPrompt
REGISTRY = sys.modules["text_system_prompt_under_test.utils._system_prompt"]


def _run(file):
    return asyncio.run(NODE.execute(file=file))


# --------------------------------------------------------------------- schema


def test_schema_exposes_one_flat_dropdown_and_one_string_output():
    schema = NODE.define_schema()
    assert schema["node_id"] == "1hew_TextSystemPrompt"
    assert schema["display_name"] == "Text System Prompt"
    assert schema["category"] == "1hewNodes/text"

    assert len(schema["inputs"]) == 1
    file_input = schema["inputs"][0]
    assert file_input["id"] == "file"
    assert file_input["type"] == "COMBO"

    assert len(schema["outputs"]) == 1
    assert schema["outputs"][0]["type"] == "STRING"
    assert schema["outputs"][0]["display_name"] == "string"


def test_dropdown_entries_are_paths_without_the_txt_suffix():
    options = NODE.define_schema()["inputs"][0]["options"]
    assert options
    assert all(not option.endswith(".txt") for option in options)
    assert all("/" in option for option in options)
    assert options == sorted(options)


def test_default_is_the_first_entry():
    schema = NODE.define_schema()
    options = schema["inputs"][0]["options"]
    assert schema["inputs"][0]["default"] == options[0]


# ------------------------------------------------------------------- loading


def test_registry_key_matches_the_file_path_without_suffix():
    for key, file_path in REGISTRY.system_prompt_registry().items():
        assert key == file_path.relative_to(REGISTRY.PROMPT_DIR).with_suffix("").as_posix()
        assert file_path.suffix == ".txt"


def test_body_is_returned_verbatim():
    key = REGISTRY.list_system_prompts()[0]
    file_path = REGISTRY.system_prompt_registry()[key]
    expected = file_path.read_text(encoding="utf-8")
    assert _run(key) == (expected,)


def test_language_tag_is_part_of_the_key():
    keys = REGISTRY.list_system_prompts()
    assert any("[en]" in key for key in keys)
    tagged = [key for key in keys if key.startswith("qwen_image_2.1/[")]
    assert tagged, "sample prompts should demonstrate the [lang] prefix"


def test_unknown_key_returns_an_empty_string_without_raising():
    assert _run("does/not/exist") == ("",)


def test_underscore_or_dot_prefixed_files_are_skipped(tmp_path, monkeypatch):
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "[en] keep.txt").write_text("prompt", encoding="utf-8")
    (tmp_path / "model" / "_ignore.txt").write_text("note", encoding="utf-8")
    (tmp_path / "model" / ".ignore.txt").write_text("note", encoding="utf-8")
    monkeypatch.setattr(REGISTRY, "PROMPT_DIR", tmp_path)
    try:
        assert REGISTRY.reload_system_prompts() == ["model/[en] keep"]
    finally:
        monkeypatch.undo()
        REGISTRY.reload_system_prompts()
