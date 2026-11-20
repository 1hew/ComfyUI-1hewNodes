"""Text Path Replace: write values back at the locations a path matches.

These tests pin the write-back contract: the path that Text Path Extract
reads is the same path this node writes; one replacement per match or a
single value broadcast to all; mismatches and read-only paths raise instead
of silently changing the wrong place; the input content is never mutated.
"""

import asyncio
import importlib
import sys
import types

import pytest

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
    "text_path_replace_under_test", "nodes/text/text_path_replace.py"
).TextPathReplace
JSON_PATH = importlib.import_module("text_path_replace_under_test.utils._json_path")
replace_all = JSON_PATH.replace_all


def _run(**kwargs):
    kwargs.setdefault("value_type", "auto")
    kwargs.setdefault("output_format", "compact")
    return asyncio.run(NODE.execute(**kwargs))


# --------------------------------------------------------------------- schema


def test_schema_declares_list_input_and_list_value_output():
    schema = NODE.define_schema()
    assert schema["node_id"] == "1hew_TextPathReplace"
    assert schema["display_name"] == "Text Path Replace"
    assert schema["category"] == "1hewNodes/text"
    assert schema["is_input_list"] is True
    assert [port["id"] for port in schema["inputs"]] == [
        "text",
        "path",
        "replace",
        "value_type",
        "output_format",
    ]
    # value is wrapped in a list so a root-level JSON array is not split
    # by ComfyUI downstream; the port renders as a list endpoint.
    value_output = schema["outputs"][1]
    assert value_output["display_name"] == "value"
    assert value_output["is_output_list"] is True


# ------------------------------------------------------------- core behaviour


def test_one_to_one_replacement_for_each_match():
    data = '{"layers":[{"index":1},{"index":2},{"index":3}]}'
    new, count = replace_all(data, "layers[*].index", [2, 3, 4])
    assert count == 3
    assert [layer["index"] for layer in new["layers"]] == [2, 3, 4]


def test_single_value_is_broadcast_to_every_match():
    data = '[{"content":"a"},{"content":"b"}]'
    new, count = replace_all(data, "[*].content", ["X"])
    assert count == 2
    assert new == [{"content": "X"}, {"content": "X"}]


def test_index_path_writes_only_the_selected_item():
    data = '[{"bbox":[1,2,3,4]},{"bbox":[5,6,7,8]}]'
    new, count = replace_all(data, "[1].bbox[0]", [50])
    assert count == 1
    assert new == [{"bbox": [1, 2, 3, 4]}, {"bbox": [50, 6, 7, 8]}]


def test_negative_index_on_list_is_supported():
    data = "[1, 2, 3]"
    new, count = replace_all(data, "[-1]", [9])
    assert count == 1
    assert new == [1, 2, 9]


def test_values_selector_on_object_writes_each_value():
    data = '{"a":{"x":1},"b":{"x":2}}'
    new, count = replace_all(data, "values[*].x", [10, 20])
    assert count == 2
    assert new == {"a": {"x": 10}, "b": {"x": 20}}


def test_values_index_on_object_writes_by_position():
    data = '{"reference_image_1":{"ids":[1]},"image_1":{"ids":[2]}}'
    new, count = replace_all(data, "values[1].ids", [[9]])
    assert count == 1
    assert new["image_1"]["ids"] == [9]
    assert new["reference_image_1"]["ids"] == [1]


def test_replacement_can_be_a_container():
    data = '[{"bbox":[1,2,3,4]},{"bbox":[5,6,7,8]}]'
    new, count = replace_all(data, "[*].bbox", [{"k": 1}])
    assert count == 2
    assert new == [{"bbox": {"k": 1}}, {"bbox": {"k": 1}}]


def test_no_match_returns_unchanged_data_and_zero():
    data = '{"layers":[{"index":1}]}'
    new, count = replace_all(data, "missing[*].index", [2])
    assert count == 0
    assert new == {"layers": [{"index": 1}]}


def test_input_object_is_not_mutated():
    original = {"layers": [{"index": 1}]}
    new, count = replace_all(original, "layers[*].index", [2])
    assert count == 1
    assert new["layers"][0]["index"] == 2
    assert original["layers"][0]["index"] == 1


def test_fenced_json_input_is_parsed_then_rewritten():
    fence = chr(96) * 3
    data = f'{fence}json\n[{{"n": 1}}]\n{fence}'
    new, count = replace_all(data, "[0].n", [7])
    assert count == 1
    assert new == [{"n": 7}]


# ------------------------------------------------------------ error handling


def test_mismatched_replacement_count_raises():
    data = '[{"n":1},{"n":2},{"n":3}]'
    with pytest.raises(ValueError, match="匹配到 3 处"):
        replace_all(data, "[*].n", [1, 2])


def test_keys_selector_cannot_be_replaced():
    with pytest.raises(ValueError, match="keys"):
        replace_all('{"a":1}', "keys[*]", ["b"])


def test_whole_values_group_needs_a_position():
    with pytest.raises(ValueError, match="整组"):
        replace_all('{"a":1}', "values", [5])


def test_count_path_is_read_only():
    with pytest.raises(ValueError, match="只能读取"):
        replace_all('[1,2]', "[*].count", [5])


def test_empty_path_raises():
    with pytest.raises(ValueError, match="path"):
        replace_all('[1]', "", [5])


# ------------------------------------------------------------------ node level


def test_node_writes_index_plus_upstream_values():
    text = '{"layers":[{"index":1},{"index":2}]}'
    rendered, value, count = _run(
        text=[text],
        path=["layers[*].index"],
        replace=[2, 3],
    )
    assert rendered == '{"layers":[{"index":2},{"index":3}]}'
    assert count == 2
    # value is a one-item list holding the edited document.
    assert isinstance(value, list) and len(value) == 1
    assert value[0]["layers"][1]["index"] == 3


def test_node_value_is_list_even_when_root_is_array():
    _, value, _ = _run(
        text=['[{"n":1}]'],
        path=["[0].n"],
        replace=[2],
    )
    assert value == [[{"n": 2}]]


def test_node_value_type_converts_replacement():
    rendered, _, count = _run(
        text=['[{"n":1}]'],
        path=["[0].n"],
        replace=["5"],
        value_type="int",
    )
    assert count == 1
    assert rendered == '[{"n":5}]'


def test_node_value_type_dict_parses_json_text():
    rendered, _, _ = _run(
        text=['[{"n":1}]'],
        path=["[0].n"],
        replace=['{"k": 1}'],
        value_type="dict",
    )
    assert rendered == '[{"n":{"k":1}}]'


def test_node_keeps_integers_without_float_suffix():
    rendered, _, _ = _run(
        text=['[{"n":1}]'],
        path=["[0].n"],
        replace=[3],
    )
    assert rendered == '[{"n":3}]'


def test_node_pretty_output_is_indented():
    rendered, _, _ = _run(
        text=['[{"n":1}]'],
        path=["[0].n"],
        replace=[2],
        output_format="pretty",
    )
    assert rendered == '[\n  {\n    "n": 2\n  }\n]'


def test_node_no_match_returns_original_text_and_zero():
    original = '{"layers":[{"index":1}]}'
    rendered, _, count = _run(
        text=[original],
        path=["nothing[*]"],
        replace=[9],
    )
    assert rendered == original
    assert count == 0


def test_node_raises_on_count_mismatch():
    with pytest.raises(ValueError, match="Text Path Replace"):
        _run(
            text=['[{"n":1},{"n":2}]'],
            path=["[*].n"],
            replace=[1, 2, 3],
        )


def test_node_rejects_python_only_replacement():
    with pytest.raises(ValueError):
        _run(
            text=['[{"n":1}]'],
            path=["[0].n"],
            replace=[{1, 2}],
        )
