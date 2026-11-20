"""Text Path Replace: write values back at the locations a path matches.

These tests pin the write-back contract: the path that Text Path Extract
reads is the same path this node writes; one value per match or a single
value broadcast to all; mismatches and read-only paths raise instead of
silently changing the wrong place; the input content is never mutated; and
the node does not render, it hands the edited document back as a value.
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
    "data_path_replace_under_test", "nodes/data/data_path_replace.py"
).DataPathReplace
JSON_PATH = importlib.import_module("data_path_replace_under_test.utils._json_path")
replace_all = JSON_PATH.replace_all


def _run(**kwargs):
    kwargs.setdefault("value_type", "auto")
    return asyncio.run(NODE.execute(**kwargs))


# --------------------------------------------------------------------- schema


def test_schema_collects_a_value_list_and_reports_a_count():
    schema = NODE.define_schema()
    assert schema["node_id"] == "1hew_DataPathReplace"
    assert schema["display_name"] == "Data Path Replace"
    assert schema["category"] == "1hewNodes/data"
    # The value is typed in, so the node no longer takes lists.
    assert not schema.get("is_input_list")
    assert [port["id"] for port in schema["inputs"]] == [
        "data",
        "data_path",
        "value",
        "value_type",
    ]
    ports = {p["id"]: p for p in schema["inputs"]}
    assert ports["value"]["type"] == "STRING"
    assert not ports["value"].get("multiline")
    # No rendering ports any more: value and count only.
    # data, matching the input port it goes back into.
    assert [o["display_name"] for o in schema["outputs"]] == ["data", "count"]
    # value is wrapped in a list so a root-level JSON array is not split by
    # ComfyUI downstream; the port renders as a list endpoint.
    # No is_output_list: the shape follows the data, so the port cannot claim
    # to always be a list.
    assert not schema["outputs"][0].get("is_output_list")
    assert schema["outputs"][1]["type"] == "INT"


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


def test_value_can_be_a_container():
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


def test_mismatched_value_count_raises():
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


def test_node_writes_one_value_broadcast_to_every_match():
    text = '{"layers":[{"index":1},{"index":2}]}'
    value, count = _run(
        data=text,
        data_path="layers[*].index",
        value="3",
        value_type="int",
    )
    assert count == 2
    # A single typed value broadcasts to every match; object in, object out.
    assert value == {"layers": [{"index": 3}, {"index": 3}]}


def test_output_keeps_the_input_shape():
    # A dict stays a dict; a list stays a list. No wrapper either way.
    value, _ = _run(data='{"n":1}', data_path="n", value="2", value_type="int")
    assert value == {"n": 2}

    value, _ = _run(data='[{"n":1}]', data_path="[0].n", value="2", value_type="int")
    assert value == [{"n": 2}]


def test_node_value_type_converts_the_written_value():
    value, count = _run(
        data='[{"n":1}]',
        data_path="[0].n",
        value="5",
        value_type="int",
    )
    assert count == 1
    # Array in, array out.
    assert value == [{"n": 5}]


def test_node_value_type_dict_parses_json_text():
    value, _ = _run(
        data='[{"n":1}]',
        data_path="[0].n",
        value='{"k": 1}',
        value_type="dict",
    )
    assert value == [{"n": {"k": 1}}]


def test_node_keeps_integers_without_float_suffix():
    value, _ = _run(
        data='[{"n":1}]',
        data_path="[0].n",
        value="3",
        value_type="int",
    )
    assert value == [{"n": 3}]


def test_node_no_match_returns_the_original_document_and_zero():
    original = '{"layers":[{"index":1}]}'
    value, count = _run(
        data=original,
        data_path="nothing[*]",
        value="9",
        value_type="int",
    )
    assert value == {"layers": [{"index": 1}]}
    assert count == 0


def test_node_broadcasts_one_value_to_every_match():
    value, count = _run(
        data='[{"n":1},{"n":2},{"n":3}]',
        data_path="[*].n",
        value="7",
        value_type="int",
    )
    assert count == 3
    assert value == [{"n": 7}, {"n": 7}, {"n": 7}]


def test_node_rejects_a_value_that_cannot_be_converted():
    with pytest.raises(ValueError):
        _run(
            data='[{"n":1}]',
            data_path="[0].n",
            value="abc",
            value_type="int",
        )


# ------------------------------------------------------------- typed text box


def test_value_auto_keeps_text_as_text():
    value, _ = _run(
        data='{"n":1}',
        data_path="n",
        value="hello",
        value_type="auto",
    )
    assert value == {"n": "hello"}


def test_value_can_be_a_json_container():
    value, _ = _run(
        data='{"n":1}',
        data_path="n",
        value='{"k": 1}',
        value_type="dict",
    )
    assert value == {"n": {"k": 1}}


def test_empty_value_writes_nothing():
    original = '{"n":1}'
    value, count = _run(
        data=original,
        data_path="n",
        value="",
    )
    assert count == 0
    assert value == {"n": 1}


# ------------------------------------------------------- structured data input


def test_writes_into_an_already_parsed_dict():
    value, count = _run(
        data={"n": 1},
        data_path="n",
        value="7",
        value_type="int",
    )
    assert count == 1
    assert value == {"n": 7}


def test_writes_into_an_already_parsed_list():
    value, count = _run(
        data=[{"n": 1}, {"n": 2}],
        data_path="[*].n",
        value="9",
        value_type="int",
    )
    assert count == 2
    assert value == [{"n": 9}, {"n": 9}]


def test_does_not_mutate_the_linked_object():
    original = {"n": 1}
    _run(data=original, data_path="n", value="7", value_type="int")
    assert original == {"n": 1}


# ------------------------------------------------------- multiple values


def test_a_json_array_writes_one_value_per_match():
    value, count = _run(
        data='[{"n":1},{"n":2},{"n":3}]',
        data_path="[*].n",
        value="[10, 20, 30]",
        value_type="int",
    )
    assert count == 3
    assert value == [{"n": 10}, {"n": 20}, {"n": 30}]


def test_a_array_of_mixed_types_follows_value_type():
    value, _ = _run(
        data='{"a":1,"b":"x"}',
        data_path="[*]",
        value='[true, "hello"]',
        value_type="auto",
    )
    # Object in, object out; the two values land in insertion order.
    assert value == {"a": True, "b": "hello"}


def test_a_single_value_still_broadcasts():
    value, count = _run(
        data='[{"n":1},{"n":2}]',
        data_path="[*].n",
        value="7",
        value_type="int",
    )
    assert count == 2
    assert value == [{"n": 7}, {"n": 7}]


def test_value_type_list_writes_the_whole_array_as_one_value():
    # value_type=list means the text IS one array to write into one slot.
    value, count = _run(
        data='{"x":1}',
        data_path="x",
        value="[1, 2]",
        value_type="list",
    )
    assert count == 1
    assert value == {"x": [1, 2]}


def test_array_value_count_mismatch_raises():
    with pytest.raises(ValueError, match="Text Path Replace|Data Path Replace"):
        _run(
            data='[{"n":1},{"n":2},{"n":3}]',
            data_path="[*].n",
            value="[10, 20]",
            value_type="int",
        )
