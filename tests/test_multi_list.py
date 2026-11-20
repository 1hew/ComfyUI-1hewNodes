"""Regression tests for the Multi List node."""

import asyncio
import json

import pytest

from _comfy_stub import install

install()

from _node_loader import load_node

Node = load_node("multi_list_under_test", "nodes/data/multi_list.py").MultiList


def run(**kwargs):
    return asyncio.run(Node.execute(**kwargs))


def test_schema_declares_mode_and_dynamic_items():
    schema = Node.define_schema()
    assert schema["node_id"] == "1hew_MultiList"
    assert schema["display_name"] == "Multi List"
    assert schema["category"] == "1hewNodes/data"
    assert schema["accept_all_inputs"] is True
    assert not schema.get("is_input_list")

    ports = {item["id"]: item for item in schema["inputs"]}
    assert ports["mode"]["type"] == "COMBO"
    assert ports["mode"]["default"] == "extend"
    assert ports["mode"]["options"] == ["extend", "append"]
    assert ports["any_1"]["type"] == "*"
    assert ports["any_1"]["optional"] is True

    # Two outputs: the whole array plus its compact JSON text. The list port
    # is a declared list output, so it renders with the list-styled socket.
    assert [item["display_name"] for item in schema["outputs"]] == ["list", "json"]
    assert schema["outputs"][0]["type"] == "*"
    assert schema["outputs"][0]["is_output_list"] is True
    assert schema["outputs"][1]["type"] == "STRING"
    ports = {item["id"]: item for item in schema["inputs"]}
    assert "output_type" not in ports


def test_extend_splices_list_inputs_one_level():
    output = run(any_1="a", any_2=[1, 2], any_3={"b": 3})
    assert output[0] == ["a", 1, 2, {"b": 3}]


def test_append_keeps_list_inputs_nested():
    output = run(mode="append", any_1="a", any_2=[1, 2], any_3={"b": 3})
    assert output[0] == ["a", [1, 2], {"b": 3}]


def test_append_keeps_scalar_items_unwrapped():
    assert run(mode="append", any_1="a")[0] == ["a"]


def test_strings_and_dicts_are_never_spliced():
    assert run(any_1="abc")[0] == ["abc"]
    assert run(any_1={"a": 1})[0] == [{"a": 1}]


def test_extend_is_one_level_only():
    assert run(any_1=[[1, 2], [3]])[0] == [[1, 2], [3]]


def test_none_inputs_are_skipped():
    assert run(any_1=None, any_2=1)[0] == [1]
    assert run()[0] == []
    assert run()[1] == "[]"


def test_invalid_mode_is_rejected():
    with pytest.raises(ValueError):
        run(mode="add", any_1=1)


def test_both_outputs_are_filled():
    out = run(any_1=[1, None], any_2={"a": [2]})
    assert out[0] == [1, None, {"a": [2]}]
    assert isinstance(out[1], str)
    assert json.loads(out[1]) == out[0]
