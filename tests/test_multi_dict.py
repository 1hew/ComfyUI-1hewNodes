"""Regression tests for the Multi Dict node."""

import asyncio
import json

import pytest

from _comfy_stub import install

install()

from _node_loader import load_node

Node = load_node("multi_dict_under_test", "nodes/data/multi_dict.py").MultiDict


def run(**kwargs):
    return asyncio.run(Node.execute(**kwargs))


def test_schema_declares_typed_text_inputs_and_one_output():
    schema = Node.define_schema()
    assert schema["node_id"] == "1hew_MultiDict"
    assert schema["display_name"] == "Multi Dict"
    assert schema["category"] == "1hewNodes/data"
    assert schema["accept_all_inputs"] is True
    # INPUT_IS_LIST would wrap every linked scalar into a single-item list.
    assert not schema.get("is_input_list")

    ports = {item["id"]: item for item in schema["inputs"]}
    assert set(ports) == {
        "dict_1", "key", "key_type", "value", "value_type", "output_type"
    }
    assert ports["dict_1"]["type"] == "*"
    assert ports["dict_1"]["optional"] is True
    # key / value are typed widgets, not sockets; value is single-line.
    assert ports["key"]["type"] == "STRING"
    assert ports["value"]["type"] == "STRING"
    assert not ports["value"].get("multiline")
    assert ports["key_type"]["options"] == ["string", "auto"]
    assert ports["value_type"]["options"] == [
        "auto", "string", "int", "float", "boolean", "dict", "list", "json", "null",
    ]
    assert ports["output_type"]["options"] == ["dict", "json"]
    assert ports["output_type"]["default"] == "dict"

    # Single data output, its value chosen by output_type.
    assert [item["display_name"] for item in schema["outputs"]] == ["data"]
    assert schema["outputs"][0]["type"] == "*"


def test_output_type_switches_between_dict_and_json():
    out = run(dict_1={"a": 1}, output_type="dict")
    assert out[0] == {"a": 1}
    out = run(dict_1={"a": 1}, output_type="json")
    assert out[0] == '{"a":1}'
    assert json.loads(out[0]) == {"a": 1}


def test_auto_keeps_text_as_text():
    assert run(key="a", value="Alice")[0] == {"a": "Alice"}
    # auto does not guess a number: that is what value_type is for.
    assert run(key="a", value="20")[0] == {"a": "20"}


def test_value_type_int_and_float():
    assert run(key="k", value="30", value_type="int")[0] == {"k": 30}
    assert run(key="k", value="3.7", value_type="int")[0] == {"k": 3}
    assert run(key="k", value="3.5", value_type="float")[0] == {"k": 3.5}
    with pytest.raises(ValueError):
        run(key="k", value="abc", value_type="int")


def test_value_type_boolean():
    assert run(key="k", value="true", value_type="boolean")[0] == {"k": True}
    assert run(key="k", value="0", value_type="boolean")[0] == {"k": False}
    with pytest.raises(ValueError):
        run(key="k", value="maybe", value_type="boolean")


def test_value_type_dict_and_list_parse_json_text():
    assert run(key="k", value='{"a": 1}', value_type="dict")[0] == {"k": {"a": 1}}
    assert run(key="k", value="[1, 2]", value_type="list")[0] == {"k": [1, 2]}
    with pytest.raises(ValueError):
        run(key="k", value="Alice", value_type="dict")


def test_value_type_json_parses_any_json():
    assert run(key="k", value="30", value_type="json")[0] == {"k": 30}
    assert run(key="k", value="null", value_type="json")[0] == {"k": None}
    assert run(key="k", value='"x"', value_type="json")[0] == {"k": "x"}


def test_empty_value_becomes_null():
    # key present, value empty: the key is added with a null value.
    assert run(key="k")[0] == {"k": None}


def test_key_type_string_keeps_the_text_as_written():
    assert run(key="01", value="x")[0] == {"01": "x"}
    assert run(key="1.0", value="x")[0] == {"1.0": "x"}


def test_key_type_auto_normalises_numbers_and_bools():
    assert run(key="1.0", value="x", key_type="auto")[0] == {"1": "x"}
    assert run(key="true", value="x", key_type="auto")[0] == {"true": "x"}
    assert run(key="abc", value="x", key_type="auto")[0] == {"abc": "x"}


def test_dict_inputs_merge_in_port_order_and_later_wins():
    out = run(dict_1={"a": 1, "b": 1}, dict_2={"a": 2}, dict_3={"b": 3, "c": 3})
    assert out[0] == {"a": 2, "b": 3, "c": 3}
    assert list(out[0]) == ["a", "b", "c"]


def test_key_and_value_are_applied_after_dicts():
    out = run(dict_1={"a": 1, "b": 2}, dict_2={"a": 3}, key="a", value="9", value_type="int")
    assert out[0] == {"a": 9, "b": 2}


def test_no_inputs_produces_empty_object():
    assert run(output_type="dict")[0] == {}
    assert run(output_type="json")[0] == "{}"


def test_empty_key_only_merges_dicts():
    out = run(dict_1={"a": 1}, key="   ", value="2")
    assert out[0] == {"a": 1}


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        run(dict_1=[{"a": 1}])
    with pytest.raises(ValueError):
        run(dict_1="not a dict")
    with pytest.raises(ValueError):
        run(dict_1={1: "non-string key"})
    with pytest.raises(ValueError):
        run(key="a", value="nan", value_type="float")


def test_json_output_round_trips():
    out = run(dict_1={"b": [1, None]}, key="a", value='{"c": true}', value_type="dict", output_type="json")
    assert json.loads(out[0]) == {"b": [1, None], "a": {"c": True}}
