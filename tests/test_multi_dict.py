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


def test_schema_declares_typed_inputs_and_two_outputs():
    schema = Node.define_schema()
    assert schema["node_id"] == "1hew_MultiDict"
    assert schema["display_name"] == "Multi Dict"
    assert schema["category"] == "1hewNodes/data"
    assert schema["accept_all_inputs"] is True
    # INPUT_IS_LIST would wrap every linked scalar into a single-item list.
    assert not schema.get("is_input_list")

    ports = {item["id"]: item for item in schema["inputs"]}
    assert set(ports) == {
        "dict_1", "key", "key_type", "value", "value_type"
    }
    # dict sockets are typed DICT, not the wildcard.
    assert ports["dict_1"]["type"] == "DICT"
    assert ports["dict_1"]["optional"] is True
    # key / value are typed widgets, not sockets; value is single-line.
    assert ports["key"]["type"] == "STRING"
    assert ports["value"]["type"] == "STRING"
    assert not ports["value"].get("multiline")
    assert ports["key_type"]["options"] == [
        "auto", "string", "int", "float", "boolean", "null",
    ]
    assert list(ports["value_type"]["options"]) == [
        "auto", "dict", "list", "string", "int", "float", "boolean", "null",
    ]
    # Two outputs, both always filled: native dict plus compact json.
    assert [item["display_name"] for item in schema["outputs"]] == ["dict", "json"]
    assert [item["type"] for item in schema["outputs"]] == ["DICT", "STRING"]


def test_both_outputs_are_filled():
    out = run(dict_1={"a": 1})
    assert out[0] == {"a": 1}
    assert out[1] == '{"a":1}'
    assert json.loads(out[1]) == out[0]


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


def test_value_type_round_trips_through_json_text():
    # The json output writes the same type the value_type produced, so a
    # downstream dict / list / number / bool / null reads back unchanged.
    cases = [
        ('{"a": 1}', "dict", {"a": 1}),
        ("[1, 2]", "list", [1, 2]),
        ("30", "int", 30),
        ("3.5", "float", 3.5),
        ("true", "boolean", True),
        ("anything", "string", "anything"),
        ("ignored", "null", None),
    ]
    for text, mode, expected in cases:
        out = run(key="k", value=text, value_type=mode)
        assert out[0] == {"k": expected}, mode
        assert json.loads(out[1]) == {"k": expected}, mode


def test_empty_value_becomes_null():
    # key present, value empty: the key is added with a null value.
    assert run(key="k")[0] == {"k": None}


def test_key_type_string_keeps_the_text_as_written():
    assert run(key="01", value="x", key_type="string")[0] == {"01": "x"}
    assert run(key="1.0", value="x", key_type="string")[0] == {"1.0": "x"}
    assert run(key="true", value="x", key_type="string")[0] == {"true": "x"}


def test_key_type_auto_infers_python_literals():
    # auto keeps text as text unless it reads as a literal.
    assert run(key="abc", value="x")[0] == {"abc": "x"}
    assert run(key="1", value="x")[0] == {1: "x"}
    assert run(key="1.5", value="x")[0] == {1.5: "x"}
    assert run(key="true", value="x")[0] == {True: "x"}
    assert run(key="null", value="x")[0] == {None: "x"}


def test_key_type_forces_the_named_type():
    assert run(key="1", value="x", key_type="int")[0] == {1: "x"}
    assert run(key="1.0", value="x", key_type="float")[0] == {1.0: "x"}
    assert run(key="true", value="x", key_type="boolean")[0] == {True: "x"}
    assert run(key="null", value="x", key_type="null")[0] == {None: "x"}


def test_key_type_rejects_text_that_does_not_fit():
    with pytest.raises(ValueError):
        run(key="abc", value="x", key_type="int")
    with pytest.raises(ValueError):
        run(key="abc", value="x", key_type="float")
    with pytest.raises(ValueError):
        run(key="abc", value="x", key_type="bogus")


def test_json_output_stringifies_scalar_keys():
    # The dict port keeps the real key type; the json port must be valid JSON.
    out = run(key="1", value="x", key_type="int")
    assert out[0] == {1: "x"}
    assert json.loads(out[1]) == {"1": "x"}

    out = run(key="true", value="x", key_type="boolean")
    assert out[0] == {True: "x"}
    assert json.loads(out[1]) == {"true": "x"}

    out = run(key="null", value="x", key_type="null")
    assert out[0] == {None: "x"}
    assert json.loads(out[1]) == {"null": "x"}
