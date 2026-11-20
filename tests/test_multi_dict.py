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


def test_schema_declares_value_type_and_sockets():
    schema = Node.define_schema()
    assert schema["node_id"] == "1hew_MultiDict"
    assert schema["display_name"] == "Multi Dict"
    assert schema["category"] == "1hewNodes/data"
    assert schema["accept_all_inputs"] is True
    # INPUT_IS_LIST would wrap every linked scalar into a single-item list,
    # turning a string value into ["string"].
    assert not schema.get("is_input_list")

    ports = {item["id"]: item for item in schema["inputs"]}
    assert set(ports) == {"value_type", "dict_1", "key", "value"}
    assert ports["value_type"]["type"] == "COMBO"
    assert ports["value_type"]["default"] == "auto"
    assert ports["value_type"]["options"] == [
        "auto",
        "string",
        "int",
        "float",
        "boolean",
        "dict",
        "list",
        "json",
        "null",
    ]
    for name in ("dict_1", "key", "value"):
        assert ports[name]["type"] == "*"
        assert ports[name]["optional"] is True

    outputs = schema["outputs"]
    assert [item["type"] for item in outputs] == ["DICT", "STRING"]
    assert [item["display_name"] for item in outputs] == ["dict", "json"]


def test_auto_keeps_native_types():
    assert run(key="a", value="Alice")[0] == {"a": "Alice"}
    assert run(key="a", value="20")[0] == {"a": "20"}
    assert run(key="a", value=1)[0] == {"a": 1}
    assert run(key="a", value=3.5)[0] == {"a": 3.5}
    assert run(key="a", value=False)[0] == {"a": False}
    assert run(key="a", value={"b": 1})[0] == {"a": {"b": 1}}
    assert run(key="a", value=["x"])[0] == {"a": ["x"]}


def test_string_value_type_forces_text():
    assert run(key="k", value=30, value_type="string")[0] == {"k": "30"}
    assert run(key="k", value=3.5, value_type="string")[0] == {"k": "3.5"}
    assert run(key="k", value=True, value_type="string")[0] == {"k": "True"}
    assert run(key="k", value={"a": 1}, value_type="string")[0] == {
        "k": '{"a":1}'
    }
    assert run(key="k", value=[1, 2], value_type="string")[0] == {
        "k": "[1,2]"
    }


def test_dict_value_type_requires_an_object():
    assert run(key="k", value='{"a": 1}', value_type="dict")[0] == {
        "k": {"a": 1}
    }
    assert run(key="k", value={"a": 1}, value_type="dict")[0] == {
        "k": {"a": 1}
    }
    with pytest.raises(ValueError):
        run(key="k", value="[1, 2]", value_type="dict")
    with pytest.raises(ValueError):
        run(key="k", value="Alice", value_type="dict")


def test_list_value_type_requires_an_array():
    assert run(key="k", value="[1, 2]", value_type="list")[0] == {
        "k": [1, 2]
    }
    assert run(key="k", value=[1, 2], value_type="list")[0] == {
        "k": [1, 2]
    }
    assert run(key="k", value=(1, 2), value_type="list")[0] == {
        "k": [1, 2]
    }
    with pytest.raises(ValueError):
        run(key="k", value='{"a": 1}', value_type="list")


def test_json_value_type_parses_json_text():
    assert run(key="k", value='{"a": 1}', value_type="json")[0] == {
        "k": {"a": 1}
    }
    assert run(key="k", value="[1, 2]", value_type="json")[0] == {
        "k": [1, 2]
    }
    assert run(key="k", value="30", value_type="json")[0] == {"k": 30}
    assert run(key="k", value="true", value_type="json")[0] == {"k": True}
    assert run(key="k", value="null", value_type="json")[0] == {"k": None}
    # Native objects pass through in json mode.
    assert run(key="k", value={"a": 1}, value_type="json")[0] == {
        "k": {"a": 1}
    }
    with pytest.raises(ValueError):
        run(key="k", value="Alice", value_type="json")


def test_int_value_type_converts_numbers_and_numeric_text():
    assert run(key="k", value="30", value_type="int")[0] == {"k": 30}
    assert run(key="k", value=3.7, value_type="int")[0] == {"k": 3}
    assert run(key="k", value="3.7", value_type="int")[0] == {"k": 3}
    assert run(key="k", value=True, value_type="int")[0] == {"k": 1}
    with pytest.raises(ValueError):
        run(key="k", value="abc", value_type="int")


def test_float_value_type_converts_numbers_and_numeric_text():
    output = run(key="k", value="3.5", value_type="float")
    assert output[0] == {"k": 3.5}
    output = run(key="k", value=2, value_type="float")
    assert output[0] == {"k": 2.0}
    assert isinstance(output[0]["k"], float)
    with pytest.raises(ValueError):
        run(key="k", value="abc", value_type="float")


def test_boolean_value_type_converts_common_text():
    assert run(key="k", value="true", value_type="boolean")[0] == {"k": True}
    assert run(key="k", value="false", value_type="boolean")[0] == {
        "k": False
    }
    assert run(key="k", value="1", value_type="boolean")[0] == {"k": True}
    assert run(key="k", value="0", value_type="boolean")[0] == {"k": False}
    assert run(key="k", value="yes", value_type="boolean")[0] == {"k": True}
    assert run(key="k", value="", value_type="boolean")[0] == {"k": False}
    assert run(key="k", value=2, value_type="boolean")[0] == {"k": True}
    with pytest.raises(ValueError):
        run(key="k", value="maybe", value_type="boolean")


def test_null_value_type_forces_null():
    assert run(key="k", value=123, value_type="null")[0] == {"k": None}


def test_unconnected_value_stays_null_for_any_type():
    for mode in ("auto", "string", "int", "float", "boolean", "json", "null"):
        assert run(key="k", value_type=mode)[0] == {"k": None}


def test_invalid_value_type_is_rejected():
    with pytest.raises(ValueError):
        run(key="k", value=1, value_type="foo")


def test_dict_inputs_merge_in_port_order_and_later_wins():
    output = run(
        dict_1={"a": 1, "b": 1},
        dict_2={"a": 2},
        dict_3={"b": 3, "c": 3},
    )
    assert output[0] == {"a": 2, "b": 3, "c": 3}
    assert list(output[0]) == ["a", "b", "c"]


def test_key_and_value_are_applied_after_dicts():
    output = run(
        dict_1={"a": 1, "b": 2},
        dict_2={"a": 3},
        key="a",
        value="9",
        value_type="int",
    )
    assert output[0] == {"a": 9, "b": 2}


def test_unconnected_value_becomes_json_null():
    output = run(key="a")
    assert output[0] == {"a": None}
    assert output[1] == '{"a":null}'
    assert json.loads(output[1]) == {"a": None}


def test_no_inputs_produces_empty_object():
    output = run()
    assert output[0] == {}
    assert output[1] == "{}"


def test_empty_key_only_merges_dicts():
    output = run(dict_1={"a": 1}, key="   ", value=2)
    assert output[0] == {"a": 1}


def test_scalar_keys_are_stringified():
    assert run(key=1, value="x")[0] == {"1": "x"}
    assert run(key=True, value="x")[0] == {"true": "x"}


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        run(key=["a"], value=1)
    with pytest.raises(ValueError):
        run(key={"a": 1}, value=1)
    with pytest.raises(ValueError):
        run(dict_1=[{"a": 1}])
    with pytest.raises(ValueError):
        run(dict_1="not a dict")
    with pytest.raises(ValueError):
        run(dict_1={1: "non-string key"})
    with pytest.raises(ValueError):
        run(key="a", value={1, 2})
    with pytest.raises(ValueError):
        run(key="a", value=float("nan"))


def test_json_output_round_trips():
    output = run(dict_1={"b": [1, None]}, key="a", value={"c": True})
    assert json.loads(output[1]) == output[0]
