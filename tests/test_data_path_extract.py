"""Text Path Extract: path grammar, loose parsing and the value contract.

These tests pin the behaviour the node advertises: a path such as
[0].content works on JSON text, fenced JSON, Python literals, relaxed JSON
and already-parsed dicts, wildcards collect every match, and the value output
is always a list with one entry per match. Formatting is not the node's job
any more, so there is no output_format to test here; the Text Custom Extract
checks guard the shared parser (utils/_json_path.py).
"""

import asyncio
import importlib
import sys
import types

from _node_loader import load_node

FENCE = chr(96) * 3


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
    "data_path_under_test", "nodes/data/data_path_extract.py"
).DataPathExtract
CUSTOM_EXTRACT = load_node(
    "data_path_under_test", "nodes/text/text_custom_extract.py"
).TextCustomExtract
JSON_PATH = importlib.import_module("data_path_under_test.utils._json_path")

SAMPLE = """[
  { "content": "犀牛", "bbox": [184, 410, 845, 716] },
  { "content": "犀牛", "bbox": [508, 679, 1000, 953] },
  { "content": "犀牛", "bbox": [690, 264, 992, 588] }
]"""


def _run(**kwargs):
    return asyncio.run(NODE.execute(**kwargs))


def _extract(json_data, key, precision_match="disabled", label_filter=""):
    return asyncio.run(
        CUSTOM_EXTRACT.execute(
            json_data=json_data,
            key=key,
            precision_match=precision_match,
            label_filter=label_filter,
        )
    )


# --------------------------------------------------------------------- schema


def test_schema_collects_values_and_reports_a_count():
    schema = NODE.define_schema()
    assert schema["node_id"] == "1hew_DataPathExtract"
    assert schema["display_name"] == "Data Path Extract"
    assert schema["category"] == "1hewNodes/data"

    assert [port["id"] for port in schema["inputs"]] == [
        "data",
        "data_path",
    ]
    ports = {p["id"]: p for p in schema["inputs"]}
    # data is a required wildcard link, so structured input passes through.
    assert ports["data"]["type"] == "*"
    assert ports["data"].get("optional", False) is False
    assert "multiline" not in ports["data"]
    # data_path, not path: "path" alone reads as a file path in ComfyUI.
    assert ports["data_path"]["default"] == "[*]"
    # value is the only shape: a list on a list-styled port, no switch.
    assert "output_type" not in ports

    # Formatting moved out of the node: value plus count, nothing else.
    assert [o["display_name"] for o in schema["outputs"]] == ["value", "count"]
    assert schema["outputs"][0]["type"] == "*"
    # value is always a list output, so the port renders as a list endpoint.
    assert schema["outputs"][0]["is_output_list"] is True
    assert schema["outputs"][1]["type"] == "INT"


# ------------------------------------------------------------------ json text


def test_index_then_key():
    output = _run(data=SAMPLE, data_path="[0].content")
    assert output[0] == ["犀牛"]


def test_negative_index_and_nested_index():
    assert _run(data=SAMPLE, data_path="[-1].bbox[0]")[0] == [690]
    assert _run(data=SAMPLE, data_path="[1].bbox[2]")[0] == [1000]


def test_a_list_valued_match_stays_one_item():
    # One match whose value is itself a list must not be flattened.
    assert _run(data=SAMPLE, data_path="[1].bbox")[0] == [[508, 679, 1000, 953]]


def test_data_types_are_preserved():
    output = _run(data=SAMPLE, data_path="[0].content")
    assert isinstance(output[0][0], str)
    assert _run(data='{"n": 7}', data_path="n")[0] == [7]
    assert _run(data='{"b": true}', data_path="b")[0] == [True]
    assert _run(data='{"z": null}', data_path="z")[0] == [None]


def test_fenced_json_is_parsed():
    fenced = f'{FENCE}json\n{{"a": 1}}\n{FENCE}'
    assert _run(data=fenced, data_path="a")[0] == [1]


def test_python_literal_and_relaxed_json():
    assert _run(data="{'a': True}", data_path="a")[0] == [True]
    assert _run(data='{"a": 1,}', data_path="a")[0] == [1]


def test_quoted_bracket_key_and_single_quotes():
    assert _run(data=SAMPLE, data_path='[0]["content"]')[0] == ["犀牛"]
    assert _run(data=SAMPLE, data_path="[0].'content'")[0] == ["犀牛"]


def test_full_width_path_characters():
    assert _run(data=SAMPLE, data_path="【0】.“content”")[0] == ["犀牛"]


def test_optional_jsonpath_root():
    data = '{"data": {"items": [{"zh": "猫"}]}}'
    assert _run(data=data, data_path="$.data.items[0].zh")[0] == ["猫"]
    assert _run(data=data, data_path="data.items[0].zh")[0] == ["猫"]
    assert _run(data='{"content": "狗"}', data_path="content")[0] == ["狗"]


def test_keys_selector():
    data = '{"reference_image_1": 1, "reference_image_2": 2}'
    assert _run(data=data, data_path="keys[*]")[0] == ["reference_image_1", "reference_image_2"]
    assert _run(data=data, data_path="keys[0]")[0] == ["reference_image_1"]
    assert _run(data=data, data_path="keys[-1]")[0] == ["reference_image_2"]


def test_values_selector():
    data = '{"reference_image_1": {"ids": [{"label": "树"}]}, "image_1": {"ids": [{"label": "人物"}]}}'
    assert _run(data=data, data_path="values[0].ids[0].label")[0] == ["树"]
    assert _run(data=data, data_path="values[-1].ids[0].label")[0] == ["人物"]


def test_singular_selector_spellings():
    data = '{"reference_image_1": {"ids": [{"label": "树"}]}}'
    assert _run(data=data, data_path="key[0]")[0] == ["reference_image_1"]
    assert _run(data=data, data_path="value[0].ids[0].label")[0] == ["树"]


def test_selector_on_a_list():
    assert _run(data=SAMPLE, data_path="values[0].content")[0] == ["犀牛"]
    assert _run(data=SAMPLE, data_path="values[1].bbox[0]")[0] == [508]


def test_object_index_follows_python_rules():
    data = '{"a": 1, "b": 2, "c": 3}'
    assert _run(data=data, data_path="[0]")[0] == [1]
    assert _run(data=data, data_path="[-1]")[0] == [3]
    assert _run(data=data, data_path="[1]")[0] == [2]
    assert _run(data=data, data_path="[3]")[0] == []


def test_real_field_wins_over_selector():
    assert _run(data='{"values": [1, 2]}', data_path="values")[0] == [[1, 2]]
    assert _run(data='{"values": [1, 2]}', data_path='["values"]')[0] == [[1, 2]]


# ----------------------------------------------------------------- wildcards


def test_wildcard_collects_every_match():
    assert _run(data=SAMPLE, data_path="[*].content")[0] == ["犀牛", "犀牛", "犀牛"]


def test_default_path_lists_every_value():
    assert _run(data=SAMPLE)[0] == [
        {"content": "犀牛", "bbox": [184, 410, 845, 716]},
        {"content": "犀牛", "bbox": [508, 679, 1000, 953]},
        {"content": "犀牛", "bbox": [690, 264, 992, 588]},
    ]


def test_empty_path_returns_the_whole_payload():
    assert _run(data='{"a": 1}', data_path="")[0] == [{"a": 1}]
    assert _run(data="你好，世界", data_path="")[0] == ["你好，世界"]


# --------------------------------------------------------------------- count


def test_count_output_is_the_number_of_matches():
    assert _run(data=SAMPLE, data_path="[*].content")[1] == 3
    assert _run(data=SAMPLE, data_path="[0].content")[1] == 1
    assert _run(data=SAMPLE, data_path="[9].content")[1] == 0
    assert _run(data="", data_path="[*].content")[1] == 0


def test_count_operator_path_also_sets_the_count_output():
    assert _run(data=SAMPLE, data_path="[*].content.count")[1] == 3
    assert _run(data=SAMPLE, data_path="[0].bbox[#]")[1] == 4


def test_count_path_value_is_the_number():
    assert _run(data=SAMPLE, data_path="[*].content.count")[0] == [3]
    assert _run(data=SAMPLE, data_path="[0].bbox[#]")[0] == [4]
    assert _run(data=SAMPLE, data_path="[0].bbox.count")[0] == [4]
    assert _run(data='{"a": [1, 2, 3]}', data_path="a.size")[0] == [3]


# ---------------------------------------------------------------- empty input


def test_missing_path_returns_an_empty_list():
    output = _run(data=SAMPLE, data_path="[9].content")
    assert output[0] == []
    assert output[1] == 0


def test_empty_input_returns_an_empty_list():
    assert _run(data="", data_path="[0].content")[0] == []
    assert _run(data="   ", data_path="[0].content")[0] == []


def test_parser_exception_returns_an_empty_list(monkeypatch):
    node_module = sys.modules[NODE.__module__]

    def boom(*_args, **_kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(node_module, "query_all", boom)
    output = _run(data=SAMPLE, data_path="[*].content")
    assert output[0] == []
    assert output[1] == 0


# ------------------------------------------------------------- list-only output


def test_value_is_always_a_list():
    value, count = _run(data=SAMPLE, data_path="[*].content")
    assert count == 3
    assert value == ["犀牛", "犀牛", "犀牛"]


def test_miss_is_an_empty_list():
    value, count = _run(data=SAMPLE, data_path="[9].content")
    assert count == 0
    assert value == []


def test_native_types_are_preserved():
    assert _run(data='{"n": 7}', data_path="n")[0] == [7]
    assert _run(data='{"b": true}', data_path="b")[0] == [True]
    assert _run(data='{"z": null}', data_path="z")[0] == [None]
    assert _run(data='{"s": "x"}', data_path="s")[0] == ["x"]


# ------------------------------------------------------- structured data input


def test_accepts_an_already_parsed_dict():
    # No JSON text at all: the wildcard input takes the object directly.
    assert _run(data={"layers": [{"index": 1}, {"index": 2}]},
                data_path="layers[*].index")[0] == [1, 2]


def test_accepts_an_already_parsed_list():
    assert _run(data=[{"n": 1}, {"n": 2}], data_path="[*].n")[0] == [1, 2]


def test_missing_data_link_returns_empty_list():
    assert _run(data=None, data_path="[0]")[0] == []
    assert _run(data="", data_path="[0]")[0] == []
