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
    "text_path_under_test", "nodes/data/text_path_extract.py"
).TextPathExtract
CUSTOM_EXTRACT = load_node(
    "text_path_under_test", "nodes/text/text_custom_extract.py"
).TextCustomExtract
JSON_PATH = importlib.import_module("text_path_under_test.utils._json_path")

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
    assert schema["node_id"] == "1hew_TextPathExtract"
    assert schema["display_name"] == "Text Path Extract"
    assert schema["category"] == "1hewNodes/data"

    assert [port["id"] for port in schema["inputs"]] == ["text", "path"]
    text, path = schema["inputs"]
    assert text["type"] == "STRING"
    assert text["multiline"] is True
    assert text.get("optional", False) is False
    assert path["default"] == "values[*]"

    # Formatting moved out of the node: value plus count, nothing else.
    assert [o["display_name"] for o in schema["outputs"]] == ["value", "count"]
    assert schema["outputs"][0]["type"] == "*"
    # value is always a list output, so the port renders as a list endpoint.
    assert schema["outputs"][0]["is_output_list"] is True
    assert schema["outputs"][1]["type"] == "INT"


# ------------------------------------------------------------------ json text


def test_index_then_key():
    output = _run(text=SAMPLE, path="[0].content")
    assert output[0] == ["犀牛"]


def test_negative_index_and_nested_index():
    assert _run(text=SAMPLE, path="[-1].bbox[0]")[0] == [690]
    assert _run(text=SAMPLE, path="[1].bbox[2]")[0] == [1000]


def test_a_list_valued_match_stays_one_item():
    # One match whose value is itself a list must not be flattened.
    assert _run(text=SAMPLE, path="[1].bbox")[0] == [[508, 679, 1000, 953]]


def test_data_types_are_preserved():
    output = _run(text=SAMPLE, path="[0].content")
    assert isinstance(output[0][0], str)
    assert _run(text='{"n": 7}', path="n")[0] == [7]
    assert _run(text='{"b": true}', path="b")[0] == [True]
    assert _run(text='{"z": null}', path="z")[0] == [None]


def test_fenced_json_is_parsed():
    fenced = f'{FENCE}json\n{{"a": 1}}\n{FENCE}'
    assert _run(text=fenced, path="a")[0] == [1]


def test_python_literal_and_relaxed_json():
    assert _run(text="{'a': True}", path="a")[0] == [True]
    assert _run(text='{"a": 1,}', path="a")[0] == [1]


def test_quoted_bracket_key_and_single_quotes():
    assert _run(text=SAMPLE, path='[0]["content"]')[0] == ["犀牛"]
    assert _run(text=SAMPLE, path="[0].'content'")[0] == ["犀牛"]


def test_full_width_path_characters():
    assert _run(text=SAMPLE, path="【0】.“content”")[0] == ["犀牛"]


def test_optional_jsonpath_root():
    data = '{"data": {"items": [{"zh": "猫"}]}}'
    assert _run(text=data, path="$.data.items[0].zh")[0] == ["猫"]
    assert _run(text=data, path="data.items[0].zh")[0] == ["猫"]
    assert _run(text='{"content": "狗"}', path="content")[0] == ["狗"]


def test_keys_selector():
    data = '{"reference_image_1": 1, "reference_image_2": 2}'
    assert _run(text=data, path="keys[*]")[0] == ["reference_image_1", "reference_image_2"]
    assert _run(text=data, path="keys[0]")[0] == ["reference_image_1"]
    assert _run(text=data, path="keys[-1]")[0] == ["reference_image_2"]


def test_values_selector():
    data = '{"reference_image_1": {"ids": [{"label": "树"}]}, "image_1": {"ids": [{"label": "人物"}]}}'
    assert _run(text=data, path="values[0].ids[0].label")[0] == ["树"]
    assert _run(text=data, path="values[-1].ids[0].label")[0] == ["人物"]


def test_singular_selector_spellings():
    data = '{"reference_image_1": {"ids": [{"label": "树"}]}}'
    assert _run(text=data, path="key[0]")[0] == ["reference_image_1"]
    assert _run(text=data, path="value[0].ids[0].label")[0] == ["树"]


def test_selector_on_a_list():
    assert _run(text=SAMPLE, path="values[0].content")[0] == ["犀牛"]
    assert _run(text=SAMPLE, path="values[1].bbox[0]")[0] == [508]


def test_object_index_follows_python_rules():
    data = '{"a": 1, "b": 2, "c": 3}'
    assert _run(text=data, path="[0]")[0] == [1]
    assert _run(text=data, path="[-1]")[0] == [3]
    assert _run(text=data, path="[1]")[0] == [2]
    assert _run(text=data, path="[3]")[0] == []


def test_real_field_wins_over_selector():
    assert _run(text='{"values": [1, 2]}', path="values")[0] == [[1, 2]]
    assert _run(text='{"values": [1, 2]}', path='["values"]')[0] == [[1, 2]]


# ----------------------------------------------------------------- wildcards


def test_wildcard_collects_every_match():
    assert _run(text=SAMPLE, path="[*].content")[0] == ["犀牛", "犀牛", "犀牛"]


def test_default_path_lists_every_value():
    assert _run(text=SAMPLE)[0] == [
        {"content": "犀牛", "bbox": [184, 410, 845, 716]},
        {"content": "犀牛", "bbox": [508, 679, 1000, 953]},
        {"content": "犀牛", "bbox": [690, 264, 992, 588]},
    ]


def test_empty_path_returns_the_whole_payload():
    assert _run(text='{"a": 1}', path="")[0] == [{"a": 1}]
    assert _run(text="你好，世界", path="")[0] == ["你好，世界"]


# --------------------------------------------------------------------- count


def test_count_output_is_the_number_of_matches():
    assert _run(text=SAMPLE, path="[*].content")[1] == 3
    assert _run(text=SAMPLE, path="[0].content")[1] == 1
    assert _run(text=SAMPLE, path="[9].content")[1] == 0
    assert _run(text="", path="[*].content")[1] == 0


def test_count_operator_path_also_sets_the_count_output():
    assert _run(text=SAMPLE, path="[*].content.count")[1] == 3
    assert _run(text=SAMPLE, path="[0].bbox[#]")[1] == 4


def test_count_path_value_is_the_number():
    assert _run(text=SAMPLE, path="[*].content.count")[0] == [3]
    assert _run(text=SAMPLE, path="[0].bbox[#]")[0] == [4]
    assert _run(text=SAMPLE, path="[0].bbox.count")[0] == [4]
    assert _run(text='{"a": [1, 2, 3]}', path="a.size")[0] == [3]


# ---------------------------------------------------------------- empty input


def test_missing_path_returns_an_empty_list():
    output = _run(text=SAMPLE, path="[9].content")
    assert output[0] == []
    assert output[1] == 0


def test_empty_input_returns_an_empty_list():
    assert _run(text="", path="[0].content")[0] == []
    assert _run(text="   ", path="[0].content")[0] == []


def test_parser_exception_returns_an_empty_list(monkeypatch):
    node_module = sys.modules[NODE.__module__]

    def boom(*_args, **_kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(node_module, "query_all", boom)
    output = _run(text=SAMPLE, path="[*].content")
    assert output[0] == []
    assert output[1] == 0
