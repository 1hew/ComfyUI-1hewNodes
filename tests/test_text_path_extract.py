"""Path grammar and loose parsing for Text Path Extract.

These tests pin the behaviour the node advertises: a path such as
[0].content works on JSON text, fenced JSON, Python literals, relaxed JSON
and already-parsed dicts, wildcards collect every match, and the output
formatting stays stable. The Text Custom Extract checks guard its shared
parser (utils/_json_path.py).
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
    "text_path_under_test", "nodes/text/text_path_extract.py"
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


def test_schema_starts_with_text_and_defaults_to_every_item():
    schema = NODE.define_schema()
    assert schema["node_id"] == "1hew_TextPathExtract"
    assert schema["display_name"] == "Text Path Extract"
    assert schema["category"] == "1hewNodes/text"

    assert [port["id"] for port in schema["inputs"]] == [
        "text",
        "path",
        "output_format",
        "separator",
        "fallback",
    ]
    text, path = schema["inputs"][:2]
    assert text["type"] == "STRING"
    assert text["multiline"] is True
    # Optional widgets are listed after required ones by ComfyUI, which
    # would push the text box below every other widget.
    assert text.get("optional", False) is False
    assert path["default"] == "[*]"

    assert schema["outputs"][0]["type"] == "STRING"
    assert schema["outputs"][1]["type"] == "*"
    assert schema["outputs"][2]["type"] == "INT"
    assert schema["outputs"][2]["display_name"] == "count"


# ------------------------------------------------------------------ json text


def test_index_then_key():
    output = _run(text=SAMPLE, path="[0].content")
    assert output[0] == "犀牛"
    assert output[1] == "犀牛"


def test_negative_index_and_nested_index():
    assert _run(text=SAMPLE, path="[-1].bbox[0]")[0] == "690"
    assert _run(text=SAMPLE, path="[1].bbox[2]")[0] == "1000"


def test_bbox_renders_as_json_by_default():
    output = _run(text=SAMPLE, path="[1].bbox")
    assert output[0] == "[508,679,1000,953]"
    assert output[1] == [508, 679, 1000, 953]


def test_output_formats():
    assert _run(text=SAMPLE, path="[1].bbox", output_format="csv")[0] == (
        "508, 679, 1000, 953"
    )
    assert _run(text=SAMPLE, path="[1].bbox", output_format="text")[0] == (
        "[508, 679, 1000, 953]"
    )
    assert _run(text=SAMPLE, path="[1].bbox", output_format="json")[0] == (
        "[508,679,1000,953]"
    )
    assert _run(text=SAMPLE, path="[0]", output_format="json")[0] == (
        '{"content":"犀牛","bbox":[184,410,845,716]}'
    )


def test_quoted_bracket_key_and_single_quotes():
    assert _run(text=SAMPLE, path='[0]["content"]')[0] == "犀牛"
    assert _run(text=SAMPLE, path="[0].'content'")[0] == "犀牛"


def test_full_width_quotes_and_brackets_are_normalised():
    assert _run(text=SAMPLE, path="【0】.“content”")[0] == "犀牛"


def test_bare_key_and_jsonpath_root():
    data = '{"data": {"items": [{"zh": "猫"}]}}'
    assert _run(text=data, path="$.data.items[0].zh")[0] == "猫"
    assert _run(text=data, path="data.items[0].zh")[0] == "猫"
    assert _run(text='{"content": "狗"}', path="content")[0] == "狗"


def test_wildcard_collects_every_match():
    output = _run(text=SAMPLE, path="[*].content")
    assert output[0] == "犀牛\n犀牛\n犀牛"
    assert output[1] == ["犀牛", "犀牛", "犀牛"]

    dot = _run(text=SAMPLE, path="*.bbox[0]")
    assert dot[0] == "184\n508\n690"

    custom = _run(text=SAMPLE, path="[*].content", separator=", ")
    assert custom[0] == "犀牛, 犀牛, 犀牛"


def test_all_contents_and_bboxes_as_one_json_array():
    assert _run(text=SAMPLE, path="[*].content", output_format="json")[0] == (
        '["犀牛","犀牛","犀牛"]'
    )
    assert _run(text=SAMPLE, path="[*].bbox", output_format="json")[0] == (
        "[[184,410,845,716],[508,679,1000,953],[690,264,992,588]]"
    )
    assert _run(text=SAMPLE, path="[*].content")[1] == ["犀牛", "犀牛", "犀牛"]


def test_count_output_reports_match_count():
    assert _run(text=SAMPLE, path="[*].content")[2] == 3
    assert _run(text=SAMPLE, path="[*].bbox")[2] == 3
    assert _run(text=SAMPLE, path="[0].content")[2] == 1
    assert _run(text=SAMPLE, path="[9].content")[2] == 0
    assert _run(text="", path="[*].content")[2] == 0
    # A count path puts the number itself on the count output, so the
    # front-end label and the INT port agree.
    assert _run(text=SAMPLE, path="[*].content.count")[2] == 3
    assert _run(text=SAMPLE, path="[0].bbox[#]")[2] == 4


def test_execute_attaches_a_count_preview():
    # Patch the io object the node module actually imported: another test
    # module may have replaced the comfy_api stub in sys.modules by now.
    node_module = sys.modules[NODE.__module__]
    original = node_module.io.NodeOutput
    seen = []

    def capture(*values, ui=None, **kwargs):
        seen.append(ui)
        return values

    node_module.io.NodeOutput = capture
    try:
        out = asyncio.run(NODE.execute(text=SAMPLE, path="[*].content"))
        counted = asyncio.run(NODE.execute(text=SAMPLE, path="[*].content.count"))
        missing = asyncio.run(NODE.execute(text=SAMPLE, path="[9].content"))
    finally:
        node_module.io.NodeOutput = original

    assert out[2] == 3
    assert counted[2] == 3
    assert missing[2] == 0
    assert seen == [
        {"text": ["3"]},
        {"text": ["3"]},
        {"text": ["0"]},
    ]


def test_count_operator_paths():
    assert _run(text=SAMPLE, path="[*].content.count")[0] == "3"
    assert _run(text=SAMPLE, path="[*].content.length")[0] == "3"
    assert _run(text=SAMPLE, path="[0].bbox[#]")[0] == "4"
    assert _run(text=SAMPLE, path="[0].bbox.count")[1] == 4
    assert _run(text='{"a": [1, 2, 3]}', path="a.size")[0] == "3"


def test_count_key_is_not_shadowed():
    assert _run(text='{"count": 5}', path="count")[0] == "5"
    assert _run(text='{"a": {"count": 7}}', path="a.count")[0] == "7"
    assert _run(
        text='[{"count": 2}, {"count": 3}]',
        path='[*]["count"]',
        output_format="json",
    )[0] == "[2,3]"
    assert _run(
        text='[{"count": 2}, {"count": 3}]',
        path='[*]["count"]',
        output_format="csv",
        separator=", ",
    )[0] == "2, 3"
    assert _run(text='[{"count": 2}, {"count": 3}]', path="[*].count")[0] == "2"


def test_default_path_lists_every_item():
    output = _run(text=SAMPLE)
    assert output[0] == (
        '{"content":"犀牛","bbox":[184,410,845,716]}'
        "\n"
        '{"content":"犀牛","bbox":[508,679,1000,953]}'
        "\n"
        '{"content":"犀牛","bbox":[690,264,992,588]}'
    )
    assert output[2] == 3


def test_plain_text_needs_an_empty_path():
    # "[*]" means "walk the items", so plain text has nothing to walk.
    assert _run(text="今天下雨")[0] == ""
    assert _run(text="今天下雨")[2] == 0
    assert _run(text="今天下雨", path="")[0] == "今天下雨"


# ------------------------------------------------------------- loose parsing


def test_fenced_json_is_accepted():
    fenced = FENCE + "json\n" + SAMPLE + "\n" + FENCE
    assert _run(text=fenced, path="[2].bbox[3]")[0] == "588"


def test_python_literal_with_trailing_commas_and_booleans():
    data = "{'flag': True, 'items': [{'content': '牛',},], 'none': None,}"
    output = _run(text=data, path="items[0].content")
    assert output[0] == "牛"
    assert _run(text=data, path="flag")[0] == "true"
    assert _run(text=data, path="none")[0] == ""


def test_relaxed_json_with_trailing_commas():
    data = '{"a": [1, 2,], "b": {"c": 3,},}'
    assert _run(text=data, path="a[1]")[0] == "2"
    assert _run(text=data, path="b.c")[0] == "3"


def test_unquoted_keys_fall_back_to_the_loose_parser():
    data = '{content: "犀牛", bbox: [1, 2, 3, 4]}'
    assert _run(text=data, path="content")[0] == "犀牛"
    assert _run(text=data, path="bbox[3]")[0] == "4"


def test_double_encoded_json_string_is_unwrapped():
    inner = '"{\\"content\\": \\"犀牛\\"}"'
    assert _run(text=inner, path="content")[0] == "犀牛"


def test_plain_text_passes_through_with_an_empty_path():
    assert _run(text="你好，世界", path="")[0] == "你好，世界"
    assert _run(text="你好", path="")[1] == "你好"


def test_missing_path_returns_fallback():
    output = _run(text=SAMPLE, path="[9].content", fallback="N/A")
    assert output[0] == "N/A"
    assert output[1] == "N/A"


def test_empty_input_returns_fallback():
    assert _run(text="", path="[0].content", fallback="")[0] == ""
    assert _run(text="", path="[0].content", fallback="none")[0] == "none"
    assert _run(text="   ", path="[0].content", fallback="none")[0] == "none"


def test_parser_exception_returns_fallback_on_both_outputs(monkeypatch):
    node_module = sys.modules[NODE.__module__]

    def boom(*_args, **_kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(node_module, "query_all", boom)
    output = _run(text=SAMPLE, path="[*].content", fallback="N/A")
    assert output[0] == "N/A"
    assert output[1] == "N/A"
    assert output[2] == 0


def test_case_sensitive_keys_are_not_guessed():
    assert _run(text='{"Content": "犀牛"}', path="content", fallback="x")[0] == "x"


# ------------------------------------------------------------ shared helpers


def test_parse_path_tokens():
    assert JSON_PATH.parse_path("[0].content") == [0, "content"]
    assert JSON_PATH.parse_path("$.a.b[1]") == ["a", "b", 1]
    assert JSON_PATH.parse_path("content") == ["content"]
    assert len(JSON_PATH.parse_path("[*].content")) == 2


def test_query_returns_default_and_first_match():
    assert JSON_PATH.query(SAMPLE, "[0].content") == "犀牛"
    assert JSON_PATH.query(SAMPLE, "[9].content", "d") == "d"
    assert JSON_PATH.query_all(SAMPLE, "[*].bbox[0]") == [184, 508, 690]


def test_parse_content_keeps_unknown_objects():
    marker = object()
    assert JSON_PATH.parse_content(marker) is marker
    assert JSON_PATH.parse_content(None) is None
    assert JSON_PATH.parse_content([1, 2]) == [1, 2]


def test_format_value_styles():
    assert JSON_PATH.format_value("a") == "a"
    assert JSON_PATH.format_value([1, 2]) == "[1,2]"
    assert JSON_PATH.format_value([1, 2], "csv") == "1, 2"
    assert JSON_PATH.format_value({"a": 1}, "text") == "{'a': 1}"
    assert JSON_PATH.format_value(None) == ""


# ----------------------------------------------------- Text Custom Extract


def test_custom_extract_still_reads_json_and_loose_text():
    data = '[{"zh": "犀牛", "bbox": [1, 2, 3, 4]}]'
    assert _extract(data, "zh")[0] == "犀牛"

    loose = "{zh: '猫', bbox: [5, 6, 7, 8],}"
    assert _extract(loose, "bbox")[0] == "5, 6, 7, 8"


def test_custom_extract_trims_fences_and_ignores_noise():
    fenced = "说明文字\n" + FENCE + "json\n{\"zh\": \"狗\"}\n" + FENCE + "\n结尾"
    assert _extract(fenced, "zh", "enabled")[0] == "狗"
