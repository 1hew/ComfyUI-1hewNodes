"""Legacy Text Path Extract compatibility node.

The node keeps the pre-rename interface for workflows that still reference
``1hew_TextPathExtract``: a ``text`` box, ``path`` / ``output_format`` /
``separator`` / ``fallback`` inputs and the three ``text`` / ``value`` /
``count`` outputs.
"""

import asyncio

from _node_loader import load_node

NODE = load_node(
    "text_path_extract_legacy_under_test",
    "nodes/data/text_path_extract.py",
).TextPathExtract


def _run(**kwargs):
    return asyncio.run(NODE.execute(**kwargs))


def test_schema_keeps_legacy_id_and_interface():
    schema = NODE.define_schema()
    assert schema["node_id"] == "1hew_TextPathExtract"
    assert schema["display_name"] == "Text Path Extract (legacy)"
    assert [item["id"] for item in schema["inputs"]] == [
        "text",
        "path",
        "output_format",
        "separator",
        "fallback",
    ]
    assert schema["inputs"][1]["default"] == "values[*]"
    assert [item["type"] for item in schema["outputs"]] == ["STRING", "*", "INT"]


def test_single_match_renders_text_and_raw_value():
    text, value, count = _run(text='{"a": 1}', path="a")
    assert text == "1"
    assert value == 1
    assert count == 1


def test_multiple_matches_join_or_emit_json():
    text, value, count = _run(text='[{"a": 1}, {"a": 2}]', path="[*].a")
    assert text == "1\n2"
    assert value == [1, 2]
    assert count == 2

    text, value, count = _run(
        text='[{"a": 1}, {"a": 2}]',
        path="[*].a",
        output_format="json",
    )
    assert text == "[1,2]"
    assert value == [1, 2]
    assert count == 2


def test_empty_input_and_missed_path_return_fallback():
    assert _run(text="", path="a", fallback="none") == ("none", "none", 0)
    assert _run(text='{"a": 1}', path="b", fallback="none") == ("none", "none", 0)
