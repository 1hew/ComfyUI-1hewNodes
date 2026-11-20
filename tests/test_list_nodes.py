"""List Math Expression and List Text: list-in, list-out behaviour.

List Math Expression must match the core Math Expression grammar while
applying it element-wise: per-position evaluation, length-1 broadcast,
an error on misaligned lengths, port-number aliases that do not shift when
a middle port is unconnected, and the same restrictions on the expression.
List Text must keep Text List to String's prefix / suffix semantics but
return one string per item.
"""

import asyncio

import pytest

from _comfy_stub import install

install()

from _node_loader import load_node

MATH = load_node("list_math_under_test", "nodes/list/list_math_expression.py").ListMathExpression
TEXT = load_node("list_text_under_test", "nodes/list/list_text.py").ListText


def _math(expression="a + b", **inputs):
    return asyncio.run(MATH.execute(expression=[expression], **inputs))


def _text(**inputs):
    return asyncio.run(TEXT.execute(**inputs))


# ---------------------------------------------------------------- schema


def test_math_schema_is_list_in_list_out():
    schema = MATH.define_schema()
    assert schema["node_id"] == "1hew_ListMathExpression"
    assert schema["display_name"] == "List Math Expression"
    assert schema["category"] == "1hewNodes/list"
    assert schema["is_input_list"] is True
    assert schema["accept_all_inputs"] is True
    outputs = schema["outputs"]
    assert [o["display_name"] for o in outputs] == ["FLOAT", "INT", "BOOL"]
    assert all(o.get("is_output_list") is True for o in outputs)


def test_text_schema_is_list_in_list_out_with_text_list_ports():
    schema = TEXT.define_schema()
    assert schema["node_id"] == "1hew_ListText"
    assert schema["display_name"] == "List Text"
    assert schema["category"] == "1hewNodes/list"
    assert schema["is_input_list"] is True
    ports = {p["id"]: p for p in schema["inputs"]}
    assert ports["prefix"]["type"] == "STRING"
    assert ports["suffix"]["type"] == "STRING"
    assert schema["outputs"][0]["is_output_list"] is True


# ---------------------------------------------------------- math: alignment


def test_elementwise_addition_of_two_lists():
    floats, ints, bools = _math("a + b", item_1=[1, 2, 3], item_2=[10, 20, 30])
    assert floats == [11.0, 22.0, 33.0]
    assert ints == [11, 22, 33]


def test_length_one_list_is_broadcast():
    floats, _, _ = _math("a + b", item_1=[1, 2, 3], item_2=[100])
    assert floats == [101.0, 102.0, 103.0]


def test_scalar_input_is_a_one_item_list():
    floats, _, _ = _math("a * 2", item_1=[1, 2, 3])
    assert floats == [2.0, 4.0, 6.0]


def test_misaligned_lengths_raise():
    with pytest.raises(ValueError, match="长度不一致"):
        _math("a + b", item_1=[1, 2, 3], item_2=[1, 2])


def test_single_position_when_every_input_is_length_one():
    floats, ints, _ = _math("a + b", item_1=[2], item_2=[3])
    assert floats == [5.0]
    assert ints == [5]


def test_index_plus_one_per_item():
    # The motivating case: Text Path Extract's [1, 2, 3] -> [2, 3, 4].
    floats, ints, _ = _math("a + 1", item_1=[1, 2, 3])
    assert ints == [2, 3, 4]
    assert floats == [2.0, 3.0, 4.0]


def test_aliases_follow_port_number_not_connection_order():
    # item_2 is unconnected; item_3 must still be "c", not "b".
    floats, _, _ = _math("a + c", item_1=[1, 2], item_3=[10, 20])
    assert floats == [11.0, 22.0]


def test_values_lists_all_inputs_at_each_position():
    floats, _, _ = _math("sum(values)", item_1=[1, 2], item_2=[10, 20], item_3=[100, 200])
    assert floats == [111.0, 222.0]


def test_empty_expression_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        asyncio.run(MATH.execute(expression=["   "], item_1=[1]))


def test_no_inputs_returns_empty_lists():
    floats, ints, bools = _math("a + 1")
    assert floats == [] and ints == [] and bools == []


def test_result_must_be_numeric():
    # A string result is rejected; a bool result is numeric (bool subclasses
    # int), exactly as the core Math Expression accepts it.
    with pytest.raises(ValueError, match="numeric"):
        _math("'abc'", item_1=[1])
    floats, _, _ = _math("a > 0", item_1=[1, -1])
    assert floats == [1.0, 0.0]


def test_bool_output_list_matches_truthiness():
    _, _, bools = _math("a - 2", item_1=[1, 2, 3])
    assert bools == [True, False, True]


# ---------------------------------------------------------- math: safety


def test_unknown_names_are_rejected():
    with pytest.raises(Exception):
        _math("open('x')", item_1=[1])


def test_exponent_cap_still_applies():
    with pytest.raises(ValueError, match="exceeds"):
        _math("pow(a, 99999)", item_1=[2])


def test_huge_result_is_reported():
    with pytest.raises(ValueError):
        _math("a ** 4000", item_1=[10])


# ------------------------------------------------------------- list text


def test_prefix_and_suffix_apply_per_item_and_stay_a_list():
    out = _text(text_list=["cat", "dog"], prefix=["a "], suffix=[" !"])
    assert out[0] == ["a cat !", "a dog !"]


def test_nested_lists_are_flattened_one_level():
    out = _text(text_list=[["x", "y"], "z"], prefix=[""], suffix=[""])
    assert out[0] == ["x", "y", "z"]


def test_none_input_gives_empty_list():
    assert _text(text_list=None, prefix=[""], suffix=[""])[0] == []


def test_numbers_are_stringified():
    out = _text(text_list=[1, 2], prefix=["#"], suffix=[""])
    assert out[0] == ["#1", "#2"]


def test_list_text_output_is_not_joined():
    out = _text(text_list=["a", "b"], prefix=[""], suffix=[""])
    assert isinstance(out[0], list) and len(out[0]) == 2
