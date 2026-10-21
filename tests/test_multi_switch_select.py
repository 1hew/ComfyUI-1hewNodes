"""Regression tests for Multi Switch Select and the shared Image Batch Index grammar."""

import asyncio
import importlib
import sys
import types
from pathlib import Path


def _install_comfy_stub():
    class _PortType:
        def __init__(self, name):
            self.name = name

        def Input(self, id, *args, **kwargs):
            return {"kind": "input", "type": self.name, "id": id, **kwargs}

        def Output(self, *args, **kwargs):
            return {"kind": "output", "type": self.name, **kwargs}

    port_types = {
        "Image": "IMAGE",
        "Mask": "MASK",
        "Int": "INT",
        "Float": "FLOAT",
        "Boolean": "BOOLEAN",
        "String": "STRING",
        "Combo": "COMBO",
    }
    io = types.SimpleNamespace(
        ComfyNode=object,
        Schema=lambda **kwargs: kwargs,
        NodeOutput=lambda *values, **kwargs: values,
    )
    for name, io_type in port_types.items():
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


def _load_nodes():
    _install_comfy_stub()

    root = Path(__file__).parents[1]
    package_name = "range_nodes_under_test"
    top = types.ModuleType(package_name)
    top.__path__ = [str(root)]
    nodes = types.ModuleType(package_name + ".nodes")
    nodes.__path__ = [str(root / "nodes")]
    batch = types.ModuleType(package_name + ".nodes.batch")
    batch.__path__ = [str(root / "nodes" / "batch")]
    logic = types.ModuleType(package_name + ".nodes.logic")
    logic.__path__ = [str(root / "nodes" / "logic")]
    utils = types.ModuleType(package_name + ".utils")
    utils.__path__ = [str(root / "utils")]
    sys.modules.update(
        {
            package_name: top,
            package_name + ".nodes": nodes,
            package_name + ".nodes.batch": batch,
            package_name + ".nodes.logic": logic,
            package_name + ".utils": utils,
        }
    )
    multi = importlib.import_module(
        package_name + ".nodes.logic.multi_switch_select"
    ).MultiSwitchSelect
    batch_index = importlib.import_module(
        package_name + ".nodes.batch.image_batch_index"
    ).ImageBatchIndex
    return multi, batch_index


MULTI, IBI = _load_nodes()


def test_schema_exposes_string_select_and_lazy_inputs():
    schema = MULTI.define_schema()
    select = schema["inputs"][0]
    assert select["id"] == "select"
    assert select["type"] == "STRING"
    assert select["default"] == "1"

    dynamic = [port for port in schema["inputs"] if port["id"].startswith("input_")]
    assert len(dynamic) == 10
    assert all(port.get("lazy") is True for port in dynamic)


def test_parses_lists_and_loose_separators():
    assert MULTI._selected_indices("1,2,4") == [1, 2, 4]
    assert MULTI._selected_indices(" 4 , 1 , 2 ") == [1, 2, 4]
    assert MULTI._selected_indices("4，2；1 3、") == [1, 2, 3, 4]
    assert MULTI._selected_indices("1.3") == [1, 3]
    assert MULTI._selected_indices("1。3") == [1, 3]
    assert MULTI._selected_indices("1.3.5") == [1, 3, 5]


def test_parses_dash_ranges_like_image_batch_index():
    assert MULTI._selected_indices("1-3") == [1, 2, 3]
    assert MULTI._selected_indices("1 - 3") == [1, 2, 3]
    assert MULTI._selected_indices("1-5:2") == [1, 3, 5]
    assert MULTI._selected_indices("5-1:2") == [1, 3, 5]
    # Only "-" is a range separator; "~" is skipped like any unknown char.
    assert MULTI._selected_indices("5~7") == [5, 7]
    assert MULTI._selected_indices("１－３") == [1, 3]


def test_parses_bracket_ranges_and_steps():
    assert MULTI._selected_indices("[1,3]") == [1, 2, 3]
    assert MULTI._selected_indices("[1,3)") == [1, 2]
    assert MULTI._selected_indices("(1,3]") == [2, 3]
    assert MULTI._selected_indices("(1,3)") == [2]
    assert MULTI._selected_indices("[1,5]:2") == [1, 3, 5]
    assert MULTI._selected_indices("【1，3】") == [1, 2, 3]
    assert MULTI._selected_indices("（1，3）") == [2]
    assert MULTI._selected_indices("【3，1】") == [1, 2, 3]


def test_colon_inside_brackets_matches_image_batch_index():
    # ":" is the step syntax, not an endpoint separator, so [1:3] is 1 and 3.
    assert IBI._parse_indices("[1:3]", 10) == [1, 3]
    assert MULTI._selected_indices("[1:3]", 10) == [1, 3]
    assert MULTI._selected_indices("[1:3)", 10) == [1, 3]
    assert MULTI._selected_indices("(1:3]", 10) == [1, 3]
    assert MULTI._selected_indices("[1:5]:2", 10) == [1, 2, 5]


def test_negative_indices_use_connected_slot_base():
    assert MULTI._selected_indices("-1", 4) == [4]
    assert MULTI._selected_indices("-4", 4) == [1]
    assert MULTI._selected_indices("-5", 4) == []
    assert MULTI._selected_indices("[-3,-1]", 4) == [2, 3, 4]
    assert MULTI._selected_indices("-1", 0) == []


def test_out_of_range_and_empty_select_nothing():
    assert MULTI._selected_indices("0,15", 10) == []
    assert MULTI._selected_indices("5", 4) == []
    assert MULTI._selected_indices("", 10) == []
    assert MULTI._selected_indices("abc", 10) == []
    assert MULTI._selected_indices(None, 10) == []
    assert MULTI._selected_indices([], 10) == []


def test_bracket_ranges_drop_slots_outside_valid_range():
    assert MULTI._selected_indices("[0,2]", 10) == [1, 2]
    assert MULTI._selected_indices("[8,12]", 10) == [8, 9, 10]


def test_legacy_numeric_select():
    assert MULTI._selected_indices(3, 10) == [3]
    assert MULTI._selected_indices(15, 10) == []
    assert MULTI._selected_indices(-1, 4) == [4]


def test_image_batch_index_uses_the_same_grammar():
    assert IBI._parse_indices("1-3", 10) == [1, 2, 3]
    assert IBI._parse_indices("[1,3]", 10) == [1, 2, 3]
    assert IBI._parse_indices("[1,3)", 10) == [1, 2]
    assert IBI._parse_indices("1.3", 10) == [1, 3]
    assert IBI._parse_indices("1~3", 10) == [1, 3]
    assert IBI._parse_indices("[1:3]", 10) == [1, 3]
    assert IBI._parse_indices("-1", 10) == [9]
    assert MULTI._selected_indices("-1", 10) == [10]


def test_lazy_status_requests_all_selected_connected_inputs():
    status = MULTI.check_lazy_status(
        select="1,2,4", input_1=None, input_2=None, input_4=None
    )
    assert status == ["input_1", "input_2", "input_4"]


def test_lazy_status_skips_resolved_inputs():
    status = MULTI.check_lazy_status(
        select="1,2,4", input_1=object(), input_2=None, input_4=object()
    )
    assert status == ["input_2"]


def test_lazy_status_uses_connected_size_for_negative_index():
    status = MULTI.check_lazy_status(select="-1", input_1=object(), input_4=None)
    assert status == ["input_4"]


def test_lazy_status_skips_unconnected_port():
    # input_3 is selected but absent from kwargs because it is not connected.
    assert MULTI.check_lazy_status(select="1,3", input_1=object()) == []


def test_lazy_status_requests_colon_form_pair():
    status = MULTI.check_lazy_status(select="[1:3]", input_1=None, input_3=None)
    assert status == ["input_1", "input_3"]


def test_execute_routes_only_selected_inputs():
    out = asyncio.run(
        MULTI.execute(
            select="1,2,4",
            input_1="a",
            input_2="b",
            input_3="c",
            input_4="d",
        )
    )
    assert out == ("a", "b", None, "d", None, None, None, None, None, None)


def test_execute_colon_form_selects_one_and_three():
    out = asyncio.run(
        MULTI.execute(select="[1:3]", input_1="a", input_2="b", input_3="c")
    )
    assert out[0] == "a"
    assert out[1] is None
    assert out[2] == "c"


def test_execute_range_selects_contiguous():
    out = asyncio.run(
        MULTI.execute(select="1-3", input_1="a", input_2="b", input_3="c")
    )
    assert out[:3] == ("a", "b", "c")


def test_execute_resolves_negative_against_connected_size():
    out = asyncio.run(MULTI.execute(select="-1", input_1="a", input_4="d"))
    assert out[0] is None
    assert out[3] == "d"


def test_execute_unmatched_select_outputs_all_none():
    out = asyncio.run(MULTI.execute(select="0", input_1="a"))
    assert all(value is None for value in out)


def test_execute_selected_but_unconnected_input_is_none():
    out = asyncio.run(MULTI.execute(select="1,3", input_1="a", input_2="b"))
    assert out[0] == "a"
    assert out[2] is None


def test_validate_inputs_accepts_input_types_compat_argument():
    assert MULTI.validate_inputs(input_types={"select": "INT"}, select=2) is True


def test_execute_accepts_legacy_numeric_select():
    out = asyncio.run(MULTI.execute(select=2, input_2="b"))
    assert out[1] == "b"
    assert sum(value is not None for value in out) == 1
