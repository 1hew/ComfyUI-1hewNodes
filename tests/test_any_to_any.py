"""Regression tests for the Any to Any wildcard adapter."""

import asyncio
import sys
import types

import torch

from _node_loader import load_node


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

ANY = load_node("any_to_any_under_test", "nodes/conversion/any_to_any.py").AnyToAny


def _run(**kwargs):
    return asyncio.run(ANY.execute(**kwargs))


def test_schema_declares_wildcard_input_and_output():
    schema = ANY.define_schema()
    assert schema["node_id"] == "1hew_AnyToAny"
    assert schema["display_name"] == "Any to Any"
    assert schema["category"] == "1hewNodes/conversion"

    input_port = schema["inputs"][0]
    assert input_port["id"] == "any"
    assert input_port["type"] == "*"
    assert input_port["optional"] is True

    output_port = schema["outputs"][0]
    assert output_port["type"] == "*"
    assert output_port["display_name"] == "any"


def test_execute_forwards_values_unchanged():
    marker = object()
    values = [
        marker,
        torch.zeros(2, 3, 4, 3),
        "text",
        7,
        3.5,
        True,
        [torch.ones(1, 2, 2), "x"],
        {"waveform": torch.zeros(1, 2, 4), "sample_rate": 44100},
    ]
    for value in values:
        output = _run(any=value)
        assert output[0] is value


def test_execute_without_connection_returns_none():
    assert _run()[0] is None
    assert _run(any=None)[0] is None


def test_validate_inputs_accepts_any_link_type():
    assert ANY.validate_inputs(input_types={"any": "IMAGE"}) is True
    assert ANY.validate_inputs(input_types={"any": "MASK"}, any=object()) is True
