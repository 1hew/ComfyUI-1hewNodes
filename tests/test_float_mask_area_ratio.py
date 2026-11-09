"""Area-ratio semantics for the Float Mask Area Ratio node.

The node turns a mask into a 0-1 float; the value must be the mean mask weight
(soft values weighted), rounded to the requested decimals, and it must stay
stable for batches, empty inputs and out-of-range values.
"""

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
        NumberDisplay=types.SimpleNamespace(number="number", slider="slider"),
    )
    for name, io_type in {
        "Image": "IMAGE",
        "Mask": "MASK",
        "Video": "VIDEO",
        "Audio": "AUDIO",
        "Int": "INT",
        "Float": "FLOAT",
        "Boolean": "BOOLEAN",
        "String": "STRING",
        "Combo": "COMBO",
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
    "float_area_ratio_under_test",
    "nodes/float/float_mask_area_ratio.py",
).FloatMaskAreaRatio


def _run(**kwargs):
    return asyncio.run(NODE.execute(**kwargs))


# --------------------------------------------------------------------- schema


def test_schema_declares_mask_area_ratio():
    schema = NODE.define_schema()
    assert schema["node_id"] == "1hew_FloatMaskAreaRatio"
    assert schema["display_name"] == "Float Mask Area Ratio"
    assert schema["category"] == "1hewNodes/float"

    mask = schema["inputs"][0]
    assert mask["id"] == "mask"
    assert mask["type"] == "MASK"

    round_input = schema["inputs"][1]
    assert round_input["id"] == "round"
    assert round_input["type"] == "INT"
    assert round_input["default"] == 2
    assert round_input["min"] == 0
    assert round_input["max"] == 10
    assert round_input["display_mode"] == "number"

    float_output = schema["outputs"][0]
    assert float_output["type"] == "FLOAT"
    assert float_output["display_name"] == "float"


# ------------------------------------------------------------------- coverage


def test_half_covered_mask_is_half():
    mask = torch.zeros(1, 8, 8)
    mask[:, :4, :] = 1.0
    assert _run(mask=mask)[0] == 0.5


def test_default_rounds_to_two_decimals():
    mask = torch.zeros(1, 3, 3)
    mask[0, 0, 0] = 1.0  # 1/9 = 0.111...
    assert _run(mask=mask)[0] == 0.11


def test_round_controls_precision():
    mask = torch.zeros(1, 3, 3)
    mask[0, 0, 0] = 1.0
    assert _run(mask=mask, round=4)[0] == 0.1111
    assert _run(mask=mask, round=0)[0] == 0.0


def test_round_clamped_and_fallback():
    mask = torch.zeros(1, 3, 3)
    mask[0, 0, 0] = 1.0
    assert _run(mask=mask, round=None)[0] == 0.11
    assert _run(mask=mask, round=99)[0] == 0.1111111111


def test_batch_is_averaged():
    batch = torch.zeros(4, 8, 8)
    batch[0] = 1.0
    batch[1] = 0.5
    # (1 + 0.5 + 0 + 0) / 4 = 0.375 -> 0.38
    assert _run(mask=batch)[0] == 0.38


def test_soft_values_are_weighted():
    soft = torch.full((1, 4, 4), 0.25)
    assert _run(mask=soft)[0] == 0.25


def test_two_dimensional_and_four_dimensional_masks():
    flat = torch.zeros(4, 4)
    flat[:2, :] = 1.0
    assert _run(mask=flat)[0] == 0.5

    channels = torch.cat([torch.ones(1, 4, 4, 1), torch.zeros(1, 4, 4, 1)], dim=-1)
    assert _run(mask=channels)[0] == 1.0


def test_flat_masks_return_bounds():
    assert _run(mask=torch.zeros(1, 8, 8))[0] == 0.0
    assert _run(mask=torch.ones(1, 8, 8))[0] == 1.0


def test_out_of_range_values_are_clipped():
    assert _run(mask=torch.full((1, 4, 4), 3.0))[0] == 1.0
    assert _run(mask=torch.full((1, 4, 4), -2.0))[0] == 0.0


def test_empty_inputs_return_zero():
    assert _run(mask=None)[0] == 0.0
    assert _run(mask=torch.zeros(0, 8, 8))[0] == 0.0
    assert _run(mask=object())[0] == 0.0


def test_input_tensor_is_not_mutated():
    mask = torch.full((1, 4, 4), 2.0)
    _run(mask=mask)
    assert bool((mask == 2.0).all())
