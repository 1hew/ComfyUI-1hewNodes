"""Binarization-threshold semantics for the Mask Fill Hole node.

The node thresholds the mask before hole filling. Pixels above ``threshold``
are foreground; the rest are background, so only background regions fully
enclosed by foreground get filled.
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
    "mask_fill_hole_under_test",
    "nodes/mask/mask_fill_hole.py",
).MaskFillHole


def _run(mask, invert_mask=False, threshold=0.2):
    return asyncio.run(NODE.execute(mask=mask, invert_mask=invert_mask, threshold=threshold))[0]


def _hole_mask(value):
    """Bright block with an enclosed block of ``value`` inside it."""
    mask = torch.zeros(1, 16, 16)
    mask[0, 2:14, 2:14] = 1.0
    mask[0, 6:10, 6:10] = value
    return mask


def _notch_mask(value):
    """Bright block with a notch of ``value`` open to the image border."""
    mask = torch.zeros(1, 16, 16)
    mask[0, 2:14, 2:14] = 1.0
    mask[0, 0:12, 7:9] = value
    return mask


def test_schema_declares_threshold():
    schema = NODE.define_schema()
    assert schema["node_id"] == "1hew_MaskFillHole"

    threshold = schema["inputs"][2]
    assert threshold["id"] == "threshold"
    assert threshold["type"] == "FLOAT"
    assert threshold["default"] == 0.2
    assert threshold["min"] == 0.0
    assert threshold["max"] == 1.0
    assert threshold["step"] == 0.01


def test_below_threshold_pixel_is_background_and_filled():
    mask = _hole_mask(0.1)
    out = _run(mask, threshold=0.2)
    assert float(out[0, 7, 7]) == 1.0
    assert sorted(set(out.flatten().tolist())) == [0.0, 1.0]


def test_above_threshold_pixel_is_foreground_and_not_filled():
    mask = _notch_mask(0.3)
    out = _run(mask, threshold=0.2)
    assert float(out[0, 8, 7]) == 1.0
    assert float(out[0, 0, 7]) == 1.0  # notch is foreground up to the border


def test_threshold_restores_previous_50_percent_cut():
    mask = _notch_mask(0.3)
    out = _run(mask, threshold=0.5)
    assert float(out[0, 8, 7]) == 0.0  # background, connected to the border


def test_invert_flips_filled_result():
    mask = _hole_mask(0.1)
    out = _run(mask, invert_mask=True, threshold=0.2)
    assert float(out[0, 7, 7]) == 0.0


def test_batch_and_two_dimensional_inputs():
    batch = torch.cat([_hole_mask(0.1), _notch_mask(0.3)], dim=0)
    out = _run(batch, threshold=0.2)
    assert tuple(out.shape) == (2, 16, 16)
    assert float(out[0, 7, 7]) == 1.0
    assert float(out[1, 0, 7]) == 1.0

    flat = _hole_mask(0.1)[0]
    assert tuple(_run(flat, threshold=0.2).shape) == (1, 16, 16)


def test_out_of_range_threshold_is_clamped():
    mask = _hole_mask(0.3)
    assert float(_run(mask, threshold=2.0)[0, 7, 7]) == 0.0
    assert float(_run(mask, threshold=-1.0)[0, 7, 7]) == 1.0


def _framed_ring_mask():
    """1px white frame with a closed ring inside it."""
    mask = torch.zeros(1, 20, 20)
    mask[0, 0, :] = 1.0
    mask[0, -1, :] = 1.0
    mask[0, :, 0] = 1.0
    mask[0, :, -1] = 1.0
    mask[0, 5, 5:15] = 1.0
    mask[0, 14, 5:15] = 1.0
    mask[0, 5:15, 5] = 1.0
    mask[0, 5:15, 14] = 1.0
    return mask


def test_border_frame_does_not_turn_outer_background_into_a_hole():
    out = _run(_framed_ring_mask(), threshold=0.2)
    assert float(out[0, 9, 9]) == 1.0  # ring interior is filled
    assert float(out[0, 2, 2]) == 0.0  # background between frame and ring stays
    assert float(out[0, 10, 2]) == 0.0
    assert float(out[0, 0, 10]) == 1.0  # the frame itself is preserved


def test_foreground_touching_the_border_is_preserved():
    mask = _hole_mask(0.1)
    mask[0, 0:2, 7] = 1.0  # bar reaching the top border
    out = _run(mask, threshold=0.2)
    assert float(out[0, 0, 7]) == 1.0
    assert float(out[0, 7, 7]) == 1.0
