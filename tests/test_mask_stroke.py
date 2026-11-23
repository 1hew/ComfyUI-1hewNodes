"""Border handling for Mask Stroke.

A frame baked into the mask (a 1px line along the image edge) must not be
dilated into a stroke ring around the whole canvas; stroke detection ignores
the outermost 1px just like Mask Fill Hole.
"""

import torch

from _node_loader import load_node

NODE = load_node(
    "mask_stroke_under_test",
    "nodes/mask/mask_stroke.py",
    categories=["mask"],
).MaskStroke


def _framed_mask():
    mask = torch.zeros(40, 40)
    mask[0, :] = 1.0
    mask[-1, :] = 1.0
    mask[:, 0] = 1.0
    mask[:, -1] = 1.0
    mask[15:25, 15:25] = 1.0
    return mask


def test_frame_does_not_stroke_the_canvas_edge():
    out = NODE._stroke_single_mask(_framed_mask(), False, 2.0, False)
    assert float(out[0].max()) == 0.0  # the frame itself is not stroke
    assert float(out[1].max()) == 0.0  # no ring just inside the frame
    assert float(out[:, 1].max()) == 0.0
    assert float(out[14, 20]) == 1.0  # the blob still gets its own stroke


def test_include_mask_keeps_the_frame_without_edge_stroke():
    out = NODE._stroke_single_mask(_framed_mask(), False, 2.0, True)
    assert float(out[0, 0]) == 1.0  # frame preserved
    assert float(out[1, 5]) == 0.0  # but not dilated into a band


def test_fill_hole_with_frame_fills_the_blob_only():
    out = NODE._stroke_single_mask(_framed_mask(), True, 2.0, True)
    assert float(out[0, 0]) == 1.0  # frame preserved
    assert float(out[20, 20]) == 1.0  # blob interior filled
    assert float(out[5, 5]) == 0.0  # outer background stays background


def test_plain_mask_stroke_is_unchanged():
    mask = torch.zeros(40, 40)
    mask[15:25, 15:25] = 1.0
    out = NODE._stroke_single_mask(mask, False, 2.0, False)
    assert float(out[13, 20]) == 1.0
    assert float(out[12, 20]) == 0.0
    assert float(out[20, 20]) == 0.0


def test_content_touching_the_canvas_edge_keeps_its_stroke():
    mask = torch.zeros(40, 40)
    mask[0:20, 10] = 1.0  # vertical line reaching the top border
    out = NODE._stroke_single_mask(mask, False, 2.0, False)
    assert float(out[0, 9]) == 1.0  # the inward half of the dilation closes it


def test_ignore_mode_drops_the_edge_stroke():
    mask = torch.zeros(40, 40)
    mask[0:20, 10] = 1.0
    out = NODE._stroke_single_mask(mask, False, 2.0, False, "ignore")
    assert float(out[0, 9]) == 0.0
    assert float(out[1, 9]) == 1.0


def test_stroke_mode_keeps_the_frame_stroke():
    out = NODE._stroke_single_mask(_framed_mask(), False, 2.0, False, "stroke")
    assert float(out[1, 5]) == 1.0  # the frame is stroked like real content


def test_schema_exposes_border_mode():
    schema = NODE.define_schema()
    mode = schema["inputs"][-1]
    assert mode["id"] == "border_mode"
    assert mode["default"] == "auto"
    assert mode["options"] == ["auto", "stroke", "ignore"]
