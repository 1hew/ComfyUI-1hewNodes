"""Border handling for Mask Stroke.

A 1px frame baked into the mask (a line along the image edge) must not be
dilated into a stroke ring around the whole canvas; a frame is detected and
ignored like Mask Fill Hole does, while real edge content still gets stroked.
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


def test_schema_exposes_close_border_only():
    schema = NODE.define_schema()
    ids = [item["id"] for item in schema["inputs"]]
    assert ids == ["mask", "fill_hole", "stroke_width", "include_mask", "close_border"]
    close = schema["inputs"][-1]
    assert close["id"] == "close_border"
    assert close["default"] is False
    include = schema["inputs"][3]
    assert include["id"] == "include_mask"
    assert include["default"] is False


def test_close_border_completes_the_clipped_side():
    mask = torch.zeros(40, 40)
    mask[10:31, 30:38] = 1.0  # block whose right side is 2px from the edge
    open_out = NODE._stroke_single_mask(mask, False, 5.0, False)
    closed_out = NODE._stroke_single_mask(mask, False, 5.0, False, True)
    assert float(open_out[20, 37]) == 0.0  # 37 is base, clipped stroke ends at 38
    assert float(open_out[20, 39]) == 1.0  # partial stroke on the edge
    assert float(closed_out[20, 37]) == 1.0  # edge band restores the width
    assert float(closed_out[20, 34]) == 0.0  # no band beyond the edge stroke
    assert float(closed_out[0, 0]) == 0.0  # far from the shape stays untouched


def test_close_border_keeps_edge_thickness_uniform():
    mask = torch.zeros(40, 40)
    mask[10:30, 30:40] = 1.0  # filled block touching the right edge
    out = NODE._stroke_single_mask(mask, False, 6.0, False, True)
    for y in (12, 20, 27):
        assert float(out[y, 39]) == 1.0  # edge pixel
        assert float(out[y, 34]) == 1.0  # 6px band (34..39)
        assert float(out[y, 33]) == 0.0  # nothing beyond the band


def test_close_border_leaves_a_centered_shape_unchanged():
    mask = torch.zeros(60, 60)
    mask[25:35, 25:35] = 1.0
    without = NODE._stroke_single_mask(mask, False, 4.0, False)
    with_close = NODE._stroke_single_mask(mask, False, 4.0, False, True)
    assert bool((without == with_close).all())
