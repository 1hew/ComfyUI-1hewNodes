"""Regression tests for Image Crop With BBox Mask's auto ratio mode."""

import asyncio
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch


def _load_node():
    """Load the node without requiring a running ComfyUI installation."""

    class _Port:
        def __init__(self, name):
            self.name = name

        def Input(self, *args, **kwargs):
            return (self.name, args, kwargs)

        def Output(self, *args, **kwargs):
            return (self.name, args, kwargs)

    io = types.SimpleNamespace(
        ComfyNode=object,
        Schema=lambda **kwargs: kwargs,
        NodeOutput=lambda *values, **kwargs: values,
    )
    for name in ("Image", "Mask", "Int", "Float", "Boolean", "String", "Combo"):
        setattr(io, name, _Port(name))
    latest = types.ModuleType("comfy_api.latest")
    latest.io = io
    sys.modules.setdefault("comfy_api", types.ModuleType("comfy_api"))
    sys.modules["comfy_api.latest"] = latest
    source = (
        Path(__file__).parents[1] / "nodes" / "image_crop" / "image_crop_with_bbox_mask.py"
    )
    spec = importlib.util.spec_from_file_location(
        "image_crop_with_bbox_mask_under_test", source
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module.ImageCropWithBBoxMask


def _mask(height, width, rects):
    mask = np.zeros((height, width), dtype=np.float32)
    for x, y, w, h in rects:
        mask[y : y + h, x : x + w] = 1.0
    return mask


def _run(node, width, height, rects, preset="auto", **overrides):
    image = torch.full((1, height, width, 3), 0.5, dtype=torch.float32)
    mask = torch.from_numpy(_mask(height, width, rects))[None]
    kwargs = {
        "image": image,
        "mask": mask,
        "preset_ratio": preset,
        "scale_strength": 0.0,
        "crop_to_side": "None",
        "crop_to_length": 1024,
        "divisible_by": 8,
        "get_crop_ratio": None,
    }
    kwargs.update(overrides)
    cropped, bbox_mask, cropped_mask = asyncio.run(node.execute(**kwargs))
    return cropped[0].numpy(), bbox_mask[0].numpy(), cropped_mask[0].numpy()


def _crop_extent(bbox_mask):
    ys, xs = np.where(bbox_mask > 0.5)
    assert len(xs) > 0
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def _fill(rects, width, height):
    """Fraction of the crop box covered by the mask bounding box."""
    x_min = min(r[0] for r in rects)
    y_min = min(r[1] for r in rects)
    x_max = max(r[0] + r[2] for r in rects)
    y_max = max(r[1] + r[3] for r in rects)
    return (x_max - x_min) * (y_max - y_min) / float(width * height)


def _preset_combo(node):
    schema = node.define_schema()
    return [item for item in schema["inputs"] if item[0] == "Combo"][0][2]


def test_preset_ratio_options_offer_auto_only():
    combo = _preset_combo(_load_node())

    assert "auto" in combo["options"]
    assert "smart" not in combo["options"]
    assert combo["default"] == "mask"
    assert len(combo["options"]) == 13


def test_auto_is_the_adaptive_mode_not_the_old_ratio_snap():
    """auto now frames from the mask; the old image-snapped 1664x936 is gone."""
    node = _load_node()
    tight, _, _ = _run(node, 1920, 1080, [(930, 90, 60, 900)], preset="auto")
    loose, _, _ = _run(
        node, 1920, 1080, [(930, 90, 60, 900)], preset="auto", scale_strength=1.0
    )

    assert tight.shape[:2] == (904, 64)
    assert loose.shape[:2] == (1080, 1920)


def test_auto_mode_keeps_a_thin_subject_filling_the_frame():
    """A 60x900 pole in a 1920x1080 frame must not be letterboxed to 16:9."""
    node = _load_node()
    cropped, bbox_mask, _ = _run(node, 1920, 1080, [(930, 90, 60, 900)])

    height, width = cropped.shape[:2]
    assert _fill([(930, 90, 60, 900)], width, height) > 0.5
    # The crop stays a portrait sliver instead of a 1664x936 landscape box.
    assert width < 200
    assert height > 800
    assert _crop_extent(bbox_mask) == (928, 88, 992, 992)


def test_explicit_ratio_still_letterboxes_that_subject():
    """The 16:9 preset keeps its documented behaviour; only auto rescues it."""
    node = _load_node()
    cropped, _, _ = _run(node, 1920, 1080, [(930, 90, 60, 900)], preset="16:9")

    height, width = cropped.shape[:2]
    assert (width, height) == (1664, 936)
    assert _fill([(930, 90, 60, 900)], width, height) < 0.05


def test_auto_mode_ignores_isolated_mask_specks():
    """A 2x2 stray pixel must not inflate the bbox to 960x540."""
    node = _load_node()
    rects = [(810, 340, 300, 400), (5, 5, 2, 2)]
    cropped, bbox_mask, cropped_mask = _run(node, 1920, 1080, rects)

    height, width = cropped.shape[:2]
    assert width <= 400 and height <= 500
    # The speck is left outside the crop, the real subject is not clipped.
    x0, y0, x1, y1 = _crop_extent(bbox_mask)
    assert bbox_mask[5, 5] == 0.0
    assert (x0, y0) <= (810, 340) and (x1, y1) >= (1110, 740)
    assert cropped_mask.sum() == 300 * 400


def test_auto_mode_crops_a_long_strip_instead_of_returning_the_frame():
    """A 2000x300 strip with an 1800x100 banner must actually be cropped."""
    node = _load_node()
    cropped, bbox_mask, _ = _run(node, 2000, 300, [(100, 100, 1800, 100)])

    height, width = cropped.shape[:2]
    assert (width, height) != (2000, 300)
    assert width >= 1700 and height <= 120
    assert _fill([(100, 100, 1800, 100)], width, height) > 0.5
    assert _crop_extent(bbox_mask) == (100, 98, 1900, 202)


def test_explicit_ratio_on_a_long_strip_still_returns_the_frame():
    """1:1 is impossible inside 2000x300; the old path silently returns the frame."""
    node = _load_node()
    cropped, bbox_mask, _ = _run(node, 2000, 300, [(100, 100, 1800, 100)], preset="1:1")

    assert cropped.shape[:2] == (300, 2000)
    assert bbox_mask.min() == 1.0


def test_auto_mode_snaps_to_a_nearby_standard_ratio():
    node = _load_node()
    cropped, _, _ = _run(node, 1200, 1000, [(300, 300, 600, 400)])

    height, width = cropped.shape[:2]
    assert abs(width / height - 3 / 2) < 0.02
    assert _fill([(300, 300, 600, 400)], width, height) > 0.9


def test_auto_mode_keeps_the_mask_ratio_when_a_standard_ratio_is_far():
    node = _load_node()
    cropped, _, _ = _run(node, 1920, 1080, [(900, 100, 100, 800)])

    height, width = cropped.shape[:2]
    assert width / height < 0.3  # not snapped to 9:16 (0.5625)
    assert _fill([(900, 100, 100, 800)], width, height) > 0.5


def test_auto_mode_respects_divisible_by():
    node = _load_node()
    cropped, _, _ = _run(
        node, 1920, 1080, [(930, 90, 60, 900)], divisible_by=16
    )

    height, width = cropped.shape[:2]
    assert width % 16 == 0 and height % 16 == 0


def test_auto_mode_scale_strength_expands_the_crop():
    node = _load_node()
    tight, _, _ = _run(node, 1200, 1000, [(300, 300, 600, 400)], scale_strength=0.0)
    loose, _, _ = _run(node, 1200, 1000, [(300, 300, 600, 400)], scale_strength=1.0)

    assert tight.shape[:2] == (400, 600)
    # 1.0 is anchored on the image, not on an integer multiple of the tight box.
    assert loose.shape[:2] == (1000, 1200)


def test_auto_mode_strength_blends_mask_and_image_ratio():
    """1663x521 frame with a 431px square subject: 1.0 must reach the whole image."""
    node = _load_node()
    rects = [(600, 36, 431, 431)]
    tight, _, _ = _run(node, 1663, 521, rects, scale_strength=0.0, divisible_by=32)
    middle, _, _ = _run(node, 1663, 521, rects, scale_strength=0.5, divisible_by=32)
    loose, _, _ = _run(node, 1663, 521, rects, scale_strength=1.0, divisible_by=32)

    assert tight.shape[:2] == (448, 448)
    assert middle.shape[:2] == (480, 864)
    assert abs(middle.shape[1] / middle.shape[0] - 1.8) < 0.02
    assert loose.shape[:2] == (512, 1632)  # largest aligned box inside 1663x521


def test_auto_mode_strength_grows_monotonically_and_geometrically():
    node = _load_node()
    sizes = [
        _run(node, 1200, 1000, [(300, 300, 600, 400)], scale_strength=s)[0].shape[:2]
        for s in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]

    widths = [width for height, width in sizes]
    heights = [height for height, width in sizes]
    assert widths == sorted(widths) and len(set(widths)) == 5
    assert heights == sorted(heights) and len(set(heights)) == 5
    assert widths[0] == 600 and widths[-1] == 1200
    assert heights[0] == 400 and heights[-1] == 1000


def test_auto_mode_keeps_a_corner_subject_in_frame():
    """A subject at the left edge still gains the empty right-hand context."""
    node = _load_node()
    cropped, bbox_mask, _ = _run(
        node, 1663, 521, [(10, 100, 300, 300)],
        scale_strength=0.5, divisible_by=32,
    )

    height, width = cropped.shape[:2]
    x0, _, x1, _ = _crop_extent(bbox_mask)
    assert x0 == 0
    assert x1 > 10 + 300
    assert width == x1 - x0 and height == cropped.shape[0]


def test_auto_mode_crop_to_side_respects_length():
    node = _load_node()
    cropped, _, _ = _run(
        node, 1200, 1000, [(300, 300, 600, 400)],
        crop_to_side="longest", crop_to_length=1024,
    )

    height, width = cropped.shape[:2]
    assert max(width, height) <= 1024
    assert width % 8 == 0 and height % 8 == 0


def test_auto_mode_returns_the_frame_for_an_empty_mask():
    node = _load_node()
    cropped, bbox_mask, _ = _run(node, 300, 200, [])

    assert cropped.shape[:2] == (200, 300)
    assert bbox_mask.min() == 1.0 and bbox_mask.max() == 1.0


def test_auto_mode_batches_mixed_shapes():
    node = _load_node()
    image = torch.full((2, 1080, 1920, 3), 0.5, dtype=torch.float32)
    masks = torch.from_numpy(
        np.stack(
            [
                _mask(1080, 1920, [(930, 90, 60, 900)]),
                _mask(1080, 1920, [(800, 400, 300, 300)]),
            ]
        )
    )
    cropped, bbox_mask, cropped_mask = asyncio.run(
        node.execute(
            image=image,
            mask=masks,
            preset_ratio="auto",
            scale_strength=0.0,
            crop_to_side="None",
            crop_to_length=1024,
            divisible_by=8,
            get_crop_ratio=None,
        )
    )

    assert cropped.shape[0] == 2 and bbox_mask.shape[0] == 2
    # Different crop sizes are padded up to one rectangular batch.
    assert cropped.shape[1] >= 900 and cropped.shape[2] >= 300
    for index in range(2):
        x0, y0, x1, y1 = _crop_extent(bbox_mask[index].numpy())
        assert bbox_mask[index, y0, x0] > 0.5
        assert cropped_mask[index].sum() > 0
