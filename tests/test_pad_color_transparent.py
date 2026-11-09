"""Regression tests for the unified empty pad_color = transparent rule.

Empty (or `transparent`) always produces RGBA output: 3-channel input is
expanded to 4 channels so the padding can be `(0,0,0,0)` while the original
content stays opaque.
"""

import asyncio
import importlib.util
import sys
import types
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).parents[1]
_MODULES = {}


def _load(relative, name):
    cached = _MODULES.get(name)
    if cached is not None:
        return cached
    io = types.SimpleNamespace(
        ComfyNode=object,
        Schema=lambda **kwargs: kwargs,
        NodeOutput=lambda *values, **kwargs: values,
    )
    latest = types.ModuleType("comfy_api.latest")
    latest.io = io
    sys.modules.setdefault("comfy_api", types.ModuleType("comfy_api"))
    sys.modules["comfy_api.latest"] = latest
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _MODULES[name] = module
    return module


def _ops():
    return _load("utils/_image_ops.py", "pad_color_image_ops_under_test")


def _bbox_node():
    return _load(
        "nodes/image_crop/image_pad_by_bbox_mask.py",
        "pad_color_bbox_node_under_test",
    ).ImagePadByBBoxMask


def _edge_node():
    return _load(
        "nodes/image_crop/image_edge_crop_pad.py",
        "pad_color_edge_node_under_test",
    ).ImageEdgeCropPad


def _bbox_mask(height, width, rect):
    mask = torch.zeros((1, height, width), dtype=torch.float32)
    x0, y0, x1, y1 = rect
    mask[:, y0:y1, x0:x1] = 1.0
    return mask


def test_parse_pad_color_treats_empty_as_transparent():
    ops = _ops()

    for value in (None, "", "   ", "transparent", " TRANSPARENT "):
        assert ops.parse_pad_color(value) == "transparent"

    assert ops.parse_pad_color("1.0") == (1.0, 1.0, 1.0)
    assert ops.parse_pad_color("#ff0000") == (1.0, 0.0, 0.0)
    assert ops.parse_pad_color("edge") == "edge"


def test_resolve_pad_fill_maps_transparent_by_channel_count():
    ops = _ops()

    assert ops.resolve_pad_fill("transparent", 4) == (0.0, 0.0, 0.0, 0.0)
    # 3 通道只是防御性兜底：调用方会先扩成 RGBA。
    assert ops.resolve_pad_fill("transparent", 3) == (1.0, 1.0, 1.0)
    assert ops.resolve_pad_fill((0.0, 0.0, 0.0), 4) == (0.0, 0.0, 0.0, 1.0)


def test_pad_color_is_transparent():
    ops = _ops()

    for value in (None, "", "  ", "transparent", " TRANSPARENT "):
        assert ops.pad_color_is_transparent(value) is True

    for value in ("1.0", "0", "white", "edge", "#fff"):
        assert ops.pad_color_is_transparent(value) is False


def test_pad_to_rgb_fills_transparent_alpha():
    ops = _ops()
    rgba = torch.ones((1, 2, 2, 4))

    out = ops.pad_to_rgb(rgba, 4, 4, "transparent")

    assert out.shape == (1, 4, 4, 4)
    assert torch.equal(out[:, :1, :, :], torch.zeros((1, 1, 4, 4)))
    assert torch.equal(out[:, 1:3, 1:3, :], rgba)


def test_pad_to_rgb_expands_rgb_to_rgba_for_transparent():
    ops = _ops()
    rgb = torch.zeros((1, 2, 2, 3))

    out = ops.pad_to_rgb(rgb, 4, 4, "transparent")

    assert out.shape == (1, 4, 4, 4)
    # The original content keeps alpha 1, the padded frame is fully transparent.
    assert torch.equal(out[:, 1:3, 1:3, 3], torch.ones((1, 2, 2)))
    assert torch.equal(out[:, :1, :, 3], torch.zeros((1, 1, 4)))
    assert torch.equal(out[:, :1, :, :3], torch.zeros((1, 1, 4, 3)))


def test_pad_to_rgb_keeps_explicit_colors_opaque():
    ops = _ops()
    rgba = torch.zeros((1, 2, 2, 4))

    out = ops.pad_to_rgb(rgba, 4, 4, ops.parse_pad_color("1.0"))

    assert out[:, 0, 0, 3].item() == 1.0
    assert torch.allclose(out[:, 0, 0, :3], torch.ones(3))


def test_fit_resize_image_pads_white_for_empty_color():
    ops = _ops()
    # A portrait 2x4 source inside a 4x4 target leaves a left/right pad band.
    source = Image.new("RGB", (2, 4), (0, 0, 0))

    out = ops.fit_resize_image(source, 4, 4, "pad", Image.LANCZOS, "")

    assert out.size == (4, 4)
    assert out.getpixel((0, 0)) == (255, 255, 255)
    assert out.getpixel((2, 2)) == (0, 0, 0)


def test_bbox_pad_empty_color_is_transparent_for_rgba():
    node = _bbox_node()
    rgba = torch.full((1, 8, 8, 4), 0.25, dtype=torch.float32)
    rgba[..., 3] = 0.5

    out = asyncio.run(
        node.execute(
            paste_image=rgba,
            bbox_mask=_bbox_mask(32, 32, (12, 12, 20, 20)),
            pad_color="",
        )
    )[0]

    assert out.shape == (1, 32, 32, 4)
    assert out[0, 0, 0, 3].item() == 0.0
    assert out[0, 16, 16, 3].item() == 0.5


def test_bbox_pad_explicit_color_stays_opaque():
    node = _bbox_node()
    rgba = torch.full((1, 8, 8, 4), 0.25, dtype=torch.float32)

    out = asyncio.run(
        node.execute(
            paste_image=rgba,
            bbox_mask=_bbox_mask(32, 32, (12, 12, 20, 20)),
            pad_color="1.0",
        )
    )[0]

    assert out[0, 0, 0, 3].item() == 1.0
    assert torch.allclose(out[0, 0, 0, :3], torch.ones(3))


def test_bbox_pad_empty_color_expands_rgb_to_rgba():
    node = _bbox_node()
    rgb = torch.zeros((1, 8, 8, 3), dtype=torch.float32)

    out = asyncio.run(
        node.execute(
            paste_image=rgb,
            bbox_mask=_bbox_mask(32, 32, (12, 12, 20, 20)),
            pad_color="",
        )
    )[0]

    assert out.shape == (1, 32, 32, 4)
    assert out[0, 0, 0, 3].item() == 0.0
    assert out[0, 16, 16, 3].item() == 1.0


def test_bbox_pad_empty_mask_canvas_is_transparent_for_rgba():
    node = _bbox_node()
    rgba = torch.full((1, 8, 8, 4), 0.25, dtype=torch.float32)
    rgba[..., 3] = 0.5

    out = asyncio.run(
        node.execute(
            paste_image=rgba,
            bbox_mask=torch.zeros((1, 16, 16), dtype=torch.float32),
            pad_color="",
        )
    )[0]

    assert out.shape == (1, 16, 16, 4)
    assert float(out.max()) == 0.0


def test_multi_image_batch_pads_transparency_for_rgba():
    from _node_loader import load_node

    node = load_node(
        "pad_color_batch_under_test", "nodes/multi/multi_image_batch.py"
    ).MultiImageBatch
    reference = torch.full((1, 4, 4, 4), 0.5, dtype=torch.float32)
    # Portrait 4x2 inside the 4x4 reference leaves a left/right pad band.
    smaller = torch.full((1, 4, 2, 4), 0.5, dtype=torch.float32)
    smaller[..., 3] = 0.75

    out = asyncio.run(
        node.execute(fit="pad", pad_color="", image_1=reference, image_2=smaller)
    )[0]

    assert out.shape == (2, 4, 4, 4)
    assert out[1, 0, 0, 3].item() == 0.0
    assert out[1, 1, 1, 3].item() == 0.75


def test_edit_stitch_pad_helper_is_transparent_for_rgba():
    from _node_loader import load_node

    node = load_node(
        "pad_color_edit_stitch_under_test", "nodes/image/image_edit_stitch.py"
    ).ImageEditStitch
    rgba = torch.ones((1, 2, 2, 4), dtype=torch.float32)

    out = node._pad_to_rgb_like(rgba, 4, 4, "transparent")

    assert out.shape == (1, 4, 4, 4)
    assert out[0, 0, 0, 3].item() == 0.0
    assert out[0, 1, 1, 3].item() == 1.0


def test_multi_image_batch_expands_rgb_to_rgba_for_empty_color():
    from _node_loader import load_node

    node = load_node(
        "pad_color_batch_rgb_under_test", "nodes/multi/multi_image_batch.py"
    ).MultiImageBatch
    reference = torch.full((1, 4, 4, 3), 0.5, dtype=torch.float32)
    smaller = torch.full((1, 4, 2, 3), 0.5, dtype=torch.float32)

    out = asyncio.run(
        node.execute(fit="pad", pad_color="", image_1=reference, image_2=smaller)
    )[0]

    assert out.shape == (2, 4, 4, 4)
    assert out[1, 0, 0, 3].item() == 0.0
    assert out[1, 1, 1, 3].item() == 1.0


def test_image_resize_universal_expands_rgb_to_rgba_for_empty_color():
    from _node_loader import load_node

    node = load_node(
        "pad_color_universal_under_test",
        "nodes/image_resize/image_resize_universal.py",
    ).ImageResizeUniversal
    image = torch.full((1, 4, 2, 3), 0.5, dtype=torch.float32)
    kwargs = {
        "preset_ratio": "1:1",
        "proportional_width": 1,
        "proportional_height": 1,
        "method": "lanczos",
        "scale_to_side": "longest",
        "scale_to_length": 4,
        "fit": "pad",
        "divisible_by": 1,
        "image": image,
    }

    padded = asyncio.run(node.execute(pad_color="", **kwargs))[0]
    opaque = asyncio.run(node.execute(pad_color="1.0", **kwargs))[0]

    assert padded.shape == (1, 4, 4, 4)
    assert padded[0, 0, 0, 3].item() == 0.0
    assert padded[0, 2, 2, 3].item() == 1.0
    assert opaque.shape == (1, 4, 4, 3)


def test_edge_crop_pad_matches_the_same_rule():
    node = _edge_node()

    for value in (None, "", "transparent"):
        assert node._parse_pad_color(value, 4) == "transparent"
        assert node._parse_pad_color(value, 3) == "transparent"

    rgb = torch.full((1, 4, 4, 3), 0.5, dtype=torch.float32)
    out, mask = asyncio.run(
        node.execute(
            image=rgb,
            uniform_amount=0.0,
            top_amount=2.0,
            bottom_amount=0.0,
            left_amount=0.0,
            right_amount=0.0,
            pad_color="",
            divisible_by=1,
        )
    )

    assert out.shape == (1, 6, 4, 4)
    assert out[0, 0, 0, 3].item() == 0.0
    assert out[0, 3, 1, 3].item() == 1.0
