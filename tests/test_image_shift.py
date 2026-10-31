"""Regression tests for Image Shift's translation and seam mask."""

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
    for name in ("Image", "Mask", "Int", "Float", "Boolean", "String"):
        setattr(io, name, _PortType(name))
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

SHIFT = load_node("image_shift_under_test", "nodes/image/image_shift.py").ImageShift


def _run(image, **kwargs):
    return asyncio.run(SHIFT.execute(image=image, **kwargs))


def test_schema_contract():
    schema = SHIFT.define_schema()

    assert schema["node_id"] == "1hew_ImageShift"
    assert schema["display_name"] == "Image Shift"
    assert schema["category"] == "1hewNodes/image"
    assert [port["id"] for port in schema["inputs"]] == [
        "image",
        "shift_x",
        "shift_y",
        "border_mask_width",
        "border_mask_blur",
        "pad",
        "cyclic",
        "background_color",
        "mask",
    ]
    assert schema["inputs"][-1]["optional"] is True
    assert [port["display_name"] for port in schema["outputs"]] == [
        "image",
        "mask",
        "border_mask",
        "pad_mask",
    ]


def test_cyclic_shift_matches_roll_and_wraps():
    image = torch.arange(4 * 5 * 3, dtype=torch.float32).reshape(1, 4, 5, 3)

    out_image, out_mask, _, _ = _run(
        image, shift_x=2, shift_y=1, cyclic=True, border_mask_width=0
    )

    torch.testing.assert_close(
        out_image, torch.roll(image, shifts=(1, 2), dims=(-3, -2))
    )
    assert out_image.shape == image.shape
    torch.testing.assert_close(out_mask, torch.ones(1, 4, 5))


def test_non_cyclic_bounds_are_pixel_exact():
    image = torch.full((1, 8, 8, 3), 0.25)

    out_image, _, _, _ = _run(
        image,
        shift_x=3,
        shift_y=2,
        cyclic=False,
        background_color="#FF00FF",
        border_mask_width=0,
    )

    background = torch.tensor([1.0, 0.0, 1.0])
    # Top two rows and the left three columns of the remaining rows are fill.
    torch.testing.assert_close(out_image[0, :2], background.expand(2, 8, 3))
    torch.testing.assert_close(out_image[0, 2:, :3], background.expand(6, 3, 3))
    # Content starts exactly at (y=2, x=3); the reference node left one extra
    # background pixel on this leading edge.
    torch.testing.assert_close(
        out_image[0, 2:, 3:], torch.full((6, 5, 3), 0.25)
    )


def test_zero_shift_is_identity_in_non_cyclic_mode():
    image = torch.rand(2, 8, 8, 3)

    out_image, out_mask, _, _ = _run(
        image, shift_x=0, shift_y=0, cyclic=False, border_mask_width=0
    )

    torch.testing.assert_close(out_image, image)
    torch.testing.assert_close(out_mask, torch.ones(2, 8, 8))


def test_negative_shift_fills_trailing_edges():
    image = torch.ones(1, 8, 8, 3)

    out_image, _, _, _ = _run(
        image,
        shift_x=-2,
        shift_y=-3,
        cyclic=False,
        background_color="#000000",
        border_mask_width=0,
    )

    torch.testing.assert_close(out_image[0, :, 6:], torch.zeros(8, 2, 3))
    torch.testing.assert_close(out_image[0, 5:, :], torch.zeros(3, 8, 3))
    torch.testing.assert_close(out_image[0, :5, :6], torch.ones(5, 6, 3))


def test_oversized_shift_leaves_only_background():
    image = torch.ones(1, 4, 4, 3)

    out_image, out_mask, _, _ = _run(
        image,
        shift_x=40,
        shift_y=0,
        cyclic=False,
        background_color="#000000",
        border_mask_width=0,
    )

    torch.testing.assert_close(out_image, torch.zeros(1, 4, 4, 3))
    torch.testing.assert_close(out_mask, torch.zeros(1, 4, 4))


def test_mask_is_matched_per_frame():
    image = torch.zeros(3, 4, 4, 3)
    masks = (
        torch.arange(3, dtype=torch.float32).view(3, 1, 1).expand(3, 4, 4) / 2.0
    )

    _, out_mask, _, _ = _run(
        image, shift_x=0, shift_y=0, cyclic=True, border_mask_width=0, mask=masks
    )

    for index in range(3):
        torch.testing.assert_close(out_mask[index], masks[index])


def test_single_mask_broadcasts_over_the_batch():
    image = torch.zeros(3, 4, 4, 3)
    masks = torch.full((1, 4, 4), 0.25)

    _, out_mask, _, _ = _run(
        image, shift_x=0, shift_y=0, cyclic=True, border_mask_width=0, mask=masks
    )

    assert out_mask.shape == (3, 4, 4)
    torch.testing.assert_close(out_mask, torch.full((3, 4, 4), 0.25))


def test_mask_is_resized_to_the_image_size():
    image = torch.zeros(1, 8, 8, 3)
    masks = torch.ones(1, 2, 2)

    _, out_mask, _, _ = _run(
        image, shift_x=0, shift_y=0, cyclic=True, border_mask_width=0, mask=masks
    )

    assert out_mask.shape == (1, 8, 8)
    torch.testing.assert_close(out_mask, torch.ones(1, 8, 8))


def test_alpha_becomes_the_default_mask_and_is_preserved():
    image = torch.zeros(1, 4, 4, 4)
    image[..., 3] = torch.linspace(0.0, 1.0, 16).reshape(4, 4)

    out_image, out_mask, _, _ = _run(
        image, shift_x=1, shift_y=0, cyclic=True, border_mask_width=0
    )

    assert out_image.shape == image.shape
    torch.testing.assert_close(
        out_mask, torch.roll(image[..., 3], shifts=1, dims=2)
    )


def test_border_mask_band_is_exact_and_follows_the_shift():
    image = torch.zeros(1, 64, 64, 3)

    _, _, border, _ = _run(
        image,
        shift_x=0,
        shift_y=0,
        cyclic=True,
        border_mask_width=8,
        border_mask_blur=0,
    )

    assert border.shape == (1, 64, 64)
    torch.testing.assert_close(border[0, :8, 32], torch.ones(8))
    torch.testing.assert_close(border[0, 8:56, 32], torch.zeros(48))
    torch.testing.assert_close(border[0, 32, :8], torch.ones(8))

    _, _, shifted, _ = _run(
        image,
        shift_x=16,
        shift_y=0,
        cyclic=True,
        border_mask_width=8,
        border_mask_blur=0,
    )

    # A 16 px wrap moves the left band to x = 16..23 and the right band
    # (x = 56..63) to x = 8..15, so x = 8..23 is the shifted seam band.
    torch.testing.assert_close(shifted[0, 32, 8:24], torch.ones(16))
    torch.testing.assert_close(shifted[0, 32, :8], torch.zeros(8))
    torch.testing.assert_close(shifted[0, 32, 24:], torch.zeros(40))


def test_border_mask_is_shared_by_the_batch_and_clamped():
    image = torch.zeros(2, 8, 8, 3)

    _, _, border, _ = _run(
        image,
        shift_x=0,
        shift_y=0,
        cyclic=True,
        border_mask_width=999,
        border_mask_blur=0,
    )

    assert border.shape == (2, 8, 8)
    torch.testing.assert_close(border[0], border[1])
    # The band is clamped to half the canvas, so the ring covers every pixel.
    torch.testing.assert_close(border[0], torch.ones(8, 8))


def test_empty_batch_returns_empty_outputs():
    image = torch.zeros(0, 8, 8, 3)

    out_image, out_mask, out_border, _ = _run(image, shift_x=3, shift_y=3)

    assert out_image.shape == (0, 8, 8, 3)
    assert out_mask.shape == (0, 8, 8)
    assert out_border.shape == (0, 8, 8)


def test_missing_image_returns_none_outputs():
    out_image, out_mask, out_border, out_pad = _run(None)

    assert out_image is None
    assert out_mask is None
    assert out_border is None
    assert out_pad is None

def test_cyclic_ignores_the_background_color():
    image = torch.zeros(1, 4, 4, 3)

    out_image, _, _, _ = _run(
        image,
        shift_x=1,
        shift_y=1,
        cyclic=True,
        background_color="#FF00FF",
        border_mask_width=0,
    )

    torch.testing.assert_close(out_image, torch.zeros(1, 4, 4, 3))


def test_default_background_is_white_like_image_alpha_split():
    image = torch.zeros(1, 4, 4, 3)

    out_image, _, _, _ = _run(
        image, shift_x=2, shift_y=0, cyclic=False, border_mask_width=0
    )

    torch.testing.assert_close(out_image[0, :, :2], torch.ones(4, 2, 3))
    torch.testing.assert_close(out_image[0, :, 2:], torch.zeros(4, 2, 3))


def test_solid_color_uses_the_alpha_split_syntax():
    image = torch.zeros(1, 4, 4, 3)

    def fill_for(text):
        out, _, _, _ = _run(
            image,
            shift_x=2,
            shift_y=0,
            cyclic=False,
            background_color=text,
            border_mask_width=0,
        )
        return out[0, 0, 0]

    torch.testing.assert_close(fill_for("0.5"), torch.tensor([0.5, 0.5, 0.5]))
    torch.testing.assert_close(fill_for("red"), torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(fill_for("#FF0000"), torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(
        fill_for("128,192,255"),
        torch.tensor([128.0 / 255.0, 192.0 / 255.0, 1.0]),
    )
    # Unknown values fall back to white exactly like Image Alpha Split.
    torch.testing.assert_close(
        fill_for("not-a-colour"), torch.tensor([1.0, 1.0, 1.0])
    )


def test_sm_matches_image_alpha_split():
    alpha_split = load_node(
        "alpha_split_ref", "nodes/image/image_alpha_split.py"
    ).ImageAlphaSplit

    frame = torch.zeros(1, 8, 8, 4)
    frame[0, 2:6, 2:6, :3] = 0.9
    frame[0, 2:6, 2:6, 3] = 1.0

    flat, _ = asyncio.run(
        alpha_split.execute(image=frame, background_color="sm")
    )
    expected = flat[0, 0, 0]

    out_image, _, _, _ = _run(
        frame,
        shift_x=2,
        shift_y=0,
        cyclic=False,
        background_color="sm",
        border_mask_width=0,
    )

    void = out_image[0, :, :2]
    torch.testing.assert_close(void[..., :3], expected.expand(8, 2, 3))
    torch.testing.assert_close(void[..., 3], torch.ones(8, 2))


def test_sm_on_rgb_input_falls_back_to_white():
    image = torch.zeros(1, 4, 4, 3)

    out_image, _, _, _ = _run(
        image,
        shift_x=2,
        shift_y=0,
        cyclic=False,
        background_color="sm",
        border_mask_width=0,
    )

    torch.testing.assert_close(out_image[0, :, :2], torch.ones(4, 2, 3))

def test_pad_zero_keeps_the_canvas_and_an_empty_pad_mask():
    image = torch.rand(1, 6, 8, 3)

    out_image, _, _, pad_mask = _run(
        image, shift_x=2, shift_y=0, cyclic=True, pad=0, border_mask_width=0
    )

    assert out_image.shape == image.shape
    assert pad_mask.shape == (1, 6, 8)
    torch.testing.assert_close(pad_mask, torch.zeros(1, 6, 8))


def test_pad_wraps_outputs_and_marks_the_added_frame():
    image = torch.zeros(1, 6, 8, 3)
    image[0, 0, 0] = 1.0

    out_image, out_mask, out_border, pad_mask = _run(
        image, shift_x=0, shift_y=0, cyclic=True, pad=2, border_mask_width=0
    )

    assert out_image.shape == (1, 10, 12, 3)
    assert out_mask.shape == (1, 10, 12)
    assert out_border.shape == (1, 10, 12)
    assert pad_mask.shape == (1, 10, 12)

    # The 2 px frame is the wrapped opposite edge of the source.
    torch.testing.assert_close(out_image[0, :2, :2], image[0, -2:, -2:])
    torch.testing.assert_close(pad_mask[0, 0, 0], torch.tensor(1.0))
    torch.testing.assert_close(pad_mask[0, 2:-2, 2:-2], torch.zeros(6, 8))

    base_image, base_mask, base_border, _ = _run(
        image, shift_x=0, shift_y=0, cyclic=True, pad=0, border_mask_width=0
    )
    torch.testing.assert_close(out_image[0, 2:-2, 2:-2], base_image[0])
    torch.testing.assert_close(out_mask[0, 2:-2, 2:-2], base_mask[0])
    torch.testing.assert_close(out_border[0, 2:-2, 2:-2], base_border[0])


def test_pad_moves_border_mask_tips_inside_the_canvas():
    image = torch.zeros(1, 16, 16, 3)

    _, _, border, _ = _run(
        image,
        shift_x=8,
        shift_y=8,
        cyclic=True,
        pad=3,
        border_mask_width=2,
        border_mask_blur=0,
    )
    _, _, unpadded, _ = _run(
        image,
        shift_x=8,
        shift_y=8,
        cyclic=True,
        pad=0,
        border_mask_width=2,
        border_mask_blur=0,
    )

    torch.testing.assert_close(border[0, 3:-3, 3:-3], unpadded[0])
    torch.testing.assert_close(border[0, :3, 3:-3], unpadded[0, -3:])
    torch.testing.assert_close(border[0, -3:, 3:-3], unpadded[0, :3])
    torch.testing.assert_close(border[0, 3:-3, :3], unpadded[0, :, -3:])
    torch.testing.assert_close(border[0, 3:-3, -3:], unpadded[0, :, :3])


def test_empty_batch_with_pad_returns_padded_empty_outputs():
    image = torch.zeros(0, 4, 4, 3)

    out_image, out_mask, out_border, out_pad = _run(
        image, shift_x=1, shift_y=1, pad=2
    )

    assert out_image.shape == (0, 8, 8, 3)
    assert out_mask.shape == (0, 8, 8)
    assert out_border.shape == (0, 8, 8)
    assert out_pad.shape == (0, 8, 8)