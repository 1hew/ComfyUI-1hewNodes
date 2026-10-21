"""Regression tests for Image Resize Area's Universal-style divisible_by / fit."""

import asyncio
import importlib
import math
import sys
import types
from pathlib import Path

import torch


def _load_node():
    """Load the node without requiring a running ComfyUI installation."""

    class _PortType:
        def __init__(self, name):
            self.name = name

        def Input(self, *args, **kwargs):
            return ("input", self.name, args, kwargs)

        def Output(self, *args, **kwargs):
            return ("output", self.name, args, kwargs)

    io = types.SimpleNamespace(
        ComfyNode=object,
        Schema=lambda **kwargs: kwargs,
        NodeOutput=lambda *values, **kwargs: values,
    )
    for name in ("Image", "Mask", "Int", "Float", "Boolean", "String", "Combo"):
        setattr(io, name, _PortType(name))
    io.Custom = lambda name: _PortType("Custom:" + name)

    latest = types.ModuleType("comfy_api.latest")
    latest.io = io
    sys.modules.setdefault("comfy_api", types.ModuleType("comfy_api"))
    sys.modules["comfy_api.latest"] = latest

    root = Path(__file__).parents[1]
    package = types.ModuleType("image_resize_area_test")
    package.__path__ = [str(root)]
    nodes_pkg = types.ModuleType("image_resize_area_test.nodes")
    nodes_pkg.__path__ = [str(root / "nodes")]
    resize_pkg = types.ModuleType("image_resize_area_test.nodes.image_resize")
    resize_pkg.__path__ = [str(root / "nodes" / "image_resize")]
    utils_pkg = types.ModuleType("image_resize_area_test.utils")

    def first_torch_tensor(value):
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, (list, tuple)):
            for item in value:
                tensor = first_torch_tensor(item)
                if tensor is not None:
                    return tensor
        return None

    utils_pkg.first_torch_tensor = first_torch_tensor
    utils_pkg.make_ui_text = lambda text: {"text": [str(text)]}
    sys.modules.update(
        {
            "image_resize_area_test": package,
            "image_resize_area_test.nodes": nodes_pkg,
            "image_resize_area_test.nodes.image_resize": resize_pkg,
            "image_resize_area_test.utils": utils_pkg,
        }
    )
    module = importlib.import_module(
        "image_resize_area_test.nodes.image_resize.image_resize_area"
    )
    return module.ImageResizeArea


def _run(node, image, **overrides):
    kwargs = {
        "min_sq_area": 1024,
        "max_sq_area": 2048,
        "method": "lanczos",
        "fit": "crop",
        "pad_color": "1.0",
        "divisible_by": 32,
        "mask": None,
    }
    kwargs.update(overrides)
    return asyncio.run(
        node.execute(
            image,
            kwargs["min_sq_area"],
            kwargs["max_sq_area"],
            kwargs["method"],
            kwargs["fit"],
            kwargs["pad_color"],
            kwargs["divisible_by"],
            mask=kwargs["mask"],
        )
    )


def test_schema_renames_int_output_to_resolution():
    node = _load_node()
    schema = node.define_schema()
    names = [port[3].get("display_name") for port in schema["outputs"]]
    assert names == ["image", "mask", "resolution"]


def test_sq_area_length_keeps_previous_values():
    node = _load_node()
    # raw = 3464 -> clamped to max 2048 -> aligned to 2048
    assert node.sq_area_length(4000, 3000, 1024, 2048, 32) == 2048
    # raw = 1732.05 -> nearest multiple of 32 -> 1728
    assert node.sq_area_length(2000, 1500, 1024, 2048, 32) == 1728
    assert node.sq_area_length(1024, 1024, 1024, 2048, 32) == 1024


def test_sq_area_length_no_longer_raises_without_multiple():
    node = _load_node()
    value = node.sq_area_length(1000, 1000, 1000, 1010, 32)
    assert 1000 <= value <= 1010


def test_target_size_is_identity_inside_range():
    node = _load_node()
    reference = math.sqrt(1000 * 800)
    assert node.target_size(1000, 800, reference, 1) == (1000, 800)
    assert node.target_size(1000, 800, reference, 32) == (1024, 800)


def test_execute_passthrough_when_size_already_aligned():
    node = _load_node()
    image = torch.zeros((1, 512, 512, 3))
    out = _run(node, image, min_sq_area=256, max_sq_area=2048, divisible_by=32)
    assert tuple(out[0].shape) == (1, 512, 512, 3)
    assert tuple(out[1].shape) == (1, 512, 512)
    assert out[2] == 512


def test_execute_always_aligns_inside_range():
    node = _load_node()
    image = torch.zeros((1, 300, 500, 3))
    out = _run(node, image, min_sq_area=256, max_sq_area=1024, divisible_by=32)
    height, width = int(out[0].shape[1]), int(out[0].shape[2])
    assert height % 32 == 0 and width % 32 == 0
    assert (height, width) == (320, 512)


def test_execute_downscales_oversized_image():
    node = _load_node()
    image = torch.zeros((1, 3000, 4000, 3))
    out = _run(node, image, min_sq_area=1024, max_sq_area=2048, divisible_by=32)
    assert tuple(out[0].shape) == (1, 1792, 2368, 3)
    assert tuple(out[1].shape) == (1, 1792, 2368)
    assert out[2] == 2048


def test_execute_upscales_undersized_image():
    node = _load_node()
    image = torch.zeros((1, 100, 200, 3))
    out = _run(node, image, min_sq_area=512, max_sq_area=2048, divisible_by=32)
    height, width = int(out[0].shape[1]), int(out[0].shape[2])
    assert height % 32 == 0 and width % 32 == 0
    assert height > 100 and width > 200


def test_execute_does_not_raise_without_multiple_in_range():
    node = _load_node()
    image = torch.zeros((1, 1000, 1000, 3))
    out = _run(node, image, min_sq_area=1000, max_sq_area=1010, divisible_by=32)
    assert out[2] == 1000
    assert int(out[0].shape[1]) % 32 == 0


def test_fit_pad_and_crop_produce_aligned_canvas():
    node = _load_node()
    white = torch.ones((1, 300, 500, 3))

    padded = _run(
        node, white, min_sq_area=256, max_sq_area=1024, divisible_by=32,
        fit="pad", pad_color="0",
    )
    assert tuple(padded[0].shape[1:3]) == (320, 512)
    assert padded[0][0, 0, 0, 0].item() == 0.0  # padding corner is black

    cropped = _run(
        node, white, min_sq_area=256, max_sq_area=1024, divisible_by=32, fit="crop"
    )
    assert tuple(cropped[0].shape[1:3]) == (320, 512)
    assert cropped[0][0, 0, 0, 0].item() == 1.0  # crop keeps content everywhere


def test_execute_synchronises_connected_mask():
    node = _load_node()
    image = torch.zeros((1, 3000, 4000, 3))
    mask = torch.ones((1, 3000, 4000))
    out = _run(
        node, image, min_sq_area=1024, max_sq_area=2048, divisible_by=32, mask=mask
    )
    assert tuple(out[0].shape) == (1, 1792, 2368, 3)
    assert tuple(out[1].shape) == (1, 1792, 2368)
