"""Regression tests for Image Area Compare's width x height comparison."""

import asyncio
import importlib
import sys
import types
from pathlib import Path

import torch


def _load_node():
    """Load the node without requiring a running ComfyUI installation."""
    io = types.SimpleNamespace(
        ComfyNode=object,
        Schema=lambda **kwargs: kwargs,
        NodeOutput=lambda *values, **kwargs: values,
    )
    latest = types.ModuleType("comfy_api.latest")
    latest.io = io
    sys.modules.setdefault("comfy_api", types.ModuleType("comfy_api"))
    sys.modules["comfy_api.latest"] = latest

    root = Path(__file__).parents[1]
    package = types.ModuleType("image_area_compare_test")
    package.__path__ = [str(root)]
    nodes_pkg = types.ModuleType("image_area_compare_test.nodes")
    nodes_pkg.__path__ = [str(root / "nodes")]
    logic_pkg = types.ModuleType("image_area_compare_test.nodes.logic")
    logic_pkg.__path__ = [str(root / "nodes" / "logic")]
    utils_pkg = types.ModuleType("image_area_compare_test.utils")
    utils_pkg.__path__ = [str(root / "utils")]

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
    sys.modules.update(
        {
            "image_area_compare_test": package,
            "image_area_compare_test.nodes": nodes_pkg,
            "image_area_compare_test.nodes.logic": logic_pkg,
            "image_area_compare_test.utils": utils_pkg,
        }
    )
    module = importlib.import_module(
        "image_area_compare_test.nodes.logic.image_area_compare"
    )
    return module.ImageAreaCompare


def test_area_is_width_times_height():
    node = _load_node()
    image = torch.zeros((1, 480, 640, 3))
    assert node._area(image) == 640 * 480


def test_area_ignores_batch_size_and_accepts_list_inputs():
    node = _load_node()
    image = torch.zeros((4, 512, 512, 3))
    assert node._area(image) == 512 * 512
    assert node._area([image]) == 512 * 512


def test_area_rejects_non_image_inputs():
    node = _load_node()
    assert node._area(None) is None
    assert node._area(torch.zeros(3)) is None
    assert node._area([]) is None


def test_execute_compares_pixel_areas():
    node = _load_node()
    big = torch.zeros((1, 1024, 1024, 3))
    small = torch.zeros((1, 512, 512, 3))

    assert asyncio.run(node.execute(big, ">", small)) == (True,)
    assert asyncio.run(node.execute(small, ">", big)) == (False,)
    assert asyncio.run(node.execute(big, ">=", big)) == (True,)
    assert asyncio.run(node.execute(big, "==", big)) == (True,)
    assert asyncio.run(node.execute(big, "!=", small)) == (True,)
    assert asyncio.run(node.execute(small, "<=", big)) == (True,)
    assert asyncio.run(node.execute(small, "<", big)) == (True,)


def test_execute_returns_false_for_invalid_inputs():
    node = _load_node()
    valid = torch.zeros((1, 64, 64, 3))

    assert asyncio.run(node.execute(None, "==", None)) == (False,)
    assert asyncio.run(node.execute(valid, "==", None)) == (False,)
