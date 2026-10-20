"""Regression tests for Detect Remove SolidBG UnMult's gamut handling.

The core is ported from ComfyUI-RH-Nodes' MekajikiUnMult, so the 2026-10-15
fix there (clip negative channels, jointly scale positive overflow, and never
revert the whole pixel to the observed RGB) must hold here as well.
"""

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
    package = types.ModuleType("solidbg_unmult_test")
    package.__path__ = [str(root)]
    nodes_pkg = types.ModuleType("solidbg_unmult_test.nodes")
    nodes_pkg.__path__ = [str(root / "nodes")]
    detect_pkg = types.ModuleType("solidbg_unmult_test.nodes.detect")
    detect_pkg.__path__ = [str(root / "nodes" / "detect")]
    image_pkg = types.ModuleType("solidbg_unmult_test.nodes.image")
    image_pkg.__path__ = [str(root / "nodes" / "image")]
    sys.modules.update(
        {
            "solidbg_unmult_test": package,
            "solidbg_unmult_test.nodes": nodes_pkg,
            "solidbg_unmult_test.nodes.detect": detect_pkg,
            "solidbg_unmult_test.nodes.image": image_pkg,
        }
    )
    module = importlib.import_module(
        "solidbg_unmult_test.nodes.detect.detect_remove_solidbg_unmult"
    )
    return module.DetectRemoveSolidBGUnMult


def test_negative_channel_keeps_positive_ratios():
    """A severe negative channel must not revert the complete pixel."""
    node = _load_node()
    observed = torch.tensor([0.8, 0.05, 0.3])
    premultiplied = torch.tensor([0.64, -0.11, 0.14])
    output = node._gamut_safe_unpremultiply(
        premultiplied, torch.tensor(0.2), torch.tensor(True)
    )
    # Raw foreground is [3.2, -0.55, 0.7]; the fix keeps the positive ratios
    # instead of returning the observed [0.8, 0.05, 0.3].
    torch.testing.assert_close(output, torch.tensor([1.0, 0.0, 0.21875]))
    assert not torch.allclose(output, observed)


def test_inactive_pixels_are_zeroed():
    node = _load_node()
    raw = torch.tensor([[0.5, -0.25, 0.75]])
    output = node._gamut_safe_unpremultiply(
        raw, torch.tensor([1.0]), torch.tensor([False])
    )
    torch.testing.assert_close(output, torch.zeros_like(output))


def test_execute_does_not_revert_pixel_on_negative_channel():
    """End-to-end auto-background path exposes the same negative channel."""
    node = _load_node()
    image = torch.full((1, 64, 64, 3), 0.2)
    # background 0.2, foreground [0.4, 0.5, 0.0] -> alpha 0.375 with a
    # premultiplied blue of -0.125 (raw -1/3), the former fallback trigger.
    image[:, 20:44, 20:44] = torch.tensor([0.4, 0.5, 0.0])

    rgba, alpha = asyncio.run(node.execute(image, threshold=0, background_color=""))

    torch.testing.assert_close(alpha[0, 30, 30], torch.tensor(0.375))
    torch.testing.assert_close(
        rgba[0, 30, 30, :3], torch.tensor([0.73333333, 1.0, 0.0])
    )
    torch.testing.assert_close(rgba[..., 3], alpha)


if __name__ == "__main__":
    import unittest

    unittest.main()
