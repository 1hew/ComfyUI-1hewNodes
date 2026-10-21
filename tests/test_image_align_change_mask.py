"""Regression tests for Image Align Change Mask's optional edit-mask gating."""

import asyncio
import importlib
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch


class _Port:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def _factory(*args, **kwargs):
    return _Port(*args, **kwargs)


def _load_node():
    """Load the node without requiring a running ComfyUI installation."""
    io = types.SimpleNamespace(
        ComfyNode=object,
        Schema=lambda **kwargs: kwargs,
        NodeOutput=lambda *values, **kwargs: values,
        Image=types.SimpleNamespace(Input=_factory, Output=_factory),
        Mask=types.SimpleNamespace(Input=_factory, Output=_factory),
        Combo=types.SimpleNamespace(Input=_factory),
        Int=types.SimpleNamespace(Input=_factory),
        Float=types.SimpleNamespace(Input=_factory),
        Boolean=types.SimpleNamespace(Input=_factory),
    )
    latest = types.ModuleType("comfy_api.latest")
    latest.io = io
    sys.modules.setdefault("comfy_api", types.ModuleType("comfy_api"))
    sys.modules["comfy_api.latest"] = latest

    root = Path(__file__).parents[1]
    package = types.ModuleType("align_mask_test")
    package.__path__ = [str(root)]
    nodes_pkg = types.ModuleType("align_mask_test.nodes")
    nodes_pkg.__path__ = [str(root / "nodes")]
    image_pkg = types.ModuleType("align_mask_test.nodes.image")
    image_pkg.__path__ = [str(root / "nodes" / "image")]
    sys.modules.update(
        {
            "align_mask_test": package,
            "align_mask_test.nodes": nodes_pkg,
            "align_mask_test.nodes.image": image_pkg,
        }
    )
    module = importlib.import_module(
        "align_mask_test.nodes.image.image_align_change_mask"
    )
    return module.ImageAlignChangeMask


def _scene(size=256):
    """Gray base plus one change island inside the mask and one far outside."""
    rng = np.random.default_rng(7)
    base = np.full((size, size, 3), 0.5, dtype=np.float32)
    base += rng.normal(0.0, 0.01, base.shape).astype(np.float32)
    moving = base.copy()
    moving[50:90, 50:90] = 1.0          # inside the edit mask
    moving[180:220, 180:220] = 0.0      # unrelated island
    mask = np.zeros((size, size), dtype=np.float32)
    mask[40:100, 40:100] = 1.0
    return base, moving, mask


def _run(node, base, moving, mask=None):
    return asyncio.run(
        node.execute(
            edit_image=torch.from_numpy(moving),
            org_image=torch.from_numpy(base),
            mode="none",
            max_offset=8,
            max_rotation=2.0,
            max_scale=0.03,
            min_align_score=0.8,
            color_compensation=True,
            sensitivity=8.0,
            min_component_area=32,
            expand=16,
            feather=8,
            mask=None if mask is None else torch.from_numpy(mask),
        )
    )


def test_schema_exposes_optional_mask_without_extra_controls():
    node = _load_node()
    schema = node.define_schema()
    inputs = {port.args[0]: port.kwargs for port in schema["inputs"] if port.args}
    assert "mask" in inputs
    assert inputs["mask"].get("optional") is True
    assert "mask_tolerance" not in inputs
    assert "mask_fill_holes" not in inputs


def test_unrelated_island_is_detected_without_a_mask():
    node = _load_node()
    base, moving, mask = _scene()
    _, _, raw, _, _ = _run(node, base, moving)
    raw = raw[0].numpy()
    assert raw[60:80, 60:80].max() > 0.5
    assert raw[190:210, 190:210].max() > 0.5


def test_mask_suppresses_unrelated_islands():
    node = _load_node()
    base, moving, mask = _scene()
    _, _, raw, _, _ = _run(node, base, moving, mask=mask)
    raw = raw[0].numpy()
    assert raw[60:80, 60:80].max() > 0.5
    assert raw[190:210, 190:210].max() == 0.0


def test_mask_keeps_whole_islands_that_overlap_it():
    node = _load_node()
    base, _, mask = _scene()
    moving = base.copy()
    moving[90:120, 50:90] = 1.0     # overlaps the mask and spills 20 px outside
    moving[140:160, 50:90] = 0.0    # entirely outside the mask
    _, _, raw, _, _ = _run(node, base, moving, mask=mask)
    raw = raw[0].numpy()
    assert raw[95:110, 60:80].max() > 0.5    # overlapping island is kept
    assert raw[105:115, 60:80].max() > 0.5   # ...whole, including the spilled part
    assert raw[145:155, 60:80].max() == 0.0  # unrelated island is dropped


def test_mask_lets_alignment_ignore_the_edit_region():
    node = _load_node()
    size = 256
    rng = np.random.default_rng(3)
    base = cv2.GaussianBlur(rng.random((size, size, 3)).astype(np.float32), (0, 0), 1.5)
    warp = np.float32([[1, 0, 2.0], [0, 1, -3.0]])
    moving = cv2.warpAffine(
        base, warp, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )
    checker = ((np.indices((110, 110)).sum(0) // 6) % 2).astype(np.float32)
    moving[60:170, 60:170] = checker[..., None]
    mask = np.zeros((size, size), dtype=bool)
    mask[50:180, 50:180] = True

    _, dx_plain, dy_plain, _, _, _, _, _ = node._select_alignment(
        base, moving, "translation", 8, 2.0, 0.03, 0.8, None
    )
    _, dx_masked, dy_masked, _, _, _, mode_masked, _ = node._select_alignment(
        base, moving, "translation", 8, 2.0, 0.03, 0.8, ~mask
    )
    assert mode_masked == "translation"
    assert abs(dx_masked - 2.0) < 0.5
    assert abs(dy_masked + 3.0) < 0.5
    assert abs(dx_plain) + abs(dy_plain) < abs(dx_masked) + abs(dy_masked)


def test_outline_mask_is_literal_and_not_filled():
    """A closed outline uses only the drawn pixels; pre-fill it upstream if needed."""
    node = _load_node()
    size = 128
    rng = np.random.default_rng(4)
    base = np.full((size, size, 3), 0.5, dtype=np.float32)
    base += rng.normal(0.0, 0.01, base.shape).astype(np.float32)
    moving = base.copy()
    moving[50:80, 50:80] = 1.0          # change strictly inside the outline
    filled = np.zeros((size, size), dtype=np.float32)
    filled[30:100, 30:100] = 1.0
    ring = filled - cv2.erode(filled, np.ones((7, 7), np.uint8))

    _, _, raw_filled, _, _ = _run(node, base, moving, mask=filled)
    _, _, raw_ring, _, _ = _run(node, base, moving, mask=ring)
    # the filled mask associates the interior change; the bare outline does not
    assert raw_filled[0].numpy()[60:70, 60:70].max() > 0.5
    assert raw_ring[0].numpy()[60:70, 60:70].max() == 0.0

    # the outline still associates a change drawn on the ring itself
    on_ring = base.copy()
    on_ring[32:36, 40:90] = 1.0
    _, _, raw_on_ring, _, _ = _run(node, base, on_ring, mask=ring)
    assert raw_on_ring[0].numpy()[32:36, 45:85].max() > 0.5


def test_mask_to_background_falls_back_when_unusable():
    node = _load_node()
    shape = (32, 32)
    assert node._mask_to_background(None, shape) is None
    assert node._mask_to_background(np.zeros(shape, dtype=np.float32), shape) is None
    assert node._mask_to_background(np.ones(shape, dtype=np.float32), shape) is None
    background = node._mask_to_background(np.eye(32, dtype=np.float32), shape)
    assert background is not None and background[0, 0] is np.False_


def test_mask_batch_validates_shape():
    node = _load_node()
    assert node._mask_batch(torch.zeros(2, 8, 8), 2).shape == (2, 8, 8)
    assert node._mask_batch(torch.zeros(1, 1, 8, 8), 4).shape == (1, 8, 8)
    assert node._mask_batch(torch.zeros(8, 8), 1).shape == (1, 8, 8)
    try:
        node._mask_batch(torch.zeros(3, 8, 8), 2)
    except ValueError:
        pass
    else:
        raise AssertionError("expected a batch-size ValueError")
