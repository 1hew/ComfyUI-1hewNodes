"""Regression tests for Image Alpha Edge's alpha-correct filtering."""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


def _load_node():
    """Load the helper without requiring a running ComfyUI installation."""
    io = types.SimpleNamespace(ComfyNode=object)
    latest = types.ModuleType("comfy_api.latest")
    latest.io = io
    sys.modules.setdefault("comfy_api", types.ModuleType("comfy_api"))
    sys.modules["comfy_api.latest"] = latest
    source = Path(__file__).parents[1] / "nodes" / "image" / "image_alpha_edge.py"
    spec = importlib.util.spec_from_file_location("image_alpha_edge_under_test", source)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module.ImageAlphaEdge


def test_noop_is_a_lossless_pass_through():
    node = _load_node()
    frame = np.random.default_rng(1).random((11, 13, 4), dtype=np.float32)

    output, alpha = node._process_single(frame, edge=0, feather=0.0)

    np.testing.assert_array_equal(output, frame)
    np.testing.assert_array_equal(alpha, frame[:, :, 3])


def test_fine_translucent_details_do_not_form_nearest_colour_facets():
    """The former EDT bleed rewrote this blue detail with nearby red pixels."""
    node = _load_node()
    frame = np.zeros((15, 15, 4), dtype=np.float32)
    frame[7, 3:12, :3] = (1.0, 0.0, 0.0)
    frame[7, 3:12, 3] = 1.0
    frame[4:11, 7, :3] = (0.0, 0.0, 1.0)
    frame[4:11, 7, 3] = 0.35

    output, alpha = node._process_single(frame, edge=1, feather=1.0)

    # The blue translucent stroke remains blue at its centre; it is not replaced
    # by a single nearest opaque red seed as in the Voronoi artifact.
    assert output[5, 7, 2] > output[5, 7, 0]
    assert alpha.min() >= 0.0 and alpha.max() <= 1.0


def test_feather_filters_associated_rgb_with_alpha():
    """Hidden RGB in fully transparent pixels must not leak into the edge."""
    node = _load_node()
    frame = np.zeros((15, 15, 4), dtype=np.float32)
    frame[:, :, :3] = (0.0, 1.0, 0.0)  # arbitrary transparent-pixel RGB
    frame[7, 7, :3] = (1.0, 0.0, 0.0)
    frame[7, 7, 3] = 1.0

    output, alpha = node._process_single(frame, edge=0, feather=1.0)

    assert alpha[7, 5] > 1.0e-6
    assert output[7, 5, 0] > 0.99
    assert output[7, 5, 1] < 1.0e-5


def test_large_feather_extends_colour_through_its_visible_alpha_band():
    """A wide feather must not expose arbitrary RGB in transparent source pixels."""
    node = _load_node()
    frame = np.zeros((101, 101, 4), dtype=np.float32)
    frame[:, :, :3] = (0.0, 1.0, 0.0)  # arbitrary hidden RGB
    frame[50, 50, :3] = (1.0, 0.0, 0.0)
    frame[50, 50, 3] = 1.0

    output, alpha = node._process_single(frame, edge=0, feather=8.0)

    # This pixel is visibly covered by the 8 px alpha feather. It needs the
    # subject's red, not the transparent input's unrelated green RGB.
    assert alpha[50, 60] > 1.0e-3
    assert output[50, 60, 0] > 0.99
    assert output[50, 60, 1] < 1.0e-5


def test_trimmed_contaminated_edge_does_not_donate_colour_to_feather():
    """Negative edge removes a green halo before its RGB can be extended."""
    node = _load_node()
    frame = np.zeros((101, 101, 4), dtype=np.float32)
    frame[:, :, :3] = (0.0, 0.0, 1.0)  # arbitrary fully transparent RGB
    frame[:, :50, :3] = (1.0, 0.0, 0.0)
    frame[:, :50, 3] = 1.0
    frame[:, 50:56, :3] = (0.0, 1.0, 0.0)  # old-background-contaminated halo
    frame[:, 50:56, 3] = 0.4

    output, alpha = node._process_single(frame, edge=-6, feather=3.0)

    # x=56 was fully transparent. The preceding halo was excluded by erosion,
    # so only the red retained core may colour this reconstructed feather band.
    assert alpha[50, 56] > 1.0e-3
    assert output[50, 56, 0] > 0.99
    assert output[50, 56, 1] < 1.0e-5
