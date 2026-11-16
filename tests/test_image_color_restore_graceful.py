"""Image Color Restore must degrade instead of failing the whole workflow.

A refinement pass must never drop a usable matting result: when the input cannot
support a trustworthy restoration (full-bleed subject, edge-cropped subject,
tiny subject, dirty background, unreliable fit), the node passes the AI edit
through unchanged with an empty mask.
"""

import asyncio

import numpy as np
import pytest
import torch

from _node_loader import load_node

MODULE = load_node("ComfyUI_1hewNodes", "nodes/color/image_color_restore.py", ["color"])


def _framed(size=1024, margin=40):
    image = np.zeros((size, size, 3), dtype=np.float64)
    image[margin : size - margin, margin : size - margin] = (0.9, 0.2, 0.2)
    return image


def test_segmentation_reports_unavailable_for_full_bleed_subject():
    with pytest.raises(MODULE._RestoreUnavailable):
        MODULE.segment_solid_background(_framed())


def test_restore_passes_edit_through_when_restoration_is_unavailable():
    edited = _framed()

    output, foreground, unchanged, report = MODULE.restore_reference_colors(
        edited, edited.copy()
    )

    assert np.allclose(output, edited)
    assert not foreground.any()
    assert not unchanged.any()
    assert report.foreground_fraction == 0.0


def test_execute_returns_the_edit_image_instead_of_raising():
    edited = torch.from_numpy(_framed()).float()[None]

    result = asyncio.run(
        MODULE.ImageColorRestore.execute(
            edited,
            edited.clone(),
            background_threshold=0.0,
            unchanged_threshold=0.0,
            seam_max_distance=96.0,
            seam_decay=64.0,
            seam_residual_blur=24.0,
            seam_color_sigma=20.0,
            occlusion_residual_blur=48.0,
            occlusion_bridge_width=48.0,
        )
    )

    image, mask = result
    assert image.shape == edited.shape
    assert torch.allclose(image, edited)
    assert float(mask.abs().sum()) == 0.0
