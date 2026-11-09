"""Tests for the shared paired-pixel colour fit kernel."""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


CF = _load(ROOT / "utils" / "_color_fit.py", "color_fit_under_test")


def _pair(seed=5, height=160, width=192):
    rng = np.random.default_rng(seed)
    reference = rng.uniform(0.15, 0.85, (height, width, 3)).astype(np.float32)
    return reference


def _rh_core():
    """Load the RH-Nodes reference core as an oracle when it is installed."""
    candidates = [
        os.environ.get("RH_NODES_CORE", ""),
        str(ROOT.parent / "ComfyUI-RH-Nodes" / "robust_masked_color_match" / "core.py"),
        r"C:\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-RH-Nodes\robust_masked_color_match\core.py",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return _load(Path(candidate), "rh_match_core")
    pytest.skip("RH-Nodes reference core not available")


# --------------------------------------------------------------------------- #
#  parity with the reference implementation
# --------------------------------------------------------------------------- #
def test_affine_fit_matches_the_reference_implementation():
    rh = _rh_core()
    reference = _pair()
    height, width = reference.shape[:2]
    matrix = np.array(
        [[0.97, 0.02, 0.01], [0.01, 1.02, -0.01], [0.01, -0.02, 1.03]], np.float32
    )
    offset = np.array([-0.01, 0.02, 0.01], np.float32)
    target = np.clip(reference @ matrix.T + offset, 0.0, 1.0).astype(np.float32)
    mask = np.zeros((height, width), np.float32)
    mask[50:120, 60:140] = 1.0
    target = target.copy()
    target[50:120, 60:140] = np.random.default_rng(9).uniform(
        0.0, 1.0, target[50:120, 60:140].shape
    ).astype(np.float32)

    expected, _train, report = rh.match_frame(
        reference, target, mask, exclude_expand=8, iterations=12, outlier_tolerance=3.0
    )
    valid = CF.sampling_mask(
        target, reference, exclude=mask > 0.5, expand=8, use_structure_gate=False
    )
    fitted = CF.fit(
        target,
        reference,
        valid,
        model="affine",
        max_gain=None,
        max_offset=None,
        iterations=12,
        refuse_on_holdout=False,
        work_long_side=None,
    )

    assert fitted["ok"] is True
    assert fitted["model"] == "affine"
    assert fitted["fit_samples"] == report["fit_samples"]
    coefficients = np.asarray(report["coefficients_rgb_rowvector"])
    assert np.allclose(fitted["matrix"], coefficients[:3].T, atol=1e-12)
    assert np.allclose(fitted["offset"], coefficients[3], atol=1e-12)
    produced = CF.apply(target, fitted["matrix"], fitted["offset"])
    assert np.max(np.abs(produced - expected)) < 1e-6


# --------------------------------------------------------------------------- #
#  model behaviour
# --------------------------------------------------------------------------- #
def test_diagonal_model_recovers_a_known_drift():
    reference = _pair()
    target = np.clip(reference * 1.07 + 0.03, 0.0, 1.0).astype(np.float32)

    fitted = CF.fit(
        target,
        reference,
        CF.sampling_mask(target, reference),
        model="diagonal",
        max_gain=None,
        max_offset=None,
        work_long_side=None,
    )

    assert fitted["ok"] is True
    assert np.allclose(np.diag(fitted["matrix"]) - 1.0, 1.0 / 1.07 - 1.0, atol=2e-3)
    assert np.allclose(fitted["offset"], -0.03 / 1.07, atol=2e-3)


def test_affine_falls_back_when_the_background_is_degenerate():
    reference = np.full((96, 96, 3), 0.5, dtype=np.float32)
    target = np.clip(reference * 1.08 + 0.03, 0.0, 1.0).astype(np.float32)

    fitted = CF.fit(
        target,
        reference,
        CF.sampling_mask(target, reference, use_structure_gate=False),
        model="affine",
        work_long_side=None,
    )

    assert fitted["model"] == "diagonal"
    assert any("affine not identifiable" in w for w in fitted["warnings"])


def test_caps_bound_matrix_and_offset():
    reference = _pair()
    target = np.clip(reference * 1.8 + 0.2, 0.0, 1.0).astype(np.float32)

    fitted = CF.fit(
        target,
        reference,
        # A drift this large fails the structure gate, which is the point of the
        # gate; disable it here so the caps themselves are what gets exercised.
        CF.sampling_mask(target, reference, use_structure_gate=False),
        model="diagonal",
        max_gain=0.1,
        max_offset=0.05,
        refuse_on_holdout=False,
        work_long_side=None,
    )

    deviation = np.abs(fitted["matrix"] - CF.identity())
    assert float(deviation.max()) <= 0.1 + 1e-9
    assert float(np.abs(fitted["offset"]).max()) <= 0.05 + 1e-9


def test_too_few_samples_is_reported_not_raised():
    reference = _pair()
    target = reference.copy()

    fitted = CF.fit(target, reference, np.zeros(reference.shape[:2], bool))

    assert fitted["ok"] is False
    assert "too few samples" in fitted["reason"]


# --------------------------------------------------------------------------- #
#  sampling gate
# --------------------------------------------------------------------------- #
def test_structure_gate_rejects_repainted_content():
    reference = _pair()
    target = reference.copy()
    target[40:110, 50:130] = np.array([0.9, 0.15, 0.2], dtype=np.float32)

    gated = CF.sampling_mask(target, reference)
    ungated = CF.sampling_mask(target, reference, use_structure_gate=False)

    assert float(gated[40:110, 50:130].mean()) < 0.02
    assert float(ungated[40:110, 50:130].mean()) == 1.0
    assert float(gated.mean()) > 0.5


def test_alpha_excludes_transparent_pixels():
    reference = _pair()
    alpha = np.ones(reference.shape[:2], np.float32)
    alpha[:, :60] = 0.0

    valid = CF.sampling_mask(reference, reference, alpha=alpha)

    assert float(valid[:, :60].max()) == 0.0
    assert float(valid[:, 60:].min()) == 1.0


def test_exclusion_expands_by_distance():
    reference = _pair(height=96, width=96)
    exclude = np.zeros((96, 96), bool)
    exclude[40:56, 40:56] = True

    tight = CF.sampling_mask(reference, reference, exclude=exclude, expand=0)
    loose = CF.sampling_mask(reference, reference, exclude=exclude, expand=8)

    assert int(np.count_nonzero(~tight)) == int(np.count_nonzero(exclude))
    # A Euclidean grow is round: the pixels lost at a corner exceed the square's.
    assert int(np.count_nonzero(~loose)) > int(np.count_nonzero(~tight))
    assert loose[48, 20] and tight[48, 20]


def test_border_inner_margin_is_excluded():
    reference = _pair(height=64, width=64)
    valid = CF.sampling_mask(reference, reference, border=2)

    assert not valid[:2].any() and not valid[-2:].any()
    assert not valid[:, :2].any() and not valid[:, -2:].any()


# --------------------------------------------------------------------------- #
#  validation and application
# --------------------------------------------------------------------------- #
def test_holdout_rejects_a_fit_that_only_helps_the_training_half():
    reference = _pair()
    rows, cols = np.indices(reference.shape[:2])
    even = ((cols // 32) + (rows // 32)) % 2 == 0
    target = reference.copy()
    target[even] = np.clip(reference[even] * 1.08 + 0.04, 0.0, 1.0)

    fitted = CF.fit(
        target,
        reference,
        CF.sampling_mask(target, reference, use_structure_gate=False),
        model="diagonal",
        work_long_side=None,
    )

    assert fitted["ok"] is False
    assert "held-out" in fitted["reason"]
    assert fitted["holdout_after"] > fitted["holdout_before"]


def test_strength_zero_is_a_no_op():
    reference = _pair()
    target = np.clip(reference * 1.05 + 0.02, 0.0, 1.0).astype(np.float32)
    fitted = CF.fit(target, reference, CF.sampling_mask(target, reference), work_long_side=None)

    untouched = CF.apply(target, fitted["matrix"], fitted["offset"], strength=0.0)

    assert np.array_equal(untouched, target)


def test_patch_print_matches_reference_drift_better_than_nothing():
    reference = _pair()
    target = np.clip(reference * 1.05 + 0.02, 0.0, 1.0).astype(np.float32)
    valid = CF.sampling_mask(target, reference)

    fitted = CF.fit(target, reference, valid, model="diagonal", work_long_side=None)
    corrected = CF.apply(target, fitted["matrix"], fitted["offset"])

    assert np.abs(corrected - reference).mean() < np.abs(target - reference).mean() * 0.05
