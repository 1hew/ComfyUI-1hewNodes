"""Shared paired-pixel robust RGB colour fit.

One implementation of the "background drift" model that several nodes need:
estimate a global RGB correction from pixels that are known to correspond,
validate it on held-out pixels, and bound how far it may move.

The estimator is ported from ComfyUI-RH-Nodes' Robust Masked Color Match core
(MIT licensed, Copyright (c) 2026 Color Correction Near contributors): 3x3
affine, identifiability guard, checkerboard holdout.  A parity test in
tests/test_color_fit_kernel.py pins our coefficients to that implementation's
on identical input.  Three additions matter for crop/paste workflows:

* a structure gate, so the fit still works when the caller cannot supply a mask;
* hard caps on how far the matrix and offset may move, so a fit can never undo
  an intentional repaint;
* the option to reject a fit that fails validation instead of only warning.

Everything is NumPy float64 internally; callers pass and receive float32 RGB in
[0, 1].
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

# A pixel below this alpha carries no usable colour.
ALPHA_OPAQUE = 0.999

# Structure gate defaults: tolerate a global gain (detail scales with it) but not
# an inserted or removed edge (whose detail is unrelated).
DETAIL_SIGMA = 1.5
LOCAL_TOL = 0.02
LOCAL_RELATIVE = 0.5
STRUCTURE_MIN = 0.8
COLOR_TOL = 0.25
GRAY_TOL = 0.20

# Robust fit defaults, matching the reference implementation.
IRLS_ITERATIONS = 12
OUTLIER_TOLERANCE = 3.0
HOLDOUT_TILE = 32
MIN_SAMPLES = 100
MIN_HOLDOUT = 64
MAX_SAMPLES = 100000
AFFINE_COND_MAX = 1e5

# Validation slack: a fit may be slightly worse before it is rejected.
HOLDOUT_MARGIN = 1.2
HOLDOUT_SLACK = 0.25 / 255.0

IDENTITY = np.eye(3, dtype=np.float64)


def identity() -> np.ndarray:
    return IDENTITY.copy()


def expand_exclusion(mask: np.ndarray, pixels: int) -> np.ndarray:
    """Grow a boolean exclusion mask by a Euclidean distance, not a square."""
    if pixels <= 0:
        return mask.astype(bool, copy=False)
    return distance_transform_edt(~mask.astype(bool)) <= float(pixels)


def illumination_gate(
    edit: np.ndarray,
    reference: np.ndarray,
    color_tol: float = COLOR_TOL,
    gray_tol: float = GRAY_TOL,
    structure_min: float = STRUCTURE_MIN,
) -> np.ndarray:
    """Pixels whose content still matches well enough to be a colour reference."""
    structure = structure_agreement(edit, reference)
    edit_gray = cv2.cvtColor(edit, cv2.COLOR_RGB2GRAY)
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
    close = np.max(np.abs(edit - reference), axis=2) <= color_tol
    close &= np.abs(edit_gray - ref_gray) <= gray_tol
    close &= structure >= structure_min
    kernel = np.ones((3, 3), np.uint8)
    close = cv2.morphologyEx(close.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    close = cv2.erode(close, kernel)
    return close.astype(bool)


def structure_agreement(edit: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """1 where local high-frequency structure agrees, 0 where it diverges."""
    edit_gray = cv2.cvtColor(edit, cv2.COLOR_RGB2GRAY)
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
    edit_detail = edit_gray - cv2.GaussianBlur(edit_gray, (0, 0), DETAIL_SIGMA)
    ref_detail = ref_gray - cv2.GaussianBlur(ref_gray, (0, 0), DETAIL_SIGMA)
    scale = LOCAL_TOL + LOCAL_RELATIVE * np.maximum(np.abs(edit_detail), np.abs(ref_detail))
    return 1.0 / (1.0 + ((edit_detail - ref_detail) / scale) ** 2)


def sampling_mask(
    edit: np.ndarray,
    reference: np.ndarray,
    *,
    exclude: np.ndarray | None = None,
    alpha: np.ndarray | None = None,
    expand: int = 0,
    border: int = 0,
    use_structure_gate: bool = True,
    color_tol: float = COLOR_TOL,
    gray_tol: float = GRAY_TOL,
    structure_min: float = STRUCTURE_MIN,
) -> np.ndarray:
    """Pixels eligible to vote on the fit.

    exclude is the region that must not vote (an inpaint mask, or ~background);
    it is grown with a Euclidean distance so repaint bleed is dropped as well.
    alpha, when given, removes transparent pixels.  The structure gate runs
    unless the caller explicitly disables it.
    """
    height, width = reference.shape[:2]
    if use_structure_gate:
        valid = illumination_gate(edit, reference, color_tol, gray_tol, structure_min)
    else:
        valid = np.ones((height, width), dtype=bool)
    if exclude is not None:
        blocked = np.asarray(exclude, dtype=bool)
        if blocked.shape[:2] != (height, width):
            blocked = cv2.resize(blocked.astype(np.uint8), (width, height),
                                 interpolation=cv2.INTER_NEAREST).astype(bool)
        if expand > 0:
            valid &= ~expand_exclusion(blocked, expand)
        else:
            valid &= ~blocked
    if alpha is not None:
        plane = np.asarray(alpha, dtype=np.float32)
        if plane.ndim == 3:
            plane = plane[..., 0]
        if plane.shape[:2] != (height, width):
            plane = cv2.resize(plane, (width, height), interpolation=cv2.INTER_AREA)
        valid &= plane >= ALPHA_OPAQUE
    if border > 0:
        valid[:border] = False
        valid[-border:] = False
        valid[:, :border] = False
        valid[:, -border:] = False
    return valid


def fit(
    target: np.ndarray,
    reference: np.ndarray,
    valid: np.ndarray,
    *,
    model: str = "diagonal",
    max_gain: float | None = 0.1,
    max_offset: float | None = 0.1,
    iterations: int = IRLS_ITERATIONS,
    outlier_tolerance: float = OUTLIER_TOLERANCE,
    holdout_tile: int = HOLDOUT_TILE,
    refuse_on_holdout: bool = True,
    work_long_side: int | None = None,
    max_samples: int = MAX_SAMPLES,
    min_samples: int = MIN_SAMPLES,
) -> dict:
    """Estimate the correction that maps target onto reference.

    Returns a dict with ok/reason/model/matrix/offset plus sampling statistics and
    warnings.  matrix is 3x3 float64, offset is length 3; both are safe to use
    only when ok is True.
    """
    if model not in ("diagonal", "affine"):
        raise ValueError("model must be 'diagonal' or 'affine'")
    target = np.ascontiguousarray(target, dtype=np.float32)
    reference = np.ascontiguousarray(reference, dtype=np.float32)
    if target.shape != reference.shape or target.shape[-1] != 3:
        raise ValueError("target and reference must be aligned [H,W,3] RGB arrays")

    result = {
        "ok": False,
        "reason": "",
        "model": model,
        "matrix": None,
        "offset": None,
        "samples": 0.0,
        "train": 0.0,
        "holdout": 0.0,
        "holdout_before": None,
        "holdout_after": None,
        "iterations": 0,
        "fit_samples": 0,
        "warnings": [],
    }

    height, width = reference.shape[:2]
    work_target, work_reference, work_valid, scale = _downscale(
        target, reference, np.asarray(valid, dtype=bool), work_long_side
    )
    total = float(work_valid.size) if work_valid.size else 1.0
    count = int(np.count_nonzero(work_valid))
    result["samples"] = count / total
    floor = max(int(min_samples), work_valid.size * 0.0)
    if count < floor:
        result["reason"] = "too few samples ({} < {})".format(count, floor)
        return result

    train, holdout = _split(work_valid, holdout_tile)
    if int(np.count_nonzero(train)) < max(MIN_HOLDOUT, min_samples):
        train, holdout = work_valid, np.zeros_like(holdout)
        result["warnings"].append("no independent holdout")
    elif int(np.count_nonzero(holdout)) < MIN_HOLDOUT:
        holdout = np.zeros_like(holdout)
        result["warnings"].append("holdout too small")
    result["train"] = float(np.count_nonzero(train)) / total
    result["holdout"] = float(np.count_nonzero(holdout)) / total

    indices = _subsample(train, max_samples)
    source = work_target.reshape(-1, 3)[indices].astype(np.float64)
    wanted = work_reference.reshape(-1, 3)[indices].astype(np.float64)
    result["fit_samples"] = int(indices.size)

    if model == "affine":
        design = np.c_[source, np.ones(source.shape[0])]
        rank = int(np.linalg.matrix_rank(design))
        condition = float(np.linalg.cond(design))
        if rank < 4 or not np.isfinite(condition) or condition > AFFINE_COND_MAX:
            model = "diagonal"
            result["model"] = model
            result["warnings"].append(
                "affine not identifiable (rank={}, cond={:.0f})".format(
                    rank, condition if np.isfinite(condition) else -1.0
                )
            )

    matrix, offset, iterations_used = _irls(
        source, wanted, model, iterations, outlier_tolerance, min_samples
    )
    result["iterations"] = iterations_used
    if matrix is None:
        result["reason"] = "weights collapsed"
        return result

    matrix, offset = _cap(matrix, offset, max_gain, max_offset)

    if holdout.any():
        indices_h = np.flatnonzero(holdout)
        raw = work_target.reshape(-1, 3)[indices_h].astype(np.float64)
        truth = work_reference.reshape(-1, 3)[indices_h].astype(np.float64)
        corrected = np.clip(raw @ matrix.T + offset, 0.0, 1.0)
        before = float(np.abs(truth - raw).mean())
        after = float(np.abs(truth - corrected).mean())
        result["holdout_before"] = before
        result["holdout_after"] = after
        if refuse_on_holdout and after > before * HOLDOUT_MARGIN + HOLDOUT_SLACK:
            result["reason"] = "held-out pixels got worse ({:.1f} -> {:.1f}/255)".format(
                255.0 * before, 255.0 * after
            )
            return result

    result["ok"] = True
    result["matrix"] = matrix
    result["offset"] = offset
    return result


def apply(
    image: np.ndarray,
    matrix: np.ndarray,
    offset: np.ndarray,
    strength: float = 1.0,
    chunk_rows: int = 256,
) -> np.ndarray:
    """Apply a fitted correction, optionally blended, in bounded memory."""
    effective = (IDENTITY + (np.asarray(matrix) - IDENTITY) * strength).astype(np.float32)
    shift = (np.asarray(offset) * strength).astype(np.float32)
    out = np.empty_like(image, dtype=np.float32)
    for start in range(0, image.shape[0], chunk_rows):
        stop = min(start + chunk_rows, image.shape[0])
        out[start:stop] = np.clip(image[start:stop] @ effective.T + shift, 0.0, 1.0)
    return out


# --------------------------------------------------------------------------- #
#  internals
# --------------------------------------------------------------------------- #
def _downscale(target, reference, valid, work_long_side):
    if not work_long_side:
        return target, reference, valid, 1.0
    height, width = reference.shape[:2]
    scale = min(1.0, float(work_long_side) / max(height, width))
    if scale >= 1.0:
        return target, reference, valid, 1.0
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    work_target = cv2.resize(target, size, interpolation=cv2.INTER_AREA)
    work_reference = cv2.resize(reference, size, interpolation=cv2.INTER_AREA)
    work_valid = cv2.resize(
        valid.astype(np.float32), size, interpolation=cv2.INTER_AREA
    ) > 0.5
    kernel = np.ones((3, 3), np.uint8)
    work_valid = cv2.erode(work_valid.astype(np.uint8), kernel).astype(bool)
    return work_target, work_reference, work_valid, scale


def _split(valid: np.ndarray, tile: int):
    rows, cols = np.indices(valid.shape)
    checker = ((cols // max(1, int(tile))) + (rows // max(1, int(tile)))) % 2 == 0
    return valid & checker, valid & ~checker


def _subsample(mask: np.ndarray, max_samples: int) -> np.ndarray:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return indices
    stride = min(7, max(1, indices.size // 100))
    indices = indices[::stride]
    if indices.size > max_samples:
        indices = indices[np.linspace(0, indices.size - 1, int(max_samples), dtype=np.int64)]
    return indices


def _irls(source, wanted, model, iterations, tolerance, min_samples):
    count = source.shape[0]
    if model == "affine":
        design = np.c_[source, np.ones(count)]
        coefficients = np.linalg.lstsq(design, wanted, rcond=None)[0]
        matrix, offset = coefficients[:3].T.copy(), coefficients[3].copy()
    else:
        matrix = IDENTITY.copy()
        offset = np.median(wanted - source, axis=0)

    used = 0
    for _ in range(max(1, int(iterations))):
        if model == "affine":
            residual = (design @ coefficients) - wanted
        else:
            residual = (source @ matrix.T + offset) - wanted
        magnitude = np.sqrt(np.mean(residual**2, axis=1))
        weights = np.minimum(1.0, (float(tolerance) / 255.0) / np.maximum(magnitude, 1e-8)) ** 0.5
        if int(np.count_nonzero(weights > 0.05)) < max(min_samples, count * 0.1):
            return None, None, used
        if model == "affine":
            scaled = weights[:, None]
            coefficients = np.linalg.lstsq(design * scaled, wanted * scaled, rcond=None)[0]
            matrix, offset = coefficients[:3].T.copy(), coefficients[3].copy()
        else:
            total = float(weights.sum())
            weighted = weights[:, None]
            source_mean = (source * weighted).sum(axis=0) / total
            wanted_mean = (wanted * weighted).sum(axis=0) / total
            variance = ((source - source_mean) ** 2 * weighted).sum(axis=0) / total
            covariance = ((source - source_mean) * (wanted - wanted_mean) * weighted).sum(axis=0) / total
            # A flat background carries no contrast: the gain regresses to 1 and
            # only the offset works, so noise is never amplified.
            gain = (covariance + 1e-4) / (variance + 1e-4)
            offset = wanted_mean - source_mean * gain
            matrix = np.diag(gain)
        used += 1
    return matrix, offset, used


def _cap(matrix, offset, max_gain, max_offset):
    if max_gain is not None:
        matrix = IDENTITY + np.clip(matrix - IDENTITY, -float(max_gain), float(max_gain))
    if max_offset is not None:
        offset = np.clip(offset, -float(max_offset), float(max_offset))
    return matrix, offset
