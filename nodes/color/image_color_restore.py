"""Image Color Restore - restore an aligned AI edit's colors to its original.

Pure OpenCV/NumPy reference-aware color restoration with local occlusion-seam
handling, ported from ComfyUI-RH-Nodes' 'Reference Color Restore (Occlusion
Seam Advanced) V0.8'.  This variant keeps the trusted affine color fit and the
continuous seam field, with multi-scale structural unchanged-mask cleanup
disabled for direct comparison.

Hard requirements:
- edit_image and org_image must be pixel-aligned; only a few pixels of drift
  are tolerated.  edit_image is resized to org_image's size when they differ.
- edit_image must place the subject on an approximately solid background, or
  foreground segmentation (and therefore the whole fit) fails.

The pure algorithm core is inlined below (ported
byte-for-byte from the original algorithm.py; retains the original MIT
attribution).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from comfy_api.latest import io

# ============================================================================
# Pure algorithm core. Ported byte-for-byte from ComfyUI-RH-Nodes algorithm.py
# (original MIT license retained). Independent of ComfyUI for standalone tests.
# ============================================================================
@dataclass(frozen=True)
class RestorationReport:
    background_rgb: tuple[int, int, int]
    foreground_fraction: float
    unchanged_fraction: float
    threshold: float
    before_mae: float
    mapped_mae: float
    structural_rescued_fraction: float
    hard_restored_fraction: float
    seam_compensated_fraction: float
    occlusion_core_fraction: float = 0.0
    occlusion_zone_fraction: float = 0.0
    occlusion_seam_compensated_fraction: float = 0.0
    component_handoff_fraction: float = 0.0


def _features(rgb: np.ndarray, quadratic: bool) -> np.ndarray:
    """Return [1, r, g, b, r2, g2, b2, rg, rb, gb]."""
    r, g, b = rgb.T
    columns = [np.ones(len(rgb)), r, g, b]
    if quadratic:
        columns.extend([r * r, g * g, b * b, r * g, r * b, g * b])
    return np.column_stack(columns)


def _solve_weighted(
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    quadratic: bool,
    ridge: float,
) -> np.ndarray:
    design = _features(source, quadratic)
    sqrt_w = np.sqrt(np.maximum(weights, 0.0))[:, None]
    a = design * sqrt_w
    y = target * sqrt_w
    regularizer = np.eye(design.shape[1])
    regularizer[0, 0] = 0.0
    identity = np.zeros((design.shape[1], 3))
    identity[1:4, :] = np.eye(3)
    return np.linalg.solve(
        a.T @ a + ridge * regularizer,
        a.T @ y + ridge * regularizer @ identity,
    )


def _robust_fit(
    source: np.ndarray,
    target: np.ndarray,
    quadratic: bool,
    base_weights: np.ndarray | None = None,
    iterations: int = 8,
) -> np.ndarray:
    """Fit with Tukey IRLS so genuinely changed pixels receive zero weight."""
    if base_weights is None:
        base_weights = np.ones(len(source), dtype=np.float64)
    weights = base_weights.copy()
    ridge = 10.0 if quadratic else 1.0
    transform = np.zeros((10 if quadratic else 4, 3))
    for _ in range(iterations):
        transform = _solve_weighted(source, target, weights, quadratic, ridge)
        residual = np.linalg.norm(
            _features(source, quadratic) @ transform - target, axis=1
        )
        residual *= 255.0
        active = residual[weights > 0.05 * np.mean(base_weights)]
        if len(active) == 0:
            raise RuntimeError("Robust fitting rejected every sample")
        median = float(np.median(active))
        mad = 1.4826 * float(np.median(np.abs(active - median)))
        scale = max(2.0, median + 2.0 * mad)
        u = residual / (4.685 * scale)
        robust = np.where(u < 1.0, (1.0 - u * u) ** 2, 0.0)
        weights = base_weights * robust
    return transform


def _apply_transform(
    image: np.ndarray, transform: np.ndarray, quadratic: bool
) -> np.ndarray:
    height, width, _ = image.shape
    result = np.empty_like(image)
    rows_per_strip = max(1, 1_000_000 // width)
    for y0 in range(0, height, rows_per_strip):
        y1 = min(height, y0 + rows_per_strip)
        pixels = image[y0:y1].reshape(-1, 3)
        result[y0:y1] = (_features(pixels, quadratic) @ transform).reshape(
            y1 - y0, width, 3
        )
    return np.clip(result, 0.0, 1.0)


def _uniform_sample(
    source: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    max_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = source.shape[:2]
    stride = max(1, math.ceil(math.sqrt(height * width / max_samples)))
    sampled_mask = mask[::stride, ::stride]
    return source[::stride, ::stride][sampled_mask], target[::stride, ::stride][
        sampled_mask
    ]


def _balanced_color_weights(samples: np.ndarray, bins: int = 8) -> np.ndarray:
    index = np.clip((samples * bins).astype(np.int32), 0, bins - 1)
    flat = (index[:, 0] * bins + index[:, 1]) * bins + index[:, 2]
    counts = np.bincount(flat, minlength=bins**3)
    weights = 1.0 / np.sqrt(counts[flat] + 1.0)
    weights /= np.mean(weights)
    return np.clip(weights, 0.25, 4.0)


def _fill_binary_holes(mask: np.ndarray) -> np.ndarray:
    inverse = (~mask).astype(np.uint8)
    count, labels = cv2.connectedComponents(inverse, 8)
    if count <= 1:
        return mask
    border_labels = np.unique(
        np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    )
    border_labels = border_labels[border_labels != 0]
    exterior_background = np.isin(labels, border_labels)
    enclosed_background = (inverse > 0) & ~exterior_background
    return mask | enclosed_background


def segment_solid_background(
    image: np.ndarray,
    threshold: float | None = None,
    border_width: int = 20,
    min_component_fraction: float = 0.0005,
) -> tuple[np.ndarray, tuple[int, int, int], float]:
    """Find sizeable foreground islands on an approximately solid background."""
    u8_rgb = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
    height, width = image.shape[:2]
    border_width = max(1, min(border_width, height // 4, width // 4))
    border = np.concatenate(
        [
            u8_rgb[:border_width].reshape(-1, 3),
            u8_rgb[-border_width:].reshape(-1, 3),
            u8_rgb[:, :border_width].reshape(-1, 3),
            u8_rgb[:, -border_width:].reshape(-1, 3),
        ]
    )
    background = np.median(border, axis=0).astype(np.uint8)
    lab = cv2.cvtColor(u8_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    background_lab = cv2.cvtColor(background[None, None, :], cv2.COLOR_RGB2LAB)[0, 0]
    distance = np.linalg.norm(lab - background_lab, axis=2)
    if threshold is None:
        border_distance = np.concatenate(
            [
                distance[:border_width].ravel(),
                distance[-border_width:].ravel(),
                distance[:, :border_width].ravel(),
                distance[:, -border_width:].ravel(),
            ]
        )
        quiet = border_distance[border_distance <= np.quantile(border_distance, 0.70)]
        median = float(np.median(quiet))
        mad = 1.4826 * float(np.median(np.abs(quiet - median)))
        threshold = float(np.clip(max(8.0, median + 6.0 * mad), 8.0, 20.0))

    raw = (distance > threshold).astype(np.uint8) * 255
    raw = cv2.morphologyEx(
        raw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
    foreground = np.zeros((height, width), dtype=bool)
    minimum_area = max(100, int(height * width * min_component_fraction))
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= minimum_area:
            foreground |= labels == label
    foreground = _fill_binary_holes(foreground)
    fraction = float(np.mean(foreground))
    if fraction < 0.005 or fraction > 0.80:
        raise RuntimeError(
            f"Implausible solid-background foreground fraction ({100*fraction:.2f}%); "
            "adjust the background threshold or verify that AI image is connected first"
        )
    return foreground, tuple(int(x) for x in background), float(threshold)


def _best_local_zncc(
    source: np.ndarray,
    reference: np.ndarray,
    patch_size: int = 13,
    max_shift: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_gray = cv2.cvtColor(source.astype(np.float32), cv2.COLOR_RGB2GRAY) * 255.0
    reference_gray = (
        cv2.cvtColor(reference.astype(np.float32), cv2.COLOR_RGB2GRAY) * 255.0
    )
    source_gray = cv2.GaussianBlur(source_gray, (0, 0), 0.7)
    reference_gray = cv2.GaussianBlur(reference_gray, (0, 0), 0.7)
    mean_source = cv2.boxFilter(
        source_gray,
        -1,
        (patch_size, patch_size),
        normalize=True,
        borderType=cv2.BORDER_REFLECT,
    )
    variance_source = np.maximum(
        cv2.boxFilter(
            source_gray * source_gray,
            -1,
            (patch_size, patch_size),
            normalize=True,
            borderType=cv2.BORDER_REFLECT,
        )
        - mean_source * mean_source,
        1e-3,
    )
    best = np.full(source_gray.shape, -1.0, dtype=np.float32)
    zero_shift = np.full(source_gray.shape, -1.0, dtype=np.float32)
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            matrix = np.float32([[1, 0, dx], [0, 1, dy]])
            shifted = cv2.warpAffine(
                reference_gray,
                matrix,
                (reference_gray.shape[1], reference_gray.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )
            mean_reference = cv2.boxFilter(
                shifted,
                -1,
                (patch_size, patch_size),
                normalize=True,
                borderType=cv2.BORDER_REFLECT,
            )
            variance_reference = np.maximum(
                cv2.boxFilter(
                    shifted * shifted,
                    -1,
                    (patch_size, patch_size),
                    normalize=True,
                    borderType=cv2.BORDER_REFLECT,
                )
                - mean_reference * mean_reference,
                1e-3,
            )
            covariance = (
                cv2.boxFilter(
                    source_gray * shifted,
                    -1,
                    (patch_size, patch_size),
                    normalize=True,
                    borderType=cv2.BORDER_REFLECT,
                )
                - mean_source * mean_reference
            )
            correlation = covariance / np.sqrt(variance_source * variance_reference)
            if dx == 0 and dy == 0:
                zero_shift = np.clip(correlation, -1.0, 1.0)
            best = np.maximum(best, np.clip(correlation, -1.0, 1.0))
    return best, zero_shift, np.sqrt(variance_source)


def _zero_shift_zncc(
    source: np.ndarray,
    reference: np.ndarray,
    patch_size: int,
) -> np.ndarray:
    """Return aligned-patch ZNCC without searching spatial offsets."""
    source_gray = cv2.cvtColor(source.astype(np.float32), cv2.COLOR_RGB2GRAY) * 255.0
    reference_gray = (
        cv2.cvtColor(reference.astype(np.float32), cv2.COLOR_RGB2GRAY) * 255.0
    )
    source_gray = cv2.GaussianBlur(source_gray, (0, 0), 0.7)
    reference_gray = cv2.GaussianBlur(reference_gray, (0, 0), 0.7)
    kernel = (patch_size, patch_size)
    mean_source = cv2.boxFilter(
        source_gray, -1, kernel, normalize=True, borderType=cv2.BORDER_REFLECT
    )
    mean_reference = cv2.boxFilter(
        reference_gray, -1, kernel, normalize=True, borderType=cv2.BORDER_REFLECT
    )
    variance_source = np.maximum(
        cv2.boxFilter(
            source_gray * source_gray,
            -1,
            kernel,
            normalize=True,
            borderType=cv2.BORDER_REFLECT,
        )
        - mean_source * mean_source,
        1e-3,
    )
    variance_reference = np.maximum(
        cv2.boxFilter(
            reference_gray * reference_gray,
            -1,
            kernel,
            normalize=True,
            borderType=cv2.BORDER_REFLECT,
        )
        - mean_reference * mean_reference,
        1e-3,
    )
    covariance = (
        cv2.boxFilter(
            source_gray * reference_gray,
            -1,
            kernel,
            normalize=True,
            borderType=cv2.BORDER_REFLECT,
        )
        - mean_source * mean_reference
    )
    return np.clip(
        covariance / np.sqrt(variance_source * variance_reference), -1.0, 1.0
    )


def _identity_transform(quadratic: bool = True) -> np.ndarray:
    transform = np.zeros((10 if quadratic else 4, 3), dtype=np.float64)
    transform[1:4] = np.eye(3)
    return transform


def _automatic_unchanged_threshold(
    local_rms: np.ndarray, interior: np.ndarray
) -> float:
    values = local_rms[interior]
    quiet = values[values <= np.quantile(values, 0.70)]
    median = float(np.median(quiet))
    mad = 1.4826 * float(np.median(np.abs(quiet - median)))
    return float(np.clip(max(16.0, median + 6.0 * mad), 16.0, 32.0))


def _trusted_affine_fit(
    edited: np.ndarray,
    reference: np.ndarray,
    interior: np.ndarray,
    unchanged_threshold: float | None,
    max_samples: int,
) -> tuple[np.ndarray, float]:
    """Fit only aligned, structurally matching pixels and validate on a holdout."""
    raw_residual = np.max(np.abs(edited - reference), axis=2) * 255.0
    raw_rms = np.sqrt(
        cv2.GaussianBlur(
            (raw_residual * raw_residual).astype(np.float32), (0, 0), 1.5
        )
    )
    threshold = (
        float(unchanged_threshold)
        if unchanged_threshold is not None
        else _automatic_unchanged_threshold(raw_rms, interior)
    )
    initial = interior & (raw_rms < threshold)
    initial = cv2.morphologyEx(
        initial.astype(np.uint8),
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    ).astype(bool)
    _, zero_shift, texture = _best_local_zncc(
        edited, reference, patch_size=13, max_shift=2
    )
    trusted = initial & (zero_shift >= 0.97) & (texture >= 3.0)
    yy, xx = np.indices(trusted.shape)
    validation = trusted & (((yy // 16 + xx // 16) % 4) == 0)
    training = trusted & ~validation
    sample_edited, sample_reference = _uniform_sample(
        edited, reference, training, max_samples
    )
    if len(sample_edited) < 100 or np.sum(validation) < 50:
        return _identity_transform(quadratic=True), threshold
    affine = _robust_fit(
        sample_edited,
        sample_reference,
        quadratic=False,
        base_weights=_balanced_color_weights(sample_edited),
        iterations=10,
    )
    transform = np.zeros((10, 3), dtype=np.float64)
    transform[:4] = affine
    candidate = _apply_transform(edited, transform, quadratic=True)
    before = float(np.mean(np.abs(edited[validation] - reference[validation])))
    after = float(np.mean(np.abs(candidate[validation] - reference[validation])))
    if not np.isfinite(after) or after >= 0.99 * before:
        transform = _identity_transform(quadratic=True)
    return transform, threshold


def _clean_unchanged_by_structure(
    unchanged: np.ndarray,
    mapped: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """Reject color coincidences lacking aligned structure at multiple scales."""
    zncc_13 = _zero_shift_zncc(mapped, reference, 13)
    zncc_25 = _zero_shift_zncc(mapped, reference, 25)
    zncc_41 = _zero_shift_zncc(mapped, reference, 41)
    structural_core = unchanged & (zncc_13 >= 0.80) & (
        (zncc_25 >= 0.50) | (zncc_41 >= 0.50)
    )
    nearby_core = cv2.dilate(
        structural_core.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
    ).astype(bool)
    candidate = unchanged & nearby_core
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate.astype(np.uint8), 8
    )
    cleaned = np.zeros_like(candidate)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= 32:
            cleaned |= labels == label
    return cleaned


_SEAM_HANDOFF_COLLAR_RADIUS = 16
_SEAM_HANDOFF_MIN_COMPONENT_AREA = 96
_SEAM_HANDOFF_MIN_ANCHOR_PIXELS = 24
_SEAM_HANDOFF_RECOVERY_START_WEIGHT = 0.050
_SEAM_HANDOFF_RECOVERY_FULL_WEIGHT = 0.010
_SEAM_HANDOFF_TARGET_DELTA_MIN_RGB = 0.05
_SEAM_HANDOFF_GUIDE_SIGMA_LAB = 12.0
_SEAM_HANDOFF_MIN_GUIDE_LINK = 0.02
_SEAM_HANDOFF_SCREEN_WEIGHT = 0.18
_SEAM_HANDOFF_ITERATIONS = 192
_SEAM_HANDOFF_CONVERGENCE = 0.002 / 255.0


def _smootherstep(values: np.ndarray) -> np.ndarray:
    """Return a C2-continuous ramp over the closed interval [0, 1]."""
    values = np.clip(values, 0.0, 1.0)
    return values**3 * (values * (values * 6.0 - 15.0) + 10.0)


def _neighbor_slices(
    height: int, width: int, dy: int, dx: int
) -> tuple[slice, slice, slice, slice]:
    """Return aligned current/neighbor slices for a pixel offset."""
    return (
        slice(max(0, -dy), height - max(0, dy)),
        slice(max(0, -dx), width - max(0, dx)),
        slice(max(0, dy), height - max(0, -dy)),
        slice(max(0, dx), width - max(0, -dx)),
    )


def _screened_correction(
    baseline: np.ndarray,
    soft_target: np.ndarray,
    domain: np.ndarray,
    anchors: np.ndarray,
    guide_lab: np.ndarray,
) -> np.ndarray:
    """Solve a compact Lab-guided correction field with fixed boundary values.

    ``baseline`` is the established V1 correction, and ``soft_target`` is a
    recovery-only target.  Pixels in ``anchors`` are never updated, so the
    field meets the existing correction continuously at the collar perimeter.
    """
    active = domain | anchors
    ys, xs = np.nonzero(active)
    if not len(ys):
        return baseline
    y0, y1 = max(0, int(ys.min()) - 1), min(
        active.shape[0], int(ys.max()) + 2
    )
    x0, x1 = max(0, int(xs.min()) - 1), min(
        active.shape[1], int(xs.max()) + 2
    )
    crop = np.s_[y0:y1, x0:x1]
    field = baseline[crop].astype(np.float64, copy=True)
    target_roi = soft_target[crop]
    update = domain[crop]
    active_roi = active[crop]
    lab = guide_lab[crop]
    height, width = update.shape
    checkerboard = (np.indices(update.shape).sum(axis=0) + x0 + y0) % 2
    directions = (
        (0, 1, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (-1, 0, 1.0),
        (1, 1, 1.0 / math.sqrt(2.0)),
        (1, -1, 1.0 / math.sqrt(2.0)),
        (-1, 1, 1.0 / math.sqrt(2.0)),
        (-1, -1, 1.0 / math.sqrt(2.0)),
    )
    for _ in range(_SEAM_HANDOFF_ITERATIONS):
        maximum_change = 0.0
        for parity in (0, 1):
            numerator = _SEAM_HANDOFF_SCREEN_WEIGHT * target_roi.copy()
            denominator = np.full(
                update.shape, _SEAM_HANDOFF_SCREEN_WEIGHT, dtype=np.float64
            )
            for dy, dx, spatial_weight in directions:
                dy0, dx0, ny0, nx0 = _neighbor_slices(height, width, dy, dx)
                can_update = (
                    update[dy0, dx0]
                    & (checkerboard[dy0, dx0] == parity)
                    & active_roi[ny0, nx0]
                )
                if not np.any(can_update):
                    continue
                lab_delta = np.linalg.norm(
                    lab[dy0, dx0] - lab[ny0, nx0], axis=2
                )
                links = spatial_weight * np.exp(
                    -0.5 * (lab_delta / _SEAM_HANDOFF_GUIDE_SIGMA_LAB) ** 2
                )
                links[links < _SEAM_HANDOFF_MIN_GUIDE_LINK] = 0.0
                links *= can_update
                numerator[dy0, dx0] += links[:, :, None] * field[ny0, nx0]
                denominator[dy0, dx0] += links
            valid = (
                update
                & (checkerboard == parity)
                & (denominator > _SEAM_HANDOFF_SCREEN_WEIGHT)
            )
            if not np.any(valid):
                continue
            replacement = numerator[valid] / denominator[valid, None]
            maximum_change = max(
                maximum_change,
                float(np.max(np.abs(replacement - field[valid]))),
            )
            field[valid] = replacement
        if maximum_change < _SEAM_HANDOFF_CONVERGENCE:
            break
    result = baseline.copy()
    result[crop][update] = field[update]
    return result


def _repair_continuous_seam_handoff(
    image: np.ndarray,
    baseline_result: np.ndarray,
    base: np.ndarray,
    reference: np.ndarray,
    foreground: np.ndarray,
    unchanged: np.ndarray,
    max_distance: float,
    decay: float,
    residual_blur: float,
    color_sigma: float,
    bridge_width: float,
    silhouette_guard: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Repair a continuous-field compatibility handoff without moving a silhouette.

    The normal V1 field remains the boundary condition.  A nearby-pixel colour
    compatibility estimate is only a soft interior target, avoiding its old
    Voronoi residual lookup.  The foreground silhouette band is a fixed
    Dirichlet boundary, rather than a zero-faded correction band.
    """
    trusted = unchanged.astype(bool)
    target = foreground.astype(bool) & ~trusted
    if not np.any(trusted) or not np.any(target) or silhouette_guard <= 0:
        return baseline_result, np.zeros_like(trusted)

    trusted_float = trusted.astype(np.float32)
    residual = (reference - base).astype(np.float32)
    denominator = cv2.GaussianBlur(
        trusted_float, (0, 0), residual_blur, borderType=cv2.BORDER_REFLECT
    )
    continuous_residual = cv2.GaussianBlur(
        residual * trusted_float[:, :, None],
        (0, 0),
        residual_blur,
        borderType=cv2.BORDER_REFLECT,
    ) / np.maximum(denominator[:, :, None], 1e-5)
    continuous_residual = np.clip(
        continuous_residual, -32.0 / 255.0, 32.0 / 255.0
    )
    continuous_base = cv2.GaussianBlur(
        base.astype(np.float32) * trusted_float[:, :, None],
        (0, 0),
        residual_blur,
        borderType=cv2.BORDER_REFLECT,
    ) / np.maximum(denominator[:, :, None], 1e-5)
    base_lab = cv2.cvtColor(
        np.clip(np.rint(base * 255.0), 0, 255).astype(np.uint8),
        cv2.COLOR_RGB2LAB,
    ).astype(np.float32)
    continuous_lab = cv2.cvtColor(
        np.clip(np.rint(continuous_base * 255.0), 0, 255).astype(np.uint8),
        cv2.COLOR_RGB2LAB,
    ).astype(np.float32)
    continuous_colour = np.exp(
        -0.5
        * (np.linalg.norm(base_lab - continuous_lab, axis=2) / color_sigma) ** 2
    )
    distance = cv2.distanceTransform(
        (~trusted).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    envelope = np.exp(-distance / decay) * np.clip(
        1.0 - (distance / max_distance) ** 2, 0.0, 1.0
    ) ** 2
    continuous_weight = continuous_colour * envelope
    _, labels = cv2.distanceTransformWithLabels(
        (~trusted).astype(np.uint8),
        cv2.DIST_L2,
        cv2.DIST_MASK_5,
        labelType=cv2.DIST_LABEL_PIXEL,
    )
    nearest_lab_lut = np.zeros((int(labels.max()) + 1, 3), dtype=np.float32)
    nearest_lab_lut[labels[trusted]] = base_lab[trusted]
    nearest_lab = nearest_lab_lut[labels]
    nearest_colour = np.exp(
        -0.5 * (np.linalg.norm(base_lab - nearest_lab, axis=2) / color_sigma) ** 2
    )
    nearest_weight = nearest_colour * envelope
    recovery = _smootherstep(
        (_SEAM_HANDOFF_RECOVERY_START_WEIGHT - continuous_weight)
        / (
            _SEAM_HANDOFF_RECOVERY_START_WEIGHT
            - _SEAM_HANDOFF_RECOVERY_FULL_WEIGHT
        )
    )
    hybrid_weight = continuous_weight + recovery * np.maximum(
        nearest_weight - continuous_weight, 0.0
    )

    direct_hybrid = image.copy()
    direct_apply = target & (distance < max_distance) & (hybrid_weight > 1e-4)
    direct_hybrid[direct_apply] += (
        continuous_residual[direct_apply] * hybrid_weight[direct_apply, None]
    )
    trusted_distance = cv2.distanceTransform(
        trusted.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    inner_weight = np.clip(1.0 - trusted_distance / bridge_width, 0.0, 1.0)
    inner_mask = trusted & (inner_weight > 0.0)
    inner_bridge = np.clip(base + continuous_residual, 0.0, 1.0)
    direct_hybrid[inner_mask] = (
        image[inner_mask] * (1.0 - inner_weight[inner_mask, None])
        + inner_bridge[inner_mask] * inner_weight[inner_mask, None]
    )
    direct_hybrid = np.clip(direct_hybrid, 0.0, 1.0)

    direct_delta = direct_hybrid - baseline_result
    seed = (
        target
        & (distance < max_distance - 4.0)
        & (
            np.max(np.abs(direct_delta), axis=2) * 255.0
            > _SEAM_HANDOFF_TARGET_DELTA_MIN_RGB
        )
    )
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        seed.astype(np.uint8), 8
    )
    collar_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * _SEAM_HANDOFF_COLLAR_RADIUS + 1, 2 * _SEAM_HANDOFF_COLLAR_RADIUS + 1),
    )
    one_pixel = np.ones((3, 3), dtype=np.uint8)
    foreground_distance = cv2.distanceTransform(
        foreground.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    silhouette = foreground_distance <= silhouette_guard
    baseline_correction = baseline_result - image
    direct_correction = direct_hybrid - image
    output = baseline_result.copy()
    continuation = np.zeros_like(target)
    for label_index in range(1, component_count):
        area = int(stats[label_index, cv2.CC_STAT_AREA])
        if area < _SEAM_HANDOFF_MIN_COMPONENT_AREA:
            continue
        component = labels == label_index
        proposed_domain = (
            cv2.dilate(component.astype(np.uint8), collar_kernel).astype(bool)
            & target
            & (distance < max_distance)
        )
        domain = proposed_domain & ~silhouette
        if not np.any(domain):
            continue
        anchors = (
            cv2.dilate(domain.astype(np.uint8), one_pixel).astype(bool)
            & ~domain
            & foreground.astype(bool)
        )
        if int(np.sum(anchors)) < _SEAM_HANDOFF_MIN_ANCHOR_PIXELS:
            continue
        solved = _screened_correction(
            baseline_correction, direct_correction, domain, anchors, base_lab
        )
        output[domain] = np.clip(image[domain] + solved[domain], 0.0, 1.0)
        continuation |= domain
    return np.clip(output, 0.0, 1.0), continuation


def _propagate_seam_color(
    image: np.ndarray,
    base: np.ndarray,
    reference: np.ndarray,
    foreground: np.ndarray,
    unchanged: np.ndarray,
    max_distance: float,
    decay: float,
    residual_blur: float,
    color_sigma: float,
    bridge_width: float,
    region: np.ndarray | None = None,
    continuous_field: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    values = (max_distance, decay, residual_blur, color_sigma, bridge_width)
    if any(value <= 0 for value in values):
        raise ValueError("Seam propagation parameters must be positive")
    trusted = unchanged.astype(bool)
    target = foreground.astype(bool) & ~trusted
    if region is not None:
        if region.shape != trusted.shape:
            raise ValueError("Seam propagation region must match image dimensions")
        region = region.astype(bool)
        target &= region
    if not np.any(trusted) or not np.any(target):
        return image, np.zeros_like(trusted)

    trusted_float = trusted.astype(np.float32)
    residual = (reference - base).astype(np.float32)
    denominator = cv2.GaussianBlur(
        trusted_float, (0, 0), residual_blur, borderType=cv2.BORDER_REFLECT
    )
    numerator = cv2.GaussianBlur(
        residual * trusted_float[:, :, None],
        (0, 0),
        residual_blur,
        borderType=cv2.BORDER_REFLECT,
    )
    smooth_residual = numerator / np.maximum(denominator[:, :, None], 1e-5)
    smooth_residual = np.clip(smooth_residual, -32.0 / 255.0, 32.0 / 255.0)

    base_u8 = np.clip(np.rint(base * 255.0), 0, 255).astype(np.uint8)
    base_lab = cv2.cvtColor(base_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    if continuous_field:
        # Sampling a single nearest trusted pixel turns the distance-transform
        # labels into a Voronoi lookup table.  V1 uses normalized Gaussian
        # estimates from all nearby trusted pixels instead, keeping the field
        # continuous across those label boundaries.
        distance = cv2.distanceTransform(
            (~trusted).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        )
        smooth_base = cv2.GaussianBlur(
            base.astype(np.float32) * trusted_float[:, :, None],
            (0, 0),
            residual_blur,
            borderType=cv2.BORDER_REFLECT,
        ) / np.maximum(denominator[:, :, None], 1e-5)
        smooth_base_u8 = np.clip(
            np.rint(smooth_base * 255.0), 0, 255
        ).astype(np.uint8)
        smooth_base_lab = cv2.cvtColor(
            smooth_base_u8, cv2.COLOR_RGB2LAB
        ).astype(np.float32)
        propagated_residual = smooth_residual
        propagated_lab = smooth_base_lab
    else:
        # Kept byte-for-byte equivalent in behavior to the 0805 backup for
        # the ordinary and non-V1 Advanced nodes.
        distance, labels = cv2.distanceTransformWithLabels(
            (~trusted).astype(np.uint8),
            cv2.DIST_L2,
            cv2.DIST_MASK_5,
            labelType=cv2.DIST_LABEL_PIXEL,
        )
        max_label = int(labels.max())
        residual_lut = np.zeros((max_label + 1, 3), dtype=np.float32)
        lab_lut = np.zeros((max_label + 1, 3), dtype=np.float32)
        trusted_labels = labels[trusted]
        residual_lut[trusted_labels] = smooth_residual[trusted]
        lab_lut[trusted_labels] = base_lab[trusted]
        propagated_residual = residual_lut[labels]
        propagated_lab = lab_lut[labels]
    lab_difference = np.linalg.norm(base_lab - propagated_lab, axis=2)
    color_weight = np.exp(-0.5 * (lab_difference / color_sigma) ** 2)
    distance_weight = np.exp(-distance / decay)
    cutoff = np.clip(1.0 - (distance / max_distance) ** 2, 0.0, 1.0) ** 2
    weight = color_weight * distance_weight * cutoff
    apply_mask = target & (distance < max_distance) & (weight > 0.01)

    corrected = image.copy()
    corrected[apply_mask] += (
        propagated_residual[apply_mask] * weight[apply_mask, None]
    )
    trusted_distance = cv2.distanceTransform(
        trusted.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    inner_weight = np.clip(1.0 - trusted_distance / bridge_width, 0.0, 1.0)
    inner_bridge = np.clip(base + smooth_residual, 0.0, 1.0)
    inner_mask = trusted & (inner_weight > 0.0)
    if region is not None:
        inner_mask &= region
    corrected[inner_mask] = (
        image[inner_mask] * (1.0 - inner_weight[inner_mask, None])
        + inner_bridge[inner_mask] * inner_weight[inner_mask, None]
    )
    return np.clip(corrected, 0.0, 1.0), apply_mask | inner_mask


def _handoff_independent_components(
    result: np.ndarray,
    base: np.ndarray,
    reference: np.ndarray,
    foreground: np.ndarray,
    unchanged: np.ndarray,
    reference_weight: np.ndarray,
    core_depth: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Join small, exact-reference islands to the existing seam field.

    The normal global seam pass remains the sole correction outside every
    component.  A qualifying independent trusted component is changed only
    internally: its exact-reference core is kept fixed, its one-pixel inner
    boundary is fixed to the *actual adjacent exterior seam field*, and a
    Lab-edge-aware harmonic interpolation joins the two.  This avoids the
    competing direct-reference/seam mechanisms without extending any new
    field to a foreground silhouette.  It works with either the historical
    nearest-pixel field or V1's continuous field.

    Detection is deliberately conservative.  Components with too little
    geometry, incomplete exterior sampling, non-exact cores, or negligible
    component/exterior residual disagreement are left untouched.
    """
    if core_depth <= 1.0:
        return result, np.zeros_like(unchanged)

    foreground = foreground.astype(bool)
    unchanged = unchanged.astype(bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        unchanged.astype(np.uint8), 8
    )
    foreground_area = max(1, int(np.sum(foreground)))
    min_area = max(64, int(round(0.00010 * foreground_area)))
    max_area = max(2_048, int(round(0.010 * foreground_area)))
    minimum_core_pixels = 16
    minimum_transition_pixels = 64
    mismatch_threshold = 2.0 / 255.0
    iterations = 128
    convergence = 0.002 / 255.0
    base_lab = cv2.cvtColor(
        np.clip(np.rint(base * 255.0), 0, 255).astype(np.uint8),
        cv2.COLOR_RGB2LAB,
    ).astype(np.float64)
    correction = (result - base).astype(np.float64)
    reference_correction = (reference - base).astype(np.float64)
    output = result.copy()
    applied = np.zeros_like(unchanged)
    height, width = unchanged.shape
    neighbors = (
        (0, 1, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (-1, 0, 1.0),
        (1, 1, 1.0 / math.sqrt(2.0)),
        (1, -1, 1.0 / math.sqrt(2.0)),
        (-1, 1, 1.0 / math.sqrt(2.0)),
        (-1, -1, 1.0 / math.sqrt(2.0)),
    )

    for label in range(1, count):
        x, y, component_width, component_height, area = stats[label]
        if area < min_area or area > max_area:
            continue
        # A one-pixel padding permits all immediate exterior samples to be
        # gathered in a compact ROI, avoiding any global nearest-pixel lookup.
        x0, x1 = max(0, x - 1), min(width, x + component_width + 1)
        y0, y1 = max(0, y - 1), min(height, y + component_height + 1)
        component = labels[y0:y1, x0:x1] == label
        foreground_roi = foreground[y0:y1, x0:x1]
        if not np.all(component <= foreground_roi):
            continue
        component_distance = cv2.distanceTransform(
            component.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        )
        core = component & (component_distance >= core_depth)
        edge = component & (component_distance <= 1.0)
        transition = component & ~core & ~edge
        if (
            np.sum(core) < minimum_core_pixels
            or np.sum(transition) < minimum_transition_pixels
            or not np.all(reference_weight[y0:y1, x0:x1][core] >= 0.999)
        ):
            continue

        correction_roi = correction[y0:y1, x0:x1]
        reference_roi = reference_correction[y0:y1, x0:x1]
        lab_roi = base_lab[y0:y1, x0:x1]
        exterior = foreground_roi & ~component
        boundary_sum = np.zeros_like(correction_roi)
        boundary_weight = np.zeros(component.shape, dtype=np.float64)
        roi_height, roi_width = component.shape
        for dy, dx, spatial_weight in neighbors:
            destination_y = slice(max(0, -dy), roi_height - max(0, dy))
            destination_x = slice(max(0, -dx), roi_width - max(0, dx))
            neighbor_y = slice(max(0, dy), roi_height - max(0, -dy))
            neighbor_x = slice(max(0, dx), roi_width - max(0, -dx))
            valid = (
                edge[destination_y, destination_x]
                & exterior[neighbor_y, neighbor_x]
            )
            if not np.any(valid):
                continue
            sum_view = boundary_sum[destination_y, destination_x]
            weight_view = boundary_weight[destination_y, destination_x]
            sum_view[valid] += (
                spatial_weight * correction_roi[neighbor_y, neighbor_x][valid]
            )
            weight_view[valid] += spatial_weight
        sampled_edge = edge & (boundary_weight > 0.0)
        # Every component-boundary pixel must have a real adjacent exterior
        # sample.  Otherwise a narrow or touching component is not safe.
        if np.sum(sampled_edge) != np.sum(edge):
            continue
        boundary_field = boundary_sum / np.maximum(
            boundary_weight[:, :, None], 1e-8
        )
        mismatch = np.max(
            np.abs(reference_roi[sampled_edge] - boundary_field[sampled_edge]),
            axis=1,
        )
        if np.percentile(mismatch, 90) < mismatch_threshold:
            continue

        field = correction_roi.copy()
        field[core] = reference_roi[core]
        field[edge] = boundary_field[edge]
        checkerboard = (
            np.indices(component.shape).sum(axis=0) + x0 + y0
        ) % 2
        # Red-black Gauss-Seidel converges to the edge-aware harmonic field
        # without SciPy.  It operates only on this small component ROI.
        for _ in range(iterations):
            maximum_change = 0.0
            for parity in (0, 1):
                numerator = np.zeros_like(field)
                denominator = np.zeros(component.shape, dtype=np.float64)
                for dy, dx, spatial_weight in neighbors:
                    destination_y = slice(max(0, -dy), roi_height - max(0, dy))
                    destination_x = slice(max(0, -dx), roi_width - max(0, dx))
                    neighbor_y = slice(max(0, dy), roi_height - max(0, -dy))
                    neighbor_x = slice(max(0, dx), roi_width - max(0, -dx))
                    connected = (
                        component[destination_y, destination_x]
                        & component[neighbor_y, neighbor_x]
                    )
                    if not np.any(connected):
                        continue
                    lab_delta = np.linalg.norm(
                        lab_roi[destination_y, destination_x]
                        - lab_roi[neighbor_y, neighbor_x],
                        axis=2,
                    )
                    weights = spatial_weight * np.exp(
                        -0.5 * (lab_delta / 12.0) ** 2
                    )
                    weights = np.maximum(weights, 1e-5) * connected
                    numerator[destination_y, destination_x] += (
                        weights[:, :, None]
                        * field[neighbor_y, neighbor_x]
                    )
                    denominator[destination_y, destination_x] += weights
                update = transition & (checkerboard == parity) & (denominator > 0.0)
                estimate = numerator / np.maximum(denominator[:, :, None], 1e-8)
                if np.any(update):
                    maximum_change = max(
                        maximum_change,
                        float(np.max(np.abs(field[update] - estimate[update]))),
                    )
                    field[update] = estimate[update]
            if maximum_change < convergence:
                break

        component_output = np.clip(base[y0:y1, x0:x1] + field, 0.0, 1.0)
        output_roi = output[y0:y1, x0:x1]
        output_roi[component] = component_output[component]
        applied[y0:y1, x0:x1] |= component
    return np.clip(output, 0.0, 1.0), applied


def _detect_occlusion_region(
    edited: np.ndarray,
    reference: np.ndarray,
    mapped: np.ndarray,
    foreground: np.ndarray,
    unchanged: np.ndarray,
    unchanged_threshold: float,
    *,
    min_area_fraction: float = 0.001,
    influence_width: float = 48.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find a removed reference object and a local seam zone around it.

    A sizeable reference-only background component seeds the detection.  Only
    connected high-residual foreground pixels touching that component become
    direct occlusion.  A narrow low-frequency lightness-only halo is retained
    as indirect evidence.  The resulting zone selects alternate seam values;
    it does not change the old trusted mask or global color fit.
    """
    shape = foreground.shape
    direct = np.zeros(shape, dtype=bool)
    indirect = np.zeros(shape, dtype=bool)
    zone = np.zeros(shape, dtype=bool)
    raw_difference = np.max(np.abs(edited - reference), axis=2) * 255.0
    reference_only = (~foreground) & (
        raw_difference > max(48.0, 2.0 * unchanged_threshold)
    )
    background_area = max(1, int(np.sum(~foreground)))
    if np.sum(reference_only) / background_area >= 0.20:
        return direct, indirect, zone

    component_count, component_labels, component_stats, _ = (
        cv2.connectedComponentsWithStats(reference_only.astype(np.uint8), 8)
    )
    min_component_area = max(
        64, int(round(shape[0] * shape[1] * min_area_fraction))
    )
    reference_only_large = np.zeros(shape, dtype=bool)
    for component in range(1, component_count):
        if component_stats[component, cv2.CC_STAT_AREA] >= min_component_area:
            reference_only_large |= component_labels == component
    if not np.any(reference_only_large):
        return direct, indirect, zone

    direct_candidates = cv2.morphologyEx(
        (raw_difference > 48.0).astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    ).astype(bool)
    _, direct_labels, _, _ = cv2.connectedComponentsWithStats(
        direct_candidates.astype(np.uint8), 8
    )
    touching_labels = np.unique(direct_labels[reference_only_large])
    touching_labels = touching_labels[touching_labels > 0]
    if touching_labels.size:
        direct = np.isin(direct_labels, touching_labels) & foreground
    if not np.any(direct):
        return direct, indirect, zone

    mapped_lab = cv2.cvtColor(
        np.clip(np.rint(mapped * 255.0), 0, 255).astype(np.uint8),
        cv2.COLOR_RGB2LAB,
    ).astype(np.float32)
    reference_lab = cv2.cvtColor(
        np.clip(np.rint(reference * 255.0), 0, 255).astype(np.uint8),
        cv2.COLOR_RGB2LAB,
    ).astype(np.float32)
    chroma_difference = cv2.GaussianBlur(
        np.linalg.norm(mapped_lab[:, :, 1:] - reference_lab[:, :, 1:], axis=2),
        (0, 0),
        2.0,
    )
    lightness_difference = cv2.GaussianBlur(
        mapped_lab[:, :, 0] - reference_lab[:, :, 0],
        (0, 0),
        3.0,
    )
    distance_to_direct = cv2.distanceTransform(
        (~direct).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    image_scale = float(min(shape))
    influence_reach = min(float(influence_width), max(8.0, 0.045 * image_scale))
    effect_candidates = (
        unchanged
        & ~direct
        & (distance_to_direct < influence_reach)
        & (np.abs(lightness_difference) > 4.0)
        & (np.abs(lightness_difference) < 48.0)
        & (chroma_difference < 18.0)
    )
    effect_candidates = cv2.morphologyEx(
        effect_candidates.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    ).astype(bool)
    effect_count, effect_labels, effect_stats, _ = (
        cv2.connectedComponentsWithStats(effect_candidates.astype(np.uint8), 8)
    )
    min_effect_area = max(32, int(round(shape[0] * shape[1] * 0.00010)))
    for component in range(1, effect_count):
        if effect_stats[component, cv2.CC_STAT_AREA] >= min_effect_area:
            indirect |= effect_labels == component

    core = direct | indirect
    if not np.any(core):
        return direct, indirect, zone
    radius = max(1, int(np.ceil(float(influence_width))))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    zone = cv2.dilate(core.astype(np.uint8), kernel).astype(bool) & foreground
    return direct, indirect, zone


def restore_reference_colors(
    edited: np.ndarray,
    reference: np.ndarray,
    *,
    max_samples: int = 300_000,
    background_threshold: float | None = None,
    unchanged_threshold: float | None = None,
    structural_rescue: bool = True,
    structural_threshold: float = 0.95,
    structural_max_shift: int = 2,
    hard_restore_high_confidence: bool = True,
    hard_restore_threshold: float = 0.995,
    small_hole_no_feather_area: int = 16,
    seam_color_propagation: bool = True,
    seam_max_distance: float = 96.0,
    seam_decay: float = 64.0,
    seam_residual_blur: float = 12.0,
    seam_color_sigma: float = 20.0,
    seam_bridge_width: float = 12.0,
    occlusion_residual_blur: float = 0.0,
    occlusion_bridge_width: float = 0.0,
    component_internal_handoff: bool = False,
    seam_handoff_smoothing: bool = False,
    seam_handoff_silhouette_guard: float = 3.0,
    continuous_seam_field: bool = False,
    boundary_guard: int = 4,
    transition: float = 6.0,
    trusted_fit: bool = False,
    structural_unchanged_cleanup: bool = False,
    return_occlusion_masks: bool = False,
    return_diagnostics: bool = False,
) -> tuple:
    """Restore aligned content while preserving newly generated regions."""
    edited = np.asarray(edited, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if edited.shape != reference.shape:
        raise ValueError(f"Image shapes differ: {edited.shape} != {reference.shape}")
    if edited.ndim != 3 or edited.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB images, got {edited.shape}")
    if max_samples < 100:
        raise ValueError("max_samples must be at least 100")
    if seam_handoff_silhouette_guard < 0:
        raise ValueError("seam_handoff_silhouette_guard must be non-negative")
    foreground, background_rgb, _ = segment_solid_background(
        edited, threshold=background_threshold
    )
    guard_size = max(1, 2 * boundary_guard + 1)
    interior = cv2.erode(
        foreground.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (guard_size, guard_size)),
    ).astype(bool)
    if trusted_fit:
        transform, fit_threshold = _trusted_affine_fit(
            edited,
            reference,
            interior,
            unchanged_threshold,
            max_samples,
        )
    else:
        sample_edited, sample_reference = _uniform_sample(
            edited, reference, interior, max_samples
        )
        if len(sample_edited) < 100:
            raise RuntimeError("Too few foreground correspondences for color correction")
        transform = _robust_fit(
            sample_edited,
            sample_reference,
            quadratic=True,
            base_weights=_balanced_color_weights(sample_edited),
            iterations=10,
        )
        fit_threshold = None
    mapped = _apply_transform(edited, transform, quadratic=True)
    residual = np.max(np.abs(mapped - reference), axis=2) * 255.0
    local_rms = np.sqrt(
        cv2.GaussianBlur((residual * residual).astype(np.float32), (0, 0), 1.5)
    )
    if unchanged_threshold is None:
        unchanged_threshold = (
            fit_threshold
            if fit_threshold is not None
            else _automatic_unchanged_threshold(local_rms, interior)
        )

    unchanged = interior & (local_rms < unchanged_threshold)
    unchanged = cv2.morphologyEx(
        unchanged.astype(np.uint8),
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    ).astype(bool)
    baseline_unchanged = unchanged.copy()
    correlation = zero_shift = texture = chroma_difference = None
    if structural_rescue or hard_restore_high_confidence:
        correlation, zero_shift, texture = _best_local_zncc(
            mapped, reference, patch_size=13, max_shift=structural_max_shift
        )
        mapped_lab = cv2.cvtColor(
            np.clip(np.rint(mapped * 255.0), 0, 255).astype(np.uint8),
            cv2.COLOR_RGB2LAB,
        ).astype(np.float32)
        reference_lab = cv2.cvtColor(
            np.clip(np.rint(reference * 255.0), 0, 255).astype(np.uint8),
            cv2.COLOR_RGB2LAB,
        ).astype(np.float32)
        chroma_difference = np.linalg.norm(
            mapped_lab[:, :, 1:] - reference_lab[:, :, 1:], axis=2
        )
        chroma_difference = cv2.GaussianBlur(chroma_difference, (0, 0), 2.0)
        if structural_rescue:
            candidates = (
                interior
                & ~unchanged
                & (correlation >= structural_threshold)
                & (texture >= 4.0)
                & (local_rms < 64.0)
                & (chroma_difference < 20.0)
            )
            for _ in range(2):
                support = cv2.dilate(
                    unchanged.astype(np.uint8),
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
                ).astype(bool)
                unchanged |= candidates & support

    if structural_unchanged_cleanup:
        unchanged = _clean_unchanged_by_structure(unchanged, mapped, reference)

    relative_unchanged = float(np.sum(unchanged) / max(1, np.sum(foreground)))
    if relative_unchanged < 0.20:
        raise RuntimeError(
            f"Only {100*relative_unchanged:.1f}% of foreground is confidently unchanged; "
            "refusing reference restoration"
        )
    foreground_distance = cv2.distanceTransform(
        foreground.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    feather_mask = unchanged.copy()
    tiny_uncertain = np.zeros_like(unchanged)
    if small_hole_no_feather_area > 0:
        uncertain = interior & ~unchanged
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            uncertain.astype(np.uint8), 8
        )
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] <= small_hole_no_feather_area:
                component = labels == label
                tiny_uncertain |= component
                feather_mask |= component
    unchanged_distance = cv2.distanceTransform(
        feather_mask.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    color_weight = np.clip(foreground_distance / 3.0, 0.0, 1.0)
    reference_weight = np.clip(
        unchanged_distance / max(1.0, transition), 0.0, 1.0
    )
    reference_weight[tiny_uncertain] = 0.0
    hard_restore = np.zeros_like(unchanged)
    if hard_restore_high_confidence:
        hard_restore = (
            unchanged
            & (zero_shift >= hard_restore_threshold)
            & (texture >= 4.0)
            & (chroma_difference < 12.0)
            & (local_rms < 48.0)
        )
        reference_weight[hard_restore] = 1.0
    base = edited * (1.0 - color_weight[:, :, None]) + mapped * color_weight[:, :, None]
    result = base * (1.0 - reference_weight[:, :, None]) + reference * reference_weight[
        :, :, None
    ]
    occlusion_core = np.zeros_like(unchanged)
    occlusion_zone = np.zeros_like(unchanged)
    occlusion_seam_compensated = np.zeros_like(unchanged)
    component_handoff = np.zeros_like(unchanged)
    if occlusion_residual_blur > 0 and occlusion_bridge_width > 0:
        direct_mask, indirect_mask, occlusion_zone = _detect_occlusion_region(
            edited,
            reference,
            mapped,
            foreground,
            unchanged,
            float(unchanged_threshold),
            influence_width=occlusion_bridge_width,
        )
        occlusion_core = direct_mask | indirect_mask

    seam_compensated = np.zeros_like(unchanged)
    if seam_color_propagation:
        seam_input = result
        global_result, global_compensated = _propagate_seam_color(
            result,
            base,
            reference,
            foreground,
            unchanged,
            seam_max_distance,
            seam_decay,
            seam_residual_blur,
            seam_color_sigma,
            seam_bridge_width,
            continuous_field=continuous_seam_field,
        )
        if seam_handoff_smoothing and continuous_seam_field:
            handoff_result, handoff_compensated = _repair_continuous_seam_handoff(
                seam_input,
                global_result,
                base,
                reference,
                foreground,
                unchanged,
                seam_max_distance,
                seam_decay,
                seam_residual_blur,
                seam_color_sigma,
                seam_bridge_width,
                seam_handoff_silhouette_guard,
            )
            global_result = handoff_result
            global_compensated |= handoff_compensated
        result = global_result
        seam_compensated = global_compensated
        if np.any(occlusion_zone):
            occlusion_result, occlusion_compensated = _propagate_seam_color(
                seam_input,
                base,
                reference,
                foreground,
                unchanged,
                seam_max_distance,
                seam_decay,
                occlusion_residual_blur,
                seam_color_sigma,
                occlusion_bridge_width,
                region=occlusion_zone,
                continuous_field=continuous_seam_field,
            )
            local_selection = occlusion_compensated & occlusion_zone
            result = np.where(
                local_selection[:, :, None], occlusion_result, global_result
            )
            seam_compensated = global_compensated | local_selection
            occlusion_seam_compensated = local_selection
    if component_internal_handoff and seam_color_propagation:
        result, component_handoff = _handoff_independent_components(
            result,
            base,
            reference,
            foreground,
            unchanged,
            reference_weight,
            seam_bridge_width,
        )
    report = RestorationReport(
        background_rgb=background_rgb,
        foreground_fraction=float(np.mean(foreground)),
        unchanged_fraction=float(np.mean(unchanged)),
        threshold=float(unchanged_threshold),
        before_mae=float(
            np.mean(np.abs(edited[unchanged] - reference[unchanged])) * 255.0
        ),
        mapped_mae=float(
            np.mean(np.abs(mapped[unchanged] - reference[unchanged])) * 255.0
        ),
        structural_rescued_fraction=float(
            np.sum(unchanged & ~baseline_unchanged) / max(1, np.sum(foreground))
        ),
        hard_restored_fraction=float(
            np.sum(hard_restore) / max(1, np.sum(foreground))
        ),
        seam_compensated_fraction=float(
            np.sum(seam_compensated) / max(1, np.sum(foreground))
        ),
        occlusion_core_fraction=float(
            np.sum(occlusion_core & foreground) / max(1, np.sum(foreground))
        ),
        occlusion_zone_fraction=float(
            np.sum(occlusion_zone & foreground) / max(1, np.sum(foreground))
        ),
        occlusion_seam_compensated_fraction=float(
            np.sum(occlusion_seam_compensated & foreground)
            / max(1, np.sum(foreground))
        ),
        component_handoff_fraction=float(
            np.sum(component_handoff & foreground) / max(1, np.sum(foreground))
        ),
    )
    output = np.clip(result, 0.0, 1.0)
    if return_diagnostics and not return_occlusion_masks:
        raise ValueError("return_diagnostics requires return_occlusion_masks=True")
    if return_occlusion_masks:
        outputs = (
            output,
            foreground,
            unchanged,
            report,
            occlusion_core,
            occlusion_zone,
            occlusion_seam_compensated,
        )
        if return_diagnostics:
            # ``base`` is the globally mapped AI image before reference blending
            # and seam propagation.  It lets V1 diagnose edges in its correction
            # field instead of taking a gradient of the final RGB image.
            return (*outputs, {"base": base})
        return outputs
    return output, foreground, unchanged, report


def _as_image_batch(image: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4 or image.shape[-1] < 3:
        raise ValueError(
            f"{name} must be a ComfyUI IMAGE in BHWC layout, got {tuple(image.shape)}"
        )
    return image[..., :3].detach().float().cpu().clamp(0.0, 1.0)


def _as_alpha_batch(image: torch.Tensor) -> torch.Tensor | None:
    """Return the alpha-channel batch ([B, H, W, 1]) or None when absent."""
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4 or image.shape[-1] <= 3:
        return None
    return image[..., 3:4].detach().float().cpu().clamp(0.0, 1.0)


def _resize_image_to(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    target_height, target_width = shape
    height, width = image.shape[:2]
    ratio_error = abs((target_width / target_height) / (width / height) - 1.0)
    if ratio_error > 0.02:
        raise ValueError(
            f"Aspect ratios differ by {100 * ratio_error:.2f}%; refusing resize"
        )
    interpolation = cv2.INTER_AREA if height > target_height else cv2.INTER_CUBIC
    return cv2.resize(
        image, (target_width, target_height), interpolation=interpolation
    )


class ImageColorRestore(io.ComfyNode):
    """Restore reference colors onto an aligned edit, with local occlusion seams."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_ImageColorRestore",
            display_name="Image Color Restore",
            category="1hewNodes/color",
            description=(
                "Restore an aligned AI edit's colors to its original reference while "
                "locally smoothing occlusion seams from object removal. Uses a trusted "
                "affine color fit and a continuous seam field (structural unchanged-mask "
                "cleanup disabled). Preserves edit_image's alpha channel when present."
            ),
            inputs=[
                io.Image.Input(
                    "edit_image",
                    tooltip=(
                        "AI-edited image on a solid background; must be pixel-aligned "
                        "with org_image (resized to org_image's size when they differ)"
                    ),
                ),
                io.Image.Input(
                    "org_image",
                    tooltip=(
                        "Aligned original/reference image (color source; no solid-",
                        "background requirement)"
                    ),
                ),
                io.Float.Input(
                    "background_threshold",
                    default=0.0,
                    min=0.0,
                    max=80.0,
                    step=0.1,
                    tooltip=("LAB distance threshold for foreground/background split; "
                             "0 = auto. Foreground must end up 0.5%-80% or the node errors."),
                ),
                io.Float.Input(
                    "unchanged_threshold",
                    default=10.0,
                    min=0.0,
                    max=80.0,
                    step=0.1,
                    tooltip=("Max RGB residual (0-255) for a pixel to count as 'unchanged'; "
                             "0 = auto. Higher restores the reference more aggressively."),
                ),
                io.Float.Input(
                    "seam_max_distance",
                    default=96.0,
                    min=1.0,
                    max=1024.0,
                    step=1.0,
                    tooltip="Max propagation distance (px) from trusted unchanged region",
                ),
                io.Float.Input(
                    "seam_decay",
                    default=64.0,
                    min=1.0,
                    max=1024.0,
                    step=1.0,
                    tooltip="Distance falloff exp(-d/decay); larger reaches farther",
                ),
                io.Float.Input(
                    "seam_residual_blur",
                    default=24.0,
                    min=0.1,
                    max=128.0,
                    step=0.5,
                    tooltip="Gaussian sigma for smoothing the residual field",
                ),
                io.Float.Input(
                    "seam_color_sigma",
                    default=20.0,
                    min=0.1,
                    max=128.0,
                    step=0.5,
                    tooltip="Lab color-similarity gate sigma (only propagates to similar colors)",
                ),
                io.Float.Input(
                    "occlusion_residual_blur",
                    default=48.0,
                    min=0.0,
                    max=128.0,
                    step=0.5,
                    tooltip="Extra residual blur used only in the detected occlusion zone (0 disables local pass)",
                ),
                io.Float.Input(
                    "occlusion_bridge_width",
                    default=48.0,
                    min=0.0,
                    max=128.0,
                    step=0.5,
                    tooltip="Occlusion detection reach / bridge width (0 disables local pass)",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="corrected_image"),
                io.Mask.Output(display_name="safe_unchanged_mask"),
            ],
        )

    @classmethod
    async def execute(
        cls,
        edit_image: torch.Tensor,
        org_image: torch.Tensor,
        background_threshold: float,
        unchanged_threshold: float,
        seam_max_distance: float,
        seam_decay: float,
        seam_residual_blur: float,
        seam_color_sigma: float,
        occlusion_residual_blur: float,
        occlusion_bridge_width: float,
    ) -> io.NodeOutput:
        result = cls._restore_batch(
            edit_image,
            org_image,
            max_samples=300_000,
            background_threshold=(
                None if background_threshold <= 0 else background_threshold
            ),
            unchanged_threshold=(
                None if unchanged_threshold <= 0 else unchanged_threshold
            ),
            boundary_guard=4,
            transition=6.0,
            structural_rescue=True,
            structural_threshold=0.95,
            structural_max_shift=2,
            hard_restore_high_confidence=True,
            hard_restore_threshold=0.995,
            small_hole_no_feather_area=16,
            seam_color_propagation=True,
            seam_max_distance=seam_max_distance,
            seam_decay=seam_decay,
            seam_residual_blur=seam_residual_blur,
            seam_color_sigma=seam_color_sigma,
            seam_bridge_width=12.0,
            occlusion_residual_blur=occlusion_residual_blur,
            occlusion_bridge_width=occlusion_bridge_width,
            component_internal_handoff=False,
            seam_handoff_smoothing=False,
            seam_handoff_silhouette_guard=3.0,
            continuous_seam_field=True,
            trusted_fit=True,
            structural_unchanged_cleanup=False,
        )
        return io.NodeOutput(*result)

    @classmethod
    def _restore_batch(
        cls,
        edit_image: torch.Tensor,
        org_image: torch.Tensor,
        **options,
    ):
        edited_batch = _as_image_batch(edit_image, "edit_image")
        edited_alpha_batch = _as_alpha_batch(edit_image)
        reference_batch = _as_image_batch(org_image, "org_image")
        edited_count = edited_batch.shape[0]
        reference_count = reference_batch.shape[0]
        if edited_count != reference_count and edited_count != 1 and reference_count != 1:
            raise ValueError(
                "Image batches must have equal sizes, or one input batch must "
                "contain one image; "
                f"got {edited_count} and {reference_count}"
            )

        corrected_items = []
        unchanged_items = []
        count = max(edited_count, reference_count)
        for index in range(count):
            edited = edited_batch[0 if edited_count == 1 else index].numpy().astype(
                np.float64, copy=False
            )
            edited_alpha = None
            if edited_alpha_batch is not None:
                edited_alpha = edited_alpha_batch[
                    0 if edited_count == 1 else index
                ].numpy().astype(np.float64, copy=False)
            reference = reference_batch[
                0 if reference_count == 1 else index
            ].numpy().astype(np.float64, copy=False)
            if edited.shape != reference.shape:
                edited = _resize_image_to(edited, reference.shape[:2])
                if edited_alpha is not None:
                    edited_alpha = _resize_image_to(edited_alpha, reference.shape[:2])
            try:
                output, _foreground, unchanged, _report = restore_reference_colors(
                    edited,
                    reference,
                    **options,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Image Color Restore failed at batch item "
                    f"{index}: {exc}"
                ) from exc
            if edited_alpha is not None:
                output = np.concatenate([output, edited_alpha], axis=-1)
            corrected_items.append(output.astype(np.float32))
            unchanged_items.append(unchanged.astype(np.float32))

        return (
            torch.from_numpy(np.stack(corrected_items)),
            torch.from_numpy(np.stack(unchanged_items)),
        )
