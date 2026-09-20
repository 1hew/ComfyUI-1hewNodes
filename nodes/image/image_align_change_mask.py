"""Extract localized AI-edit changes while rejecting small global drift.

The node intentionally uses only constrained global registration.  It never uses
optical flow, because a dense warp can make a real generated edit disappear.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from comfy_api.latest import io


class ImageAlignChangeMask(io.ComfyNode):
    """Create diagnostic and production masks from original and edited images."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_ImageAlignChangeMask",
            display_name="Image Align Change Mask",
            category="1hewNodes/image",
            description=(
                "Align an AI-edited image to its original, reject small global color "
                "drift using stable background pixels, and output raw/clean change masks."
            ),
            inputs=[
                io.Image.Input("edit_image"),
                io.Image.Input("org_image"),
                io.Combo.Input(
                    "mode",
                    options=["auto", "none", "translation", "similarity"],
                    default="auto",
                    tooltip="Auto evaluates identity, translation, and constrained similarity candidates.",
                ),
                io.Int.Input("max_offset", default=8, min=0, max=128, step=1),
                io.Float.Input("max_rotation", default=2.0, min=0.0, max=30.0, step=0.1),
                io.Float.Input("max_scale", default=0.03, min=0.0, max=0.25, step=0.005),
                io.Float.Input("min_align_score", default=0.80, min=0.0, max=1.0, step=0.01),
                io.Boolean.Input("color_compensation", default=True),
                io.Float.Input("sensitivity", default=8.0, min=0.5, max=20.0, step=0.1),
                io.Int.Input("min_component_area", default=32, min=0, max=100000, step=1),
                io.Int.Input("expand", default=16, min=0, max=128, step=1),
                io.Int.Input("feather", default=8, min=0, max=128, step=1),
            ],
            outputs=[
                io.Image.Output(display_name="align_image"),
                io.Mask.Output(display_name="clean_mask"),
                io.Mask.Output(display_name="raw_mask"),
                io.Mask.Output(display_name="change_score"),
                io.Image.Output(display_name="overlay_image"),
            ],
        )

    @classmethod
    async def execute(
        cls, edit_image: torch.Tensor, org_image: torch.Tensor, mode: str,
        max_offset: int, max_rotation: float, max_scale: float,
        min_align_score: float, color_compensation: bool, sensitivity: float,
        min_component_area: int, expand: int, feather: int,
    ) -> io.NodeOutput:
        org_image = cls._image_batch(org_image, "org_image")
        edit_image = cls._image_batch(edit_image, "edit_image")
        if org_image.shape[0] != edit_image.shape[0] and org_image.shape[0] != 1 and edit_image.shape[0] != 1:
            raise ValueError("org_image and edit_image must have equal batch sizes, or one input must have batch size 1")
        count = max(org_image.shape[0], edit_image.shape[0])
        align_images, clean_masks, raw_masks, overlays, score_maps = [], [], [], [], []
        for index in range(count):
            base = org_image[0 if org_image.shape[0] == 1 else index].numpy()
            moving = edit_image[0 if edit_image.shape[0] == 1 else index].numpy()
            if base.shape != moving.shape:
                raise ValueError(f"Batch item {index}: org_image and edit_image must have the same dimensions")
            result = cls._process_one(
                base, moving, mode, int(max_offset), float(max_rotation),
                float(max_scale), float(min_align_score), bool(color_compensation),
                float(sensitivity), int(min_component_area), int(expand), int(feather),
            )
            raw, clean, score_map, aligned, overlay = result[:5]
            align_images.append(aligned)
            clean_masks.append(clean)
            raw_masks.append(raw)
            overlays.append(overlay)
            score_maps.append(score_map)
        device = edit_image.device
        align_t = torch.from_numpy(np.stack(align_images)).to(device=device)
        clean_t = torch.from_numpy(np.stack(clean_masks)).to(device=device)
        raw_t = torch.from_numpy(np.stack(raw_masks)).to(device=device)
        overlay_t = torch.from_numpy(np.stack(overlays)).to(device=device)
        score_t = torch.from_numpy(np.stack(score_maps)).to(device=device)
        return io.NodeOutput(align_t, clean_t, raw_t, score_t, overlay_t)

    @staticmethod
    def _image_batch(image: torch.Tensor, name: str) -> torch.Tensor:
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"{name} must be an IMAGE tensor")
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4 or image.shape[-1] < 3:
            raise ValueError(f"{name} must be BHWC with at least 3 channels, got {tuple(image.shape)}")
        return image[..., :3].detach().float().cpu().clamp(0.0, 1.0)

    @staticmethod
    def _gray_gradient(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (0, 0), 1.1)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        return cv2.GaussianBlur(cv2.magnitude(gx, gy), (0, 0), 0.8)

    @classmethod
    def _translation_ecc(cls, base: np.ndarray, moving: np.ndarray, max_shift: int) -> tuple[np.ndarray, float, float, float, str]:
        """Phase correlation initializes ECC; identity is returned if validation fails."""
        h, w = base.shape[:2]
        scale = min(1.0, 512.0 / max(h, w))
        size = (max(32, int(round(w * scale))), max(32, int(round(h * scale))))
        ref = cls._gray_gradient(base)
        mov = cls._gray_gradient(moving)
        if scale < 1.0:
            ref_small = cv2.resize(ref, size, interpolation=cv2.INTER_AREA)
            mov_small = cv2.resize(mov, size, interpolation=cv2.INTER_AREA)
        else:
            ref_small, mov_small = ref, mov
        window = cv2.createHanningWindow((ref_small.shape[1], ref_small.shape[0]), cv2.CV_32F)
        (px, py), phase_response = cv2.phaseCorrelate(ref_small, mov_small, window)
        initial = np.float32([[1, 0, px / scale], [0, 1, py / scale]])
        if abs(initial[0, 2]) > max_shift or abs(initial[1, 2]) > max_shift:
            return moving, 0.0, 0.0, float(phase_response), "phase_shift_exceeds_limit"
        try:
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
            ecc, warp = cv2.findTransformECC(
                cls._gray_gradient(base), cls._gray_gradient(moving), initial,
                cv2.MOTION_TRANSLATION, criteria, None, 5,
            )
            dx, dy = float(warp[0, 2]), float(warp[1, 2])
            if abs(dx) > max_shift or abs(dy) > max_shift:
                return moving, 0.0, 0.0, float(ecc), "ecc_shift_exceeds_limit"
            aligned = cv2.warpAffine(
                moving, warp, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            return aligned.astype(np.float32), dx, dy, float(ecc), "aligned"
        except cv2.error:
            # Phase has already supplied a useful sub-pixel estimate; use it only if plausible.
            aligned = cv2.warpAffine(
                moving, initial, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            return aligned.astype(np.float32), float(initial[0, 2]), float(initial[1, 2]), float(phase_response), "phase_fallback"

    @staticmethod
    def _warp_inverse(moving: np.ndarray, warp: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        height, width = shape
        return cv2.warpAffine(
            moving, warp.astype(np.float32), (width, height),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT_101,
        ).astype(np.float32)

    @staticmethod
    def _similarity_parameters(warp: np.ndarray) -> tuple[float, float]:
        """Return rotation degrees and isotropic scale from template-to-moving warp."""
        a, b = float(warp[0, 0]), float(warp[0, 1])
        scale = max(1e-8, float(np.hypot(a, b)))
        rotation = float(np.degrees(np.arctan2(-b, a)))
        return rotation, scale

    @classmethod
    def _motion_parameters(cls, warp: np.ndarray, shape: tuple[int, int]) -> tuple[float, float, float, float]:
        """Return center displacement, rotation, and scale for an affine warp."""
        height, width = shape
        rotation, scale = cls._similarity_parameters(warp)
        center = np.array([width * 0.5, height * 0.5], dtype=np.float64)
        moved = warp[:, :2].astype(np.float64) @ center + warp[:, 2]
        dx, dy = moved - center
        return float(dx), float(dy), rotation, scale

    @classmethod
    def _similarity_candidate(
        cls, base: np.ndarray, moving: np.ndarray, max_shift: int,
        max_rotation: float, max_scale_change: float,
    ) -> tuple[np.ndarray, np.ndarray, float, str]:
        """Estimate a constrained similarity transform with AKAZE/RANSAC, then ECC."""
        height, width = base.shape[:2]
        base_u8 = np.clip(cv2.cvtColor(base, cv2.COLOR_RGB2GRAY) * 255.0, 0, 255).astype(np.uint8)
        moving_u8 = np.clip(cv2.cvtColor(moving, cv2.COLOR_RGB2GRAY) * 255.0, 0, 255).astype(np.uint8)
        if hasattr(cv2, "AKAZE_create"):
            detector, norm, feature_name = cv2.AKAZE_create(), cv2.NORM_HAMMING, "akaze"
        elif hasattr(cv2, "SIFT_create"):
            detector, norm, feature_name = cv2.SIFT_create(nfeatures=5000), cv2.NORM_L2, "sift"
        else:
            detector, norm, feature_name = cv2.ORB_create(nfeatures=5000), cv2.NORM_HAMMING, "orb"
        key_base, desc_base = detector.detectAndCompute(base_u8, None)
        key_moving, desc_moving = detector.detectAndCompute(moving_u8, None)
        if desc_base is None or desc_moving is None or len(key_base) < 8 or len(key_moving) < 8:
            raise ValueError(f"insufficient_{feature_name}_features")
        pairs = cv2.BFMatcher(norm).knnMatch(desc_moving, desc_base, k=2)
        good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.78 * pair[1].distance]
        if len(good) < 8:
            raise ValueError("insufficient_similarity_matches")
        source = np.float32([key_moving[m.queryIdx].pt for m in good])
        target = np.float32([key_base[m.trainIdx].pt for m in good])
        direct, inliers = cv2.estimateAffinePartial2D(
            source, target, method=cv2.RANSAC, ransacReprojThreshold=2.5,
            maxIters=5000, confidence=0.995, refineIters=20,
        )
        if direct is None or inliers is None or int(np.count_nonzero(inliers)) < 6:
            raise ValueError("similarity_ransac_failed")
        # ECC expects a template-to-input warp, while the feature fit maps moving to base.
        initial = cv2.invertAffineTransform(direct).astype(np.float32)
        try:
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
            ecc, affine = cv2.findTransformECC(
                cls._gray_gradient(base), cls._gray_gradient(moving), initial,
                cv2.MOTION_AFFINE, criteria, None, 5,
            )
            # Project the affine linear part to its nearest rotation + isotropic scale,
            # preventing shear from explaining away a real local edit.
            linear = affine[:, :2].astype(np.float64)
            u, singular, vt = np.linalg.svd(linear)
            rotation_matrix = u @ vt
            if np.linalg.det(rotation_matrix) < 0:
                u[:, -1] *= -1
                rotation_matrix = u @ vt
            scale = float(np.mean(singular))
            warp = affine.copy().astype(np.float32)
            warp[:, :2] = (scale * rotation_matrix).astype(np.float32)
            quality = float(ecc)
            note = f"similarity_{feature_name}_ecc"
        except cv2.error:
            warp = initial
            quality = float(np.count_nonzero(inliers) / max(1, len(good)))
            note = f"similarity_{feature_name}_ransac"
        dx, dy, rotation, scale = cls._motion_parameters(warp, (height, width))
        if abs(dx) > max_shift or abs(dy) > max_shift:
            raise ValueError("similarity_shift_exceeds_limit")
        if abs(rotation) > max_rotation:
            raise ValueError("similarity_rotation_exceeds_limit")
        if abs(scale - 1.0) > max_scale_change:
            raise ValueError("similarity_scale_exceeds_limit")
        return cls._warp_inverse(moving, warp, (height, width)), warp, quality, note

    @classmethod
    def _alignment_loss(cls, base: np.ndarray, aligned: np.ndarray) -> float:
        """Robust held-out-like loss: local edits are discarded as upper-tail outliers."""
        residual = np.abs(cls._gray_gradient(base) - cls._gray_gradient(aligned))
        finite = residual[np.isfinite(residual)]
        if finite.size < 100:
            return float("inf")
        cutoff = float(np.quantile(finite, 0.70))
        stable = finite[finite <= cutoff]
        return float(np.mean(stable)) if stable.size else float("inf")

    @classmethod
    def _select_alignment(
        cls, base: np.ndarray, moving: np.ndarray, mode: str, max_shift: int,
        max_rotation: float, max_scale_change: float, min_score: float,
    ) -> tuple[np.ndarray, float, float, float, float, float, str, str]:
        identity_loss = cls._alignment_loss(base, moving)
        candidates = [{
            "mode": "none", "image": moving, "warp": np.eye(2, 3, dtype=np.float32),
            "quality": 1.0, "loss": identity_loss, "note": "alignment_disabled",
        }]
        errors: list[str] = []
        if mode in {"auto", "translation"} and max_shift > 0:
            image, dx, dy, quality, note = cls._translation_ecc(base, moving, max_shift)
            if note in {"aligned", "phase_fallback"} and quality >= min_score:
                warp = np.float32([[1, 0, dx], [0, 1, dy]])
                candidates.append({"mode": "translation", "image": image, "warp": warp,
                                   "quality": quality, "loss": cls._alignment_loss(base, image), "note": note})
            else:
                errors.append(note if quality >= min_score else "translation_score_below_minimum")
        if mode in {"auto", "similarity"}:
            try:
                image, warp, quality, note = cls._similarity_candidate(
                    base, moving, max_shift, max_rotation, max_scale_change,
                )
                if quality >= min_score:
                    candidates.append({"mode": "similarity", "image": image, "warp": warp,
                                       "quality": quality, "loss": cls._alignment_loss(base, image), "note": note})
                else:
                    errors.append("similarity_score_below_minimum")
            except (ValueError, cv2.error, AttributeError) as exc:
                errors.append(str(exc))
        if mode == "none":
            selected = candidates[0]
        elif mode == "translation":
            candidate = next((x for x in candidates if x["mode"] == "translation"), None)
            selected = candidate if candidate is not None and candidate["loss"] < identity_loss * 0.99 else candidates[0]
        elif mode == "similarity":
            candidate = next((x for x in candidates if x["mode"] == "similarity"), None)
            selected = candidate if candidate is not None and candidate["loss"] < identity_loss * 0.99 else candidates[0]
        else:
            # A more complex model must materially improve the robust residual.
            selected = candidates[0]
            translation = next((x for x in candidates if x["mode"] == "translation"), None)
            similarity = next((x for x in candidates if x["mode"] == "similarity"), None)
            if translation is not None and translation["loss"] < selected["loss"] * 0.99:
                selected = translation
            if similarity is not None and similarity["loss"] < selected["loss"] * 0.975:
                selected = similarity
        warp = selected["warp"]
        dx, dy, rotation, scale = cls._motion_parameters(warp, base.shape[:2])
        note = f"selected={selected['mode']};loss={selected['loss']:.6g};identity_loss={identity_loss:.6g};{selected['note']}"
        if errors:
            note += ";rejected=" + ",".join(errors)
        return (selected["image"], dx, dy, float(selected["quality"]),
                rotation, scale, str(selected["mode"]), note)

    @staticmethod
    def _local_zncc(a: np.ndarray, b: np.ndarray, size: int = 13) -> np.ndarray:
        ga = cv2.GaussianBlur(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), (0, 0), 0.7)
        gb = cv2.GaussianBlur(cv2.cvtColor(b, cv2.COLOR_RGB2GRAY), (0, 0), 0.7)
        ma = cv2.boxFilter(ga, -1, (size, size), normalize=True, borderType=cv2.BORDER_REFLECT)
        mb = cv2.boxFilter(gb, -1, (size, size), normalize=True, borderType=cv2.BORDER_REFLECT)
        va = np.maximum(cv2.boxFilter(ga * ga, -1, (size, size)) - ma * ma, 1e-5)
        vb = np.maximum(cv2.boxFilter(gb * gb, -1, (size, size)) - mb * mb, 1e-5)
        cov = cv2.boxFilter(ga * gb, -1, (size, size)) - ma * mb
        return np.clip(cov / np.sqrt(va * vb), -1.0, 1.0)

    @staticmethod
    def _robust_diagonal_map(moving: np.ndarray, base: np.ndarray, structural: np.ndarray) -> tuple[np.ndarray, bool]:
        """Fit only stable pixels, then require a held-out improvement before use."""
        mapped = moving.copy()
        height, width = base.shape[:2]
        yy, xx = np.indices((height, width))
        valid = np.ones((height, width), dtype=bool)
        valid[:2] = valid[-2:] = False; valid[:, :2] = valid[:, -2:] = False
        lum = cv2.cvtColor(moving, cv2.COLOR_RGB2GRAY)
        # Local correlation is unreliable in flat areas, so it is not a hard gate
        # here.  The trimmed residual fit below already rejects edited outliers.
        used = False
        for _ in range(2):
            residual = np.max(np.abs(mapped - base), axis=2)
            candidates = residual[valid]
            if candidates.size < 200:
                break
            cutoff = float(np.quantile(candidates, 0.60))
            trusted = valid & (residual <= cutoff)
            if np.count_nonzero(trusted) < 200:
                break
            # Per-channel gain+bias is deliberately safer than a full 3x3 color matrix.
            candidate = np.empty_like(moving)
            for channel in range(3):
                x = moving[..., channel][trusted].astype(np.float64)
                y = base[..., channel][trusted].astype(np.float64)
                variance = float(np.var(x))
                if variance < 1e-6:
                    slope, intercept = 1.0, float(np.mean(y - x))
                else:
                    slope = float(np.mean((x - np.mean(x)) * (y - np.mean(y))) / variance)
                    intercept = float(np.mean(y) - slope * np.mean(x))
                slope = float(np.clip(slope, 0.75, 1.25))
                intercept = float(np.clip(intercept, -0.12, 0.12))
                candidate[..., channel] = np.clip(moving[..., channel] * slope + intercept, 0.0, 1.0)
            holdout = trusted & (((yy // 16 + xx // 16) % 4) == 0)
            if np.count_nonzero(holdout) < 50:
                break
            before = float(np.mean(np.abs(mapped[holdout] - base[holdout])))
            after = float(np.mean(np.abs(candidate[holdout] - base[holdout])))
            if np.isfinite(after) and after < before * 0.99:
                mapped = candidate.astype(np.float32)
                used = True
            else:
                break
        return mapped, used

    @staticmethod
    def _robust_z(value: np.ndarray, stable: np.ndarray) -> np.ndarray:
        samples = value[stable]
        if samples.size < 100:
            samples = value.ravel()
        median = float(np.median(samples))
        mad = 1.4826 * float(np.median(np.abs(samples - median)))
        # 8-bit Lab often has an exactly-zero MAD for clean, quantized regions.
        # Fall back to an interquantile scale rather than turning every nonzero
        # residual into an astronomically large z-score.
        if mad < 1e-3:
            mad = float(np.quantile(samples, 0.90) - median) / 1.2816
        # A one-unit floor is appropriate for the 8-bit-Lab/color residual and
        # deliberately makes sub-unit floating-point resampling noise harmless.
        return np.maximum(0.0, (value - median) / max(mad, 1.0)).astype(np.float32)

    @staticmethod
    def _remove_small(mask: np.ndarray, minimum: int) -> np.ndarray:
        if minimum <= 0:
            return mask
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        result = np.zeros_like(mask, dtype=np.uint8)
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] >= minimum:
                result[labels == label] = 255
        return result

    @classmethod
    def _process_one(
        cls, base: np.ndarray, moving: np.ndarray, align_mode: str, max_shift: int,
        max_rotation: float, max_scale_change: float, min_score: float,
        compensation: bool, sensitivity: float, minimum: int, expand: int, feather: int,
    ) -> tuple:
        (aligned, dx, dy, alignment_score, rotation, scale,
         selected_mode, note) = cls._select_alignment(
            base, moving, align_mode, max_shift, max_rotation, max_scale_change, min_score,
        )
        structural = cls._local_zncc(aligned, base)
        mapped, used_compensation = cls._robust_diagonal_map(aligned, base, structural) if compensation else (aligned, False)
        lab_a = cv2.cvtColor(np.clip(base * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        lab_b = cv2.cvtColor(np.clip(mapped * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        color = np.linalg.norm(lab_a - lab_b, axis=2)
        grad_a = cls._gray_gradient(base)
        grad_b = cls._gray_gradient(mapped)
        gradient = np.abs(grad_a - grad_b)
        structure = np.maximum(0.0, 1.0 - structural)
        # Stable background is intentionally conservative: it defines noise, not edit intent.
        # Flat areas make local correlation numerically meaningless.  Only treat
        # structural disagreement as evidence where either image has enough local
        # gradient energy; color difference remains valid everywhere.
        texture = np.maximum(grad_a, grad_b)
        texture_floor = max(float(np.quantile(texture, 0.25)), 1e-3)
        provisional = (color <= np.quantile(color, 0.60)) & ((structural >= 0.80) | (texture < texture_floor))
        # OpenCV Lab is 8-bit here; four Lab units is a conservative noise scale
        # after a sub-pixel warp.  MAD remains useful for gradients, whose scale
        # depends strongly on the source texture.
        z_color = color / 4.0
        z_gradient = cls._robust_z(gradient, provisional)
        # ZNCC is used to identify stable fitting pixels.  It is intentionally
        # not a direct score term: low-texture/quantized regions can otherwise
        # report unstable correlation despite no visible edit.  Gradient residual
        # supplies the structural-change term with well-behaved statistics.
        score = np.maximum(z_color, 0.80 * z_gradient)
        score_norm = np.clip(score / max(sensitivity * 2.0, 1e-4), 0.0, 1.0)
        high = score >= sensitivity
        low = score >= sensitivity * 0.55
        # Hysteresis: weak pixels are retained only where connected to strong edits.
        count, labels = cv2.connectedComponents(low.astype(np.uint8), 8)
        raw = np.zeros_like(low, dtype=np.uint8)
        for label in range(1, count):
            component = labels == label
            if np.any(high & component):
                raw[component] = 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        clean = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel)
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)
        clean = cls._remove_small(clean, minimum)
        if expand > 0:
            size = 2 * expand + 1
            clean = cv2.dilate(clean, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
        if feather > 0:
            clean_float = cv2.GaussianBlur(clean.astype(np.float32) / 255.0, (0, 0), max(0.1, feather / 3.0))
        else:
            clean_float = clean.astype(np.float32) / 255.0
        overlay = base.copy()
        alpha = np.clip(clean_float[..., None] * 0.45, 0.0, 0.45)
        tint = np.zeros_like(overlay); tint[..., 0] = 1.0; tint[..., 1] = 0.15
        overlay = np.clip(overlay * (1.0 - alpha) + tint * alpha, 0.0, 1.0)
        return (
            raw.astype(np.float32) / 255.0, clean_float.astype(np.float32), score_norm.astype(np.float32),
            aligned.astype(np.float32), overlay.astype(np.float32), dx, dy, alignment_score,
            rotation, scale, selected_mode, used_compensation, note,
        )
