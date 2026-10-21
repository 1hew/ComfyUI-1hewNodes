"""Shared resize-family helpers for 1hewNodes.

Preset matching and the resize/pad pipeline shared by the model-specific
Image Resize nodes. Each function takes the node class as its first
argument so every node keeps its own preset tables.

This is a library module, not a node: files under nodes/<category>/ must
define exactly one ComfyNode.
"""
from __future__ import annotations

import math

import torch



def resolve_target_size(cls, source_w: int, source_h: int, preset_size: str) -> tuple[int, int]:
        preset = cls._find_preset_resolution(preset_size)
        if preset is not None:
            return preset

        if cls._is_auto_preset(preset_size):
            # auto（不带档位）：先按输入面积定档（1k/2k/4k），再在该档内选最接近比例的预设。
            # auto (1k)/(2k)/(4k)：直接限定在对应档内选比例。
            if cls._normalize_preset_size(preset_size) == "auto":
                target_key = cls._resolve_target_key(source_w, source_h, preset_size)
                candidates = cls._preset_candidates(f"auto ({target_key})")
            else:
                candidates = cls._preset_candidates(preset_size)
            return cls._find_best_resolution(source_w, source_h, candidates)

        target_key = cls._resolve_target_key(source_w, source_h, preset_size)
        max_pixels = cls.TARGET_PIXELS[target_key]
        ratio = float(max(int(source_w), 1)) / float(max(int(source_h), 1))
        ratio = min(max(ratio, cls.MIN_SAFE_ASPECT_RATIO), cls.MAX_SAFE_ASPECT_RATIO)
        width = int(round((float(max_pixels) * ratio) ** 0.5))
        height = int(round((float(max_pixels) / ratio) ** 0.5))
        return cls._normalize_size(width, height, max_pixels)


def native_pair(
        cls,
        image: torch.Tensor,
        mask: torch.Tensor,
        target_w: int,
        target_h: int,
        fit: str,
        pad_color,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Second output: same aspect ratio as the main output, anchored to the source scale.

        pad     -> native content + pad to target ratio (area grows)
        crop    -> native center-crop to target ratio (area shrinks)
        stretch -> non-uniform stretch to target ratio, area preserved (W*H == w*h, approx)
        """
        b, h, w, _ = image.shape
        ratio = float(target_w) / float(max(target_h, 1))
        source_aspect = float(w) / float(max(h, 1))

        if fit == "pad":
            # Keep the source pixels at native scale; no implicit 16 alignment.
            if source_aspect > ratio:
                cw, ch = w, max(h, int(math.ceil(w / ratio)))
            else:
                cw, ch = max(w, int(math.ceil(h * ratio))), h
            native_img = cls._pad_to_rgb(image, ch, cw, pad_color)
            top = max((ch - h) // 2, 0)
            left = max((cw - w) // 2, 0)
            native_msk = torch.zeros((b, ch, cw), dtype=torch.float32, device=mask.device)
            native_msk[:, top : top + h, left : left + w] = mask
            return cls._cap_native(native_img, native_msk)

        if fit == "crop":
            if source_aspect > ratio:
                cw, ch = max(int(round(h * ratio)), 1), h
            else:
                cw, ch = w, max(int(round(w / ratio)), 1)
            cw = max(1, min(int(cw), w))
            ch = max(1, min(int(ch), h))
            left = max((w - cw) // 2, 0)
            top = max((h - ch) // 2, 0)
            native_img = image[:, top : top + ch, left : left + cw, :].contiguous()
            native_msk = mask[:, top : top + ch, left : left + cw].contiguous()
            return cls._cap_native(native_img, native_msk)

        # stretch: preserve area, target ratio
        area = float(w * h)
        cw = max(int(round((area * ratio) ** 0.5)), 1)
        ch = max(int(round((area / ratio) ** 0.5)), 1)
        native_img = cls._resize_image(image, ch, cw)
        native_msk = cls._resize_mask(mask, ch, cw)
        return cls._cap_native(native_img, native_msk)


def resize_pair(
        cls,
        image: torch.Tensor,
        mask: torch.Tensor,
        target_w: int,
        target_h: int,
        fit: str,
        pad_color,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b, h, w, _ = image.shape
        if fit == "stretch":
            out_img = cls._resize_image(image, target_h, target_w)
            out_msk = cls._resize_mask(mask, target_h, target_w)
        elif fit == "pad":
            scale = min(target_w / max(w, 1), target_h / max(h, 1))
            new_w = max(int(round(w * scale)), 1)
            new_h = max(int(round(h * scale)), 1)
            resized_img = cls._resize_image(image, new_h, new_w)
            resized_msk = cls._resize_mask(mask, new_h, new_w)
            out_img = cls._pad_to_rgb(resized_img, target_h, target_w, pad_color)
            top = max((target_h - new_h) // 2, 0)
            left = max((target_w - new_w) // 2, 0)
            out_msk = torch.zeros((b, target_h, target_w), dtype=torch.float32, device=resized_msk.device)
            out_msk[:, top : top + new_h, left : left + new_w] = resized_msk
        else:
            target_aspect = target_w / max(target_h, 1)
            source_aspect = w / max(h, 1)
            if source_aspect > target_aspect:
                crop_w = max(int(round(h * target_aspect)), 1)
                crop_h = h
            else:
                crop_h = max(int(round(w / target_aspect)), 1)
                crop_w = w
            left = max((w - crop_w) // 2, 0)
            top = max((h - crop_h) // 2, 0)
            cropped_img = image[:, top : top + crop_h, left : left + crop_w, :]
            cropped_msk = mask[:, top : top + crop_h, left : left + crop_w]
            out_img = cls._resize_image(cropped_img, target_h, target_w)
            out_msk = cls._resize_mask(cropped_msk, target_h, target_w)

        native_img, native_msk = native_pair(cls, image, mask, target_w, target_h, fit, pad_color)
        return out_img, out_msk, native_img, native_msk
