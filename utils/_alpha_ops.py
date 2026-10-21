"""Shared alpha/background helpers for 1hewNodes.

Estimates the background colour of an RGBA image and parses colour
strings for the alpha nodes. Shared by Image Alpha Split and Detect
Remove SolidBG UnMult; this is a library module, not a node.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import ImageColor
from scipy import ndimage


OPTIMIZE_ALPHA = 32
DELTA_E_THRESHOLD = 0.10
CONTRAST_THRESHOLD = 3.0
# The supplied refiner uses #7F7F7F; the alpha nodes intentionally use the
# requested #808080 candidate while retaining its selection algorithm.
ALLOWED_BACKGROUNDS = ((0, 0, 0), (255, 255, 255), (128, 128, 128))
GRAY_MIN_SCORE_GAIN = 0.015


def _srgb_to_linear(x):
    x = np.asarray(x, dtype=np.float64)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _oklab(rgb):
    x = _srgb_to_linear(rgb)
    l = 0.4122214708 * x[..., 0] + 0.5363325363 * x[..., 1] + 0.0514459929 * x[..., 2]
    m = 0.2119034982 * x[..., 0] + 0.6806995451 * x[..., 1] + 0.1073969566 * x[..., 2]
    s = 0.0883024619 * x[..., 0] + 0.2817188376 * x[..., 1] + 0.6299787005 * x[..., 2]
    l, m, s = np.cbrt(l), np.cbrt(m), np.cbrt(s)
    return np.stack(
        (
            0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
        ),
        axis=-1,
    )


def _luminance(rgb):
    return _srgb_to_linear(rgb) @ np.array([0.2126, 0.7152, 0.0722])


def _visibility_margin(rgba, background):
    fg = np.asarray(rgba, dtype=np.float64)
    bg = np.asarray(background, dtype=np.float64) / 255
    alpha = fg[..., 3:4] / 255
    out = np.floor(fg[..., :3] * alpha + (255 * bg) * (1 - alpha) + 0.5) / 255
    delta = np.linalg.norm(_oklab(out) - _oklab(bg), axis=-1)
    lo, lb = _luminance(out), _luminance(bg)
    ratio = (np.maximum(lo, lb) + 0.05) / (np.minimum(lo, lb) + 0.05)
    return np.maximum(delta / DELTA_E_THRESHOLD, (ratio - 1) / (CONTRAST_THRESHOLD - 1))


def _boundary_mask(alpha):
    result = np.zeros(alpha.shape, dtype=bool)
    for threshold in (16, 128, 230):
        mask = alpha >= threshold
        inner = ndimage.binary_erosion(mask, iterations=3, border_value=0)
        result |= mask & ~inner
    return result


def _weighted_quantile(x, weights, q):
    order = np.argsort(x, kind="stable")
    at = np.searchsorted(np.cumsum(weights[order]), q * weights.sum(), side="left")
    return float(x[order[min(at, len(order) - 1)]])


def _sample_pixels(rgba, edge, limit=14000):
    mask = rgba[..., 3] >= OPTIMIZE_ALPHA
    if not mask.any():
        mask = rgba[..., 3] > 0
    if not mask.any():
        return None

    pixels = rgba[mask]
    edge_sample = edge[mask].astype(bool, copy=False)
    key = pixels[:, 0].astype(np.uint16)
    key >>= 5
    key *= 8
    key += pixels[:, 1] >> 5
    key *= 8
    key += pixels[:, 2] >> 5
    key *= 8
    key += pixels[:, 3] >> 5
    key *= 2
    key += edge_sample

    group_count = np.bincount(key, minlength=8192)
    total = len(pixels)
    truncated = total > limit
    if truncated:
        first = np.full(8192, total, dtype=np.int64)
        chunk_size = 262_144
        for start in range(0, total, chunk_size):
            stop = min(total, start + chunk_size)
            positions = np.arange(start, stop, dtype=np.int64)
            np.minimum.at(first, key[start:stop], positions)
        reserved = first[group_count > 0]
        remaining = max(0, limit - len(reserved))
        spread = (
            np.linspace(0, total - 1, remaining, dtype=np.int64)
            if remaining
            else np.empty(0, dtype=np.int64)
        )
        chosen = np.unique(np.concatenate((reserved, spread)))
        pixels = pixels[chosen]
        edge_sample = edge_sample[chosen]
        key = key[chosen]

    selected_per_group = np.bincount(key, minlength=8192)
    weight = np.sqrt(group_count[key]) / selected_per_group[key]
    weight *= np.clip(pixels[:, 3] / 128, 0.05, 1.0)
    for is_edge, share in ((False, 0.35), (True, 0.65)):
        here = edge_sample == is_edge
        if here.any():
            weight[here] *= share / weight[here].sum()
    weight /= weight.sum()
    _, groups = np.unique(key, return_inverse=True)
    return pixels, weight, groups, truncated


def _candidate_score(pixels, weight, groups, bg):
    margin = _visibility_margin(pixels, bg)
    bounded = np.minimum(margin, 2.0)
    group_weight = np.bincount(groups, weights=weight)
    group_score = (
        np.bincount(groups, weights=weight * np.minimum(margin, 1.0)) / group_weight
    )
    return float(
        0.40 * np.sum(weight * np.minimum(margin, 1.0))
        + 0.25 * _weighted_quantile(bounded, weight, 0.05) / 2
        + 0.20 * group_score.min()
        + 0.15 * np.sum(weight * bounded) / 2
    )


def _choose_background(rgba):
    alpha = rgba[..., 3]
    if not np.any(alpha) or np.all(alpha == 255):
        return (255, 255, 255)
    sample = _sample_pixels(rgba, _boundary_mask(alpha))
    # This only occurs when all alpha values are zero, already handled above.
    if sample is None:
        return (255, 255, 255)
    pixels, weight, groups, _truncated = sample
    palette = np.array(ALLOWED_BACKGROUNDS, dtype=np.uint8)
    scores = np.array([_candidate_score(pixels, weight, groups, b) for b in palette])
    tie_margin = np.array(
        [np.sum(weight * np.log1p(_visibility_margin(pixels, b))) for b in palette]
    )
    endpoint = int(np.lexsort((-tie_margin[:2], -np.round(scores[:2], 10)))[0])
    gray = 2
    index = gray if scores[gray] > scores[endpoint] + GRAY_MIN_SCORE_GAIN else endpoint
    return tuple(int(v) for v in palette[index])


def choose_smart_background(frame: torch.Tensor) -> torch.Tensor:
    """Run the supplied full refiner scoring algorithm on one alpha frame."""
    current = frame.detach().to(torch.float32).clamp(0.0, 1.0)
    if current.ndim != 3:
        raise ValueError("image frame must be [H,W,C]")
    channels = int(current.shape[2])
    if channels >= 4:
        rgba = current[:, :, :4]
    elif channels == 2:
        rgba = torch.cat((current[:, :, :1].repeat(1, 1, 3), current[:, :, 1:2]), dim=2)
    else:
        return torch.ones((1, 1, 3), dtype=torch.float32, device=current.device)
    # The reference algorithm is defined for straight-alpha 8-bit sRGB.
    rgba_u8 = torch.floor(rgba.cpu() * 255.0 + 0.5).to(torch.uint8).numpy()
    return (
        torch.tensor(
            _choose_background(rgba_u8), dtype=torch.float32, device=current.device
        )
        .div_(255.0)
        .view(1, 1, 3)
    )


def parse_color(color_str: str) -> tuple[float, float, float]:
    if color_str is None:
        return (1.0, 1.0, 1.0)
    text = str(color_str).strip().lower()
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()
    single = {
        "r": "red",
        "g": "lime",
        "b": "blue",
        "c": "cyan",
        "m": "magenta",
        "y": "yellow",
        "k": "black",
        "w": "white",
        "o": "orange",
        "p": "purple",
        "n": "brown",
        "s": "silver",
        "l": "lime",
        "i": "indigo",
        "v": "violet",
        "t": "turquoise",
        "f": "fuchsia",
        "h": "hotpink",
        "d": "darkblue",
    }
    if len(text) == 1 and text in single:
        text = single[text]
    try:
        value = float(text)
        if 0.0 <= value <= 1.0:
            return (value, value, value)
    except Exception:
        pass
    if "," in text:
        try:
            parts = [part.strip() for part in text.split(",")]
            if len(parts) >= 3:
                r, g, b = float(parts[0]), float(parts[1]), float(parts[2])
                return (
                    (r, g, b)
                    if max(r, g, b) <= 1.0
                    else (r / 255.0, g / 255.0, b / 255.0)
                )
        except Exception:
            pass
    if text.startswith("#") and len(text) in (4, 7):
        try:
            hex_str = text[1:]
            if len(hex_str) == 3:
                hex_str = "".join(ch * 2 for ch in hex_str)
            return (
                int(hex_str[0:2], 16) / 255.0,
                int(hex_str[2:4], 16) / 255.0,
                int(hex_str[4:6], 16) / 255.0,
            )
        except Exception:
            pass
    try:
        rgb = ImageColor.getrgb(text)
        return (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)
    except Exception:
        return (1.0, 1.0, 1.0)
