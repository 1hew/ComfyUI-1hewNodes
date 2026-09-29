from __future__ import annotations

import numpy as np
import torch
from PIL import ImageColor
from scipy import ndimage
from comfy_api.latest import io


OPTIMIZE_ALPHA = 32
DELTA_E_THRESHOLD = 0.10
CONTRAST_THRESHOLD = 3.0
# The supplied refiner uses #7F7F7F; this node intentionally uses the
# requested #808080 candidate while retaining its selection algorithm.
ALLOWED_BACKGROUNDS = ((0, 0, 0), (255, 255, 255), (128, 128, 128))
GRAY_MIN_SCORE_GAIN = 0.015


def _srgb_to_linear(x):
    x = np.asarray(x, dtype=np.float64)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _oklab(rgb):
    x = _srgb_to_linear(rgb)
    l = .4122214708*x[..., 0] + .5363325363*x[..., 1] + .0514459929*x[..., 2]
    m = .2119034982*x[..., 0] + .6806995451*x[..., 1] + .1073969566*x[..., 2]
    s = .0883024619*x[..., 0] + .2817188376*x[..., 1] + .6299787005*x[..., 2]
    l, m, s = np.cbrt(l), np.cbrt(m), np.cbrt(s)
    return np.stack((.2104542553*l + .7936177850*m - .0040720468*s,
                     1.9779984951*l - 2.4285922050*m + .4505937099*s,
                     .0259040371*l + .7827717662*m - .8086757660*s), axis=-1)


def _luminance(rgb):
    return _srgb_to_linear(rgb) @ np.array([.2126, .7152, .0722])


def _visibility_margin(rgba, background):
    fg = np.asarray(rgba, dtype=np.float64)
    bg = np.asarray(background, dtype=np.float64) / 255
    alpha = fg[..., 3:4] / 255
    out = np.floor(fg[..., :3] * alpha + (255*bg) * (1-alpha) + .5) / 255
    delta = np.linalg.norm(_oklab(out) - _oklab(bg), axis=-1)
    lo, lb = _luminance(out), _luminance(bg)
    ratio = (np.maximum(lo, lb) + .05) / (np.minimum(lo, lb) + .05)
    return np.maximum(delta / DELTA_E_THRESHOLD,
                      (ratio - 1) / (CONTRAST_THRESHOLD - 1))


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
    return float(x[order[min(at, len(order)-1)]])


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
        spread = (np.linspace(0, total - 1, remaining, dtype=np.int64)
                  if remaining else np.empty(0, dtype=np.int64))
        chosen = np.unique(np.concatenate((reserved, spread)))
        pixels = pixels[chosen]
        edge_sample = edge_sample[chosen]
        key = key[chosen]

    selected_per_group = np.bincount(key, minlength=8192)
    weight = np.sqrt(group_count[key]) / selected_per_group[key]
    weight *= np.clip(pixels[:, 3]/128, .05, 1.)
    for is_edge, share in ((False, .35), (True, .65)):
        here = edge_sample == is_edge
        if here.any():
            weight[here] *= share / weight[here].sum()
    weight /= weight.sum()
    _, groups = np.unique(key, return_inverse=True)
    return pixels, weight, groups, truncated


def _candidate_score(pixels, weight, groups, bg):
    margin = _visibility_margin(pixels, bg)
    bounded = np.minimum(margin, 2.)
    group_weight = np.bincount(groups, weights=weight)
    group_score = np.bincount(groups, weights=weight*np.minimum(margin, 1.)) / group_weight
    return float(.40*np.sum(weight*np.minimum(margin, 1.)) +
                 .25*_weighted_quantile(bounded, weight, .05)/2 +
                 .20*group_score.min() + .15*np.sum(weight*bounded)/2)


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
    tie_margin = np.array([np.sum(weight*np.log1p(_visibility_margin(pixels, b))) for b in palette])
    endpoint = int(np.lexsort((-tie_margin[:2], -np.round(scores[:2], 10)))[0])
    gray = 2
    index = gray if scores[gray] > scores[endpoint] + GRAY_MIN_SCORE_GAIN else endpoint
    return tuple(int(v) for v in palette[index])


class ImageAlphaSplit(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_ImageAlphaSplit",
            display_name="Image Alpha Split",
            category="1hewNodes/image",
            inputs=[io.Image.Input("image"), io.String.Input("background_color", default="1.0")],
            outputs=[io.Image.Output(display_name="image"), io.Mask.Output(display_name="mask")],
        )

    @classmethod
    async def execute(cls, image: torch.Tensor, background_color: str = "1.0") -> io.NodeOutput:
        if not isinstance(image, torch.Tensor):
            return io.NodeOutput(None, None)
        images = image.detach().to(torch.float32)
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.ndim != 4:
            raise ValueError("image tensor shape must be [H,W,C] or [B,H,W,C]")
        if int(images.shape[0]) <= 0:
            return io.NodeOutput(None, None)

        smart_background = str(background_color).strip().lower() == "sm"
        bg_rgb = None if smart_background else torch.tensor(cls.parse_color(background_color), dtype=torch.float32, device=images.device).view(1, 1, 3)
        output_images, output_masks = [], []
        for frame in images:
            frame_bg_rgb = cls._choose_smart_background(frame) if smart_background else bg_rgb
            flat_image, alpha_mask = cls._split_single(frame, frame_bg_rgb)
            output_images.append(flat_image)
            output_masks.append(alpha_mask)
        return io.NodeOutput(torch.stack(output_images).to(dtype=torch.float32, device=image.device), torch.stack(output_masks).to(dtype=torch.float32, device=image.device))

    @classmethod
    def _choose_smart_background(cls, frame: torch.Tensor) -> torch.Tensor:
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
        return torch.tensor(_choose_background(rgba_u8), dtype=torch.float32, device=current.device).div_(255.0).view(1, 1, 3)

    @classmethod
    def _split_single(cls, frame: torch.Tensor, bg_rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        current = frame.to(torch.float32).clamp(0.0, 1.0)
        if current.ndim != 3:
            raise ValueError("image frame must be [H,W,C]")
        height, width, channels = int(current.shape[0]), int(current.shape[1]), int(current.shape[2])
        if channels >= 4:
            rgb, alpha = current[:, :, :3], current[:, :, 3]
            return (rgb * alpha.unsqueeze(2) + bg_rgb * (1.0 - alpha.unsqueeze(2))).clamp(0.0, 1.0), alpha
        if channels == 3:
            return current[:, :, :3], torch.ones((height, width), dtype=torch.float32, device=current.device)
        if channels == 2:
            gray, alpha = current[:, :, :1].repeat(1, 1, 3), current[:, :, 1]
            return (gray * alpha.unsqueeze(2) + bg_rgb * (1.0 - alpha.unsqueeze(2))).clamp(0.0, 1.0), alpha
        if channels == 1:
            return current[:, :, :1].repeat(1, 1, 3), torch.ones((height, width), dtype=torch.float32, device=current.device)
        raise ValueError("image must have at least one channel")

    @staticmethod
    def parse_color(color_str: str) -> tuple[float, float, float]:
        if color_str is None:
            return (1.0, 1.0, 1.0)
        text = str(color_str).strip().lower()
        if text.startswith("(") and text.endswith(")"):
            text = text[1:-1].strip()
        single = {"r": "red", "g": "lime", "b": "blue", "c": "cyan", "m": "magenta", "y": "yellow", "k": "black", "w": "white", "o": "orange", "p": "purple", "n": "brown", "s": "silver", "l": "lime", "i": "indigo", "v": "violet", "t": "turquoise", "f": "fuchsia", "h": "hotpink", "d": "darkblue"}
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
                    return (r, g, b) if max(r, g, b) <= 1.0 else (r / 255.0, g / 255.0, b / 255.0)
            except Exception:
                pass
        if text.startswith("#") and len(text) in (4, 7):
            try:
                hex_str = text[1:]
                if len(hex_str) == 3:
                    hex_str = "".join(ch * 2 for ch in hex_str)
                return (int(hex_str[0:2], 16) / 255.0, int(hex_str[2:4], 16) / 255.0, int(hex_str[4:6], 16) / 255.0)
            except Exception:
                pass
        try:
            rgb = ImageColor.getrgb(text)
            return (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)
        except Exception:
            return (1.0, 1.0, 1.0)
