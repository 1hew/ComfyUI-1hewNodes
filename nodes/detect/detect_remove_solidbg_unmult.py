from __future__ import annotations

import torch
from comfy_api.latest import io

from ...utils._alpha_ops import choose_smart_background, parse_color

# Fixed internal settings copied from ComfyUI-RH-Nodes' MekajikiUnMult core,
# including the 2026-10-15 gamut fix: negative channels are clipped and the
# positive channels are jointly scaled instead of reverting the whole pixel.
_BACKGROUND_QUANTIZATION_LEVELS = 31
_BACKGROUND_QUANTIZATION_RADIX = 32
_BACKGROUND_MIN_SUPPORT_PERCENT = 20
_BACKGROUND_HIGH_CONFIDENCE_PERCENT = 30
_BACKGROUND_SIDE_SUPPORT_PERCENT = 5
_DARK_BACKGROUND_MAX_CODE = 64


class DetectRemoveSolidBGUnMult(io.ComfyNode):
    """Port of the fixed auto-background UnMult core with output compositing."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_DetectRemoveSolidBGUnMult",
            display_name="Detect Remove SolidBG UnMult",
            category="1hewNodes/detect",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input(
                    "threshold", default=0, min=0, max=255, step=1,
                    tooltip="8-bit alpha black point. 0 disables cleanup; higher values suppress low alpha and remap remaining alpha.",
                ),
                io.String.Input(
                    "background_color", default="",
                    tooltip="Post-processing background: use Image Alpha Split color syntax, 'sm' to smart-select per image, or leave empty to retain transparent RGBA.",
                ),
            ],
            outputs=[io.Image.Output(display_name="image"), io.Mask.Output(display_name="mask")],
        )

    @staticmethod
    def _split_image(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(image, torch.Tensor):
            raise TypeError("image must be a torch.Tensor")
        if image.ndim != 4:
            raise ValueError(f"IMAGE must be BHWC, received shape {tuple(image.shape)}")
        if image.shape[0] < 1 or image.shape[1] < 1 or image.shape[2] < 1:
            raise ValueError("IMAGE batch and spatial dimensions must be nonempty")
        channels = int(image.shape[-1])
        work = torch.nan_to_num(image.to(dtype=torch.float32), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        if channels == 1:
            rgb = work.repeat(1, 1, 1, 3)
            alpha = torch.ones_like(work[..., 0])
        elif channels == 2:
            rgb = work[..., :1].repeat(1, 1, 1, 3)
            alpha = work[..., 1]
        elif channels == 3:
            rgb = work
            alpha = torch.ones_like(work[..., 0])
        elif channels == 4:
            rgb = work[..., :3]
            alpha = work[..., 3]
        else:
            raise ValueError(f"IMAGE must have 1, 2, 3, or 4 channels, received {channels}")
        return rgb, alpha

    @staticmethod
    def _gamut_safe_unpremultiply(
        premultiplied_rgb: torch.Tensor,
        divisor: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        """Clip negatives, then jointly scale positive overflow.

        A negative channel no longer reverts the whole pixel to the observed RGB.
        No background-direction rollback or neighborhood recoloring is applied.
        """
        raw = premultiplied_rgb / divisor.unsqueeze(-1)
        nonnegative = raw.clamp_min(0.0)
        peak = torch.amax(nonnegative, dim=-1, keepdim=True)
        protected = nonnegative / peak.clamp_min(1.0)
        return torch.where(active.unsqueeze(-1), protected, torch.zeros_like(protected))

    @staticmethod
    def _estimate_background(rgb: torch.Tensor) -> torch.Tensor:
        """Same quantized joint-RGB border-cluster estimator as the source node."""
        batch, height, width, _ = rgb.shape
        if height < 1 or width < 1:
            raise ValueError("Cannot estimate a background from an empty IMAGE")
        border = max(1, min(32, (min(height, width) + 49) // 50))
        strips = [rgb[:, :border].reshape(batch, -1, 3), rgb[:, max(height - border, 0):].reshape(batch, -1, 3)]
        if height > 2 * border:
            middle = rgb[:, border:height-border]
            strips.extend((middle[:, :, :border].reshape(batch, -1, 3), middle[:, :, max(width-border, 0):].reshape(batch, -1, 3)))
        side_lengths = [int(strip.shape[1]) for strip in strips]
        samples = torch.cat(strips, dim=1)
        levels, radix = _BACKGROUND_QUANTIZATION_LEVELS, _BACKGROUND_QUANTIZATION_RADIX
        backgrounds = []
        for image_samples in samples:
            quantized = torch.floor(image_samples.clamp(0.0, 1.0) * levels + 0.5).to(torch.int64)
            codes = (quantized[:, 0] * radix + quantized[:, 1]) * radix + quantized[:, 2]
            unique_codes, inverse, counts = torch.unique(codes, sorted=True, return_inverse=True, return_counts=True)
            red = unique_codes // (radix * radix)
            green = (unique_codes // radix) % radix
            blue = unique_codes % radix
            darkness = 3 * levels - (red + green + blue)
            sample_count = int(image_samples.shape[0])
            minimum_population = (sample_count * _BACKGROUND_MIN_SUPPORT_PERCENT + 99) // 100
            high_population = (sample_count * _BACKGROUND_HIGH_CONFIDENCE_PERCENT + 99) // 100
            side_counts, offset = [], 0
            cluster_count = int(unique_codes.shape[0])
            for side_length in side_lengths:
                side_inverse = inverse[offset:offset + side_length]
                side_counts.append(torch.bincount(side_inverse, minlength=cluster_count))
                offset += side_length
            side_counts = torch.stack(side_counts)
            side_lengths_tensor = counts.new_tensor(side_lengths)[:, None]
            meaningful_sides = side_counts * 100 >= side_lengths_tensor * _BACKGROUND_SIDE_SUPPORT_PERCENT
            meaningful_side_count = torch.sum(meaningful_sides, dim=0)
            opposite_sides = meaningful_sides[0] & meaningful_sides[1]
            if meaningful_sides.shape[0] == 4:
                opposite_sides = opposite_sides | (meaningful_sides[2] & meaningful_sides[3])
            low_confidence_coverage = opposite_sides | (meaningful_side_count >= 3)
            high_confidence_coverage = meaningful_side_count >= 2
            high_confidence = counts >= high_population
            spatially_supported = torch.where(high_confidence, high_confidence_coverage, low_confidence_coverage)
            eligible = (counts >= minimum_population) & spatially_supported
            if bool(torch.any(eligible).item()):
                scores = darkness * (sample_count + 1) + counts.to(torch.int64)
                winner = torch.argmax(torch.where(eligible, scores, torch.full_like(scores, -1)))
            else:
                winner = torch.argmax(counts)
            selected = image_samples[inverse == winner]
            backgrounds.append(selected.median(dim=0).values.clamp(0.0, 1.0))
        return torch.stack(backgrounds)

    @classmethod
    async def execute(
        cls, image: torch.Tensor, threshold: int = 0, background_color: str = "1.0"
    ) -> io.NodeOutput:
        rgb, input_alpha = cls._split_image(image)
        # auto_background is fixed ON; dither is fixed OFF, as requested.
        estimated_background = cls._estimate_background(rgb)
        background = estimated_background[:, None, None, :]
        epsilon = torch.finfo(rgb.dtype).eps
        delta = rgb - background
        toward_white = delta / (1.0 - background).clamp_min(epsilon)
        toward_black = -delta / background.clamp_min(epsilon)
        two_sided_alpha = torch.where(delta >= 0.0, toward_white, toward_black)
        background_code = torch.floor(estimated_background.clamp(0.0, 1.0) * 255.0 + 0.5).to(torch.int64)
        light_only_background = torch.amax(background_code, dim=-1) <= _DARK_BACKGROUND_MAX_CODE
        light_only_alpha = toward_white.clamp_min(0.0)
        channel_alpha = torch.where(light_only_background[:, None, None, None], light_only_alpha, two_sided_alpha)
        matte_alpha = torch.amax(channel_alpha, dim=-1).clamp(0.0, 1.0)

        gains = []
        for sample_alpha in matte_alpha:
            positive_alpha = sample_alpha[sample_alpha > 0.0]
            if positive_alpha.numel() > 0:
                gain = (1.0 + 1.55 * torch.sqrt(positive_alpha.median())).clamp(1.0, 2.5)
            else:
                gain = sample_alpha.new_tensor(1.0)
            gains.append(gain)
        if gains:
            gain = torch.stack(gains)[:, None, None]
            white_background = torch.all(estimated_background >= (254.5 / 255.0), dim=-1)[:, None, None]
            matte_alpha = torch.where(white_background, (matte_alpha * gain).clamp(0.0, 1.0), matte_alpha)

        divisor = torch.where(matte_alpha > 0.0, matte_alpha, torch.ones_like(matte_alpha))
        premultiplied_rgb = rgb - background * (1.0 - matte_alpha.unsqueeze(-1))
        output_rgb = cls._gamut_safe_unpremultiply(premultiplied_rgb, divisor, matte_alpha > 0.0)

        threshold_code = min(max(int(threshold), 0), 255)
        if threshold_code == 0:
            alpha_factor = matte_alpha
        elif threshold_code >= 255:
            alpha_factor = torch.zeros_like(matte_alpha)
        else:
            t = threshold_code / 255.0
            alpha_factor = ((matte_alpha - t) / (1.0 - t)).clamp(0.0, 1.0)
        output_alpha = input_alpha * alpha_factor
        discarded = alpha_factor <= 0.0
        output_rgb = torch.where(discarded.unsqueeze(-1), torch.zeros_like(output_rgb), output_rgb)
        output_alpha = torch.where(discarded, torch.zeros_like(output_alpha), output_alpha)

        # background_color is a post-processing composite target, not the matte estimate.
        color_text = "" if background_color is None else str(background_color).strip()
        if color_text:
            if color_text.lower() == "sm":
                output_images = []
                for frame_rgb, frame_alpha in zip(output_rgb, output_alpha):
                    rgba = torch.cat((frame_rgb, frame_alpha.unsqueeze(-1)), dim=-1)
                    bg_rgb = choose_smart_background(rgba)
                    composited = (frame_rgb * frame_alpha.unsqueeze(-1) + bg_rgb * (1.0 - frame_alpha.unsqueeze(-1))).clamp(0.0, 1.0)
                    output_images.append(composited)
                output_image = torch.stack(output_images, dim=0)
            else:
                bg_rgb = torch.tensor(
                    parse_color(color_text), dtype=output_rgb.dtype, device=output_rgb.device
                ).clamp(0.0, 1.0).view(1, 1, 1, 3)
                output_image = (output_rgb * output_alpha.unsqueeze(-1) + bg_rgb * (1.0 - output_alpha.unsqueeze(-1))).clamp(0.0, 1.0)
        else:
            output_image = torch.cat((output_rgb, output_alpha.unsqueeze(-1)), dim=-1)
        return io.NodeOutput(output_image.to(torch.float32), output_alpha.to(torch.float32))
