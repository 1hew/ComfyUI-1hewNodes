"""Image Color Match node.

Replicates Easy-Use's ``easy imageColorMatch`` node: it transfers the color
characteristics of a reference image onto a source image using one of several
color transfer algorithms.

- ``wavelet`` / ``adain`` are implemented natively (pure PyTorch, ported from
  Easy-Use's ``libs/colorfix.py``).
- ``mkl`` / ``hm`` / ``reinhard`` / ``mvgd`` / ``hm-mvgd-hm`` / ``hm-mkl-hm``
  use the third-party ``color-matcher`` package, exactly like the original node.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from comfy_api.latest import io

_METHODS = [
    "wavelet",
    "adain",
    "mkl",
    "hm",
    "reinhard",
    "mvgd",
    "hm-mvgd-hm",
    "hm-mkl-hm",
]


class ImageColorMatch(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_ImageColorMatch",
            display_name="Image Color Match",
            category="1hewNodes/color",
            description=(
                "Transfer the color of a reference image onto a source image. "
                "Supports batches; reference images are cycled (i % ref_count) "
                "when their count differs from the source batch. wavelet/adain "
                "run natively; mkl/hm/reinhard/mvgd and their hybrids use the "
                "'color-matcher' package."
            ),
            inputs=[
                io.Image.Input(
                    "source_image",
                    tooltip="Image that will be recolored",
                ),
                io.Image.Input(
                    "reference_image",
                    tooltip="Reference image whose colors are transferred",
                ),
                io.Combo.Input("method", options=_METHODS, default="mkl"),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
            ],
        )

    @classmethod
    async def execute(cls, source_image, reference_image, method):
        if not isinstance(source_image, torch.Tensor) or not isinstance(
            reference_image, torch.Tensor
        ):
            raise TypeError(
                "source_image and reference_image must be torch.Tensor"
            )

        source = cls._to_batch(source_image).cpu()
        reference = cls._to_batch(reference_image).cpu()

        source_count = int(source.shape[0])
        ref_count = int(reference.shape[0])

        # Color transfer operates on RGB; keep alpha untouched when present.
        source_rgb, source_alpha = cls._split_rgb_alpha(source)
        reference_rgb, _ = cls._split_rgb_alpha(reference)

        device = source_image.device
        out_frames = []
        for i in range(source_count):
            # Cycle the reference batch (i % ref_count), matching the project's
            # broadcast/cycling convention: 1 ref -> all frames, equal batch ->
            # frame-wise, smaller batch -> repeats, larger batch -> truncated.
            ref_frame = reference_rgb[i % ref_count]
            src_frame = source_rgb[i]
            if method in ("wavelet", "adain"):
                result = cls._color_fix_native(src_frame, ref_frame, method)
            else:
                result = cls._color_match_package(src_frame, ref_frame, method)
            out_frames.append(result)

        matched = torch.stack(out_frames, dim=0).to(
            device=device, dtype=torch.float32
        )
        if source_alpha is not None:
            matched = torch.cat([matched, source_alpha], dim=-1)

        return io.NodeOutput(matched)

    # ------------------------------------------------------------------ #
    #  tensor helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_batch(image: torch.Tensor) -> torch.Tensor:
        image = image.detach().to(torch.float32).clamp(0.0, 1.0)
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4:
            raise ValueError(
                f"expected IMAGE tensor [B, H, W, C], got {tuple(image.shape)}"
            )
        return image

    @staticmethod
    def _split_rgb_alpha(batch: torch.Tensor):
        rgb = batch[..., :3].contiguous()
        alpha = batch[..., 3:4].contiguous() if int(batch.shape[-1]) > 3 else None
        return rgb, alpha

    @staticmethod
    def _resize_rgb(frame: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Resize an [H, W, 3] float tensor with PIL LANCZOS (matches wavelet fix)."""
        np_u8 = (frame.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        pil = Image.fromarray(np_u8, mode="RGB").resize(
            (width, height), Image.LANCZOS
        )
        return torch.from_numpy(np.asarray(pil).astype(np.float32) / 255.0)

    # ------------------------------------------------------------------ #
    #  native wavelet / adain (ported from Easy-Use libs/colorfix.py)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _calc_mean_std(feat: torch.Tensor, eps: float = 1e-5):
        b, c = feat.shape[:2]
        feat_var = feat.view(b, c, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(b, c, 1, 1)
        feat_mean = feat.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
        return feat_mean, feat_std

    @classmethod
    def _adaptive_instance_norm(cls, content: torch.Tensor, style: torch.Tensor):
        size = content.size()
        style_mean, style_std = cls._calc_mean_std(style)
        content_mean, content_std = cls._calc_mean_std(content)
        normalized = (content - content_mean.expand(size)) / content_std.expand(size)
        return normalized * style_std.expand(size) + style_mean.expand(size)

    @staticmethod
    def _wavelet_blur(image: torch.Tensor, radius: int) -> torch.Tensor:
        kernel_vals = [
            [0.0625, 0.125, 0.0625],
            [0.125, 0.25, 0.125],
            [0.0625, 0.125, 0.0625],
        ]
        kernel = torch.tensor(kernel_vals, dtype=image.dtype, device=image.device)
        kernel = kernel[None, None].repeat(3, 1, 1, 1)
        image = F.pad(image, (radius, radius, radius, radius), mode="replicate")
        return F.conv2d(image, kernel, groups=3, dilation=radius)

    @classmethod
    def _wavelet_decomposition(cls, image: torch.Tensor, levels: int = 5):
        high_freq = torch.zeros_like(image)
        for i in range(levels):
            radius = 2 ** i
            low_freq = cls._wavelet_blur(image, radius)
            high_freq += image - low_freq
            image = low_freq
        return high_freq, low_freq

    @classmethod
    def _wavelet_reconstruct(cls, content: torch.Tensor, style: torch.Tensor):
        content_high, _ = cls._wavelet_decomposition(content)
        _, style_low = cls._wavelet_decomposition(style)
        return content_high + style_low

    @classmethod
    def _color_fix_native(cls, src: torch.Tensor, ref: torch.Tensor, method: str):
        """Apply wavelet or adain color fix to a single [H, W, 3] frame."""
        if method == "wavelet":
            th, tw = src.shape[:2]
            ref = cls._resize_rgb(ref, th, tw)

        content = src.permute(2, 0, 1).unsqueeze(0).contiguous()  # [1, 3, H, W]
        style = ref.permute(2, 0, 1).unsqueeze(0).contiguous()

        if method == "adain":
            result = cls._adaptive_instance_norm(content, style)
        else:
            result = cls._wavelet_reconstruct(content, style)

        return result.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0)  # [H, W, 3]

    # ------------------------------------------------------------------ #
    #  color-matcher package (mkl / hm / reinhard / mvgd / hybrids)
    # ------------------------------------------------------------------ #
    @classmethod
    def _color_match_package(cls, src: torch.Tensor, ref: torch.Tensor, method: str):
        try:
            from color_matcher import ColorMatcher
        except Exception as exc:
            raise RuntimeError(
                "Image Color Match methods 'mkl', 'hm', 'reinhard', 'mvgd', "
                "'hm-mvgd-hm' and 'hm-mkl-hm' require the 'color-matcher' "
                "package. Install it with: python -m pip install color-matcher"
            ) from exc

        src_np = src.detach().cpu().numpy().astype(np.float32)  # [H, W, 3]
        ref_np = ref.detach().cpu().numpy().astype(np.float32)

        matcher = ColorMatcher()
        result_np = matcher.transfer(src=src_np, ref=ref_np, method=method)
        result_np = np.clip(result_np, 0.0, 1.0).astype(np.float32)
        return torch.from_numpy(result_np)
