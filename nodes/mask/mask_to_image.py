from __future__ import annotations

import json
from typing import Any

import numpy as np
from scipy import ndimage
import torch
from PIL import ImageColor
from comfy_api.latest import io


class MaskToImage(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_MaskToImage",
            display_name="Mask To Image",
            category="1hewNodes/mask",
            inputs=[
                io.Mask.Input("mask"),
                io.Boolean.Input(
                    "fill_hole",
                    default=False,
                    tooltip="开启后会先填充 mask 内部封闭黑洞，再进行颜色映射。",
                ),
                io.String.Input(
                    "white_area_color",
                    default="1.0",
                    tooltip="mask 白色区域映射到的颜色；输入格式与 Image Solid 的 color 一致。",
                ),
                io.String.Input(
                    "black_area_color",
                    default="0.0",
                    tooltip="mask 黑色区域映射到的颜色；输入格式与 Image Solid 的 color 一致。",
                ),
                io.Boolean.Input(
                    "output_alpha",
                    default=False,
                    tooltip="开启后输出 RGBA，RGB 仍保持黑白区域颜色映射，同时使用原始 mask 作为 alpha 通道。",
                ),
            ],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    async def execute(
        cls,
        mask: torch.Tensor,
        fill_hole: bool = False,
        white_area_color: str = "1.0",
        black_area_color: str = "0.0",
        output_alpha: bool = False,
    ) -> io.NodeOutput:
        masks = cls._normalize_masks(mask)
        if bool(fill_hole):
            masks = cls._fill_holes(masks)
        white_rgb = cls.parse_color(white_area_color)
        black_rgb = cls.parse_color(black_area_color)

        output_images: list[torch.Tensor] = []
        for index in range(int(masks.shape[0])):
            output_images.append(
                cls._map_single_mask(
                    mask_2d=masks[index],
                    black_rgb=black_rgb,
                    white_rgb=white_rgb,
                    output_alpha=bool(output_alpha),
                )
            )
        result = torch.cat(output_images, dim=0).to(dtype=torch.float32, device=mask.device)
        return io.NodeOutput(result)

    @classmethod
    def fingerprint_inputs(
        cls,
        mask: torch.Tensor,
        fill_hole: bool = False,
        white_area_color: str = "1.0",
        black_area_color: str = "0.0",
        output_alpha: bool = False,
    ) -> str:
        shape = tuple(mask.shape) if isinstance(mask, torch.Tensor) else ()
        total = float(mask.detach().to(torch.float32).sum().item()) if isinstance(mask, torch.Tensor) else 0.0
        return json.dumps(
            {
                "shape": shape,
                "sum": round(total, 6),
                "fill_hole": bool(fill_hole),
                "white_area_color": str(white_area_color),
                "black_area_color": str(black_area_color),
                "output_alpha": bool(output_alpha),
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    @staticmethod
    def _normalize_masks(mask: Any) -> torch.Tensor:
        if not isinstance(mask, torch.Tensor):
            raise ValueError("mask is required")
        current = mask.detach().to(torch.float32)
        if current.ndim == 2:
            current = current.unsqueeze(0)
        elif current.ndim == 4 and int(current.shape[-1]) >= 1:
            current = current[:, :, :, 0]
        if current.ndim != 3:
            raise ValueError("mask tensor shape must be [H,W], [B,H,W], or [B,H,W,C]")
        return torch.clamp(current, 0.0, 1.0)

    @classmethod
    def _fill_holes(cls, masks: torch.Tensor) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        for index in range(int(masks.shape[0])):
            mask_np = masks[index].detach().cpu().numpy().astype(np.float32)
            binary = mask_np > 0.5
            structure = ndimage.generate_binary_structure(2, 2)
            filled = ndimage.binary_fill_holes(binary, structure=structure)
            outputs.append(torch.from_numpy(filled.astype(np.float32)))
        return torch.stack(outputs, dim=0).to(dtype=torch.float32, device=masks.device)

    @classmethod
    def _map_single_mask(
        cls,
        *,
        mask_2d: torch.Tensor,
        black_rgb: tuple[float, float, float],
        white_rgb: tuple[float, float, float],
        output_alpha: bool,
    ) -> torch.Tensor:
        mask_np = mask_2d.detach().cpu().numpy().astype(np.float32)
        mask_np = np.clip(mask_np, 0.0, 1.0)
        mask_3 = mask_np[:, :, None]

        black = np.asarray(black_rgb, dtype=np.float32)
        white = np.asarray(white_rgb, dtype=np.float32)
        rgb = ((1.0 - mask_3) * black[None, None, :]) + (mask_3 * white[None, None, :])
        if bool(output_alpha):
            alpha = mask_3
            out = np.concatenate([rgb, alpha], axis=2)
        else:
            out = rgb
        return torch.from_numpy(out.astype(np.float32)).unsqueeze(0)

    @staticmethod
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
                    r = float(parts[0])
                    g = float(parts[1])
                    b = float(parts[2])
                    if max(r, g, b) <= 1.0:
                        return (r, g, b)
                    return (r / 255.0, g / 255.0, b / 255.0)
            except Exception:
                pass
        if text.startswith("#") and len(text) in (4, 7):
            try:
                hex_str = text[1:]
                if len(hex_str) == 3:
                    hex_str = "".join(ch * 2 for ch in hex_str)
                r = int(hex_str[0:2], 16) / 255.0
                g = int(hex_str[2:4], 16) / 255.0
                b = int(hex_str[4:6], 16) / 255.0
                return (r, g, b)
            except Exception:
                pass
        try:
            rgb = ImageColor.getrgb(text)
            return (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)
        except Exception:
            return (1.0, 1.0, 1.0)
