from __future__ import annotations

import torch
from PIL import ImageColor
from comfy_api.latest import io


class ImageAlphaSplit(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_ImageAlphaSplit",
            display_name="Image Alpha Split",
            category="1hewNodes/image",
            inputs=[
                io.Image.Input("image"),
                io.String.Input("background_color", default="1.0"),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Mask.Output(display_name="mask"),
            ],
        )

    @classmethod
    async def execute(
        cls,
        image: torch.Tensor,
        background_color: str = "1.0",
    ) -> io.NodeOutput:
        if not isinstance(image, torch.Tensor):
            raise ValueError("image is required")

        images = image.detach().to(torch.float32)
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.ndim != 4:
            raise ValueError("image tensor shape must be [H,W,C] or [B,H,W,C]")

        bg_rgb = torch.tensor(
            cls.parse_color(background_color),
            dtype=torch.float32,
            device=images.device,
        ).view(1, 1, 3)

        output_images: list[torch.Tensor] = []
        output_masks: list[torch.Tensor] = []
        for index in range(int(images.shape[0])):
            flat_image, alpha_mask = cls._split_single(images[index], bg_rgb)
            output_images.append(flat_image)
            output_masks.append(alpha_mask)

        result_images = torch.stack(output_images, dim=0).to(dtype=torch.float32, device=image.device)
        result_masks = torch.stack(output_masks, dim=0).to(dtype=torch.float32, device=image.device)
        return io.NodeOutput(result_images, result_masks)

    @classmethod
    def _split_single(
        cls,
        frame: torch.Tensor,
        bg_rgb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current = frame.to(torch.float32).clamp(0.0, 1.0)
        if current.ndim != 3:
            raise ValueError("image frame must be [H,W,C]")

        height, width, channels = int(current.shape[0]), int(current.shape[1]), int(current.shape[2])

        if channels >= 4:
            rgb = current[:, :, :3]
            alpha = current[:, :, 3]
            flat = (rgb * alpha.unsqueeze(2) + bg_rgb * (1.0 - alpha.unsqueeze(2))).clamp(0.0, 1.0)
            return flat, alpha

        if channels == 3:
            mask = torch.ones((height, width), dtype=torch.float32, device=current.device)
            return current[:, :, :3], mask

        if channels == 2:
            gray = current[:, :, :1].repeat(1, 1, 3)
            alpha = current[:, :, 1]
            flat = (gray * alpha.unsqueeze(2) + bg_rgb * (1.0 - alpha.unsqueeze(2))).clamp(0.0, 1.0)
            return flat, alpha

        if channels == 1:
            mask = torch.ones((height, width), dtype=torch.float32, device=current.device)
            gray = current[:, :, :1].repeat(1, 1, 3).clamp(0.0, 1.0)
            return gray, mask

        raise ValueError("image must have at least one channel")

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
