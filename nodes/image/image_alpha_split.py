from __future__ import annotations

import torch
from comfy_api.latest import io

from ...utils._alpha_ops import choose_smart_background, parse_color


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
        bg_rgb = None if smart_background else torch.tensor(parse_color(background_color), dtype=torch.float32, device=images.device).view(1, 1, 3)
        output_images, output_masks = [], []
        for frame in images:
            frame_bg_rgb = choose_smart_background(frame) if smart_background else bg_rgb
            flat_image, alpha_mask = cls._split_single(frame, frame_bg_rgb)
            output_images.append(flat_image)
            output_masks.append(alpha_mask)
        return io.NodeOutput(torch.stack(output_images).to(dtype=torch.float32, device=image.device), torch.stack(output_masks).to(dtype=torch.float32, device=image.device))


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
