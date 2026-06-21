from __future__ import annotations

import torch
from comfy_api.latest import io


class ImageToMask(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_ImageToMask",
            display_name="Image to Mask",
            category="1hewNodes/conversion",
            inputs=[
                io.Image.Input("image", optional=True),
            ],
            outputs=[
                io.Mask.Output(display_name="mask"),
            ],
        )

    @classmethod
    async def execute(cls, image: torch.Tensor | None = None) -> io.NodeOutput:
        if image is None:
            return io.NodeOutput(torch.zeros((1, 64, 64), dtype=torch.float32))

        images = image.detach().to(torch.float32).clamp(0.0, 1.0)
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.ndim != 4:
            raise ValueError("image tensor shape must be [H,W,C] or [B,H,W,C]")

        channels = int(images.shape[-1])
        if channels <= 0:
            raise ValueError("image must have at least one channel")
        if channels == 1:
            mask = images[:, :, :, 0]
        else:
            rgb = images[:, :, :, :3]
            weights = torch.tensor((0.299, 0.587, 0.114), dtype=torch.float32, device=images.device)
            mask = (rgb * weights.view(1, 1, 1, 3)).sum(dim=3)

        return io.NodeOutput(mask.to(dtype=torch.float32, device=image.device))
