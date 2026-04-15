from __future__ import annotations

import torch
import torch.nn.functional as F
from comfy_api.latest import io


class ImageAlphaJoin(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_ImageAlphaJoin",
            display_name="Image Alpha Join",
            category="1hewNodes/image",
            inputs=[
                io.Image.Input("image", optional=True),
                io.Mask.Input("mask", optional=True),
                io.Boolean.Input("invert_mask", default=True),
            ],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    async def execute(
        cls,
        image: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        invert_mask: bool = True,
    ) -> io.NodeOutput:
        images = cls._normalize_images(image) if isinstance(image, torch.Tensor) else None
        masks = cls._normalize_masks(mask) if isinstance(mask, torch.Tensor) else None

        if masks is not None and bool(invert_mask):
            masks = 1.0 - masks

        if images is None and masks is None:
            return io.NodeOutput(None)

        batch_size = max(
            int(images.shape[0]) if images is not None else 0,
            int(masks.shape[0]) if masks is not None else 0,
        )
        if batch_size <= 0:
            return io.NodeOutput(None)

        output_images: list[torch.Tensor] = []
        output_device = image.device if isinstance(image, torch.Tensor) else mask.device

        for index in range(batch_size):
            frame = images[index % int(images.shape[0])] if images is not None else None
            alpha = masks[index % int(masks.shape[0])] if masks is not None else None
            output_images.append(cls._join_single(frame=frame, alpha=alpha))

        result = torch.stack(output_images, dim=0).to(dtype=torch.float32, device=output_device)
        return io.NodeOutput(result)

    @classmethod
    def _join_single(
        cls,
        *,
        frame: torch.Tensor | None,
        alpha: torch.Tensor | None,
    ) -> torch.Tensor:
        if frame is None:
            if alpha is None:
                raise ValueError("image and mask cannot both be empty")
            height, width = int(alpha.shape[0]), int(alpha.shape[1])
            return torch.zeros((height, width, 4), dtype=torch.float32, device=alpha.device)

        rgb = cls._to_rgb(frame)
        height, width = int(rgb.shape[0]), int(rgb.shape[1])

        if alpha is None:
            alpha = torch.ones((height, width), dtype=torch.float32, device=rgb.device)
        else:
            alpha = cls._fit_mask(alpha, height, width)

        return torch.cat([rgb, alpha.unsqueeze(2)], dim=2).clamp(0.0, 1.0)

    @staticmethod
    def _normalize_images(image: torch.Tensor) -> torch.Tensor:
        current = image.detach().to(torch.float32)
        if current.ndim == 3:
            current = current.unsqueeze(0)
        if current.ndim != 4:
            raise ValueError("image tensor shape must be [H,W,C] or [B,H,W,C]")
        return torch.clamp(current, 0.0, 1.0)

    @staticmethod
    def _normalize_masks(mask: torch.Tensor) -> torch.Tensor:
        current = mask.detach().to(torch.float32)
        if current.ndim == 2:
            current = current.unsqueeze(0)
        elif current.ndim == 4 and int(current.shape[-1]) >= 1:
            current = current[:, :, :, 0]
        if current.ndim != 3:
            raise ValueError("mask tensor shape must be [H,W], [B,H,W], or [B,H,W,C]")
        return torch.clamp(current, 0.0, 1.0)

    @staticmethod
    def _to_rgb(frame: torch.Tensor) -> torch.Tensor:
        current = frame.to(torch.float32).clamp(0.0, 1.0)
        if current.ndim != 3:
            raise ValueError("image frame must be [H,W,C]")

        channels = int(current.shape[2])
        if channels >= 3:
            return current[:, :, :3]
        if channels == 2:
            return current[:, :, :1].repeat(1, 1, 3)
        if channels == 1:
            return current.repeat(1, 1, 3)
        raise ValueError("image must have at least one channel")

    @staticmethod
    def _fit_mask(mask_2d: torch.Tensor, target_height: int, target_width: int) -> torch.Tensor:
        current = mask_2d.to(torch.float32).clamp(0.0, 1.0)
        if current.ndim != 2:
            raise ValueError("mask frame must be [H,W]")
        if int(current.shape[0]) == target_height and int(current.shape[1]) == target_width:
            return current

        resized = F.interpolate(
            current.unsqueeze(0).unsqueeze(0),
            size=(target_height, target_width),
            mode="nearest",
        )
        return resized[0, 0]
