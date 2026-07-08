from __future__ import annotations

import asyncio
import os

import cv2
import numpy as np
import torch
from comfy_api.latest import io

from .mask_fill_hole import MaskFillHole


class MaskStroke(io.ComfyNode):
    MAX_RESOLUTION = 8192

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_MaskStroke",
            display_name="Mask Stroke",
            category="1hewNodes/mask",
            inputs=[
                io.Mask.Input("mask"),
                io.Boolean.Input("fill_hole", default=False),
                io.Float.Input(
                    "stroke_width",
                    default=20.0,
                    min=0.0,
                    max=cls.MAX_RESOLUTION,
                    step=0.01,
                ),
                io.Boolean.Input("include_mask", default=True),
            ],
            outputs=[io.Mask.Output(display_name="mask")],
        )

    @classmethod
    async def execute(
        cls,
        mask: torch.Tensor,
        fill_hole: bool = False,
        stroke_width: float = 20.0,
        include_mask: bool = True,
    ) -> io.NodeOutput:
        if not isinstance(mask, torch.Tensor):
            empty = torch.zeros((0, 64, 64), dtype=torch.float32)
            return io.NodeOutput(empty)

        masks = cls._normalize_masks(mask)
        if masks is None:
            empty = torch.zeros((0, 64, 64), dtype=torch.float32, device=mask.device)
            return io.NodeOutput(empty)

        batch_size = int(masks.shape[0])
        if batch_size == 0:
            return io.NodeOutput(masks)

        concurrency = max(1, min(batch_size, os.cpu_count() or 1))
        sem = asyncio.Semaphore(concurrency)
        tasks = []
        for batch_index in range(batch_size):

            async def run_one(index=batch_index):
                async with sem:
                    return await asyncio.to_thread(
                        cls._stroke_single_mask,
                        masks[index],
                        bool(fill_hole),
                        float(stroke_width),
                        bool(include_mask),
                    )

            tasks.append(run_one())

        results = await asyncio.gather(*tasks)
        output = torch.stack(results, dim=0).to(dtype=torch.float32, device=masks.device)
        return io.NodeOutput(output)

    @staticmethod
    def _normalize_masks(mask: torch.Tensor) -> torch.Tensor | None:
        current = mask.detach().to(torch.float32)
        if current.ndim == 2:
            current = current.unsqueeze(0)
        elif current.ndim == 4 and int(current.shape[-1]) >= 1:
            current = current[:, :, :, 0]
        if current.ndim != 3:
            return None
        return current.clamp(0.0, 1.0)

    @classmethod
    def _stroke_single_mask(
        cls,
        mask_2d: torch.Tensor,
        fill_hole: bool,
        stroke_width: float,
        include_mask: bool,
    ) -> torch.Tensor:
        base_mask = mask_2d
        if fill_hole:
            base_mask = MaskFillHole._fill_one(mask_2d, False)

        base_mask = base_mask.to(torch.float32).clamp(0.0, 1.0)
        height = int(base_mask.shape[0])
        width = int(base_mask.shape[1])
        stroke_width_px = cls._resolve_stroke_width(stroke_width, height, width)

        base_np = np.clip(
            np.rint(base_mask.detach().cpu().numpy() * 255.0),
            0.0,
            255.0,
        ).astype(np.uint8)
        stroke_np = cls._create_stroke_mask(base_np, stroke_width_px)

        if include_mask:
            output_np = np.maximum(base_np, stroke_np)
        else:
            output_np = stroke_np

        return torch.from_numpy(output_np.astype(np.float32) / 255.0).to(
            dtype=mask_2d.dtype,
            device=mask_2d.device,
        )

    @staticmethod
    def _resolve_stroke_width(
        stroke_width: float,
        height: int,
        width: int,
    ) -> int:
        value = max(0.0, float(stroke_width))
        if 0.0 < value < 1.0:
            return max(0, int(min(int(height), int(width)) * value))
        return max(0, int(value))

    @staticmethod
    def _create_stroke_mask(mask_np: np.ndarray, stroke_width_px: int) -> np.ndarray:
        if stroke_width_px <= 0:
            return np.zeros_like(mask_np, dtype=np.uint8)

        kernel_size = stroke_width_px * 2 + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )
        dilated_mask = cv2.dilate(mask_np, kernel, iterations=1)
        return cv2.subtract(dilated_mask, mask_np)
