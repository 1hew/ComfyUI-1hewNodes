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
                io.Combo.Input(
                    "border_mode",
                    options=["auto", "stroke", "ignore"],
                    default="auto",
                ),
                io.Boolean.Input("close_border", default=False),
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
        border_mode: str = "auto",
        close_border: bool = False,
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
                        str(border_mode),
                        bool(close_border),
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
        border_mode: str = "auto",
        close_border: bool = False,
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
        # Border handling: real content touching the canvas edge should still
        # get its stroke (the inward half of the dilation closes the outline),
        # but a 1px frame baked into the mask - which would otherwise become a
        # ring around the whole canvas - is an artifact and is ignored.
        mode = str(border_mode or "auto").strip().lower()
        if mode == "ignore":
            stroke_source = MaskFillHole._mask_without_border(base_np)
        elif mode == "auto":
            stroke_source = MaskFillHole._mask_ignoring_frame(base_np)
        else:
            stroke_source = base_np
        stroke_np = cls._create_stroke_mask(stroke_source, stroke_width_px)
        if mode == "ignore":
            # "ignore" also refuses to draw on the canvas edge itself, even
            # when content sits one pixel inside it.
            stroke_np = MaskFillHole._mask_without_border(stroke_np)
        if close_border and mode != "ignore":
            # The outward stroke is clipped by the canvas. Complete the cut
            # segment with a band of the same width along the nearest edge so
            # the outline closes with a consistent thickness inside the frame.
            closing = cls._close_stroke_on_edge(
                stroke_source,
                stroke_np,
                stroke_width_px,
            )
            stroke_np = np.maximum(stroke_np, closing)

        if include_mask:
            output_np = np.maximum(base_np, stroke_np)
        else:
            output_np = stroke_np

        return torch.from_numpy(output_np.astype(np.float32) / 255.0).to(
            dtype=mask_2d.dtype,
            device=mask_2d.device,
        )

    @classmethod
    def _close_stroke_on_edge(
        cls,
        stroke_source: np.ndarray,
        stroke_np: np.ndarray,
        stroke_width_px: int,
    ) -> np.ndarray:
        """Complete a stroke cut by the canvas with a same-width edge band.

        Where the mask or its stroke meets the canvas border, the outward part
        of the stroke is clipped. Drawing that cut segment as a stroke of the
        same width along the edge closes the outline; a per-pixel mirror would
        overlap the existing band unevenly and thin out at curved corners.
        """
        if stroke_width_px <= 0 or stroke_np.size == 0:
            return np.zeros_like(stroke_np)

        contact = (stroke_source > 0) | (stroke_np > 0)
        border_line = np.zeros_like(stroke_np)
        border_line[0, :] = np.where(contact[0, :], 255, 0)
        border_line[-1, :] = np.where(contact[-1, :], 255, 0)
        border_line[:, 0] = np.where(contact[:, 0], 255, 0)
        border_line[:, -1] = np.where(contact[:, -1], 255, 0)
        if not border_line.any():
            return border_line

        kernel_size = max(1, int(stroke_width_px) * 2 - 1)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )
        return cv2.dilate(border_line, kernel, iterations=1)

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
