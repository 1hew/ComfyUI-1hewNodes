from __future__ import annotations

import numpy as np
import torch
from comfy_api.latest import io
from skimage.measure import label, regionprops


class MaskToBBoxMask(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_MaskToBBoxMask",
            display_name="Mask To BBox Mask",
            category="1hewNodes/mask",
            inputs=[
                io.Mask.Input("mask"),
                io.Combo.Input(
                    "output_mode",
                    options=["merge", "separate"],
                    default="merge",
                ),
                io.Int.Input("divisible_by", default=8, min=1, max=1024, step=1),
            ],
            outputs=[
                io.Mask.Output(display_name="bbox_mask"),
            ],
        )

    @classmethod
    async def execute(
        cls,
        mask: torch.Tensor,
        output_mode: str = "merge",
        divisible_by: int = 8,
    ) -> io.NodeOutput:
        if not isinstance(mask, torch.Tensor):
            empty = torch.zeros((0, 64, 64), dtype=torch.float32)
            return io.NodeOutput(empty)

        masks = cls._normalize_masks(mask)
        if masks is None:
            empty = torch.zeros((0, 64, 64), dtype=torch.float32, device=mask.device)
            return io.NodeOutput(empty)

        mode = str(output_mode).strip().lower()
        divisible_value = max(1, int(divisible_by))

        output_masks: list[torch.Tensor] = []
        for idx in range(int(masks.shape[0])):
            boxes = cls._boxes_for_mask(
                masks[idx],
                output_mode=mode,
                divisible_by=divisible_value,
            )
            if mode == "separate":
                for box in boxes:
                    output_masks.append(cls._make_bbox_mask(masks[idx], box))
            else:
                output_masks.append(cls._make_bbox_mask(masks[idx], boxes[0] if boxes else None))

        if not output_masks:
            height = int(masks.shape[1])
            width = int(masks.shape[2])
            empty = torch.zeros((0, height, width), dtype=torch.float32, device=masks.device)
            return io.NodeOutput(empty)

        result = torch.stack(output_masks, dim=0).to(dtype=torch.float32, device=masks.device)
        return io.NodeOutput(result)

    @staticmethod
    def _normalize_masks(mask: torch.Tensor) -> torch.Tensor | None:
        current = mask.detach().to(torch.float32)
        if current.ndim == 2:
            current = current.unsqueeze(0)
        elif current.ndim == 4 and int(current.shape[-1]) >= 1:
            current = current[:, :, :, 0]
        if current.ndim != 3:
            return None
        return torch.clamp(current, 0.0, 1.0)

    @classmethod
    def _boxes_for_mask(
        cls,
        mask_2d: torch.Tensor,
        *,
        output_mode: str,
        divisible_by: int,
    ) -> list[tuple[int, int, int, int]]:
        mask_np = mask_2d.detach().cpu().numpy().astype(np.float32)
        binary = np.clip(mask_np, 0.0, 1.0) > 0.0
        if not np.any(binary):
            return []

        height = int(mask_np.shape[0])
        width = int(mask_np.shape[1])
        labeled = label(binary, connectivity=2)
        regions = list(regionprops(labeled))
        if not regions:
            return []

        if output_mode == "separate":
            regions = sorted(
                regions,
                key=lambda region: (
                    float(region.centroid[0]),
                    float(region.centroid[1]),
                    -int(region.area),
                ),
            )
            return [
                cls._expand_box_to_multiple(
                    cls._region_to_box(region),
                    height=height,
                    width=width,
                    divisible_by=divisible_by,
                )
                for region in regions
            ]

        rows: list[int] = []
        cols: list[int] = []
        for region in regions:
            min_row, min_col, max_row, max_col = cls._region_to_box(region)
            rows.extend([min_row, max_row])
            cols.extend([min_col, max_col])
        return [
            cls._expand_box_to_multiple(
                (min(rows), min(cols), max(rows), max(cols)),
                height=height,
                width=width,
                divisible_by=divisible_by,
            )
        ]

    @staticmethod
    def _region_to_box(region) -> tuple[int, int, int, int]:
        min_row, min_col, max_row, max_col = region.bbox
        return int(min_row), int(min_col), int(max_row), int(max_col)

    @classmethod
    def _expand_box_to_multiple(
        cls,
        box: tuple[int, int, int, int],
        *,
        height: int,
        width: int,
        divisible_by: int,
    ) -> tuple[int, int, int, int]:
        if divisible_by <= 1:
            return box

        min_row, min_col, max_row, max_col = box
        box_h = max(1, max_row - min_row)
        box_w = max(1, max_col - min_col)
        target_h = min(height, cls._round_up_to_multiple(box_h, divisible_by))
        target_w = min(width, cls._round_up_to_multiple(box_w, divisible_by))

        new_min_row = cls._place_expanded_span(min_row, max_row, target_h, height)
        new_min_col = cls._place_expanded_span(min_col, max_col, target_w, width)
        return (
            new_min_row,
            new_min_col,
            new_min_row + target_h,
            new_min_col + target_w,
        )

    @staticmethod
    def _round_up_to_multiple(value: int, multiple: int) -> int:
        return ((int(value) + int(multiple) - 1) // int(multiple)) * int(multiple)

    @staticmethod
    def _place_expanded_span(start: int, end: int, target: int, limit: int) -> int:
        current = max(1, int(end) - int(start))
        before = (int(target) - current) // 2
        placed = int(start) - before
        return max(0, min(placed, int(limit) - int(target)))

    @staticmethod
    def _make_bbox_mask(
        mask_2d: torch.Tensor,
        box: tuple[int, int, int, int] | None,
    ) -> torch.Tensor:
        out = torch.zeros_like(mask_2d, dtype=torch.float32)
        if box is None:
            return out
        min_row, min_col, max_row, max_col = box
        if max_row > min_row and max_col > min_col:
            out[min_row:max_row, min_col:max_col] = 1.0
        return out
