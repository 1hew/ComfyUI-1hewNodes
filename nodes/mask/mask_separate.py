from __future__ import annotations

import numpy as np
import torch
from comfy_api.latest import io
from skimage.measure import label, regionprops

from ...utils import make_ui_text


class MaskSeparate(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_MaskSeparate",
            display_name="Mask Separate",
            category="1hewNodes/mask",
            inputs=[
                io.Mask.Input("mask"),
                io.Float.Input("threshold", default=0.5, min=0.0, max=1.0, step=0.01),
                io.Int.Input("min_area", default=1, min=1, max=100000000, step=1),
                io.Combo.Input(
                    "sort_mode",
                    options=["top_to_bottom", "left_to_right", "area_desc"],
                    default="top_to_bottom",
                ),
                io.Combo.Input(
                    "connectivity",
                    options=["8", "4"],
                    default="8",
                ),
            ],
            outputs=[
                io.Mask.Output(display_name="mask"),
                io.Int.Output(display_name="count"),
            ],
        )

    @classmethod
    async def execute(
        cls,
        mask: torch.Tensor,
        threshold: float = 0.5,
        min_area: int = 1,
        sort_mode: str = "top_to_bottom",
        connectivity: str = "8",
    ) -> io.NodeOutput:
        if not isinstance(mask, torch.Tensor):
            empty = torch.zeros((0, 64, 64), dtype=torch.float32)
            return io.NodeOutput(empty, 0, ui=make_ui_text("0"))

        masks = cls._normalize_masks(mask)
        if masks is None:
            empty = torch.zeros((0, 64, 64), dtype=torch.float32)
            return io.NodeOutput(empty, 0, ui=make_ui_text("0"))

        threshold_value = max(0.0, min(1.0, float(threshold)))
        min_area_value = max(1, int(min_area))
        connectivity_value = 2 if str(connectivity) == "8" else 1

        split_masks: list[torch.Tensor] = []
        for idx in range(int(masks.shape[0])):
            split_masks.extend(
                cls._split_single_mask(
                    masks[idx],
                    threshold=threshold_value,
                    min_area=min_area_value,
                    sort_mode=sort_mode,
                    connectivity=connectivity_value,
                )
            )

        if not split_masks:
            height = int(masks.shape[1]) if masks.ndim == 3 else 64
            width = int(masks.shape[2]) if masks.ndim == 3 else 64
            empty = torch.zeros(
                (0, height, width), dtype=torch.float32, device=masks.device
            )
            return io.NodeOutput(empty, 0, ui=make_ui_text("0"))

        result = torch.stack(split_masks, dim=0).to(torch.float32)
        count = int(result.shape[0])
        return io.NodeOutput(result, count, ui=make_ui_text(str(count)))

    @staticmethod
    def _normalize_masks(mask: torch.Tensor) -> torch.Tensor | None:
        if mask.ndim == 2:
            return mask.unsqueeze(0).to(torch.float32)
        if mask.ndim == 3:
            return mask.to(torch.float32)
        if mask.ndim == 4 and int(mask.shape[-1]) >= 1:
            return mask[:, :, :, 0].to(torch.float32)
        return None

    @classmethod
    def _split_single_mask(
        cls,
        mask_2d: torch.Tensor,
        *,
        threshold: float,
        min_area: int,
        sort_mode: str,
        connectivity: int,
    ) -> list[torch.Tensor]:
        mask_np = mask_2d.detach().cpu().numpy().astype(np.float32)
        mask_np = np.clip(mask_np, 0.0, 1.0)
        binary = mask_np > float(threshold)
        if not np.any(binary):
            return []

        labeled = label(binary, connectivity=connectivity)
        regions = [
            region for region in regionprops(labeled) if int(region.area) >= min_area
        ]
        regions = cls._sort_regions(regions, sort_mode)

        outputs: list[torch.Tensor] = []
        for region in regions:
            region_mask = np.zeros_like(mask_np, dtype=np.float32)
            coords = region.coords
            region_mask[coords[:, 0], coords[:, 1]] = mask_np[
                coords[:, 0], coords[:, 1]
            ]
            outputs.append(
                torch.from_numpy(region_mask).to(
                    dtype=mask_2d.dtype, device=mask_2d.device
                )
            )
        return outputs

    @staticmethod
    def _sort_regions(regions: list, sort_mode: str) -> list:
        mode = str(sort_mode).strip().lower()
        if mode == "left_to_right":
            return sorted(
                regions,
                key=lambda region: (
                    float(region.centroid[1]),
                    float(region.centroid[0]),
                    -int(region.area),
                ),
            )
        if mode == "area_desc":
            return sorted(
                regions,
                key=lambda region: (
                    -int(region.area),
                    float(region.centroid[0]),
                    float(region.centroid[1]),
                ),
            )
        return sorted(
            regions,
            key=lambda region: (
                float(region.centroid[0]),
                float(region.centroid[1]),
                -int(region.area),
            ),
        )
