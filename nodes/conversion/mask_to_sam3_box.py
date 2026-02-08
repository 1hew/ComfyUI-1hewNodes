import numpy as np
import torch
from comfy_api.latest import io
from skimage.measure import label, regionprops


class MaskToSam3Box(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_MaskToSam3Box",
            display_name="Mask to SAM3 Box",
            category="1hewNodes/conversion",
            inputs=[
                io.Mask.Input("mask"),
                io.Combo.Input(
                    "condition",
                    options=["positive", "negative"],
                    default="positive",
                ),
                io.Combo.Input(
                    "output_mode",
                    options=["merge", "separate"],
                    default="merge",
                ),
            ],
            outputs=[
                io.Custom("SAM3_BOXES_PROMPT").Output(
                    display_name="sam3_box"
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        mask: torch.Tensor,
        condition: str,
        output_mode: str,
    ) -> io.NodeOutput:
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        batch, height, width = mask.shape[:3]
        prompts: list[dict[str, list]] = []
        is_positive = condition == "positive"
        for b in range(batch):
            boxes_xyxy = cls._boxes_from_mask(
                mask[b],
                width=width,
                height=height,
                output_mode=output_mode,
            )
            formatted = cls._to_collector_boxes(
                boxes_xyxy,
                width=width,
                height=height,
            )
            prompts.append(
                {
                    "boxes": formatted,
                    "labels": [is_positive] * len(formatted),
                }
            )

        if batch == 1:
            return io.NodeOutput(prompts[0])
        return io.NodeOutput(prompts)

    @classmethod
    def _boxes_from_mask(
        cls,
        mask_2d: torch.Tensor,
        width: int,
        height: int,
        output_mode: str,
    ) -> list[list[int]]:
        if mask_2d.numel() == 0:
            return []

        mask_np = (mask_2d.detach().cpu().numpy() > 0.5).astype(np.uint8)
        if not np.any(mask_np):
            return []

        if output_mode == "merge":
            ys, xs = np.where(mask_np > 0)
            x_min = int(xs.min())
            x_max = int(xs.max()) + 1
            y_min = int(ys.min())
            y_max = int(ys.max()) + 1
            box = cls._clamp_box(
                [x_min, y_min, x_max, y_max],
                width=width,
                height=height,
            )
            return [box] if box else []

        labeled = label(mask_np > 0)
        regions = regionprops(labeled)
        boxes: list[list[int]] = []
        for region in regions:
            min_row, min_col, max_row, max_col = region.bbox
            x_min = int(min_col)
            x_max = int(max_col)
            y_min = int(min_row)
            y_max = int(max_row)
            box = cls._clamp_box(
                [x_min, y_min, x_max, y_max],
                width=width,
                height=height,
            )
            if box:
                boxes.append(box)
        return boxes

    @staticmethod
    def _clamp_box(
        box_xyxy: list[int],
        width: int,
        height: int,
    ) -> list[int] | None:
        x_min, y_min, x_max, y_max = box_xyxy
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(width, x_max)
        y_max = min(height, y_max)
        if x_max <= x_min or y_max <= y_min:
            return None
        return [int(x_min), int(y_min), int(x_max), int(y_max)]

    @staticmethod
    def _to_collector_boxes(
        boxes_xyxy: list[list[int]],
        width: int,
        height: int,
    ) -> list[list[float]]:
        w = float(width) if width else 1.0
        h = float(height) if height else 1.0
        formatted: list[list[float]] = []
        for x1, y1, x2, y2 in boxes_xyxy:
            cx = (float(x1) + float(x2)) / 2.0 / w
            cy = (float(y1) + float(y2)) / 2.0 / h
            bw = (float(x2) - float(x1)) / w
            bh = (float(y2) - float(y1)) / h
            formatted.append([cx, cy, bw, bh])
        return formatted
