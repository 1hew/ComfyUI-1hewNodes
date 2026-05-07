from __future__ import annotations

from typing import Any

import torch
from comfy_api.latest import io


class ImageListInterleave(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_ImageListInterleave",
            display_name="Image List Interleave",
            category="1hewNodes/image",
            is_input_list=True,
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("segment_count", default=2, min=1, max=100000, step=1),
            ],
            outputs=[
                io.Image.Output(display_name="image_list", is_output_list=True),
            ],
        )

    @classmethod
    async def execute(
        cls,
        image: Any,
        segment_count: Any,
    ) -> io.NodeOutput:
        image_list = cls._to_image_list(image)
        item_count = len(image_list)
        if item_count <= 0:
            return io.NodeOutput([])

        segment_count_value = cls._resolve_int(segment_count) or 2
        actual_segments = max(1, min(segment_count_value, item_count))
        indices = cls._build_interleave_indices(item_count, actual_segments)
        return io.NodeOutput([image_list[index] for index in indices])

    @classmethod
    def _to_image_list(cls, value: Any) -> list[torch.Tensor]:
        if isinstance(value, (list, tuple)):
            images: list[torch.Tensor] = []
            for item in value:
                images.extend(cls._to_image_list(item))
            return images

        if not isinstance(value, torch.Tensor) or value.numel() == 0:
            return []

        image = value.to(torch.float32).clamp(0.0, 1.0)
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4:
            return []
        return [image[index : index + 1] for index in range(int(image.shape[0]))]

    @classmethod
    def _resolve_int(cls, value: Any) -> int | None:
        if isinstance(value, (list, tuple)):
            for item in value:
                resolved = cls._resolve_int(item)
                if resolved is not None:
                    return resolved
            return None
        try:
            return int(value)
        except Exception:
            return None

    @staticmethod
    def _build_interleave_indices(item_count: int, segment_count: int) -> list[int]:
        base_size = item_count // segment_count
        remainder = item_count % segment_count

        segments: list[list[int]] = []
        start = 0
        for segment_idx in range(segment_count):
            current_size = base_size + (1 if segment_idx < remainder else 0)
            end = start + current_size
            segments.append(list(range(start, end)))
            start = end

        indices: list[int] = []
        max_segment_length = max((len(segment) for segment in segments), default=0)
        for offset in range(max_segment_length):
            for segment in segments:
                if offset < len(segment):
                    indices.append(segment[offset])
        return indices
