from __future__ import annotations

from typing import Any

import torch
from comfy_api.latest import io


class MultiImageList(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_MultiImageList",
            display_name="Multi Image List",
            category="1hewNodes/multi",
            is_input_list=True,
            inputs=[
                io.Image.Input("image_1", optional=True),
            ],
            outputs=[
                io.Image.Output(display_name="image_list", is_output_list=True),
            ],
        )

    @classmethod
    async def execute(
        cls,
        image_1: Any = None,
        **kwargs,
    ) -> io.NodeOutput:
        image_list: list[torch.Tensor] = []
        for _, value in cls._collect_ordered_images(image_1, kwargs):
            image_list.extend(cls._split_image_value(value))
        return io.NodeOutput(image_list)

    @classmethod
    def _collect_ordered_images(
        cls,
        image_1: Any,
        kwargs: dict[str, Any],
    ) -> list[tuple[int, Any]]:
        images: list[tuple[int, Any]] = [(1, image_1)]
        for key, value in kwargs.items():
            if not isinstance(key, str) or not key.startswith("image_"):
                continue
            suffix = key[len("image_") :]
            if not suffix.isdigit():
                continue
            index = int(suffix)
            if index == 1:
                continue
            images.append((index, value))
        images.sort(key=lambda item: item[0])
        return images

    @staticmethod
    def _split_image_value(value: Any) -> list[torch.Tensor]:
        if isinstance(value, (list, tuple)):
            images: list[torch.Tensor] = []
            for item in value:
                images.extend(MultiImageList._split_image_value(item))
            return images

        if not isinstance(value, torch.Tensor) or value.numel() == 0:
            return []

        image = value.to(torch.float32).clamp(0.0, 1.0)
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4:
            return []
        return [image[index : index + 1] for index in range(int(image.shape[0]))]
