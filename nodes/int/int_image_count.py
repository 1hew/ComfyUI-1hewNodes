from typing import Any

from comfy_api.latest import io
import torch

from ...utils import make_ui_text


class IntImageCount(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_IntImageCount",
            display_name="Int Image Count",
            category="1hewNodes/int",
            inputs=[
                io.Image.Input("image_1", optional=True),
            ],
            outputs=[io.Int.Output(display_name="int")],
        )

    @classmethod
    async def execute(
        cls,
        image_1: torch.Tensor | None = None,
        **kwargs,
    ) -> io.NodeOutput:
        count = 0
        images = cls._collect_ordered_images(image_1, kwargs)
        for _, value in images:
            count += cls._count_valid_images(value)

        return io.NodeOutput(
            int(count),
            ui=make_ui_text(str(int(count))),
        )

    @classmethod
    def _collect_ordered_images(
        cls,
        image_1: torch.Tensor | None,
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

    @classmethod
    def _count_valid_images(cls, value: Any) -> int:
        if not isinstance(value, torch.Tensor):
            return 0
        if value.ndim == 4 and int(value.shape[0]) > 1:
            return sum(
                1
                for batch_index in range(int(value.shape[0]))
                if cls._is_valid_image_tensor(value[batch_index : batch_index + 1])
            )
        return 1 if cls._is_valid_image_tensor(value) else 0

    @staticmethod
    def _is_valid_image_tensor(value: Any) -> bool:
        if not isinstance(value, torch.Tensor):
            return False
        if value.numel() == 0:
            return False
        try:
            return not bool(torch.all(value == 0).item())
        except Exception:
            return False
