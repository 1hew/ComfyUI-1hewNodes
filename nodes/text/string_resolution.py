import math

import torch
from comfy_api.latest import io


class StringResolution(io.ComfyNode):
    RESOLUTION_OPTIONS = [
        ("0.5k", 512 * 512),
        ("1k", 1024 * 1024),
        ("2k", 2048 * 2048),
        ("4k", 4096 * 4096),
    ]
    RESOLUTION_LABELS = [label for label, _ in RESOLUTION_OPTIONS]

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_StringResolution",
            display_name="String Resolution",
            category="1hewNodes/text",
            inputs=[
                io.Combo.Input("selection", options=cls.RESOLUTION_LABELS, default="1k"),
                io.Image.Input("image", optional=True),
            ],
            outputs=[
                io.String.Output(display_name="string"),
            ],
        )

    @classmethod
    async def execute(
        cls,
        selection: str,
        image: torch.Tensor | None = None,
    ) -> io.NodeOutput:
        if not isinstance(image, torch.Tensor) or image.ndim != 4:
            return io.NodeOutput(
                selection if selection in cls.RESOLUTION_LABELS else "1k"
            )

        labels = []
        batch = int(image.shape[0])
        for index in range(batch):
            height = max(int(image[index].shape[0]), 1)
            width = max(int(image[index].shape[1]), 1)
            labels.append(cls._match_resolution(width, height))

        return io.NodeOutput("\n".join(labels))

    @classmethod
    def _match_resolution(cls, width: int, height: int) -> str:
        area = max(width * height, 1)
        best_label = cls.RESOLUTION_OPTIONS[0][0]
        best_diff = float("inf")

        for label, target_area in cls.RESOLUTION_OPTIONS:
            diff = abs(math.log(area) - math.log(target_area))
            if diff < best_diff:
                best_diff = diff
                best_label = label

        return best_label
