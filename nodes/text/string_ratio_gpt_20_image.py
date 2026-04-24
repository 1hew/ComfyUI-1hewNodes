import math

import torch
from comfy_api.latest import io


class StringRatioGpt20Image(io.ComfyNode):
    RATIO_OPTIONS = [
        ("3:2", 3, 2),
        ("1:1", 1, 1),
        ("2:3", 2, 3),
        ("5:4", 5, 4),
        ("4:5", 4, 5),
        ("16:9", 16, 9),
        ("9:16", 9, 16),
        ("21:9", 21, 9),
        ("3:4", 3, 4),
        ("4:3", 4, 3),
    ]
    RATIO_LABELS = [label for label, _, _ in RATIO_OPTIONS]

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_StringRatioGpt20Image",
            display_name="String Ratio Gpt20Image",
            category="1hewNodes/text",
            inputs=[
                io.Combo.Input("selection", options=cls.RATIO_LABELS, default="3:2"),
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
            return io.NodeOutput(selection if selection in cls.RATIO_LABELS else "3:2")

        ratios = []
        batch = int(image.shape[0])
        for index in range(batch):
            height = max(int(image[index].shape[0]), 1)
            width = max(int(image[index].shape[1]), 1)
            ratios.append(cls._match_ratio(width, height))

        return io.NodeOutput("\n".join(ratios))

    @classmethod
    def _match_ratio(cls, width: int, height: int) -> str:
        image_ratio = width / max(height, 1)
        best_label = cls.RATIO_OPTIONS[0][0]
        best_diff = float("inf")

        for label, rw, rh in cls.RATIO_OPTIONS:
            ratio = rw / rh
            diff = abs(math.log(image_ratio) - math.log(ratio))
            if diff < best_diff:
                best_diff = diff
                best_label = label

        return best_label
