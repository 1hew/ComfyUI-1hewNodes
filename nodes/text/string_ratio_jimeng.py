
import torch
from comfy_api.latest import io

from ...utils._text_ops import (
    match_ratio,
)


class StringRatioJimeng(io.ComfyNode):
    RATIO_OPTIONS = [
        ("21:9", 21, 9),
        ("16:9", 16, 9),
        ("3:2", 3, 2),
        ("4:3", 4, 3),
        ("1:1", 1, 1),
        ("3:4", 3, 4),
        ("2:3", 2, 3),
        ("9:16", 9, 16),
    ]
    RATIO_LABELS = [label for label, _, _ in RATIO_OPTIONS]

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_StringRatioJimeng",
            display_name="String Ratio Jimeng",
            category="1hewNodes/text",
            inputs=[
                io.Combo.Input("selection", options=cls.RATIO_LABELS, default="1:1"),
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
            return io.NodeOutput(selection if selection in cls.RATIO_LABELS else "1:1")

        ratios = []
        batch = int(image.shape[0])
        for index in range(batch):
            height = max(int(image[index].shape[0]), 1)
            width = max(int(image[index].shape[1]), 1)
            ratios.append(match_ratio(cls, width, height))

        return io.NodeOutput("\n".join(ratios))
