"""即梦（Jimeng）专用分辨率标签节点。

与即梦系列节点（jimeng_image_30 等）及本仓库的 Image Resize Jimeng 成对使用：
按图像面积推断出档位之一（1k / 2k / 4k），未连接图像时直接透传手动选择的标签。

目标面积取自即梦各档 1:1 官方尺寸：1k = 1328x1328，2k = 2048x2048，4k = 4096x4096。
2.0_pro 档位（1024 基准）不在本节点范围内，与 Image Resize Jimeng 的 auto 行为一致。
"""

import math

import torch
from comfy_api.latest import io


class StringResolutionJimeng(io.ComfyNode):
    # 各档位目标面积取该模型 1:1 官方尺寸的面积，与 API 节点的分辨率档位一一对应。
    RESOLUTION_OPTIONS = [
        ("1k", 1328 * 1328),
        ("2k", 2048 * 2048),
        ("4k", 4096 * 4096),
    ]
    RESOLUTION_LABELS = [label for label, _ in RESOLUTION_OPTIONS]

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_StringResolutionJimeng",
            display_name="String Resolution Jimeng",
            category="1hewNodes/text",
            description=(
                "从输入图像推断最接近的即梦分辨率档位（1k / 2k / 4k）；未连接图像时透传所选标签。与即梦系列节点及 Image Resize Jimeng 成对使用。"
            ),
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
            return io.NodeOutput(selection if selection in cls.RESOLUTION_LABELS else "1k")

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
