"""Doubao Seedream 5.0 Pro 专用比例字符串节点。

与 1hewOffice 的 1hewOffice_doubao_Seedream50Pro 成对使用：为每张输入图像推断出
API 节点 aspect_ratio 支持的 8 个比例之一（21:9 / 16:9 / 3:2 / 4:3 / 1:1 / 3:4 / 2:3 / 9:16），
未连接图像时直接透传手动选择的比例。

比例集合与 1hewOffice 的 doubao_seedream_50_pro.py 中 DoubaoSeedream50Pro.ASPECT_RATIO_OPTIONS
逐项一致（去掉 auto；该文件属于 1hewOffice，不在本仓库内）。
"""

import math

import torch
from comfy_api.latest import io


class StringRatioDoubaoSeedream50Pro(io.ComfyNode):
    # 顺序：最宽 -> 正方形 -> 最高，与 API 节点 aspect_ratio 的选项顺序一致。
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
            node_id="1hew_StringRatioDoubaoSeedream50Pro",
            display_name="String Ratio Doubao Seedream 5.0 Pro",
            category="1hewNodes/text",
            description=(
                "从输入图像推断最接近的 Doubao Seedream 5.0 Pro 支持比例；"
                "未连接图像时透传所选比例。与 1hewOffice 的 Doubao Seedream 5.0 Pro 节点成对使用。"
            ),
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
