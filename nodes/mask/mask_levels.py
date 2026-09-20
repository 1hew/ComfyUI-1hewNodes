from __future__ import annotations

import torch
from comfy_api.latest import io


class MaskLevels(io.ComfyNode):
    """色阶（Levels）调整遮罩。

    将灰度范围 0-255 做“拉一拉”处理：

    - 小于等于 black_point 的值被压缩为 0；
    - 大于等于 white_point 的值被压缩为 255；
    - 中间区间做线性插值过渡。

    例如 black_point=3、white_point=235 时，
    0-3 -> 0，235-255 -> 255，中间线性过渡。
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_MaskLevels",
            display_name="Mask Levels",
            category="1hewNodes/mask",
            inputs=[
                io.Mask.Input("mask"),
                io.Int.Input(
                    "black_point",
                    default=0,
                    min=0,
                    max=255,
                    step=1,
                ),
                io.Int.Input(
                    "white_point",
                    default=255,
                    min=0,
                    max=255,
                    step=1,
                ),
            ],
            outputs=[io.Mask.Output(display_name="mask")],
        )

    @classmethod
    async def execute(
        cls,
        mask: torch.Tensor,
        black_point: int = 0,
        white_point: int = 255,
    ) -> io.NodeOutput:
        if not isinstance(mask, torch.Tensor):
            empty = torch.zeros((0, 64, 64), dtype=torch.float32)
            return io.NodeOutput(empty)

        m = mask.detach().to(torch.float32)
        if m.ndim == 2:
            m = m.unsqueeze(0)
        elif m.ndim == 4 and int(m.shape[-1]) >= 1:
            m = m[:, :, :, 0]
        if m.ndim != 3:
            empty = torch.zeros(
                (0, 64, 64), dtype=torch.float32, device=mask.device
            )
            return io.NodeOutput(empty)

        low = min(max(float(black_point) / 255.0, 0.0), 1.0)
        high = min(max(float(white_point) / 255.0, 0.0), 1.0)

        if abs(high - low) < 1e-6:
            # 两个点重合时退化为二值阈值：>= 该值 -> 255，否则 -> 0
            out = (m >= low).to(torch.float32)
        else:
            # 线性拉伸并夹取到 [0, 1]；当 low > high 时自然得到反转效果
            out = ((m - low) / (high - low)).clamp(0.0, 1.0)

        return io.NodeOutput(out.to(dtype=mask.dtype, device=mask.device))
