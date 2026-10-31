from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from comfy_api.latest import io

from ...utils._alpha_ops import choose_smart_background, parse_color


class ImageShift(io.ComfyNode):
    """整体平移图像，并输出平移后的遮罩与接缝遮罩。

    对标 LayerStyle 的 `LayerUtility: ImageShift`：`shift_x` / `shift_y` 为正时
    内容向右 / 向下移动；`cyclic` 为真时移出画布的部分从对侧卷回，否则用
    `background_color` 填充。`border_mask` 是原图四边（即卷回后的接缝）的
    模糊环带，可直接接到重绘 / inpaint 节点的 mask 上做无缝贴图。

    `background_color` 与 `Image Alpha Split` 完全一致：同一套 `utils._alpha_ops`
    解析（`1.0` / `0.5` / `#RRGGBB` / 颜色名 / `r,g,b`），同样支持 `sm`
    智能背景（按 alpha 主体在 黑 / 白 / #808080 中选择）。

    与参考实现的有意差异（均为修正）：
    - 非循环模式按精确像素取边界，不再多留 1 像素背景（零位移即恒等变换）；
    - `mask` 按帧序号匹配（1 帧自动广播，不足时沿用最后一帧），不再全部取最后一帧；
    - 保留输入通道数（含 alpha），不再强制转成 RGB；
    - `mask` 尺寸与图像不一致时先缩放对齐；
    - 接缝带宽度精确为 `border_mask_width`，不再被 expand + resize 缩放。

    `pad` 为 0 时输出与输入同尺寸；大于 0 时把平移后的图像、`mask`、
    `border_mask` 一起环向扩展 `pad` 像素（`pad_mask` 标出补出来的边框），
    让顶到画布边界的接缝端点在 inpaint 时拿到边界外上下文，补完再裁回。
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_ImageShift",
            display_name="Image Shift",
            category="1hewNodes/image",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("shift_x", default=256, min=-9999, max=9999, step=1),
                io.Int.Input("shift_y", default=256, min=-9999, max=9999, step=1),
                io.Int.Input(
                    "border_mask_width", default=20, min=0, max=999, step=1
                ),
                io.Int.Input(
                    "border_mask_blur", default=12, min=0, max=999, step=1
                ),
                io.Int.Input("pad", default=0, min=0, max=8192, step=1),
                io.Boolean.Input("cyclic", default=True),
                io.String.Input("background_color", default="1.0"),
                io.Mask.Input("mask", optional=True),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Mask.Output(display_name="mask"),
                io.Mask.Output(display_name="border_mask"),
                io.Mask.Output(display_name="pad_mask"),
            ],
        )

    @classmethod
    async def execute(
        cls,
        image: torch.Tensor,
        shift_x: int = 256,
        shift_y: int = 256,
        border_mask_width: int = 20,
        border_mask_blur: int = 12,
        pad: int = 0,
        cyclic: bool = True,
        background_color: str = "1.0",
        mask: torch.Tensor | None = None,
    ) -> io.NodeOutput:
        if not isinstance(image, torch.Tensor):
            return io.NodeOutput(None, None, None, None)

        images = image.detach().to(torch.float32)
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.ndim != 4:
            raise ValueError("image tensor shape must be [H,W,C] or [B,H,W,C]")

        batch, height, width, channels = (int(v) for v in images.shape)
        device = images.device
        amount = max(0, min(int(pad), 8192))
        out_height, out_width = height + 2 * amount, width + 2 * amount
        if batch <= 0:
            empty_image = torch.zeros(
                (0, out_height, out_width, channels),
                dtype=images.dtype,
                device=device,
            )
            empty_mask = torch.zeros(
                (0, out_height, out_width), dtype=torch.float32, device=device
            )
            return io.NodeOutput(empty_image, empty_mask, empty_mask, empty_mask)

        sx, sy = int(shift_x), int(shift_y)
        wrap = bool(cyclic)

        if wrap:
            shifted_image = torch.roll(images, shifts=(sy, sx), dims=(-3, -2))
        else:
            shifted_image = cls._translate(images, sx, sy, background_color)

        base_mask = cls._base_mask(images, mask)
        shifted_mask = cls._shift4d(
            base_mask.unsqueeze(-1),
            sx,
            sy,
            wrap,
            torch.zeros((1, 1, 1, 1), dtype=torch.float32, device=device),
        ).squeeze(-1)

        # 接缝遮罩只由画布尺寸与位移决定，与图像内容无关，算一次后广播到整批。
        border = cls._border_mask(
            height,
            width,
            int(border_mask_width),
            int(border_mask_blur),
            sx,
            sy,
            wrap,
            device,
        ).expand(batch, height, width)

        shifted_image = shifted_image.contiguous()
        shifted_mask = shifted_mask.clamp(0.0, 1.0)
        border = border.contiguous()

        if amount > 0:
            rows = torch.arange(-amount, out_height - amount, device=device) % height
            cols = torch.arange(-amount, out_width - amount, device=device) % width
            shifted_image = shifted_image[:, rows][:, :, cols]
            shifted_mask = shifted_mask[:, rows][:, :, cols]
            border = border[:, rows][:, :, cols]
            pad_mask = torch.zeros(
                (batch, out_height, out_width),
                dtype=torch.float32,
                device=device,
            )
            pad_mask[:, :amount] = 1.0
            pad_mask[:, out_height - amount :] = 1.0
            pad_mask[:, :, :amount] = 1.0
            pad_mask[:, :, out_width - amount :] = 1.0
        else:
            pad_mask = torch.zeros(
                (batch, height, width), dtype=torch.float32, device=device
            )

        return io.NodeOutput(
            shifted_image,
            shifted_mask,
            border,
            pad_mask,
        )

    @classmethod
    def _translate(
        cls,
        images: torch.Tensor,
        sx: int,
        sy: int,
        background_color: str,
    ) -> torch.Tensor:
        """非循环平移：空出的区域用 Image Alpha Split 同款颜色 / sm 填充。"""
        smart_background = str(background_color).strip().lower() == "sm"
        if smart_background:
            fill = cls._smart_fill(images)
        else:
            fill = cls._solid_fill(parse_color(background_color), images)
        return cls._shift4d(images, sx, sy, False, fill)

    @staticmethod
    def _fill_values(rgb, channels: int) -> list[float]:
        r, g, b = (float(value) for value in rgb[:3])
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        if channels == 1:
            return [gray]
        if channels == 2:
            return [gray, 1.0]
        if channels == 3:
            return [r, g, b]
        if channels == 4:
            return [r, g, b, 1.0]
        return ([r, g, b] + [1.0] * channels)[:channels]

    @classmethod
    def _solid_fill(cls, rgb, images: torch.Tensor) -> torch.Tensor:
        channels = int(images.shape[-1])
        values = cls._fill_values(rgb, channels)
        return torch.tensor(
            values, dtype=images.dtype, device=images.device
        ).view(1, 1, 1, channels)

    @classmethod
    def _smart_fill(cls, images: torch.Tensor) -> torch.Tensor:
        """逐帧运行 choose_smart_background，与 Image Alpha Split 的 sm 同一实现。"""
        batch = int(images.shape[0])
        channels = int(images.shape[-1])
        fills = []
        for index in range(batch):
            color = choose_smart_background(images[index]).reshape(-1)
            values = cls._fill_values(color.tolist(), channels)
            fills.append(
                torch.tensor(values, dtype=images.dtype, device=images.device)
            )
        return torch.stack(fills).view(batch, 1, 1, channels)

    @staticmethod
    def _shift4d(
        tensor: torch.Tensor,
        sx: int,
        sy: int,
        cyclic: bool,
        fill: torch.Tensor,
    ) -> torch.Tensor:
        """平移 (B,H,W,C) 张量；正 sx 向右、正 sy 向下，与参考实现一致。"""
        if cyclic:
            return torch.roll(tensor, shifts=(sy, sx), dims=(-3, -2))

        height, width = int(tensor.shape[-3]), int(tensor.shape[-2])
        out = torch.empty_like(tensor)
        out[...] = fill

        # 目标区间 [ys0, ys1) 采样源区间 [ys0 - sy, ys1 - sy)
        ys0, ys1 = max(0, sy), min(height, height + sy)
        xs0, xs1 = max(0, sx), min(width, width + sx)
        if ys0 < ys1 and xs0 < xs1:
            out[..., ys0:ys1, xs0:xs1, :] = tensor[
                ..., ys0 - sy : ys1 - sy, xs0 - sx : xs1 - sx, :
            ]
        return out

    @classmethod
    def _base_mask(
        cls, images: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        batch, height, width = (
            int(images.shape[0]),
            int(images.shape[1]),
            int(images.shape[2]),
        )
        device = images.device

        if isinstance(mask, torch.Tensor) and mask.numel() > 0:
            current = mask.detach().to(device=device, dtype=torch.float32)
            if current.ndim == 2:
                current = current.unsqueeze(0)
            if current.ndim != 3:
                raise ValueError("mask tensor shape must be [H,W] or [B,H,W]")
            if tuple(current.shape[-2:]) != (height, width):
                current = F.interpolate(
                    current.unsqueeze(1),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
            if int(current.shape[0]) == batch:
                return current
            if int(current.shape[0]) == 1:
                return current.expand(batch, -1, -1)
            index = torch.arange(batch, device=device).clamp(
                max=int(current.shape[0]) - 1
            )
            return current[index]

        if int(images.shape[3]) == 4:
            return images[..., 3].to(torch.float32)
        return torch.ones((batch, height, width), dtype=torch.float32, device=device)

    @classmethod
    def _border_mask(
        cls,
        height: int,
        width: int,
        band: int,
        blur: int,
        sx: int,
        sy: int,
        cyclic: bool,
        device: torch.device,
    ) -> torch.Tensor:
        """原图四边环带 -> 平移 -> 高斯模糊，返回 (1,H,W)。"""
        width_px = max(0, min(int(band), width // 2, height // 2))
        ring = torch.zeros((1, height, width, 1), dtype=torch.float32)
        if width_px > 0:
            ring[..., :width_px, :, :] = 1.0
            ring[..., height - width_px :, :, :] = 1.0
            ring[..., :, :width_px, :] = 1.0
            ring[..., :, width - width_px :, :] = 1.0

        ring = cls._shift4d(
            ring,
            sx,
            sy,
            cyclic,
            torch.zeros((1, 1, 1, 1), dtype=torch.float32),
        )

        if int(blur) > 0:
            gray = (
                ring[0, :, :, 0].numpy() * 255.0
            ).round().clip(0, 255).astype(np.uint8)
            blurred = Image.fromarray(gray).filter(
                ImageFilter.GaussianBlur(radius=float(blur))
            )
            ring = torch.from_numpy(
                np.asarray(blurred, dtype=np.float32) / 255.0
            ).view(1, height, width, 1)

        return ring.squeeze(-1).to(device=device)