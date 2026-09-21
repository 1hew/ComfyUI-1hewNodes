from __future__ import annotations

import asyncio
import os

import cv2
import numpy as np
import torch
from comfy_api.latest import io
from scipy import ndimage


class ImageAlphaEdge(io.ComfyNode):
    """调整带 alpha 图像的不透明区域边缘：向内收边 / 向外扩边 + 羽化。

    典型场景：抠图（去背景）后元素边缘仍残留一圈被背景色污染的
    半透明“毛边 / 彩边”。本节点先用形态学向内收边，把被污染的
    边缘像素直接裁掉，再做高斯羽化得到干净的过渡；羽化前会固定把
    不透明区域的颜色向过渡带扩散（bleed），避免羽化后出现暗边 / 彩边。

    处理顺序：收边 / 扩边 → 颜色外扩 → 羽化。
    输出带 alpha 的 4 通道图像，并同时给出处理后的 alpha 遮罩。

    收边 / 扩边是灰度形态学（min / max 滤波），直接作用在 alpha 上：
    对局部线性的软边，半径为 r 的滤波恰好等于把该斜坡平移 r 像素，
    因此 alpha 的每一条等值线都被整体平移 r，matte 自带的抗锯齿完整
    保留，不会退化成 1px 台阶。填孔属于遮罩修补，交给别的节点处理。
    """

    MAX_RADIUS = 256
    SOLID_ALPHA = 0.9

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_ImageAlphaEdge",
            display_name="Image Alpha Edge",
            category="1hewNodes/image",
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="Input image, single or batch; uses the alpha "
                    "channel when present (>=4 channels), otherwise treats "
                    "the image as fully opaque",
                ),
                io.Int.Input(
                    "edge",
                    default=0,
                    min=-cls.MAX_RADIUS,
                    max=cls.MAX_RADIUS,
                    step=1,
                    tooltip="Signed edge adjustment in pixels: negative "
                    "shrinks the opaque region inward, positive expands it "
                    "outward. Runs as grayscale min/max filtering on the "
                    "alpha, so the matte's anti-aliasing is preserved",
                ),
                io.Float.Input(
                    "feather",
                    default=0.0,
                    min=0.0,
                    max=float(cls.MAX_RADIUS),
                    step=0.1,
                    tooltip="Gaussian blur sigma applied to the alpha after "
                    "the edge adjustment; 0 disables it",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Mask.Output(display_name="mask"),
            ],
        )

    @classmethod
    async def execute(
        cls,
        image: torch.Tensor,
        edge: int = 0,
        feather: float = 0.0,
    ) -> io.NodeOutput:
        if not isinstance(image, torch.Tensor):
            empty_image = torch.zeros((0, 64, 64, 4), dtype=torch.float32)
            empty_mask = torch.zeros((0, 64, 64), dtype=torch.float32)
            return io.NodeOutput(empty_image, empty_mask)

        images = image.detach().to(torch.float32)
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.ndim != 4:
            raise ValueError("image tensor shape must be [H,W,C] or [B,H,W,C]")

        batch_size = int(images.shape[0])
        if batch_size == 0:
            empty_image = torch.zeros(
                (0, 64, 64, 4), dtype=torch.float32, device=image.device
            )
            empty_mask = torch.zeros(
                (0, 64, 64), dtype=torch.float32, device=image.device
            )
            return io.NodeOutput(empty_image, empty_mask)

        edge_value = int(edge)
        feather_value = float(feather)

        concurrency = max(1, min(batch_size, os.cpu_count() or 1))
        sem = asyncio.Semaphore(concurrency)

        async def run_one(index: int):
            async with sem:
                return await asyncio.to_thread(
                    cls._process_single,
                    images[index].detach().cpu().numpy().astype(np.float32),
                    edge_value,
                    feather_value,
                )

        results = await asyncio.gather(*[run_one(i) for i in range(batch_size)])

        out_images = torch.stack(
            [torch.from_numpy(item[0]) for item in results], dim=0
        ).to(dtype=torch.float32, device=image.device)
        out_masks = torch.stack(
            [torch.from_numpy(item[1]) for item in results], dim=0
        ).to(dtype=torch.float32, device=image.device)
        return io.NodeOutput(out_images, out_masks)

    @classmethod
    def _process_single(
        cls,
        frame: np.ndarray,
        edge: int,
        feather: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        current = np.clip(frame.astype(np.float32), 0.0, 1.0)
        if current.ndim != 3:
            raise ValueError("image frame must be [H,W,C]")

        rgb = cls._to_rgb(current)
        alpha = cls._to_alpha(current)

        if edge != 0:
            alpha = cls._morph(alpha, edge)
        if feather > 0.0:
            # Colour extension only matters for the band the blur will spread
            # the alpha into; the kernel radius of a float Gaussian is ~4*sigma.
            rgb = cls._bleed_rgb(rgb, alpha, int(np.ceil(4.0 * feather)) + 1)
            alpha = cls._feather(alpha, feather)

        alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)
        rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)
        rgba = np.concatenate([rgb, alpha[:, :, None]], axis=2).astype(np.float32)
        return rgba, alpha

    @staticmethod
    def _to_rgb(frame: np.ndarray) -> np.ndarray:
        channels = int(frame.shape[2])
        if channels >= 3:
            return frame[:, :, :3].copy()
        if channels == 2:
            return np.repeat(frame[:, :, :1], 3, axis=2)
        if channels == 1:
            return np.repeat(frame[:, :, :1], 3, axis=2)
        raise ValueError("image must have at least one channel")

    @staticmethod
    def _to_alpha(frame: np.ndarray) -> np.ndarray:
        channels = int(frame.shape[2])
        if channels >= 4:
            return frame[:, :, 3].astype(np.float32)
        return np.ones(frame.shape[:2], dtype=np.float32)

    @staticmethod
    def _morph(alpha: np.ndarray, edge: int) -> np.ndarray:
        """Grayscale min/max filtering: shift every alpha level set by edge px.

        For a locally linear ramp a disk-shaped min filter is exactly a shift of
        that ramp, so running on the grayscale alpha (instead of a binarized
        mask) moves the edge while keeping the matte's anti-aliasing intact.
        """
        radius = abs(int(edge))
        if radius <= 0:
            return alpha.astype(np.float32)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (radius * 2 + 1, radius * 2 + 1),
        )
        source = alpha.astype(np.float32)
        if edge > 0:
            return cv2.dilate(source, kernel, iterations=1)
        return cv2.erode(source, kernel, iterations=1)

    @classmethod
    def _bleed_rgb(
        cls, rgb: np.ndarray, alpha: np.ndarray, radius: int
    ) -> np.ndarray:
        """Push the nearest *clean* colour outward, but only where it matters.

        The colour source is the solidly opaque region (alpha >= SOLID_ALPHA).
        On a soft matte the semi-transparent ramp is itself blended with the
        old background, so copying from the nearest alpha > 0 pixel would just
        spread that contamination outward and leave a halo. Targets are
        limited to the band the feather blur can actually reach; everything
        else keeps its own RGB. rgb is a private copy and is filled in place.
        """
        if radius <= 0:
            return rgb

        solid = alpha >= cls.SOLID_ALPHA
        if not np.any(solid) or np.all(solid):
            return rgb

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (radius * 2 + 1, radius * 2 + 1),
        )
        reach = (
            cv2.dilate((alpha > 0.0).astype(np.uint8), kernel, iterations=1) > 0
        )
        targets = reach & ~solid
        if not np.any(targets):
            return rgb

        _, indices = ndimage.distance_transform_edt(~solid, return_indices=True)
        ys, xs = np.nonzero(targets)
        rgb[ys, xs] = rgb[indices[0][ys, xs], indices[1][ys, xs]]
        return rgb

    @staticmethod
    def _feather(alpha: np.ndarray, radius: float) -> np.ndarray:
        sigma = float(radius)
        if sigma <= 0.0:
            return alpha.astype(np.float32)
        blurred = cv2.GaussianBlur(
            alpha.astype(np.float32),
            (0, 0),
            sigmaX=sigma,
            sigmaY=sigma,
        )
        return blurred.astype(np.float32)
