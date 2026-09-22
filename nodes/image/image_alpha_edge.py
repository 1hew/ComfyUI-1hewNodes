from __future__ import annotations

import asyncio
import os

import cv2
import numpy as np
import torch
from comfy_api.latest import io


class ImageAlphaEdge(io.ComfyNode):
    """调整带 alpha 图像的不透明区域边缘：向内收边 / 向外扩边 + 羽化。

    典型场景：抠图（去背景）后元素边缘仍残留一圈被背景色污染的
    半透明“毛边 / 彩边”。本节点先用形态学向内收边，把被污染的
    边缘像素直接裁掉，再做高斯羽化得到干净的过渡。

    采用设计软件（如 Photoshop 层遮罩）的模型：收边 / 扩边与羽化
    只修改 alpha 通道，主体的 RGB 颜色不会被模糊或重染，从而在
    烟花、发丝等细小多彩细节上不会产生多边色块或条状色带；只有
    原本完全透明、本次新出现覆盖的像素，才会得到平滑的 alpha 感知
    颜色延展，保证羽化过渡带合成时不带暗边 / 彩边。
    输出带 alpha 的 4 通道图像，并同时给出处理后的 alpha 遮罩。

    收边 / 扩边是灰度形态学（min / max 滤波），直接作用在 alpha 上：
    对局部线性的软边，半径为 r 的滤波恰好等于把该斜坡平移 r 像素，
    因此 alpha 的每一条等值线都被整体平移 r，matte 自带的抗锯齿完整
    保留，不会退化成 1px 台阶。填孔属于遮罩修补，交给别的节点处理。
    """

    MAX_RADIUS = 256

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

        # Preserve the exact sample values for the documented pass-through case.
        if edge == 0 and feather <= 0.0:
            alpha = cls._to_alpha(current)
            return current.copy(), alpha.copy()

        rgb = cls._to_rgb(current)
        alpha = cls._to_alpha(current)

        # Design-software model: the edge adjustment and feather modify the alpha
        # channel only.  RGB is the subject's own colour and is never bent,
        # blurred or recoloured - blurring or recolouring straight RGB is what
        # produced the Voronoi facets and the opaque tube-like bands on fine,
        # multi-coloured detail.  Only pixels that become newly visible (their
        # source alpha was zero) receive a colour, extended smoothly from nearby
        # covered pixels so the feathered band stays clean when composited.
        source_alpha = alpha
        source_visible = source_alpha > 1.0e-6

        if edge != 0:
            alpha = cls._morph(alpha, edge)
        # The post-edge matte is the only colour donor: if a negative edge
        # removes a contaminated fringe, that fringe must not be sampled again
        # while reconstructing the feather band.
        donor_alpha = alpha.copy()
        if feather > 0.0:
            alpha = cls._feather(alpha, feather)

        newly_visible = (alpha > 1.0e-6) & ~(source_visible)
        if np.any(newly_visible):
            rgb = cls._extend_rgb(rgb, donor_alpha, newly_visible, feather)

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

    @staticmethod
    def _extend_rgb(
        rgb: np.ndarray,
        donor_alpha: np.ndarray,
        targets: np.ndarray,
        feather: float,
    ) -> np.ndarray:
        """Fill newly visible pixels with smooth colour from the retained matte.

        A normalized convolution of RGB weighted by the post-edge alpha makes a
        continuous colour field without hard nearest-source Voronoi cells. Only
        targets that were fully transparent in the input are written, so existing
        translucent detail keeps its exact RGB. The colour-field blur matches the
        alpha feather; otherwise a large feather can reveal arbitrary RGB stored
        in transparent input pixels.
        """
        weight = donor_alpha.astype(np.float32)
        sigma = max(1.0, float(feather))
        weights = cv2.GaussianBlur(weight, (0, 0), sigmaX=sigma, sigmaY=sigma)
        usable = targets & (weights > 1.0e-6)
        if not np.any(usable):
            return rgb
        for channel in range(3):
            weighted = cv2.GaussianBlur(
                rgb[:, :, channel] * weight, (0, 0), sigmaX=sigma, sigmaY=sigma
            )
            rgb[:, :, channel][usable] = weighted[usable] / weights[usable]
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
