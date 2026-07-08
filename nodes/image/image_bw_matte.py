from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageFilter
from comfy_api.latest import io


class ImageBWMatte(io.ComfyNode):
    LEVELS_LOW = 0.5
    LEVELS_HIGH = 99.5

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_ImageBWMatte",
            display_name="Image BW Matte",
            category="1hewNodes/image",
            inputs=[
                io.Image.Input("image"),
                io.Float.Input("gamma", default=1.0, min=0.1, max=5.0, step=0.01),
                io.Int.Input("shrink_radius", default=0, min=0, max=128, step=1),
                io.Float.Input("blur_radius", default=0.0, min=0.0, max=32.0, step=0.1),
            ],
            outputs=[
                io.Mask.Output(display_name="mask"),
            ],
        )

    @classmethod
    async def execute(
        cls,
        image: torch.Tensor,
        gamma: float = 1.0,
        shrink_radius: int = 0,
        blur_radius: float = 0.0,
    ) -> io.NodeOutput:
        if not isinstance(image, torch.Tensor):
            raise ValueError("image is required")

        images = image.detach().to(torch.float32).clamp(0.0, 1.0)
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.ndim != 4:
            raise ValueError("image tensor shape must be [H,W,C] or [B,H,W,C]")

        mask_batch: list[torch.Tensor] = []
        for index in range(int(images.shape[0])):
            mask_np = cls._process_single(
                frame=images[index].detach().cpu().numpy().astype(np.float32),
                gamma=gamma,
                shrink_radius=shrink_radius,
                blur_radius=blur_radius,
            )
            mask_batch.append(torch.from_numpy(mask_np))

        out_mask = torch.stack(mask_batch, dim=0).to(dtype=torch.float32, device=image.device)
        return io.NodeOutput(out_mask)

    @classmethod
    def _process_single(
        cls,
        *,
        frame: np.ndarray,
        gamma: float,
        shrink_radius: int,
        blur_radius: float,
    ) -> np.ndarray:
        scalar = cls._extract_auto_scalar(frame)
        matte = cls._apply_levels(
            scalar=scalar,
            levels_low=cls.LEVELS_LOW,
            levels_high=cls.LEVELS_HIGH,
            gamma=gamma,
        )
        if shrink_radius > 0:
            matte = cls._shrink_mask(matte, shrink_radius)
        if blur_radius > 0.0:
            matte = cls._blur_mask(matte, blur_radius)
        return np.clip(matte, 0.0, 1.0).astype(np.float32)

    @classmethod
    def _extract_auto_scalar(cls, frame: np.ndarray) -> np.ndarray:
        current = np.clip(frame.astype(np.float32), 0.0, 1.0)
        if current.ndim != 3:
            raise ValueError("image frame must be [H,W,C]")

        channels = int(current.shape[2])

        if channels == 1:
            return current[:, :, 0].astype(np.float32)

        if channels == 2:
            gray = current[:, :, 0]
            alpha = current[:, :, 1]
            if cls._alpha_has_signal(alpha):
                return alpha.astype(np.float32)
            return gray.astype(np.float32)

        rgb = current[:, :, :3]
        alpha = current[:, :, 3] if channels >= 4 else None
        if alpha is not None and cls._alpha_has_signal(alpha):
            return alpha.astype(np.float32)
        if cls._border_is_uniform(rgb):
            return cls._background_diff(rgb)
        return cls._value(rgb)

    @staticmethod
    def _alpha_has_signal(alpha: np.ndarray) -> bool:
        if alpha.size == 0:
            return False
        return float(np.max(alpha) - np.min(alpha)) > 1e-4

    @classmethod
    def _border_is_uniform(cls, rgb: np.ndarray) -> bool:
        border = cls._collect_border(rgb)
        if border.size == 0:
            return True
        gray = border @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        gray_std = float(np.std(gray))
        color_std = float(np.max(np.std(border, axis=0)))
        return max(gray_std, color_std) <= 0.08

    @staticmethod
    def _collect_border(rgb: np.ndarray) -> np.ndarray:
        height, width = rgb.shape[:2]
        border = int(np.clip(min(height, width) // 12, 2, min(height, width)))
        return np.concatenate(
            [
                rgb[:border, :, :].reshape(-1, 3),
                rgb[-border:, :, :].reshape(-1, 3),
                rgb[:, :border, :].reshape(-1, 3),
                rgb[:, -border:, :].reshape(-1, 3),
            ],
            axis=0,
        ).astype(np.float32)

    @staticmethod
    def _background_diff(rgb: np.ndarray) -> np.ndarray:
        border = ImageBWMatte._collect_border(rgb)
        bg_rgb = np.median(border, axis=0).astype(np.float32)
        diff = np.abs(rgb - bg_rgb.reshape(1, 1, 3)).astype(np.float32)
        return np.max(diff, axis=2).astype(np.float32)

    @staticmethod
    def _value(rgb: np.ndarray) -> np.ndarray:
        return np.max(rgb[:, :, :3], axis=2).astype(np.float32)

    @staticmethod
    def _apply_levels(
        *,
        scalar: np.ndarray,
        levels_low: float,
        levels_high: float,
        gamma: float,
    ) -> np.ndarray:
        matte = np.clip(scalar.astype(np.float32), 0.0, 1.0)
        low_value = float(np.percentile(matte, float(levels_low)))
        high_value = float(np.percentile(matte, float(levels_high)))
        if not np.isfinite(low_value) or not np.isfinite(high_value):
            return matte
        if high_value - low_value <= 1e-6:
            return matte

        matte = np.clip((matte - low_value) / max(high_value - low_value, 1e-6), 0.0, 1.0)
        gamma_value = max(0.1, float(gamma))
        if abs(gamma_value - 1.0) > 1e-6:
            matte = np.power(matte, gamma_value).astype(np.float32)
        return np.clip(matte, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _shrink_mask(matte: np.ndarray, shrink_radius: int) -> np.ndarray:
        size = max(1, int(shrink_radius) * 2 + 1)
        if size <= 1:
            return matte.astype(np.float32)
        matte_u8 = np.clip(np.rint(matte * 255.0), 0.0, 255.0).astype(np.uint8)
        matte_pil = Image.fromarray(matte_u8, mode="L")
        shrunk = matte_pil.filter(ImageFilter.MinFilter(size=size))
        return (np.asarray(shrunk).astype(np.float32) / 255.0).astype(np.float32)

    @staticmethod
    def _blur_mask(matte: np.ndarray, blur_radius: float) -> np.ndarray:
        matte_u8 = np.clip(np.rint(matte * 255.0), 0.0, 255.0).astype(np.uint8)
        matte_pil = Image.fromarray(matte_u8, mode="L")
        blurred = matte_pil.filter(ImageFilter.GaussianBlur(radius=float(blur_radius)))
        return (np.asarray(blurred).astype(np.float32) / 255.0).astype(np.float32)
