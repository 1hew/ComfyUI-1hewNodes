import math

from PIL import ImageColor
import torch
import torch.nn.functional as F
from comfy_api.latest import io


class MultiImageOverlay(io.ComfyNode):
    """
    多图层叠加：
    - image_1 为最上层，数字越大越靠底层
    - RGBA 输入使用 alpha 做正常图层合成
    - RGB 输入视为不透明图层
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_MultiImageOverlay",
            display_name="Multi Image Overlay",
            category="1hewNodes/multi",
            inputs=[
                io.Combo.Input(
                    "fit_mode",
                    options=["top_left", "center", "stretch"],
                    default="center",
                ),
                io.String.Input("color", default="1.0"),
                io.Image.Input("image_1"),
            ],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    async def execute(
        cls,
        fit_mode: str,
        color: str = "1.0",
        **kwargs,
    ) -> io.NodeOutput:
        ordered = []
        for key in kwargs.keys():
            if key.startswith("image_"):
                suffix = key[len("image_") :]
                if suffix.isdigit():
                    ordered.append((int(suffix), key))
        ordered.sort(key=lambda item: item[0])

        images = []
        for _, key in ordered:
            value = kwargs.get(key)
            if isinstance(value, torch.Tensor):
                images.append(value.to(dtype=torch.float32).clamp(0.0, 1.0))

        if not images:
            empty = torch.zeros((0, 64, 64, 3), dtype=torch.float32)
            return io.NodeOutput(empty)

        preserve_alpha = any(int(image.shape[3]) == 4 for image in images)
        batch_size = max(int(image.shape[0]) for image in images)
        images = [cls._broadcast_image(image, batch_size) for image in images]
        stack_images = list(reversed(images))

        canvas = cls._ensure_rgba(stack_images[0])
        target_h = int(canvas.shape[1])
        target_w = int(canvas.shape[2])

        result = canvas
        for image in stack_images[1:]:
            overlay = cls._ensure_rgba(image)
            overlay = cls._fit_to_canvas(overlay, target_h, target_w, fit_mode)
            result = cls._alpha_composite(result, overlay)

        background_rgb, background_enabled = cls._parse_color_with_enabled(
            color, default=(1.0, 1.0, 1.0)
        )
        result = cls._finalize_output(
            result,
            background_rgb=background_rgb,
            background_enabled=background_enabled,
            preserve_alpha=preserve_alpha,
        )

        result = result.clamp(0.0, 1.0).to(torch.float32)
        return io.NodeOutput(result)

    @staticmethod
    def _broadcast_image(image: torch.Tensor, batch_size: int) -> torch.Tensor:
        current_batch = int(image.shape[0])
        if current_batch == batch_size:
            return image
        if current_batch == 1:
            return image.repeat(batch_size, 1, 1, 1)
        repeat_times = int(math.ceil(batch_size / current_batch))
        return image.repeat(repeat_times, 1, 1, 1)[:batch_size]

    @staticmethod
    def _ensure_rgba(image: torch.Tensor) -> torch.Tensor:
        channels = int(image.shape[3])
        if channels == 4:
            return image
        if channels == 3:
            alpha = torch.ones(
                (image.shape[0], image.shape[1], image.shape[2], 1),
                dtype=image.dtype,
                device=image.device,
            )
            return torch.cat([image, alpha], dim=3)
        if channels == 1:
            rgb = image.repeat(1, 1, 1, 3)
            alpha = torch.ones(
                (image.shape[0], image.shape[1], image.shape[2], 1),
                dtype=image.dtype,
                device=image.device,
            )
            return torch.cat([rgb, alpha], dim=3)
        if channels > 4:
            return image[:, :, :, :4]

        out = torch.zeros(
            (image.shape[0], image.shape[1], image.shape[2], 4),
            dtype=image.dtype,
            device=image.device,
        )
        out[:, :, :, :channels] = image
        if channels < 4:
            out[:, :, :, 3] = 1.0
        return out

    @classmethod
    def _fit_to_canvas(
        cls,
        image: torch.Tensor,
        target_h: int,
        target_w: int,
        fit_mode: str,
    ) -> torch.Tensor:
        _, image_h, image_w, _ = image.shape
        if image_h == target_h and image_w == target_w:
            return image

        if fit_mode == "stretch":
            nchw = image.permute(0, 3, 1, 2)
            resized = F.interpolate(
                nchw,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )
            return resized.permute(0, 2, 3, 1)

        canvas = torch.zeros(
            (image.shape[0], target_h, target_w, 4),
            dtype=image.dtype,
            device=image.device,
        )

        if fit_mode == "center":
            src_top = max((image_h - target_h) // 2, 0)
            src_left = max((image_w - target_w) // 2, 0)
            dst_top = max((target_h - image_h) // 2, 0)
            dst_left = max((target_w - image_w) // 2, 0)
        else:
            src_top = 0
            src_left = 0
            dst_top = 0
            dst_left = 0

        copy_h = min(image_h, target_h)
        copy_w = min(image_w, target_w)
        canvas[
            :,
            dst_top : dst_top + copy_h,
            dst_left : dst_left + copy_w,
            :,
        ] = image[
            :,
            src_top : src_top + copy_h,
            src_left : src_left + copy_w,
            :,
        ]
        return canvas

    @staticmethod
    def _alpha_composite(base: torch.Tensor, overlay: torch.Tensor) -> torch.Tensor:
        base_rgb = base[:, :, :, :3]
        base_alpha = base[:, :, :, 3:4]
        overlay_rgb = overlay[:, :, :, :3]
        overlay_alpha = overlay[:, :, :, 3:4]

        out_alpha = overlay_alpha + base_alpha * (1.0 - overlay_alpha)
        out_rgb_premult = (
            overlay_rgb * overlay_alpha
            + base_rgb * base_alpha * (1.0 - overlay_alpha)
        )

        safe_alpha = torch.where(
            out_alpha > 1e-6,
            out_alpha,
            torch.ones_like(out_alpha),
        )
        out_rgb = torch.where(
            out_alpha > 1e-6,
            out_rgb_premult / safe_alpha,
            torch.zeros_like(out_rgb_premult),
        )
        return torch.cat([out_rgb, out_alpha], dim=3)

    @classmethod
    def _finalize_output(
        cls,
        image: torch.Tensor,
        background_rgb: tuple[float, float, float],
        background_enabled: bool,
        preserve_alpha: bool,
    ) -> torch.Tensor:
        image = image.clamp(0.0, 1.0)
        if preserve_alpha:
            return image

        background = background_rgb if background_enabled else (1.0, 1.0, 1.0)
        bg = torch.tensor(
            background,
            dtype=image.dtype,
            device=image.device,
        ).view(1, 1, 1, 3)
        rgb = image[:, :, :, :3]
        alpha = image[:, :, :, 3:4]
        return rgb * alpha + bg * (1.0 - alpha)

    @staticmethod
    def _parse_color_with_enabled(
        color_str: str | None,
        default: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> tuple[tuple[float, float, float], bool]:
        text = "" if color_str is None else str(color_str).strip().lower()
        if text in {"", "none", "off", "disable", "disabled", "transparent", "null"}:
            return (0.0, 0.0, 0.0), False
        return MultiImageOverlay._parse_color_rgb(color_str, default=default), True

    @staticmethod
    def _parse_color_rgb(
        color_str: str | None,
        default: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> tuple[float, float, float]:
        if color_str is None:
            return default

        text = str(color_str).strip().lower()
        if not text:
            return default

        if text.startswith("(") and text.endswith(")"):
            text = text[1:-1].strip()

        single = {
            "r": "red",
            "g": "lime",
            "b": "blue",
            "c": "cyan",
            "m": "magenta",
            "y": "yellow",
            "k": "black",
            "w": "white",
            "o": "orange",
            "p": "purple",
            "n": "brown",
            "s": "silver",
            "l": "lime",
            "i": "indigo",
            "v": "violet",
            "t": "turquoise",
            "f": "fuchsia",
            "h": "hotpink",
            "d": "darkblue",
        }
        if len(text) == 1 and text in single:
            text = single[text]

        try:
            value = float(text)
            if 0.0 <= value <= 1.0:
                return (value, value, value)
        except Exception:
            pass

        if "," in text:
            try:
                parts = [part.strip() for part in text.split(",")]
                if len(parts) >= 3:
                    r, g, b = [float(parts[idx]) for idx in range(3)]
                    if max(r, g, b) <= 1.0:
                        return (r, g, b)
                    return (r / 255.0, g / 255.0, b / 255.0)
            except Exception:
                pass

        hex_text = text[1:] if text.startswith("#") else text
        if len(hex_text) in (3, 6):
            try:
                if len(hex_text) == 3:
                    hex_text = "".join(ch * 2 for ch in hex_text)
                r = int(hex_text[0:2], 16) / 255.0
                g = int(hex_text[2:4], 16) / 255.0
                b = int(hex_text[4:6], 16) / 255.0
                return (r, g, b)
            except Exception:
                pass

        try:
            rgb = ImageColor.getrgb(text)
            return (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)
        except Exception:
            return default
