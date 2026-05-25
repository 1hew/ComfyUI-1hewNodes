import asyncio

from comfy_api.latest import io
from PIL import ImageColor
import torch
import torch.nn.functional as F


class ImagePadByBBoxMask(io.ComfyNode):
    """
    图像边界框填充器 - 按 bbox_mask 的外接框放置图像，并将其余区域填充为指定颜色。
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_ImagePadByBBoxMask",
            display_name="Image Pad By BBox Mask",
            category="1hewNodes/image/crop",
            inputs=[
                io.Image.Input("paste_image"),
                io.Mask.Input("bbox_mask"),
                io.String.Input("pad_color", default="1.0"),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
            ],
        )

    @classmethod
    async def execute(
        cls,
        paste_image: torch.Tensor,
        bbox_mask: torch.Tensor,
        pad_color: str = "1.0",
    ) -> io.NodeOutput:
        paste_image = paste_image.to(torch.float32).clamp(0.0, 1.0)
        bbox_mask = cls._ensure_mask_3d(bbox_mask.to(torch.float32).clamp(0.0, 1.0))

        image_bs = paste_image.shape[0]
        mask_bs = bbox_mask.shape[0]
        max_bs = max(image_bs, mask_bs)

        async def _proc(b: int):
            def _do():
                image_idx = b % image_bs
                mask_idx = b % mask_bs
                image = paste_image[image_idx]
                mask = bbox_mask[mask_idx]
                out_h, out_w = int(mask.shape[0]), int(mask.shape[1])
                bbox = cls.get_bbox_from_mask(mask)

                if bbox is None:
                    return cls._make_solid_background(image, out_h, out_w, pad_color)

                resized, paste_x, paste_y = cls._fit_image_to_bbox(image, bbox)
                background = cls._make_background(
                    resized,
                    image,
                    out_h,
                    out_w,
                    paste_x,
                    paste_y,
                    pad_color,
                )
                paste_h, paste_w = int(resized.shape[0]), int(resized.shape[1])
                background[paste_y:paste_y + paste_h, paste_x:paste_x + paste_w, :] = resized
                return background

            return await asyncio.to_thread(_do)

        results = await asyncio.gather(*[_proc(b) for b in range(max_bs)])
        images_t = torch.stack(results).to(paste_image.device).to(torch.float32).clamp(0.0, 1.0)
        return io.NodeOutput(images_t)

    @staticmethod
    def _ensure_mask_3d(mask: torch.Tensor) -> torch.Tensor:
        if mask.dim() == 4:
            if mask.shape[-1] == 1:
                return mask[..., 0]
            if mask.shape[1] == 1:
                return mask[:, 0, :, :]
            return mask[..., 0]
        if mask.dim() == 2:
            return mask.unsqueeze(0)
        return mask

    @staticmethod
    def get_bbox_from_mask(mask: torch.Tensor):
        mask_np = (mask.detach().cpu().numpy() * 255).astype("uint8")
        rows = (mask_np > 10).any(axis=1)
        cols = (mask_np > 10).any(axis=0)
        if not rows.any() or not cols.any():
            return None
        y_min, y_max = rows.nonzero()[0][[0, -1]]
        x_min, x_max = cols.nonzero()[0][[0, -1]]
        return (int(x_min), int(y_min), int(x_max) + 1, int(y_max) + 1)

    @staticmethod
    def _fit_image_to_bbox(image: torch.Tensor, bbox: tuple[int, int, int, int]):
        x_min, y_min, x_max, y_max = bbox
        bbox_w = max(x_max - x_min, 1)
        bbox_h = max(y_max - y_min, 1)
        image_h, image_w = int(image.shape[0]), int(image.shape[1])
        image_ratio = image_w / max(image_h, 1)
        bbox_ratio = bbox_w / max(bbox_h, 1)

        if image_ratio > bbox_ratio:
            new_w = bbox_w
            new_h = max(int(bbox_w / max(image_ratio, 1e-6)), 1)
        else:
            new_h = bbox_h
            new_w = max(int(bbox_h * image_ratio), 1)

        resized = ImagePadByBBoxMask._resize_image(image, new_h, new_w)
        paste_x = x_min + (bbox_w - new_w) // 2
        paste_y = y_min + (bbox_h - new_h) // 2
        return resized, paste_x, paste_y

    @staticmethod
    def _resize_image(image: torch.Tensor, height: int, width: int) -> torch.Tensor:
        if int(image.shape[0]) == height and int(image.shape[1]) == width:
            return image
        nchw = image.permute(2, 0, 1).unsqueeze(0)
        resized = F.interpolate(nchw, size=(height, width), mode="bilinear", align_corners=False)
        return resized.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0)

    @classmethod
    def _make_background(
        cls,
        resized: torch.Tensor,
        source: torch.Tensor,
        out_h: int,
        out_w: int,
        paste_x: int,
        paste_y: int,
        pad_color: str,
    ) -> torch.Tensor:
        fill_spec = cls._parse_pad_color(pad_color)
        if fill_spec == "extend":
            return cls._pad_from_resized(resized, out_h, out_w, paste_x, paste_y, "replicate")
        if fill_spec == "mirror":
            return cls._pad_from_resized(resized, out_h, out_w, paste_x, paste_y, "reflect")
        if fill_spec == "edge":
            color = cls._edge_color(source)
            return cls._fill_with_color(source, out_h, out_w, color)
        if fill_spec == "average":
            color = source.mean(dim=(0, 1))
            return cls._fill_with_color(source, out_h, out_w, color)
        return cls._fill_with_color(source, out_h, out_w, fill_spec)

    @classmethod
    def _make_solid_background(
        cls,
        source: torch.Tensor,
        out_h: int,
        out_w: int,
        pad_color: str,
    ) -> torch.Tensor:
        fill_spec = cls._parse_pad_color(pad_color)
        if fill_spec == "edge":
            fill_spec = cls._edge_color(source)
        elif fill_spec in ("extend", "mirror", "average"):
            fill_spec = source.mean(dim=(0, 1))
        return cls._fill_with_color(source, out_h, out_w, fill_spec)

    @staticmethod
    def _pad_from_resized(
        resized: torch.Tensor,
        out_h: int,
        out_w: int,
        paste_x: int,
        paste_y: int,
        mode: str,
    ) -> torch.Tensor:
        paste_h, paste_w = int(resized.shape[0]), int(resized.shape[1])
        pad_left = paste_x
        pad_right = out_w - paste_x - paste_w
        pad_top = paste_y
        pad_bottom = out_h - paste_y - paste_h
        nchw = resized.permute(2, 0, 1).unsqueeze(0)

        if mode == "replicate":
            padded = F.pad(nchw, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")
            return padded.squeeze(0).permute(1, 2, 0)

        padded = nchw
        left, right, top, bottom = pad_left, pad_right, pad_top, pad_bottom
        while left > 0 or right > 0 or top > 0 or bottom > 0:
            height = int(padded.shape[2])
            width = int(padded.shape[3])
            if height <= 1 or width <= 1:
                padded = F.pad(padded, (left, right, top, bottom), mode="replicate")
                break
            step_left = min(left, width - 1)
            step_right = min(right, width - 1)
            step_top = min(top, height - 1)
            step_bottom = min(bottom, height - 1)
            padded = F.pad(
                padded,
                (step_left, step_right, step_top, step_bottom),
                mode="reflect",
            )
            left -= step_left
            right -= step_right
            top -= step_top
            bottom -= step_bottom
        return padded.squeeze(0).permute(1, 2, 0)

    @staticmethod
    def _fill_with_color(source: torch.Tensor, out_h: int, out_w: int, color) -> torch.Tensor:
        channels = int(source.shape[2])
        if isinstance(color, torch.Tensor):
            color_t = color.to(dtype=source.dtype, device=source.device).flatten()
        else:
            color_t = torch.tensor(color, dtype=source.dtype, device=source.device).flatten()

        if color_t.numel() == 1:
            color_t = color_t.repeat(channels)
        elif color_t.numel() < channels:
            pad_value = torch.ones(1, dtype=source.dtype, device=source.device)
            color_t = torch.cat([color_t, pad_value.repeat(channels - color_t.numel())])
        elif color_t.numel() > channels:
            color_t = color_t[:channels]

        return color_t.view(1, 1, channels).expand(out_h, out_w, channels).clone()

    @staticmethod
    def _edge_color(image: torch.Tensor) -> torch.Tensor:
        top_edge = image[0, :, :]
        bottom_edge = image[-1, :, :]
        left_edge = image[:, 0, :]
        right_edge = image[:, -1, :]
        return torch.cat(
            [
                top_edge.reshape(-1, image.shape[2]),
                bottom_edge.reshape(-1, image.shape[2]),
                left_edge.reshape(-1, image.shape[2]),
                right_edge.reshape(-1, image.shape[2]),
            ],
            dim=0,
        ).mean(dim=0)

    @staticmethod
    def _parse_pad_color(color_str):
        if color_str is None:
            return (1.0, 1.0, 1.0)
        text = str(color_str).strip().lower()
        if text in ("edge", "e"):
            return "edge"
        if text in ("extend", "ex"):
            return "extend"
        if text in ("mirror", "mr"):
            return "mirror"
        if text in ("average", "avg", "a"):
            return "average"
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
            "q": "aqua",
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
            if 1.0 < value <= 255.0:
                normalized = value / 255.0
                return (normalized, normalized, normalized)
        except Exception:
            pass
        if "," in text:
            try:
                parts = [p.strip() for p in text.split(",")]
                if len(parts) >= 3:
                    r, g, b = [float(parts[i]) for i in range(3)]
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
            return (1.0, 1.0, 1.0)
