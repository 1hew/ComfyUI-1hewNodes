import asyncio
import math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import ImageColor
from comfy_api.latest import io


class ImageMainStitch(io.ComfyNode):
    """
    ImageMainStitch：主画面拼接。支持 image_2..image_N 动态输入；
    先将 image_2..image_N 顺序拼成组合，再按 direction 与 image_1 合并。
    """
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_ImageMainStitch",
            display_name="Image Main Stitch",
            category="1hewNodes/image",
            inputs=[
                io.Image.Input("image_1"),
                io.Image.Input("image_2", optional=True),
                io.Image.Input("image_3", optional=True),
                io.Combo.Input("direction", options=["top", "bottom", "left", "right"], default="left"),
                io.Boolean.Input("match_image_size", default=True),
                io.Int.Input("spacing_width", default=10, min=0, max=1000, step=1),
                io.String.Input("spacing_color", default="1.0"),
                io.String.Input("pad_color", default="1.0"),
            ],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    async def execute(
        cls,
        image_1: torch.Tensor,
        direction: str,
        match_image_size: bool,
        spacing_width: int,
        spacing_color: str,
        pad_color: str,
        image_2: torch.Tensor | None = None,
        image_3: torch.Tensor | None = None,
        **kwargs,
    ) -> io.NodeOutput:
        a = image_1.to(dtype=torch.float32).clamp(0.0, 1.0)
        device = a.device
        others = []
        if image_2 is not None:
            others.append(image_2.to(dtype=torch.float32, device=device).clamp(0.0, 1.0))
        if image_3 is not None:
            others.append(image_3.to(dtype=torch.float32, device=device).clamp(0.0, 1.0))
        ordered = []
        for k in kwargs.keys():
            if k.startswith("image_"):
                suf = k[len("image_") :]
                if suf.isdigit():
                    idx = int(suf)
                    if idx >= 4:
                        ordered.append((idx, k))
        ordered.sort(key=lambda x: x[0])
        for _, key in ordered:
            val = kwargs.get(key)
            if val is None:
                continue
            if isinstance(val, torch.Tensor):
                others.append(val.to(dtype=torch.float32, device=device).clamp(0.0, 1.0))
            else:
                others.append(torch.zeros_like(a))

        if len(others) == 0:
            image = a.to(dtype=torch.float32).clamp(0.0, 1.0)
            return io.NodeOutput(image)

        bs = a.shape[0]
        for t in others:
            bs = max(bs, t.shape[0])
        a = cls._broadcast_image(a, bs)
        others = [cls._broadcast_image(t, bs) for t in others]

        space_rgb = cls._parse_spacing_color(spacing_color)
        pad_rgb = cls._parse_pad_color(pad_color)

        if direction in ("top", "bottom"):
            def build_pair_h():
                cur = others[0]
                for t in others[1:]:
                    cur = cls._combine_horizontal(
                        cur,
                        t,
                        match_image_size,
                        spacing_width,
                        space_rgb,
                        pad_rgb,
                        bs,
                    )
                return cur
            pair = await asyncio.to_thread(build_pair_h)
            out = await asyncio.to_thread(
                cls._stack_vertical,
                a,
                pair,
                direction,
                match_image_size,
                spacing_width,
                space_rgb,
                pad_rgb,
                bs,
            )
        else:
            def build_pair_v():
                cur = others[0]
                for t in others[1:]:
                    cur = cls._combine_vertical(
                        cur,
                        t,
                        match_image_size,
                        spacing_width,
                        space_rgb,
                        pad_rgb,
                        bs,
                    )
                return cur
            pair = await asyncio.to_thread(build_pair_v)
            out = await asyncio.to_thread(
                cls._stack_horizontal,
                a,
                pair,
                direction,
                match_image_size,
                spacing_width,
                space_rgb,
                pad_rgb,
                bs,
            )

        image = torch.clamp(out, min=0.0, max=1.0).to(torch.float32)
        return io.NodeOutput(image)

    # --- 合并 2 与 3 ---
    @classmethod
    def _combine_horizontal(
        cls, img2, img3, match, spacing_width, space_rgb, pad_rgb, batch_size
    ):
        _, h2, w2, _ = img2.shape
        _, h3, w3, _ = img3.shape
        # 合并阶段：
        # - match=True 按较大高度等比缩放；
        # - match=False 统一到较大高度，使用居中填充保持原图尺寸不变。
        if match:
            target_h = max(h2, h3)
            img2 = cls._resize_keep_ratio(img2, None, target_h)
            img3 = cls._resize_keep_ratio(img3, None, target_h)
        else:
            target_h = max(h2, h3)
            img2 = cls._pad_to_rgb(img2, target_h, w2, pad_rgb)
            img3 = cls._pad_to_rgb(img3, target_h, w3, pad_rgb)

        _, hh2, ww2, _ = img2.shape
        _, hh3, ww3, _ = img3.shape
        strip = cls._make_strip(
            hh2, spacing_width, space_rgb, axis="v", dtype=img2.dtype,
            device=img2.device, batch_size=batch_size,
        )

        return torch.cat([img2, strip, img3], dim=2)

    @classmethod
    def _combine_vertical(
        cls, img2, img3, match, spacing_width, space_rgb, pad_rgb, batch_size
    ):
        _, h2, w2, _ = img2.shape
        _, h3, w3, _ = img3.shape
        # 合并阶段：
        # - match=True 按较大宽度等比缩放；
        # - match=False 统一到较大宽度，使用居中填充保持原图尺寸不变。
        if match:
            target_w = max(w2, w3)
            img2 = cls._resize_keep_ratio(img2, target_w, None)
            img3 = cls._resize_keep_ratio(img3, target_w, None)
        else:
            target_w = max(w2, w3)
            img2 = cls._pad_to_rgb(img2, h2, target_w, pad_rgb)
            img3 = cls._pad_to_rgb(img3, h3, target_w, pad_rgb)

        _, hh2, ww2, _ = img2.shape
        _, hh3, ww3, _ = img3.shape
        strip = cls._make_strip(
            spacing_width, ww2, space_rgb, axis="h", dtype=img2.dtype,
            device=img2.device, batch_size=batch_size,
        )

        return torch.cat([img2, strip, img3], dim=1)

    # --- 将 (2,3) 组合贴到 1 ---
    @classmethod
    def _stack_vertical(
        cls,
        img1,
        pair,
        direction,
        match,
        spacing_width,
        space_rgb,
        pad_rgb,
        batch_size,
    ):
        _, h1, w1, _ = img1.shape
        _, hp, wp, _ = pair.shape
        # 外部贴合：以 image_1 的宽度为基准
        target_w = w1

        if match:
            img1 = cls._resize_keep_ratio(img1, target_w, None)
            pair = cls._resize_keep_ratio(pair, target_w, None)
        else:
            unified_w = max(w1, wp)
            img1 = cls._pad_to_rgb(img1, h1, unified_w, pad_rgb)
            pair = cls._pad_to_rgb(pair, hp, unified_w, pad_rgb)

        _, h1, w1, _ = img1.shape
        _, hp, wp, _ = pair.shape
        # 间隔条宽度与最终统一宽度一致
        strip = cls._make_strip(spacing_width, w1 if match else unified_w, space_rgb, axis="h", dtype=img1.dtype, device=img1.device, batch_size=batch_size)

        if direction == "bottom":
            return torch.cat([img1, strip, pair], dim=1)
        return torch.cat([pair, strip, img1], dim=1)

    @classmethod
    def _stack_horizontal(
        cls,
        img1,
        pair,
        direction,
        match,
        spacing_width,
        space_rgb,
        pad_rgb,
        batch_size,
    ):
        _, h1, w1, _ = img1.shape
        _, hp, wp, _ = pair.shape
        # 外部贴合：以 image_1 的高度为基准
        target_h = h1

        if match:
            img1 = cls._resize_keep_ratio(img1, None, target_h)
            pair = cls._resize_keep_ratio(pair, None, target_h)
        else:
            unified_h = max(h1, hp)
            img1 = cls._pad_to_rgb(img1, unified_h, w1, pad_rgb)
            pair = cls._pad_to_rgb(pair, unified_h, wp, pad_rgb)

        _, h1, w1, _ = img1.shape
        _, hp, wp, _ = pair.shape
        # 间隔条高度与最终统一高度一致
        strip = cls._make_strip(h1 if match else unified_h, spacing_width, space_rgb, axis="v", dtype=img1.dtype, device=img1.device, batch_size=batch_size)

        if direction == "right":
            return torch.cat([img1, strip, pair], dim=2)
        return torch.cat([pair, strip, img1], dim=2)

    # --- 工具函数 ---

    @staticmethod
    def _resize_keep_ratio(img, target_w, target_h):
        """按目标宽或高等比缩放，另一维自适应；不拉伸。"""
        _, h, w, _ = img.shape
        if target_w is None:
            scale = target_h / max(h, 1)
            new_w = max(int(round(w * scale)), 1)
            new_h = target_h
        elif target_h is None:
            scale = target_w / max(w, 1)
            new_h = max(int(round(h * scale)), 1)
            new_w = target_w
        else:
            new_w, new_h = target_w, target_h

        nchw = img.permute(0, 3, 1, 2)
        out = F.interpolate(
            nchw, size=(new_h, new_w), mode="bicubic", align_corners=False
        ).permute(0, 2, 3, 1)
        return out

    @staticmethod
    def _pad_to_rgb(img, target_h, target_w, fill_rgb):
        """居中填充到指定尺寸，使用 RGB 填充颜色。"""
        b, h, w, c = img.shape
        top = max((target_h - h) // 2, 0)
        left = max((target_w - w) // 2, 0)
        h_end = min(top + h, target_h)
        w_end = min(left + w, target_w)
        if isinstance(fill_rgb, str) and fill_rgb == "extend":
            pad_top = top
            pad_bottom = max(target_h - (top + h), 0)
            pad_left = left
            pad_right = max(target_w - (left + w), 0)
            nchw = img.permute(0, 3, 1, 2)
            padded = F.pad(
                nchw, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate"
            )
            return padded.permute(0, 2, 3, 1)
        if isinstance(fill_rgb, str) and fill_rgb == "mirror":
            pad_top = top
            pad_bottom = max(target_h - (top + h), 0)
            pad_left = left
            pad_right = max(target_w - (left + w), 0)
            nchw = img.permute(0, 3, 1, 2)
            while pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
                hh = int(nchw.shape[2])
                ww = int(nchw.shape[3])
                st = min(pad_top, max(hh - 1, 0))
                sb = min(pad_bottom, max(hh - 1, 0))
                sl = min(pad_left, max(ww - 1, 0))
                sr = min(pad_right, max(ww - 1, 0))
                nchw = F.pad(nchw, (sl, sr, st, sb), mode="reflect")
                pad_top -= st
                pad_bottom -= sb
                pad_left -= sl
                pad_right -= sr
            return nchw.permute(0, 2, 3, 1)
        out = torch.zeros((b, target_h, target_w, c), dtype=img.dtype, device=img.device)
        if isinstance(fill_rgb, str) and fill_rgb == "edge":
            if top > 0 or (target_h - h_end) > 0:
                top_row = img[:, 0:1, :, :].mean(dim=(1, 2))
                top_color = top_row.view(b, 1, 1, c)
                bottom_row = img[:, -1:, :, :].mean(dim=(1, 2))
                bottom_color = bottom_row.view(b, 1, 1, c)
                if top > 0:
                    out[:, :top, :, :] = top_color.expand(b, top, target_w, c)
                if (target_h - h_end) > 0:
                    bottom_pad = target_h - h_end
                    out[:, h_end:, :, :] = bottom_color.expand(b, bottom_pad, target_w, c)
            if left > 0 or (target_w - w_end) > 0:
                left_col = img[:, :, 0:1, :].mean(dim=(1, 2))
                left_color = left_col.view(b, 1, 1, c)
                right_col = img[:, :, -1:, :].mean(dim=(1, 2))
                right_color = right_col.view(b, 1, 1, c)
                if left > 0:
                    out[:, :, :left, :] = left_color.expand(b, target_h, left, c)
                if (target_w - w_end) > 0:
                    right_pad = target_w - w_end
                    out[:, :, w_end:, :] = right_color.expand(b, target_h, right_pad, c)
        elif isinstance(fill_rgb, str) and fill_rgb == "average":
            avg = img.mean(dim=(1, 2))
            avg_color = avg.view(b, 1, 1, c)
            out[:] = avg_color.expand(b, target_h, target_w, c)
        else:
            fill_t = torch.tensor(fill_rgb, dtype=img.dtype, device=img.device)
            out[:] = fill_t
        out[:, top:h_end, left:w_end, :] = img[:, : h_end - top, : w_end - left, :]
        return out

    @staticmethod
    def _pad_or_crop_to_rgb(img, target_h, target_w, fill_rgb):
        """居中对齐到指定尺寸：
        - 若源尺寸小于目标尺寸，则填充居中；
        - 若源尺寸大于目标尺寸，则居中裁剪到目标尺寸；
        使用 RGB 填充颜色处理需要填充的区域。
        """
        b, h, w, c = img.shape
        out = torch.zeros(
            (b, target_h, target_w, c), dtype=img.dtype, device=img.device
        )
        fill_t = torch.tensor(fill_rgb, dtype=img.dtype, device=img.device)
        out[:] = fill_t

        # 计算源裁剪窗口（居中裁剪）
        src_top = max((h - target_h) // 2, 0)
        src_left = max((w - target_w) // 2, 0)
        copy_h = min(h, target_h)
        copy_w = min(w, target_w)

        # 计算目标粘贴位置（居中填充）
        dst_top = max((target_h - h) // 2, 0)
        dst_left = max((target_w - w) // 2, 0)

        out[:, dst_top : dst_top + copy_h, dst_left : dst_left + copy_w, :] = (
            img[:, src_top : src_top + copy_h, src_left : src_left + copy_w, :]
        )
        return out

    @staticmethod
    def _make_strip(h, w, fill_rgb, axis, dtype, device, batch_size=1):
        """生成间隔条：axis='v' 垂直条，axis='h' 水平条。"""
        if w <= 0:
            return torch.zeros((batch_size, h, 0, 3), dtype=dtype, device=device)
        if axis == "v":
            out = torch.zeros((batch_size, h, w, 3), dtype=dtype, device=device)
            out[:] = torch.tensor(fill_rgb, dtype=dtype, device=device)
            return out
        out = torch.zeros((batch_size, w, h, 3), dtype=dtype, device=device)
        out[:] = torch.tensor(fill_rgb, dtype=dtype, device=device)
        return out.permute(0, 2, 1, 3)

    @staticmethod
    def _broadcast_image(img, batch_size):
        b = img.shape[0]
        if b == batch_size:
            return img
        if b == 1:
            return img.repeat(batch_size, 1, 1, 1)
        reps = int(math.ceil(batch_size / b))
        tiled = img.repeat(reps, 1, 1, 1)[:batch_size]
        return tiled
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
        if text in ("average", "a"):
            return "average"
        if text.startswith("(") and text.endswith(")"):
            text = text[1:-1].strip()
        single = {
            "r": "red",
            "g": "green",
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
            v = float(text)
            if 0.0 <= v <= 1.0:
                return (v, v, v)
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
        if text.startswith("#") and len(text) in (4, 7):
            try:
                hex_str = text[1:]
                if len(hex_str) == 3:
                    hex_str = "".join(ch * 2 for ch in hex_str)
                r = int(hex_str[0:2], 16) / 255.0
                g = int(hex_str[2:4], 16) / 255.0
                b = int(hex_str[4:6], 16) / 255.0
                return (r, g, b)
            except Exception:
                pass
        try:
            rgb = ImageColor.getrgb(text)
            return (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)
        except Exception:
            return (1.0, 1.0, 1.0)

    @staticmethod
    def _parse_spacing_color(color_str):
        if color_str is None:
            return (1.0, 1.0, 1.0)
        text = str(color_str).strip().lower()
        if text.startswith("(") and text.endswith(")"):
            text = text[1:-1].strip()
        single = {
            "r": "red",
            "g": "green",
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
            v = float(text)
            if 0.0 <= v <= 1.0:
                return (v, v, v)
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
        if text.startswith("#") and len(text) in (4, 7):
            try:
                hex_str = text[1:]
                if len(hex_str) == 3:
                    hex_str = "".join(ch * 2 for ch in hex_str)
                r = int(hex_str[0:2], 16) / 255.0
                g = int(hex_str[2:4], 16) / 255.0
                b = int(hex_str[4:6], 16) / 255.0
                return (r, g, b)
            except Exception:
                pass
        try:
            rgb = ImageColor.getrgb(text)
            return (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)
        except Exception:
            return (1.0, 1.0, 1.0)
