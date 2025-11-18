import math
from PIL import ImageColor
import torch
import torch.nn.functional as F
from comfy_api.latest import io


class MultiImageStitch(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="1hew_MultiImageStitch",
            display_name="Multi Image Stitch",
            category="1hewNodes/multi",
            inputs=[
                io.Combo.Input("direction", options=["top", "bottom", "left", "right"], default="right"),
                io.Boolean.Input("match_image_size", default=True),
                io.Int.Input("spacing_width", default=10, min=0, max=1000, step=1),
                io.String.Input("spacing_color", default="1.0"),
                io.String.Input("pad_color", default="1.0"),
                io.Image.Input("image_1"),
                io.Image.Input("image_2", optional=True),
            ],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    async def execute(
        cls,
        direction,
        match_image_size,
        spacing_width,
        spacing_color,
        pad_color,
        **kwargs,
    ) -> io.NodeOutput:
        ordered = []
        for key in kwargs.keys():
            if key.startswith("image_"):
                suf = key[len("image_") :]
                if suf.isdigit():
                    ordered.append((int(suf), key))
        ordered.sort(key=lambda x: x[0])

        images = []
        for _, key in ordered:
            val = kwargs.get(key)
            if val is None:
                continue
            images.append(val)

        if not images:
            empty = torch.zeros((0, 64, 64, 3), dtype=torch.float32)
            return io.NodeOutput(empty)
        if len(images) == 1:
            single = torch.clamp(images[0], min=0.0, max=1.0).to(torch.float32)
            return io.NodeOutput(single)

        current = images[0].cpu()
        for img in images[1:]:
            next_img = img.cpu() if img is not None else current.new_zeros(
                current.shape
            )

            bs = max(current.shape[0], next_img.shape[0])
            current = cls._broadcast_image(current, bs)
            next_img = cls._broadcast_image(next_img, bs)

            current = cls._stitch_pair(
                current,
                next_img,
                direction,
                match_image_size,
                spacing_width,
                spacing_color,
                pad_color,
            ).cpu()

        image = torch.clamp(current, min=0.0, max=1.0).to(torch.float32)
        return io.NodeOutput(image)

    @classmethod
    def _stitch_pair(
        cls,
        a,
        b,
        direction,
        match_image_size,
        spacing_width,
        spacing_color,
        pad_color,
    ):
        space_rgb = cls._parse_spacing_color(spacing_color)
        pad_rgb = cls._parse_pad_color(pad_color)

        bs = max(a.shape[0], b.shape[0])
        a = cls._broadcast_image(a, bs)
        b = cls._broadcast_image(b, bs)

        _, ha, wa, _ = a.shape
        _, hb, wb, _ = b.shape

        if direction in ("left", "right"):
            if match_image_size:
                target_h = ha
                a = cls._resize_keep_ratio(a, None, target_h)
                b = cls._resize_keep_ratio(b, None, target_h)
            else:
                unified_h = max(ha, hb)
                a = cls._pad_to_rgb(a, unified_h, wa, pad_rgb)
                b = cls._pad_to_rgb(b, unified_h, wb, pad_rgb)

            _, ha2, wa2, _ = a.shape
            spacer = cls._make_strip(
                ha2,
                spacing_width,
                space_rgb,
                axis="v",
                dtype=a.dtype,
                device=a.device,
                batch_size=bs,
            )

            if direction == "right":
                out = torch.cat([a, spacer, b], dim=2)
            else:
                out = torch.cat([b, spacer, a], dim=2)
            return out

        if match_image_size:
            target_w = wa
            a = cls._resize_keep_ratio(a, target_w, None)
            b = cls._resize_keep_ratio(b, target_w, None)
        else:
            unified_w = max(wa, wb)
            a = cls._pad_to_rgb(a, ha, unified_w, pad_rgb)
            b = cls._pad_to_rgb(b, hb, unified_w, pad_rgb)

        _, ha2, wa2, _ = a.shape
        spacer = cls._make_strip(
            spacing_width,
            wa2,
            space_rgb,
            axis="h",
            dtype=a.dtype,
            device=a.device,
            batch_size=bs,
        )

        if direction == "bottom":
            out = torch.cat([a, spacer, b], dim=1)
        else:
            out = torch.cat([b, spacer, a], dim=1)
        return out

    @staticmethod
    def _resize_keep_ratio(img, target_w, target_h):
        b, h, w, c = img.shape
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
        img_nchw = img.permute(0, 3, 1, 2)
        out = F.interpolate(
            img_nchw, size=(new_h, new_w), mode="bicubic", align_corners=False
        ).permute(0, 2, 3, 1)
        return out

    @staticmethod
    def _pad_to_rgb(img, target_h, target_w, fill_rgb):
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
    def _make_strip(h, w, fill_rgb, axis, dtype, device, batch_size=1):
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
