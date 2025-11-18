import math
from PIL import ImageColor
import torch
import torch.nn.functional as F
from comfy_api.latest import io


class MultiImageBatch(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="1hew_MultiImageBatch",
            display_name="Multi Image Batch",
            category="1hewNodes/multi",
            inputs=[
                io.Combo.Input("fit", options=["crop", "pad", "stretch"], default="pad"),
                io.String.Input("pad_color", default="1.0"),
                io.Image.Input("image_1"),
            ],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    async def execute(cls, fit, pad_color, **kwargs) -> io.NodeOutput:
        ordered = []
        for k in kwargs.keys():
            if k.startswith("image_"):
                suf = k[len("image_") :]
                if suf.isdigit():
                    ordered.append((int(suf), k))
        ordered.sort(key=lambda x: x[0])

        images = []
        for _, key in ordered:
            val = kwargs.get(key)
            if val is None:
                continue
            images.append(val.cpu())

        if not images:
            empty = torch.zeros((0, 64, 64, 3), dtype=torch.float32)
            return io.NodeOutput(empty)

        ref_h = int(images[0].shape[1])
        ref_w = int(images[0].shape[2])
        pad_rgb = cls._parse_color_string(pad_color)

        aligned = []
        for img in images:
            h = int(img.shape[1])
            w = int(img.shape[2])
            if fit == "stretch":
                aligned.append(cls._resize_to(img, ref_h, ref_w))
            elif fit == "crop":
                sw = ref_w / max(w, 1)
                sh = ref_h / max(h, 1)
                scale = max(sw, sh)
                new_w = max(int(round(w * scale)), 1)
                new_h = max(int(round(h * scale)), 1)
                resized = cls._resize_to(img, new_h, new_w)
                aligned.append(cls._center_crop(resized, ref_h, ref_w))
            else:
                sw = ref_w / max(w, 1)
                sh = ref_h / max(h, 1)
                scale = min(sw, sh)
                new_w = max(int(round(w * scale)), 1)
                new_h = max(int(round(h * scale)), 1)
                resized = cls._resize_to(img, new_h, new_w)
                aligned.append(
                    cls._pad_to_rgb(resized, ref_h, ref_w, pad_rgb)
                )

        result = (
            torch.cat(aligned, dim=0) if len(aligned) > 1 else aligned[0]
        )
        result = torch.clamp(result, min=0.0, max=1.0).to(torch.float32)
        return io.NodeOutput(result)

    @staticmethod
    def _resize_to(img, target_h, target_w):
        b, h, w, c = img.shape
        img_nchw = img.permute(0, 3, 1, 2)
        out = F.interpolate(
            img_nchw,
            size=(target_h, target_w),
            mode="bicubic",
            align_corners=False,
        ).permute(0, 2, 3, 1)
        return out

    @staticmethod
    def _center_crop(img, target_h, target_w):
        b, h, w, c = img.shape
        top = max((h - target_h) // 2, 0)
        left = max((w - target_w) // 2, 0)
        top_end = min(top + target_h, h)
        left_end = min(left + target_w, w)
        cropped = img[:, top:top_end, left:left_end, :]
        if cropped.shape[1] != target_h or cropped.shape[2] != target_w:
            cropped = MultiImageBatch._pad_to_rgb(
                cropped, target_h, target_w, (0.0, 0.0, 0.0)
            )
        return cropped

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
    def _parse_color_string(color_str):
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
