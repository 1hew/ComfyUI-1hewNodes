from comfy_api.latest import io
import math
import numpy as np
import torch
import torch.nn.functional as F


class ImageResizeQwenImage(io.ComfyNode):
    PRESET_RESOLUTIONS = [
        ("928×1664 [1:1.79]", 928, 1664),
        ("1056×1584 [1:1.50] (2:3)", 1056, 1584),
        ("1140×1472 [1:1.29]", 1140, 1472),
        ("1328×1328 [1:1.00] (1:1)", 1328, 1328),
        ("1472×1140 [1.29:1]", 1472, 1140),
        ("1584×1056 [1.50:1] (3:2)", 1584, 1056),
        ("1664×928 [1.79:1]", 1664, 928),
    ]
    PRESET_OPTIONS = ["auto"] + [name for name, _, _ in PRESET_RESOLUTIONS]

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_ImageResizeQwenImage",
            display_name="Image Resize QwenImage",
            category="1hewNodes/image",
            inputs=[
                io.Combo.Input("preset_size", options=cls.PRESET_OPTIONS, default="auto"),
                io.Combo.Input("fit", options=["crop", "pad", "stretch"], default="crop"),
                io.String.Input("pad_color", default="1.0"),
                io.Image.Input("image", optional=True),
                io.Mask.Input("mask", optional=True),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Mask.Output(display_name="mask"),
            ],
        )

    @classmethod
    def validate_inputs(
        cls,
        preset_size: str,
        fit: str,
        pad_color: str,
        image: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ):
        if preset_size not in cls.PRESET_OPTIONS:
            return "invalid preset_size"
        if fit not in ("crop", "pad", "stretch"):
            return "invalid fit"
        return True

    @classmethod
    def fingerprint_inputs(
        cls,
        preset_size: str,
        fit: str,
        pad_color: str,
        image: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ):
        ib = int(image.shape[0]) if isinstance(image, torch.Tensor) else 0
        ih = int(image.shape[1]) if isinstance(image, torch.Tensor) else 0
        iw = int(image.shape[2]) if isinstance(image, torch.Tensor) else 0
        mb = int(mask.shape[0]) if isinstance(mask, torch.Tensor) else 0
        mh = int(mask.shape[1]) if isinstance(mask, torch.Tensor) else 0
        mw = int(mask.shape[2]) if isinstance(mask, torch.Tensor) else 0
        return f"{preset_size}|fit={fit}|pad={pad_color}|img={ib}x{ih}x{iw}|mask={mb}x{mh}x{mw}"

    @classmethod
    async def execute(
        cls,
        preset_size: str,
        fit: str,
        pad_color: str,
        image: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> io.NodeOutput:
        if preset_size == "auto":
            if isinstance(image, torch.Tensor):
                iw = max(int(image.shape[2]), 1)
                ih = max(int(image.shape[1]), 1)
                ar = iw / ih
                _, width, height = min(
                    ((abs(ar - w / h), w, h) for _, w, h in cls.PRESET_RESOLUTIONS),
                    key=lambda x: x[0],
                )
            elif isinstance(mask, torch.Tensor):
                iw = max(int(mask.shape[2]), 1)
                ih = max(int(mask.shape[1]), 1)
                ar = iw / ih
                _, width, height = min(
                    ((abs(ar - w / h), w, h) for _, w, h in cls.PRESET_RESOLUTIONS),
                    key=lambda x: x[0],
                )
            else:
                width, height = 1328, 1328
        else:
            found = [p for p in cls.PRESET_RESOLUTIONS if p[0] == preset_size]
            if found:
                _, width, height = found[0]
            else:
                width, height = 1328, 1328

        device = None
        if isinstance(image, torch.Tensor):
            device = image.device
        elif isinstance(mask, torch.Tensor):
            device = mask.device

        tw, th = int(width), int(height)
        pc = cls._parse_pad_color(pad_color)

        if not isinstance(image, torch.Tensor) and not isinstance(mask, torch.Tensor):
            rgb = (1.0, 1.0, 1.0) if isinstance(pc, str) else pc
            base = torch.ones((1, th, tw, 3), dtype=torch.float32)
            color_t = torch.tensor(rgb, dtype=torch.float32)
            out_img = base * color_t.view(1, 1, 1, 3)
            out_msk = torch.ones((1, th, tw), dtype=torch.float32)
            if device is not None:
                out_img = out_img.to(device)
                out_msk = out_msk.to(device)
            return io.NodeOutput(out_img, out_msk)

        if isinstance(image, torch.Tensor) and (not isinstance(mask, torch.Tensor)):
            b, h, w, _ = image.shape
            if fit == "stretch":
                img_nchw = image.permute(0, 3, 1, 2)
                resized = F.interpolate(img_nchw, size=(th, tw), mode="bicubic", align_corners=False).permute(0, 2, 3, 1)
                out_img = torch.clamp(resized, 0.0, 1.0).to(torch.float32)
                out_msk = torch.ones((b, th, tw), dtype=torch.float32, device=image.device)
                return io.NodeOutput(out_img, out_msk)
            if fit == "pad":
                sw = tw / max(w, 1)
                sh = th / max(h, 1)
                scale = min(sw, sh)
                new_w = max(int(round(w * scale)), 1)
                new_h = max(int(round(h * scale)), 1)
                img_nchw = image.permute(0, 3, 1, 2)
                resized = F.interpolate(img_nchw, size=(new_h, new_w), mode="bicubic", align_corners=False).permute(0, 2, 3, 1)
                out_img = cls._pad_to_rgb(resized, th, tw, pc)
                top = max((th - new_h) // 2, 0)
                left = max((tw - new_w) // 2, 0)
                out_msk = torch.zeros((b, th, tw), dtype=torch.float32, device=out_img.device)
                out_msk[:, top : top + new_h, left : left + new_w] = 1.0
                return io.NodeOutput(out_img, out_msk)
            ta = tw / max(th, 1)
            oa = w / max(h, 1)
            if oa > ta:
                cw = max(int(round(h * ta)), 1)
                ch = h
            else:
                ch = max(int(round(w / ta)), 1)
                cw = w
            left = max((w - cw) // 2, 0)
            top = max((h - ch) // 2, 0)
            cropped = image[:, top : top + ch, left : left + cw, :]
            img_nchw = cropped.permute(0, 3, 1, 2)
            resized = F.interpolate(img_nchw, size=(th, tw), mode="bicubic", align_corners=False).permute(0, 2, 3, 1)
            out_img = torch.clamp(resized, 0.0, 1.0).to(torch.float32)
            out_msk = torch.zeros((b, h, w), dtype=torch.float32, device=image.device)
            out_msk[:, top : top + ch, left : left + cw] = 1.0
            return io.NodeOutput(out_img, out_msk)

        if (not isinstance(image, torch.Tensor)) and isinstance(mask, torch.Tensor):
            mb, mh, mw = int(mask.shape[0]), int(mask.shape[1]), int(mask.shape[2])
            if fit == "stretch":
                m4 = mask.unsqueeze(1)
                out_msk = F.interpolate(m4, size=(th, tw), mode="nearest").squeeze(1)
                out_msk = torch.clamp(out_msk, 0.0, 1.0).to(torch.float32)
                rgb = (1.0, 1.0, 1.0) if isinstance(pc, str) else pc
                base = torch.ones((1, th, tw, 3), dtype=torch.float32)
                color_t = torch.tensor(rgb, dtype=torch.float32)
                out_img = base * color_t.view(1, 1, 1, 3)
                if device is not None:
                    out_img = out_img.to(device)
                    out_msk = out_msk.to(device)
                return io.NodeOutput(out_img, out_msk)
            if fit == "pad":
                sw = tw / max(mw, 1)
                sh = th / max(mh, 1)
                scale = min(sw, sh)
                new_w = max(int(round(mw * scale)), 1)
                new_h = max(int(round(mh * scale)), 1)
                m4 = mask.unsqueeze(1)
                resized_m = F.interpolate(m4, size=(new_h, new_w), mode="nearest").squeeze(1)
                top = max((th - new_h) // 2, 0)
                left = max((tw - new_w) // 2, 0)
                out_msk = torch.zeros((mb, th, tw), dtype=torch.float32, device=resized_m.device)
                out_msk[:, top : top + new_h, left : left + new_w] = torch.clamp(resized_m, 0.0, 1.0)
                rgb = (1.0, 1.0, 1.0) if isinstance(pc, str) else pc
                base = torch.ones((1, th, tw, 3), dtype=torch.float32)
                color_t = torch.tensor(rgb, dtype=torch.float32)
                out_img = base * color_t.view(1, 1, 1, 3)
                if device is not None:
                    out_img = out_img.to(device)
                    out_msk = out_msk.to(device)
                return io.NodeOutput(out_img, out_msk)
            ta = tw / max(th, 1)
            oa = mw / max(mh, 1)
            if oa > ta:
                cw = max(int(round(mh * ta)), 1)
                ch = mh
            else:
                ch = max(int(round(mw / ta)), 1)
                cw = mw
            left = max((mw - cw) // 2, 0)
            top = max((mh - ch) // 2, 0)
            rgb = (1.0, 1.0, 1.0) if isinstance(pc, str) else pc
            base = torch.ones((1, th, tw, 3), dtype=torch.float32)
            color_t = torch.tensor(rgb, dtype=torch.float32)
            out_img = base * color_t.view(1, 1, 1, 3)
            cropped_m = mask[:, top : top + ch, left : left + cw]
            m4 = cropped_m.unsqueeze(1)
            out_msk = F.interpolate(m4, size=(th, tw), mode="nearest").squeeze(1)
            out_msk = torch.clamp(out_msk, 0.0, 1.0).to(torch.float32)
            if device is not None:
                out_img = out_img.to(device)
                out_msk = out_msk.to(device)
            return io.NodeOutput(out_img, out_msk)

        b, h, w, _ = image.shape
        m = cls._ensure_mask_3d(mask)
        if fit == "stretch":
            img_nchw = image.permute(0, 3, 1, 2)
            out_img = F.interpolate(img_nchw, size=(th, tw), mode="bicubic", align_corners=False).permute(0, 2, 3, 1)
            out_img = torch.clamp(out_img, 0.0, 1.0).to(torch.float32)
            if isinstance(m, torch.Tensor):
                m4 = m.unsqueeze(1)
                out_msk = F.interpolate(m4, size=(th, tw), mode="nearest").squeeze(1)
                out_msk = torch.clamp(out_msk, 0.0, 1.0).to(torch.float32)
            else:
                out_msk = torch.ones((b, th, tw), dtype=torch.float32, device=image.device)
            return io.NodeOutput(out_img, out_msk)
        if fit == "pad":
            sw = tw / max(w, 1)
            sh = th / max(h, 1)
            scale = min(sw, sh)
            new_w = max(int(round(w * scale)), 1)
            new_h = max(int(round(h * scale)), 1)
            img_nchw = image.permute(0, 3, 1, 2)
            resized = F.interpolate(img_nchw, size=(new_h, new_w), mode="bicubic", align_corners=False).permute(0, 2, 3, 1)
            out_img = cls._pad_to_rgb(resized, th, tw, pc)
            top = max((th - new_h) // 2, 0)
            left = max((tw - new_w) // 2, 0)
            out_msk = torch.zeros((b, th, tw), dtype=torch.float32, device=out_img.device)
            if isinstance(m, torch.Tensor):
                m4 = m.unsqueeze(1)
                m_res = F.interpolate(m4, size=(new_h, new_w), mode="nearest").squeeze(1)
                out_msk[:, top : top + new_h, left : left + new_w] = torch.clamp(m_res, 0.0, 1.0)
            else:
                out_msk[:, top : top + new_h, left : left + new_w] = 1.0
            return io.NodeOutput(out_img, out_msk)
        ta = tw / max(th, 1)
        oa = w / max(h, 1)
        if oa > ta:
            cw = max(int(round(h * ta)), 1)
            ch = h
        else:
            ch = max(int(round(w / ta)), 1)
            cw = w
        left = max((w - cw) // 2, 0)
        top = max((h - ch) // 2, 0)
        cropped_img = image[:, top : top + ch, left : left + cw, :]
        img_nchw = cropped_img.permute(0, 3, 1, 2)
        out_img = F.interpolate(img_nchw, size=(th, tw), mode="bicubic", align_corners=False).permute(0, 2, 3, 1)
        out_img = torch.clamp(out_img, 0.0, 1.0).to(torch.float32)
        if isinstance(m, torch.Tensor):
            cropped_m = m[:, top : top + ch, left : left + cw]
            m4 = cropped_m.unsqueeze(1)
            out_msk = F.interpolate(m4, size=(th, tw), mode="nearest").squeeze(1)
            out_msk = torch.clamp(out_msk, 0.0, 1.0).to(torch.float32)
        else:
            out_msk = torch.ones((b, th, tw), dtype=torch.float32, device=image.device)
        return io.NodeOutput(out_img, out_msk)

    @staticmethod
    def _ensure_mask_3d(mask):
        if mask is None:
            return None
        if mask.dim() == 4:
            if mask.shape[1] == 1:
                return mask.squeeze(1)
            return mask[:, 0, :, :]
        if mask.dim() == 2:
            return mask.unsqueeze(0)
        return mask

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
            from PIL import ImageColor
            rgb = ImageColor.getrgb(text)
            return (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)
        except Exception:
            return (1.0, 1.0, 1.0)

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
            padded = F.pad(nchw, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")
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
            top_row = img[:, 0:1, :, :].mean(dim=(1, 2))
            top_color = top_row.view(b, 1, 1, c)
            bottom_row = img[:, -1:, :, :].mean(dim=(1, 2))
            bottom_color = bottom_row.view(b, 1, 1, c)
            if top > 0:
                out[:, :top, :, :] = top_color.expand(b, top, target_w, c)
            if (target_h - h_end) > 0:
                bottom_pad = target_h - h_end
                out[:, h_end:, :, :] = bottom_color.expand(b, bottom_pad, target_w, c)
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
