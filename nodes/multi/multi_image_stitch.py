import math
import torch
import torch.nn.functional as F
from comfy_api.latest import io

from ...utils._image_ops import (
    adapt_fill_to_channels,
    ensure_channels,
    pad_to_rgb,
    parse_pad_color,
    parse_spacing_color,
)


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
            images.append(val.to(dtype=torch.float32).clamp(0.0, 1.0))

        if not images:
            empty = torch.zeros((0, 64, 64, 3), dtype=torch.float32)
            return io.NodeOutput(empty)

        target_channels = 4 if any(int(img.shape[3]) == 4 for img in images) else 3
        images = [ensure_channels(img, target_channels) for img in images]

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
        space_rgb = parse_spacing_color(spacing_color)
        pad_rgb = parse_pad_color(pad_color)

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
                a = pad_to_rgb(a, unified_h, wa, pad_rgb)
                b = pad_to_rgb(b, unified_h, wb, pad_rgb)

            _, ha2, wa2, _ = a.shape
            spacer = cls._make_strip(
                ha2,
                spacing_width,
                space_rgb,
                axis="v",
                channels=int(a.shape[3]),
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
            a = pad_to_rgb(a, ha, unified_w, pad_rgb)
            b = pad_to_rgb(b, hb, unified_w, pad_rgb)

        _, ha2, wa2, _ = a.shape
        spacer = cls._make_strip(
            spacing_width,
            wa2,
            space_rgb,
            axis="h",
            channels=int(a.shape[3]),
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
    def _make_strip(h, w, fill_rgb, axis, channels, dtype, device, batch_size=1):
        if w <= 0:
            return torch.zeros((batch_size, h, 0, channels), dtype=dtype, device=device)
        fill_value = adapt_fill_to_channels(fill_rgb, channels)
        fill_tensor = torch.tensor(fill_value, dtype=dtype, device=device)
        if axis == "v":
            out = torch.zeros((batch_size, h, w, channels), dtype=dtype, device=device)
            out[:] = fill_tensor
            return out
        out = torch.zeros((batch_size, w, h, channels), dtype=dtype, device=device)
        out[:] = fill_tensor
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
