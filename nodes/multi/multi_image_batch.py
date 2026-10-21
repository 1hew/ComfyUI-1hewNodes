import torch
import torch.nn.functional as F
from comfy_api.latest import io

from ...utils._image_ops import (
    ensure_channels,
    pad_to_rgb,
)


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

        # 任一输入含 alpha 时，统一所有输入为 RGBA，确保输出保留 alpha
        target_channels = 4 if any(int(img.shape[3]) == 4 for img in images) else 3
        images = [ensure_channels(img, target_channels) for img in images]

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
                    pad_to_rgb(resized, ref_h, ref_w, pad_rgb)
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
            cropped = pad_to_rgb(
                cropped, target_h, target_w, (0.0, 0.0, 0.0)
            )
        return cropped
