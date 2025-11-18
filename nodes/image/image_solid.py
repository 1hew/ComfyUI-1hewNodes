from comfy_api.latest import io
import math
import numpy as np
import torch
from PIL import ImageColor


class ImageSolid(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_ImageSolid",
            display_name="Image Solid",
            category="1hewNodes/image",
            inputs=[
                io.Image.Input("get_image_size", optional=True),
                io.Combo.Input(
                    "preset_size",
                    options=[
                        "custom",
                        "512×512 [1:1.00] (1:1)",
                        "768×768 [1:1.00] (1:1)",
                        "1024×1024 [1:1.00] (1:1)",
                        "1328×1328 [1:1.00] (1:1)",
                        "1408×1408 [1:1.00] (1:1)",
                        "768×1024 [1:1.33] (3:4)",
                        "1140×1472 [1:1.29]",
                        "1216×1664 [1:1.37]",
                        "512×768 [1:1.50] (2:3)",
                        "832×1248 [1:1.50] (2:3)",
                        "1056×1584 [1:1.50] (2:3)",
                        "1152×1728 [1:1.50] (2:3)",
                        "480×832 [1:1.73]",
                        "576×1024 [1:1.78] (9:16)",
                        "720×1280 [1:1.78] (9:16)",
                        "928×1664 [1:1.79]",
                        "1080×1920 [1:1.78] (9:16)",
                        "1088×1920 [1:1.76]",
                        "672×1568 [1:2.33] (3:7)",
                        "960×2176 [1:2.27]",
                        "1024×768 [1.33:1] (4:3)",
                        "1472×1140 [1.29:1]",
                        "1664×1216 [1.37:1]",
                        "768×512 [1.50:1] (3:2)",
                        "1248×832 [1.50:1] (3:2)",
                        "1584×1056 [1.50:1] (3:2)",
                        "1728×1152 [1.50:1] (3:2)",
                        "832×480 [1.73:1]",
                        "1024×576 [1.78:1] (16:9)",
                        "1280×720 [1.78:1] (16:9)",
                        "1664×928 [1.79:1]",
                        "1920×1080 [1.78:1] (16:9)",
                        "1920×1088 [1.76:1]",
                        "1568×672 [2.33:1] (7:3)",
                        "2176×960 [2.27:1]",
                    ],
                    default="custom",
                ),
                io.Int.Input("width", default=1024, min=1, max=8192, step=1),
                io.Int.Input("height", default=1024, min=1, max=8192, step=1),
                io.String.Input("color", default="1.0"),
                io.Float.Input("alpha", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Boolean.Input("invert", default=False),
                io.Float.Input("mask_opacity", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Int.Input("divisible_by", default=8, min=1, max=1024, step=1),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Mask.Output(display_name="mask"),
            ],
        )

    @classmethod
    async def execute(
        cls,
        preset_size: str,
        width: int,
        height: int,
        color: str,
        alpha: float,
        invert: bool,
        mask_opacity: float,
        divisible_by: int,
        get_image_size: torch.Tensor | None = None,
    ) -> io.NodeOutput:
        images = []
        masks = []

        if get_image_size is not None:
            for ref_img in get_image_size:
                h, w, _ = ref_img.shape
                img_w = w
                img_h = h

                rgb = cls.parse_color(color)
                r = float(rgb[0])
                g = float(rgb[1])
                b = float(rgb[2])
                if invert:
                    r = 1.0 - r
                    g = 1.0 - g
                    b = 1.0 - b
                r *= alpha
                g *= alpha
                b *= alpha

                arr = np.zeros((img_h, img_w, 3), dtype=np.float32)
                arr[:, :, 0] = r
                arr[:, :, 1] = g
                arr[:, :, 2] = b
                m = np.ones((img_h, img_w), dtype=np.float32) * mask_opacity

                images.append(torch.from_numpy(arr).unsqueeze(0))
                masks.append(torch.from_numpy(m).unsqueeze(0))
        else:
            if preset_size != "custom":
                dims = preset_size.split(" ")[0].split("×")
                img_w = int(dims[0])
                img_h = int(dims[1])
            else:
                img_w = width
                img_h = height
            if divisible_by > 1:
                img_w = math.ceil(img_w / divisible_by) * divisible_by
                img_h = math.ceil(img_h / divisible_by) * divisible_by

            rgb = cls.parse_color(color)
            r = float(rgb[0])
            g = float(rgb[1])
            b = float(rgb[2])
            if invert:
                r = 1.0 - r
                g = 1.0 - g
                b = 1.0 - b
            r *= alpha
            g *= alpha
            b *= alpha

            arr = np.zeros((img_h, img_w, 3), dtype=np.float32)
            arr[:, :, 0] = r
            arr[:, :, 1] = g
            arr[:, :, 2] = b
            m = np.ones((img_h, img_w), dtype=np.float32) * mask_opacity

            images.append(torch.from_numpy(arr).unsqueeze(0))
            masks.append(torch.from_numpy(m).unsqueeze(0))

        final_images = torch.cat(images, dim=0)
        final_masks = torch.cat(masks, dim=0)
        return io.NodeOutput(final_images, final_masks)

    @staticmethod
    def parse_color(color_str):
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
                    r = float(parts[0])
                    g = float(parts[1])
                    b = float(parts[2])
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