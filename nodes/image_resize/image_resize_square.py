from comfy_api.latest import io

from .image_resize_gemini_30_pro_image import ImageResizeGemini30ProImage


class ImageResizeSquare(ImageResizeGemini30ProImage):
    PRESET_RESOLUTIONS = [
        ("[0.25k] 256x256 (1:1)", 256, 256),
        ("[0.5k] 512x512 (1:1)", 512, 512),
        ("[1k] 1024x1024 (1:1)", 1024, 1024),
        ("[2k] 2048x2048 (1:1)", 2048, 2048),
        ("[4k] 4096x4096 (1:1)", 4096, 4096),
    ]
    PRESET_OPTIONS = ["auto", "auto (0.5k | 1k)"] + [name for name, _, _ in PRESET_RESOLUTIONS]

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_ImageResizeSquare",
            display_name="Image Resize Square",
            category="1hewNodes/image/resize",
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
    async def execute(
        cls,
        preset_size,
        fit,
        pad_color,
        image=None,
        mask=None,
    ):
        if preset_size == "auto (0.5k | 1k)":
            if image is not None:
                iw = max(int(image.shape[2]), 1)
                ih = max(int(image.shape[1]), 1)
            elif mask is not None:
                iw = max(int(mask.shape[2]), 1)
                ih = max(int(mask.shape[1]), 1)
            else:
                iw, ih = 1024, 1024

            candidates = [
                r for r in cls.PRESET_RESOLUTIONS if r[0].startswith("[0.5k]") or r[0].startswith("[1k]")
            ]
            width, height = cls._find_best_resolution(iw, ih, candidates)
            matched = [name for name, w, h in candidates if w == width and h == height]
            if matched:
                preset_size = matched[0]

        return await super().execute(preset_size, fit, pad_color, image=image, mask=mask)
