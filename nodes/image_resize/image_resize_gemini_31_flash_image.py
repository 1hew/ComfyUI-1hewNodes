from comfy_api.latest import io

from .image_resize_gemini_30_pro_image import ImageResizeGemini30ProImage


class ImageResizeGemini31FlashImage(ImageResizeGemini30ProImage):
    PRESET_RESOLUTIONS = [
        # 0.5k_ratios
        ("[0.5k] 512x512 (1:1)", 512, 512),
        ("[0.5k] 256x1024 (1:4)", 256, 1024),
        ("[0.5k] 192x1536 (1:8)", 192, 1536),
        ("[0.5k] 424x632 (2:3)", 424, 632),
        ("[0.5k] 632x424 (3:2)", 632, 424),
        ("[0.5k] 448x600 (3:4)", 448, 600),
        ("[0.5k] 1024x256 (4:1)", 1024, 256),
        ("[0.5k] 600x448 (4:3)", 600, 448),
        ("[0.5k] 464x576 (4:5)", 464, 576),
        ("[0.5k] 576x464 (5:4)", 576, 464),
        ("[0.5k] 1536x192 (8:1)", 1536, 192),
        ("[0.5k] 384x688 (9:16)", 384, 688),
        ("[0.5k] 688x384 (16:9)", 688, 384),
        ("[0.5k] 792x168 (21:9)", 792, 168),
        # 1k_ratios
        ("[1k] 1024x1024 (1:1)", 1024, 1024),
        ("[1k] 512x2048 (1:4)", 512, 2048),
        ("[1k] 384x3072 (1:8)", 384, 3072),
        ("[1k] 848x1264 (2:3)", 848, 1264),
        ("[1k] 1264x848 (3:2)", 1264, 848),
        ("[1k] 896x1200 (3:4)", 896, 1200),
        ("[1k] 2048x512 (4:1)", 2048, 512),
        ("[1k] 1200x896 (4:3)", 1200, 896),
        ("[1k] 928x1152 (4:5)", 928, 1152),
        ("[1k] 1152x928 (5:4)", 1152, 928),
        ("[1k] 3072x384 (8:1)", 3072, 384),
        ("[1k] 768x1376 (9:16)", 768, 1376),
        ("[1k] 1376x768 (16:9)", 1376, 768),
        ("[1k] 1584x672 (21:9)", 1584, 672),
        # 2k_ratios
        ("[2k] 2048x2048 (1:1)", 2048, 2048),
        ("[2k] 1024x4096 (1:4)", 1024, 4096),
        ("[2k] 768x6144 (1:8)", 768, 6144),
        ("[2k] 1696x2528 (2:3)", 1696, 2528),
        ("[2k] 2528x1696 (3:2)", 2528, 1696),
        ("[2k] 1792x2400 (3:4)", 1792, 2400),
        ("[2k] 4096x1024 (4:1)", 4096, 1024),
        ("[2k] 2400x1792 (4:3)", 2400, 1792),
        ("[2k] 1856x2304 (4:5)", 1856, 2304),
        ("[2k] 2304x1856 (5:4)", 2304, 1856),
        ("[2k] 6144x768 (8:1)", 6144, 768),
        ("[2k] 1536x2752 (9:16)", 1536, 2752),
        ("[2k] 2752x1536 (16:9)", 2752, 1536),
        ("[2k] 3168x1344 (21:9)", 3168, 1344),
        # 4k_ratios
        ("[4k] 4096x4096 (1:1)", 4096, 4096),
        ("[4k] 2048x8192 (1:4)", 2048, 8192),
        ("[4k] 1536x12288 (1:8)", 1536, 12288),
        ("[4k] 3392x5056 (2:3)", 3392, 5056),
        ("[4k] 5056x3392 (3:2)", 5056, 3392),
        ("[4k] 3584x4800 (3:4)", 3584, 4800),
        ("[4k] 8192x2048 (4:1)", 8192, 2048),
        ("[4k] 4800x3584 (4:3)", 4800, 3584),
        ("[4k] 3712x4608 (4:5)", 3712, 4608),
        ("[4k] 4608x3712 (5:4)", 4608, 3712),
        ("[4k] 12288x1536 (8:1)", 12288, 1536),
        ("[4k] 3072x5504 (9:16)", 3072, 5504),
        ("[4k] 5504x3072 (16:9)", 5504, 3072),
        ("[4k] 6336x2688 (21:9)", 6336, 2688),
    ]
    PRESET_OPTIONS = ["auto", "auto (0.5k)", "auto (1k | 2k)", "auto (2k | 4k)"] + [name for name, _, _ in PRESET_RESOLUTIONS]

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_ImageResizeGemini31FlashImage",
            display_name="Image Resize Gemini31FlashImage",
            category="1hewNodes/image/resize",
            inputs=[
                io.Combo.Input("preset_size", options=cls.PRESET_OPTIONS, default="auto (2k | 4k)"),
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
        if preset_size == "auto (0.5k)":
            if image is not None:
                iw = max(int(image.shape[2]), 1)
                ih = max(int(image.shape[1]), 1)
            elif mask is not None:
                iw = max(int(mask.shape[2]), 1)
                ih = max(int(mask.shape[1]), 1)
            else:
                iw, ih = 512, 512

            candidates = [r for r in cls.PRESET_RESOLUTIONS if r[0].startswith("[0.5k]")]
            width, height = cls._find_best_resolution(iw, ih, candidates)
            matched = [name for name, w, h in candidates if w == width and h == height]
            if matched:
                preset_size = matched[0]

        return await super().execute(preset_size, fit, pad_color, image=image, mask=mask)
