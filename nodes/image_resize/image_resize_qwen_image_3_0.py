from comfy_api.latest import io
import math

import torch
import torch.nn.functional as F


class ImageResizeQwenImage30(io.ComfyNode):
    PRESET_RESOLUTIONS = [
        ("[1k] 1728x576 (3:1)", 1728, 576),
        ("[1k] 1344x576 (21:9)", 1344, 576),
        ("[1k] 1440x720 (2:1)", 1440, 720),
        ("[1k] 1280x720 (16:9)", 1280, 720),
        ("[1k] 1248x832 (3:2)", 1248, 832),
        ("[1k] 1152x864 (4:3)", 1152, 864),
        ("[1k] 1120x896 (5:4)", 1120, 896),
        ("[1k] 1024x1024 (1:1)", 1024, 1024),
        ("[1k] 896x1120 (4:5)", 896, 1120),
        ("[1k] 864x1152 (3:4)", 864, 1152),
        ("[1k] 832x1248 (2:3)", 832, 1248),
        ("[1k] 720x1280 (9:16)", 720, 1280),
        ("[1k] 720x1440 (1:2)", 720, 1440),
        ("[1k] 576x1344 (9:21)", 576, 1344),
        ("[1k] 576x1728 (1:3)", 576, 1728),
        ("[2k] 3456x1152 (3:1)", 3456, 1152),
        ("[2k] 2688x1152 (21:9)", 2688, 1152),
        ("[2k] 2880x1440 (2:1)", 2880, 1440),
        ("[2k] 2560x1440 (16:9)", 2560, 1440),
        ("[2k] 2496x1664 (3:2)", 2496, 1664),
        ("[2k] 2304x1728 (4:3)", 2304, 1728),
        ("[2k] 2240x1792 (5:4)", 2240, 1792),
        ("[2k] 2048x2048 (1:1)", 2048, 2048),
        ("[2k] 1792x2240 (4:5)", 1792, 2240),
        ("[2k] 1728x2304 (3:4)", 1728, 2304),
        ("[2k] 1664x2496 (2:3)", 1664, 2496),
        ("[2k] 1440x2560 (9:16)", 1440, 2560),
        ("[2k] 1440x2880 (1:2)", 1440, 2880),
        ("[2k] 1152x2688 (9:21)", 1152, 2688),
        ("[2k] 1152x3456 (1:3)", 1152, 3456),
    ]
    PRESET_OPTIONS = [
        "auto",
        "auto (1k)",
        "auto (2k)",
        "dynamic",
        "dynamic (1k)",
        "dynamic (2k)",
    ] + [name for name, _, _ in PRESET_RESOLUTIONS]
    TARGET_PIXELS = {
        "1k": 1024 * 1024,
        "2k": 2048 * 2048,
    }
    # Qwen-Image 官方宽高比范围 1:8 ~ 8:1（比 GPT 的 1:3 ~ 3:1 更宽）。
    MIN_SAFE_ASPECT_RATIO = 1.0 / 8.0
    MAX_SAFE_ASPECT_RATIO = 8.0
    # 上游实测长边 4096 可用；qwen 官方面积上限 2048x2048，故无 4k 档。
    MAX_EDGE = 4096

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_ImageResizeQwenImage30",
            display_name="Image Resize Qwen Image 3.0",
            category="1hewNodes/image/resize",
            inputs=[
                io.Combo.Input("preset_size", options=cls.PRESET_OPTIONS, default="auto (2k)"),
                io.Combo.Input("fit", options=["crop", "pad", "stretch"], default="crop"),
                io.String.Input("pad_color", default="1.0"),
                io.Image.Input("image", optional=True),
                io.Mask.Input("mask", optional=True),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Mask.Output(display_name="mask"),
                io.Image.Output(display_name="native_image"),
                io.Mask.Output(display_name="native_mask"),
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
        if not cls._is_prompt_link(preset_size) and cls._normalize_preset_size(preset_size) not in cls.PRESET_OPTIONS:
            return f"invalid preset_size: {preset_size!r}"
        if fit not in ("crop", "pad", "stretch"):
            return "invalid fit"
        return True

    @classmethod
    async def execute(
        cls,
        preset_size: str,
        fit: str,
        pad_color: str,
        image: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> io.NodeOutput:
        preset_size = cls._normalize_preset_size(preset_size)
        if preset_size not in cls.PRESET_OPTIONS:
            raise ValueError(f"invalid preset_size: {preset_size!r}")

        source_w, source_h = cls._source_dimensions(image, mask)
        tw, th = cls._resolve_target_size(source_w, source_h, preset_size)
        pc = cls._parse_pad_color(pad_color)

        device = None
        if isinstance(image, torch.Tensor):
            device = image.device
        elif isinstance(mask, torch.Tensor):
            device = mask.device

        if not isinstance(image, torch.Tensor) and not isinstance(mask, torch.Tensor):
            out_img = cls._blank_image(1, th, tw, pc, device)
            out_msk = torch.ones((1, th, tw), dtype=torch.float32, device=device)
            return io.NodeOutput(out_img, out_msk, out_img, out_msk)

        if not isinstance(image, torch.Tensor):
            m = cls._ensure_mask_3d(mask)
            b, h, w = int(m.shape[0]), int(m.shape[1]), int(m.shape[2])
            image_device = m.device
            image = cls._blank_image(b, h, w, pc, image_device)
            m = cls._ensure_mask_3d(mask)
            out_img, out_msk, native_img, native_msk = cls._resize_pair(image, m, tw, th, fit, pc)
            return cls._output(out_img, out_msk, native_img, native_msk)

        b, h, w, _ = image.shape
        m = cls._ensure_mask_3d(mask)
        has_mask = isinstance(m, torch.Tensor)
        if not has_mask:
            m = torch.ones((b, h, w), dtype=torch.float32, device=image.device)
        out_img, out_msk, native_img, native_msk = cls._resize_pair(image, m, tw, th, fit, pc)

        if fit == "crop" and not has_mask:
            # Match the other resize nodes: with crop and no mask input, the
            # default mask marks the kept region at the SOURCE resolution.
            target_aspect = tw / max(th, 1)
            source_aspect = w / max(h, 1)
            if source_aspect > target_aspect:
                crop_w = max(int(round(h * target_aspect)), 1)
                crop_h = h
            else:
                crop_h = max(int(round(w / target_aspect)), 1)
                crop_w = w
            left = max((w - crop_w) // 2, 0)
            top = max((h - crop_h) // 2, 0)
            out_msk = torch.zeros((b, h, w), dtype=torch.float32, device=image.device)
            out_msk[:, top : top + crop_h, left : left + crop_w] = 1.0

        return cls._output(out_img, out_msk, native_img, native_msk)

    @classmethod
    def fingerprint_inputs(
        cls,
        preset_size: str,
        fit: str,
        pad_color: str,
        image: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ):
        preset_size = cls._normalize_preset_size(preset_size)
        ib = int(image.shape[0]) if isinstance(image, torch.Tensor) else 0
        ih = int(image.shape[1]) if isinstance(image, torch.Tensor) else 0
        iw = int(image.shape[2]) if isinstance(image, torch.Tensor) else 0
        mb = int(mask.shape[0]) if isinstance(mask, torch.Tensor) else 0
        mh = int(mask.shape[1]) if isinstance(mask, torch.Tensor) else 0
        mw = int(mask.shape[2]) if isinstance(mask, torch.Tensor) else 0
        tw, th = cls._resolve_target_size(iw or mw or 1, ih or mh or 1, preset_size)
        return f"{preset_size}|target={tw}x{th}|fit={fit}|pad={pad_color}|img={ib}x{ih}x{iw}|mask={mb}x{mh}x{mw}"

    @classmethod
    def _source_dimensions(
        cls,
        image: torch.Tensor | None,
        mask: torch.Tensor | None,
    ) -> tuple[int, int]:
        if isinstance(image, torch.Tensor) and image.ndim == 4:
            return max(int(image.shape[2]), 1), max(int(image.shape[1]), 1)
        if isinstance(mask, torch.Tensor):
            m = cls._ensure_mask_3d(mask)
            if isinstance(m, torch.Tensor) and m.ndim == 3:
                return max(int(m.shape[2]), 1), max(int(m.shape[1]), 1)
        return 1024, 1024

    @classmethod
    def _normalize_preset_size(cls, preset_size) -> str:
        if isinstance(preset_size, (list, tuple)):
            preset_size = preset_size[0] if len(preset_size) > 0 else ""
        return "" if preset_size is None else str(preset_size).strip()

    @staticmethod
    def _is_prompt_link(value) -> bool:
        if value is None:
            return True
        if isinstance(value, (list, tuple)) and len(value) == 2 and isinstance(value[1], int):
            return True
        if isinstance(value, str) and value.strip() in ("*", "STRING", "COMBO"):
            return True
        return False

    @classmethod
    def _resolve_target_size(cls, source_w: int, source_h: int, preset_size: str) -> tuple[int, int]:
        preset = cls._find_preset_resolution(preset_size)
        if preset is not None:
            return preset

        if cls._is_auto_preset(preset_size):
            # auto（不带档位）：先按输入面积定档（1k/2k），再在该档内选最接近比例的预设。
            # auto (1k)/(2k)：直接限定在对应档内选比例。
            if cls._normalize_preset_size(preset_size) == "auto":
                target_key = cls._resolve_target_key(source_w, source_h, preset_size)
                candidates = cls._preset_candidates(f"auto ({target_key})")
            else:
                candidates = cls._preset_candidates(preset_size)
            return cls._find_best_resolution(source_w, source_h, candidates)

        target_key = cls._resolve_target_key(source_w, source_h, preset_size)
        max_pixels = cls.TARGET_PIXELS[target_key]
        ratio = float(max(int(source_w), 1)) / float(max(int(source_h), 1))
        ratio = min(max(ratio, cls.MIN_SAFE_ASPECT_RATIO), cls.MAX_SAFE_ASPECT_RATIO)
        width = int(round((float(max_pixels) * ratio) ** 0.5))
        height = int(round((float(max_pixels) / ratio) ** 0.5))
        return cls._normalize_size(width, height, max_pixels)

    @staticmethod
    def _is_auto_preset(preset_size: str) -> bool:
        return str(preset_size or "").strip().lower() in {
            "auto",
            "auto (1k)",
            "auto (2k)",
        }

    @classmethod
    def _preset_candidates(cls, preset_size: str) -> list[tuple[str, int, int]]:
        normalized = str(preset_size or "").strip().lower()
        if normalized == "auto (1k)":
            prefixes = ("[1k]",)
        elif normalized == "auto (2k)":
            prefixes = ("[2k]",)
        else:
            prefixes = ("[1k]", "[2k]")
        return [item for item in cls.PRESET_RESOLUTIONS if item[0].startswith(prefixes)]

    @classmethod
    def _find_best_resolution(cls, source_w: int, source_h: int, candidates: list[tuple[str, int, int]]) -> tuple[int, int]:
        source_ratio = float(max(int(source_w), 1)) / float(max(int(source_h), 1))
        ratio_candidates = []
        for _, width, height in candidates:
            ratio_diff = abs(math.log(source_ratio) - math.log(float(width) / float(height)))
            ratio_candidates.append((ratio_diff, int(width), int(height)))

        min_ratio_diff = min(item[0] for item in ratio_candidates)
        tolerance = 0.02
        best_ratio_candidates = [item for item in ratio_candidates if item[0] <= min_ratio_diff + tolerance]
        source_area = max(int(source_w), 1) * max(int(source_h), 1)
        _, width, height = min(
            best_ratio_candidates,
            key=lambda item: abs(math.log(max(source_area, 1)) - math.log(item[1] * item[2])),
        )
        return width, height

    @classmethod
    def _find_preset_resolution(cls, preset_size: str) -> tuple[int, int] | None:
        for name, width, height in cls.PRESET_RESOLUTIONS:
            if name == preset_size:
                return int(width), int(height)
        return None

    @classmethod
    def _resolve_target_key(cls, source_w: int, source_h: int, preset_size: str) -> str:
        normalized = str(preset_size or "").strip().lower()
        if normalized in ("auto (1k)", "dynamic (1k)"):
            return "1k"
        if normalized in ("auto (2k)", "dynamic (2k)"):
            return "2k"

        area = max(int(source_w), 1) * max(int(source_h), 1)
        return min(
            cls.TARGET_PIXELS.keys(),
            key=lambda key: abs(math.log(max(area, 1)) - math.log(cls.TARGET_PIXELS[key])),
        )

    @classmethod
    def _normalize_size(cls, width: int, height: int, max_pixels: int) -> tuple[int, int]:
        width = max(16, int(width))
        height = max(16, int(height))
        longest = max(width, height)
        if longest > cls.MAX_EDGE:
            scale = float(cls.MAX_EDGE) / float(longest)
            width = int(width * scale)
            height = int(height * scale)

        width = cls._round_to_multiple_of_16(width)
        height = cls._round_to_multiple_of_16(height)
        while width * height > max_pixels:
            if width >= height:
                width -= 16
            else:
                height -= 16
        return max(16, width), max(16, height)

    @staticmethod
    def _round_to_multiple_of_16(value: int) -> int:
        return max(16, int(round(float(value) / 16.0)) * 16)

    @classmethod
    def _blank_image(cls, batch: int, height: int, width: int, pad_color, device) -> torch.Tensor:
        rgb = (1.0, 1.0, 1.0) if isinstance(pad_color, str) else pad_color
        base = torch.ones((int(batch), int(height), int(width), 3), dtype=torch.float32, device=device)
        color_t = torch.tensor(rgb, dtype=torch.float32, device=device)
        return base * color_t.view(1, 1, 1, 3)

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
        from .image_resize_gemini_30_pro_image import ImageResizeGemini30ProImage

        return ImageResizeGemini30ProImage._parse_pad_color(color_str)

    @staticmethod
    def _pad_to_rgb(img, target_h, target_w, fill_rgb):
        from .image_resize_gemini_30_pro_image import ImageResizeGemini30ProImage

        return ImageResizeGemini30ProImage._pad_to_rgb(img, target_h, target_w, fill_rgb)

    @classmethod
    def _resize_pair(
        cls,
        image: torch.Tensor,
        mask: torch.Tensor,
        target_w: int,
        target_h: int,
        fit: str,
        pad_color,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b, h, w, _ = image.shape
        if fit == "stretch":
            out_img = cls._resize_image(image, target_h, target_w)
            out_msk = cls._resize_mask(mask, target_h, target_w)
        elif fit == "pad":
            scale = min(target_w / max(w, 1), target_h / max(h, 1))
            new_w = max(int(round(w * scale)), 1)
            new_h = max(int(round(h * scale)), 1)
            resized_img = cls._resize_image(image, new_h, new_w)
            resized_msk = cls._resize_mask(mask, new_h, new_w)
            out_img = cls._pad_to_rgb(resized_img, target_h, target_w, pad_color)
            top = max((target_h - new_h) // 2, 0)
            left = max((target_w - new_w) // 2, 0)
            out_msk = torch.zeros((b, target_h, target_w), dtype=torch.float32, device=resized_msk.device)
            out_msk[:, top : top + new_h, left : left + new_w] = resized_msk
        else:
            target_aspect = target_w / max(target_h, 1)
            source_aspect = w / max(h, 1)
            if source_aspect > target_aspect:
                crop_w = max(int(round(h * target_aspect)), 1)
                crop_h = h
            else:
                crop_h = max(int(round(w / target_aspect)), 1)
                crop_w = w
            left = max((w - crop_w) // 2, 0)
            top = max((h - crop_h) // 2, 0)
            cropped_img = image[:, top : top + crop_h, left : left + crop_w, :]
            cropped_msk = mask[:, top : top + crop_h, left : left + crop_w]
            out_img = cls._resize_image(cropped_img, target_h, target_w)
            out_msk = cls._resize_mask(cropped_msk, target_h, target_w)

        native_img, native_msk = cls._native_pair(image, mask, target_w, target_h, fit, pad_color)
        return out_img, out_msk, native_img, native_msk

    @classmethod
    def _native_pair(
        cls,
        image: torch.Tensor,
        mask: torch.Tensor,
        target_w: int,
        target_h: int,
        fit: str,
        pad_color,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Second output: same aspect ratio as the main output, anchored to the source scale.

        pad     -> native content + pad to target ratio (area grows)
        crop    -> native center-crop to target ratio (area shrinks)
        stretch -> non-uniform stretch to target ratio, area preserved (W*H == w*h, approx)
        """
        b, h, w, _ = image.shape
        ratio = float(target_w) / float(max(target_h, 1))
        source_aspect = float(w) / float(max(h, 1))

        if fit == "pad":
            # Keep the source pixels at native scale; no implicit 16 alignment.
            if source_aspect > ratio:
                cw, ch = w, max(h, int(math.ceil(w / ratio)))
            else:
                cw, ch = max(w, int(math.ceil(h * ratio))), h
            native_img = cls._pad_to_rgb(image, ch, cw, pad_color)
            top = max((ch - h) // 2, 0)
            left = max((cw - w) // 2, 0)
            native_msk = torch.zeros((b, ch, cw), dtype=torch.float32, device=mask.device)
            native_msk[:, top : top + h, left : left + w] = mask
            return cls._cap_native(native_img, native_msk)

        if fit == "crop":
            if source_aspect > ratio:
                cw, ch = max(int(round(h * ratio)), 1), h
            else:
                cw, ch = w, max(int(round(w / ratio)), 1)
            cw = max(1, min(int(cw), w))
            ch = max(1, min(int(ch), h))
            left = max((w - cw) // 2, 0)
            top = max((h - ch) // 2, 0)
            native_img = image[:, top : top + ch, left : left + cw, :].contiguous()
            native_msk = mask[:, top : top + ch, left : left + cw].contiguous()
            return cls._cap_native(native_img, native_msk)

        # stretch: preserve area, target ratio
        area = float(w * h)
        cw = max(int(round((area * ratio) ** 0.5)), 1)
        ch = max(int(round((area / ratio) ** 0.5)), 1)
        native_img = cls._resize_image(image, ch, cw)
        native_msk = cls._resize_mask(mask, ch, cw)
        return cls._cap_native(native_img, native_msk)

    @staticmethod
    def _ceil_to_multiple_of_16(value: int) -> int:
        return max(16, ((int(value) + 15) // 16) * 16)

    @classmethod
    def _cap_native(
        cls,
        native_img: torch.Tensor,
        native_msk: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hh, ww = int(native_img.shape[1]), int(native_img.shape[2])
        longest = max(hh, ww)
        if longest <= cls.MAX_EDGE:
            return native_img, native_msk
        scale = float(cls.MAX_EDGE) / float(longest)
        nw = max(1, int(round(ww * scale)))
        nh = max(1, int(round(hh * scale)))
        return cls._resize_image(native_img, nh, nw), cls._resize_mask(native_msk, nh, nw)

    @staticmethod
    def _resize_image(image: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        img_nchw = image.permute(0, 3, 1, 2)
        resized = F.interpolate(img_nchw, size=(int(target_h), int(target_w)), mode="bicubic", align_corners=False)
        return torch.clamp(resized.permute(0, 2, 3, 1), 0.0, 1.0).to(torch.float32)

    @staticmethod
    def _resize_mask(mask: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        m4 = mask.unsqueeze(1)
        resized = F.interpolate(m4, size=(int(target_h), int(target_w)), mode="nearest").squeeze(1)
        return torch.clamp(resized, 0.0, 1.0).to(torch.float32)

    @classmethod
    def _output(cls, out_img, out_msk, native_img=None, native_msk=None):
        # Native is a proportional copy of the final main output. This keeps
        # crop/padding/content identical; only the resolution is different.
        main_h, main_w = int(out_img.shape[1]), int(out_img.shape[2])
        hint_h = int(native_img.shape[1]) if native_img is not None else main_h
        hint_w = int(native_img.shape[2]) if native_img is not None else main_w
        gcd = math.gcd(main_w, main_h)
        ratio_w, ratio_h = main_w // gcd, main_h // gcd
        scale = max(1, int(round(math.sqrt((hint_w * hint_h) / float(ratio_w * ratio_h)))))
        native_w, native_h = ratio_w * scale, ratio_h * scale
        native_img = cls._resize_image(out_img, native_h, native_w)
        native_msk = cls._resize_mask(out_msk, native_h, native_w)
        return io.NodeOutput(out_img, out_msk, native_img, native_msk)
