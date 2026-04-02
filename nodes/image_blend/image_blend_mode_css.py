from comfy_api.latest import io
import numpy as np
from PIL import Image
import torch

class ImageBlendModeByCSS(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_ImageBlendModeByCSS",
            display_name="Image Blend Mode by CSS",
            category="1hewNodes/image/blend",
            inputs=[
                io.Image.Input("overlay_image"),
                io.Image.Input("base_image"),
                io.Combo.Input(
                    "blend_mode",
                    options=[
                        "normal",
                        "multiply",
                        "screen",
                        "overlay",
                        "darken",
                        "lighten",
                        "color_dodge",
                        "color_burn",
                        "hard_light",
                        "soft_light",
                        "difference",
                        "exclusion",
                        "hue",
                        "saturation",
                        "color",
                        "luminosity",
                    ],
                    default="normal",
                ),
                io.Float.Input(
                    "blend_percentage", default=100.0, min=0.0, max=100.0, step=1.0
                ),
                io.Mask.Input("overlay_mask", optional=True),
                io.Boolean.Input("invert_mask", default=False),
            ],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    async def execute(
        cls,
        overlay_image: torch.Tensor,
        base_image: torch.Tensor,
        blend_mode: str,
        blend_percentage: float,
        overlay_mask: torch.Tensor | None = None,
        invert_mask: bool = False,
    ) -> io.NodeOutput:
        pct = float(max(0.0, min(100.0, blend_percentage)))
        opacity = pct / 100.0
        base_image = base_image.to(torch.float32).clamp(0.0, 1.0)
        overlay_image = overlay_image.to(torch.float32).clamp(0.0, 1.0)
        base_has_alpha = int(base_image.shape[-1]) == 4

        base_rgba = cls._ensure_rgba(base_image)
        overlay_rgba = cls._ensure_rgba(overlay_image)

        bs_base = base_rgba.shape[0]
        bs_ov = overlay_rgba.shape[0]
        max_bs = max(bs_base, bs_ov)

        device = base_rgba.device
        out_imgs: list[torch.Tensor] = []

        for b in range(max_bs):
            i_base = b % bs_base
            i_ov = b % bs_ov

            base = base_rgba[i_base]
            ov = overlay_rgba[i_ov]

            if base.shape[:2] != ov.shape[:2]:
                ov = cls._resize_tensor_image(ov, (base.shape[1], base.shape[0]))
                ov = ov.to(base.device)

            base_rgb = base[:, :, :3]
            base_alpha = base[:, :, 3:4]
            ov_rgb = ov[:, :, :3]
            ov_alpha = ov[:, :, 3:4]

            blended_rgb = cls._apply_css_blend(base_rgb, ov_rgb, blend_mode)
            effective_alpha = ov_alpha * opacity

            if overlay_mask is not None:
                current_mask_batch = overlay_mask
                if current_mask_batch.ndim == 2:
                    current_mask_batch = current_mask_batch.unsqueeze(0)
                if current_mask_batch.ndim == 4 and int(current_mask_batch.shape[-1]) >= 1:
                    current_mask_batch = current_mask_batch[:, :, :, 0]
                if current_mask_batch.ndim != 3:
                    raise ValueError("overlay_mask shape must be [H,W], [B,H,W], or [B,H,W,C]")
                bs_mask = current_mask_batch.shape[0]
                m = current_mask_batch[b % bs_mask]
                if invert_mask:
                    m = 1.0 - m
                if m.shape[:2] != base.shape[:2]:
                    m = cls._resize_tensor_mask(m, (base.shape[1], base.shape[0]))
                m = m.to(base.device).unsqueeze(-1)
                effective_alpha = effective_alpha * m

            out_alpha = effective_alpha + base_alpha * (1.0 - effective_alpha)
            out_rgb_premult = (
                blended_rgb * effective_alpha
                + base_rgb * base_alpha * (1.0 - effective_alpha)
            )
            safe_alpha = torch.where(
                out_alpha > 1e-6,
                out_alpha,
                torch.ones_like(out_alpha),
            )
            out_rgb = torch.where(
                out_alpha > 1e-6,
                out_rgb_premult / safe_alpha,
                torch.zeros_like(out_rgb_premult),
            )
            result_rgba = torch.cat([out_rgb, out_alpha], dim=2)
            out_imgs.append(result_rgba if base_has_alpha else result_rgba[:, :, :3])

        result = torch.stack(out_imgs).to(device).clamp(0.0, 1.0).to(torch.float32)
        return io.NodeOutput(result)

    @staticmethod
    def _ensure_rgba(img: torch.Tensor) -> torch.Tensor:
        channels = int(img.shape[-1])
        if channels == 4:
            return img
        if channels == 3:
            alpha = torch.ones(
                (img.shape[0], img.shape[1], img.shape[2], 1),
                dtype=img.dtype,
                device=img.device,
            )
            return torch.cat([img, alpha], dim=3)
        if channels == 1:
            rgb = img.repeat(1, 1, 1, 3)
            alpha = torch.ones(
                (img.shape[0], img.shape[1], img.shape[2], 1),
                dtype=img.dtype,
                device=img.device,
            )
            return torch.cat([rgb, alpha], dim=3)
        if channels > 4:
            return img[:, :, :, :4]
        raise ValueError(f"unsupported image channels: {channels}")

    @staticmethod
    def _resize_tensor_image(t: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        np_img = (t.detach().cpu().numpy() * 255).astype(np.uint8)
        pil = Image.fromarray(np_img)
        if int(t.shape[-1]) == 4:
            pil = pil.convert("RGBA")
        elif pil.mode != "RGB":
            pil = pil.convert("RGB")
        pil = pil.resize(size, Image.Resampling.LANCZOS)
        out = np.array(pil).astype(np.float32) / 255.0
        return torch.from_numpy(out)

    @staticmethod
    def _resize_tensor_mask(t: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        np_mask = (t.detach().cpu().numpy() * 255).astype(np.uint8)
        pil = Image.fromarray(np_mask).convert("L")
        pil = pil.resize(size, Image.Resampling.LANCZOS)
        out = np.array(pil).astype(np.float32) / 255.0
        return torch.from_numpy(out)

    @staticmethod
    def _apply_css_blend(base: torch.Tensor, overlay: torch.Tensor, mode: str) -> torch.Tensor:
        overlay = overlay.to(base.device)
        if mode == "normal":
            return overlay
        if mode == "multiply":
            return base * overlay
        if mode == "screen":
            return 1.0 - (1.0 - base) * (1.0 - overlay)
        if mode == "overlay":
            mask = base > 0.5
            return torch.where(
                mask,
                1.0 - 2.0 * (1.0 - base) * (1.0 - overlay),
                2.0 * base * overlay,
            )
        if mode == "darken":
            return torch.minimum(base, overlay)
        if mode == "lighten":
            return torch.maximum(base, overlay)
        if mode == "color_dodge":
            mask = overlay < 1.0
            return torch.where(
                mask,
                torch.minimum(torch.ones_like(base), base / (1.0 - overlay)),
                torch.ones_like(base),
            )
        if mode == "color_burn":
            mask = overlay > 0.0
            return torch.where(
                mask,
                1.0 - torch.minimum(
                    torch.ones_like(base), (1.0 - base) / overlay
                ),
                torch.zeros_like(base),
            )
        if mode == "hard_light":
            mask = overlay > 0.5
            return torch.where(
                mask,
                1.0 - 2.0 * (1.0 - overlay) * (1.0 - base),
                2.0 * overlay * base,
            )
        if mode == "soft_light":
            mask = overlay > 0.5
            return torch.where(
                mask,
                base + (2.0 * overlay - 1.0) * (torch.sqrt(base) - base),
                base - (1.0 - 2.0 * overlay) * base * (1.0 - base),
            )
        if mode == "difference":
            return torch.abs(base - overlay)
        if mode == "exclusion":
            return base + overlay - 2.0 * base * overlay
        if mode in {"hue", "saturation", "color", "luminosity"}:
            use = {
                "hue": ("h"),
                "saturation": ("s"),
                "color": ("h", "s"),
                "luminosity": ("l"),
            }[mode]
            return ImageBlendModeByCSS._blend_hsl(base, overlay, use=use)
        return overlay

    @staticmethod
    def _blend_hsl(
        base: torch.Tensor, overlay: torch.Tensor, use: tuple[str, ...] | str
    ) -> torch.Tensor:
        b_rgb = base[:, :, :, :3].clamp(0.0, 1.0)
        o_rgb = overlay[:, :, :, :3].clamp(0.0, 1.0)
        b_hsl = ImageBlendModeByCSS._rgb_to_hsl(b_rgb)
        o_hsl = ImageBlendModeByCSS._rgb_to_hsl(o_rgb)
        if isinstance(use, str):
            use = (use,)
        r_hsl = b_hsl.clone()
        if "h" in use:
            r_hsl[:, :, :, 0] = o_hsl[:, :, :, 0]
        if "s" in use:
            r_hsl[:, :, :, 1] = o_hsl[:, :, :, 1]
        if "l" in use:
            r_hsl[:, :, :, 2] = o_hsl[:, :, :, 2]
        r_rgb = ImageBlendModeByCSS._hsl_to_rgb(r_hsl)
        out = base.clone()
        out[:, :, :, :3] = r_rgb
        return out

    @staticmethod
    def _rgb_to_hsl(rgb: torch.Tensor) -> torch.Tensor:
        rgb = rgb.clamp(0.0, 1.0)
        r = rgb[:, :, :, 0]
        g = rgb[:, :, :, 1]
        b = rgb[:, :, :, 2]
        max_v = torch.maximum(torch.maximum(r, g), b)
        min_v = torch.minimum(torch.minimum(r, g), b)
        diff = max_v - min_v
        l = (max_v + min_v) / 2.0
        s = torch.zeros_like(l)
        nz = diff != 0
        s = torch.where(nz & (l < 0.5), diff / (max_v + min_v), s)
        s = torch.where(nz & (l >= 0.5), diff / (2.0 - max_v - min_v), s)
        h = torch.zeros_like(l)
        mask_r = (max_v == r) & nz
        h = torch.where(mask_r, ((g - b) / diff) % 6, h)
        mask_g = (max_v == g) & nz
        h = torch.where(mask_g, (b - r) / diff + 2, h)
        mask_b = (max_v == b) & nz
        h = torch.where(mask_b, (r - g) / diff + 4, h)
        h = h / 6.0
        return torch.stack([h, s, l], dim=-1)

    @staticmethod
    def _hsl_to_rgb(hsl: torch.Tensor) -> torch.Tensor:
        h = hsl[:, :, :, 0] % 1.0
        s = hsl[:, :, :, 1]
        l = hsl[:, :, :, 2]
        c = (1.0 - torch.abs(2.0 * l - 1.0)) * s
        x = c * (1.0 - torch.abs((h * 6.0) % 2.0 - 1.0))
        m = l - c / 2.0
        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)
        m1 = (h >= 0.0) & (h < 1.0 / 6.0)
        r = torch.where(m1, c, r)
        g = torch.where(m1, x, g)
        m2 = (h >= 1.0 / 6.0) & (h < 2.0 / 6.0)
        r = torch.where(m2, x, r)
        g = torch.where(m2, c, g)
        m3 = (h >= 2.0 / 6.0) & (h < 3.0 / 6.0)
        g = torch.where(m3, c, g)
        b = torch.where(m3, x, b)
        m4 = (h >= 3.0 / 6.0) & (h < 4.0 / 6.0)
        g = torch.where(m4, x, g)
        b = torch.where(m4, c, b)
        m5 = (h >= 4.0 / 6.0) & (h < 5.0 / 6.0)
        r = torch.where(m5, x, r)
        b = torch.where(m5, c, b)
        m6 = (h >= 5.0 / 6.0) & (h <= 1.0)
        r = torch.where(m6, c, r)
        b = torch.where(m6, x, b)
        r = r + m
        g = g + m
        b = b + m
        return torch.stack([r, g, b], dim=-1)
