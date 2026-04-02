from comfy_api.latest import io
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


class ImageBlendModeByAlpha(io.ComfyNode):
    """
    图层叠加模式 - 支持基础图层输入，控制叠加模式和透明度强度
    """
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_ImageBlendModeByAlpha",
            display_name="Image Blend Mode by Alpha",
            category="1hewNodes/image/blend",
            inputs=[
                io.Image.Input("overlay_image"),
                io.Image.Input("base_image"),
                io.Mask.Input("overlay_mask", optional=True),
                io.Combo.Input(
                    "blend_mode",
                    options=[
                        "normal", "dissolve", "darken", "multiply", "color_burn", "linear_burn", 
                        "add", "lighten", "screen", "color_dodge", "linear_dodge",
                        "overlay", "soft_light", "hard_light", "linear_light", "vivid_light", "pin_light", "hard_mix",
                        "difference", "exclusion",  "subtract", "divide", 
                        "hue", "saturation", "color", "luminosity",
                    ],
                    default="normal",
                ),
                io.Combo.Input(
                    "overlay_fit",
                    options=["stretch", "center"],
                    default="stretch",
                ),
                io.Float.Input("opacity", default=1.0, min=0.0, max=1.0, step=0.01),
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
        opacity: float,
        overlay_fit: str,
        invert_mask: bool,
        overlay_mask: torch.Tensor | None = None,
    ) -> io.NodeOutput:
        base_image = torch.clamp(base_image, 0.0, 1.0).to(torch.float32)
        overlay_image = torch.clamp(overlay_image, 0.0, 1.0).to(torch.float32)
        base_has_alpha = int(base_image.shape[-1]) == 4

        base_rgba = cls._ensure_rgba(base_image)
        overlay_rgba = cls._ensure_rgba(overlay_image)

        batch_size = max(base_rgba.shape[0], overlay_rgba.shape[0])
        base_rgba = cls._repeat_to_batch_size(base_rgba, batch_size)
        overlay_rgba = cls._repeat_to_batch_size(overlay_rgba, batch_size)

        base_height, base_width = base_rgba.shape[1:3]
        overlay_rgba = cls._fit_rgba_to_base(
            overlay_rgba, base_height, base_width, overlay_fit
        ).to(base_rgba.device)

        base_rgb = base_rgba[:, :, :, :3]
        base_alpha = base_rgba[:, :, :, 3:4]
        overlay_rgb = overlay_rgba[:, :, :, :3]
        overlay_alpha = overlay_rgba[:, :, :, 3:4]

        blended_rgb = cls._apply_blend(
            base_rgb,
            overlay_rgb,
            blend_mode,
            1.0,
            "stretch",
        )

        effective_alpha = overlay_alpha * float(opacity)
        if overlay_mask is not None:
            prepared_mask = cls._prepare_overlay_mask(
                overlay_mask=overlay_mask,
                batch_size=batch_size,
                target_height=base_height,
                target_width=base_width,
                invert_mask=bool(invert_mask),
                device=base_rgba.device,
            )
            effective_alpha = effective_alpha * prepared_mask

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

        result_rgba = torch.cat([out_rgb, out_alpha], dim=3).clamp(0.0, 1.0)
        if base_has_alpha:
            return io.NodeOutput(result_rgba.to(torch.float32))
        return io.NodeOutput(result_rgba[:, :, :, :3].to(torch.float32))

    @staticmethod
    def _repeat_to_batch_size(images: torch.Tensor, batch_size: int) -> torch.Tensor:
        current_batch_size = images.shape[0]
        if current_batch_size == batch_size:
            return images

        repeat_factor = batch_size // current_batch_size
        remainder = batch_size % current_batch_size

        repeat_dims = [repeat_factor] + [1] * (images.dim() - 1)
        repeated = images.repeat(*repeat_dims)
        if remainder > 0:
            repeated = torch.cat([repeated, images[:remainder]], dim=0)

        return repeated

    @staticmethod
    def _ensure_rgba(image: torch.Tensor) -> torch.Tensor:
        channels = int(image.shape[3])
        if channels == 4:
            return image
        if channels == 3:
            alpha = torch.ones(
                (image.shape[0], image.shape[1], image.shape[2], 1),
                dtype=image.dtype,
                device=image.device,
            )
            return torch.cat([image, alpha], dim=3)
        if channels == 1:
            rgb = image.repeat(1, 1, 1, 3)
            alpha = torch.ones(
                (image.shape[0], image.shape[1], image.shape[2], 1),
                dtype=image.dtype,
                device=image.device,
            )
            return torch.cat([rgb, alpha], dim=3)
        if channels > 4:
            return image[:, :, :, :4]
        raise ValueError(f"unsupported image channels: {channels}")

    @classmethod
    def _fit_rgba_to_base(
        cls,
        overlay_rgba: torch.Tensor,
        base_height: int,
        base_width: int,
        overlay_fit: str,
    ) -> torch.Tensor:
        overlay_height, overlay_width = overlay_rgba.shape[1:3]
        if overlay_height == base_height and overlay_width == base_width:
            return overlay_rgba
        if overlay_fit == "center":
            centered_overlay, _ = cls._center_fit_overlay(
                overlay_rgba, base_height, base_width
            )
            return centered_overlay
        return cls._stretch_overlay_to_base(overlay_rgba, base_height, base_width)

    @classmethod
    def _prepare_overlay_mask(
        cls,
        overlay_mask: torch.Tensor,
        batch_size: int,
        target_height: int,
        target_width: int,
        invert_mask: bool,
        device: torch.device,
    ) -> torch.Tensor:
        mask = overlay_mask.to(torch.float32)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim == 4 and int(mask.shape[-1]) >= 1:
            mask = mask[:, :, :, 0]
        if mask.ndim != 3:
            raise ValueError("overlay_mask shape must be [H,W], [B,H,W], or [B,H,W,C]")
        mask = cls._repeat_to_batch_size(mask, batch_size)
        prepared: list[torch.Tensor] = []
        for current_mask in mask:
            if bool(invert_mask):
                current_mask = 1.0 - current_mask
            if current_mask.shape[:2] != (target_height, target_width):
                mask_np = (current_mask.detach().cpu().numpy() * 255).astype(np.uint8)
                mask_pil = Image.fromarray(mask_np).convert("L")
                mask_pil = mask_pil.resize(
                    (target_width, target_height), Image.Resampling.LANCZOS
                )
                mask_np = np.array(mask_pil).astype(np.float32) / 255.0
                current_mask = torch.from_numpy(mask_np)
            prepared.append(current_mask.unsqueeze(0).unsqueeze(-1))
        return torch.cat(prepared, dim=0).to(device=device, dtype=torch.float32)
    
    @classmethod
    def _apply_blend(cls, base, overlay, blend_mode, opacity, overlay_fit_mode):
        # 确保两个图像具有相同的尺寸
        base_height, base_width = base.shape[1:3]
        overlay_height, overlay_width = overlay.shape[1:3]
        blend_region_mask = None
        
        if base_height != overlay_height or base_width != overlay_width:
            if overlay_fit_mode == "center":
                overlay, blend_region_mask = cls._center_fit_overlay(overlay, base_height, base_width)
            else:
                overlay = cls._stretch_overlay_to_base(overlay, base_height, base_width)
        
        # 确保 overlay 在与 base 相同的设备上
        overlay = overlay.to(base.device)
        
        # 应用混合模式
        result = base.clone()
        
        # 根据不同的混合模式应用不同的操作
        if blend_mode == "normal":
            # 正常模式：直接叠加
            result = cls._normal_blend(base, overlay, opacity)
        elif blend_mode == "dissolve":
            result = cls._dissolve_blend(base, overlay, opacity)
        elif blend_mode == "darken":
            result = cls._darken_blend(base, overlay, opacity)
        elif blend_mode == "multiply":
            result = cls._multiply_blend(base, overlay, opacity)
        elif blend_mode == "color_burn":
            result = cls._color_burn_blend(base, overlay, opacity)
        elif blend_mode == "linear_burn":
            result = cls._linear_burn_blend(base, overlay, opacity)
        elif blend_mode == "add":
            result = cls._add_blend(base, overlay, opacity)
        elif blend_mode == "lighten":
            result = cls._lighten_blend(base, overlay, opacity)
        elif blend_mode == "screen":
            result = cls._screen_blend(base, overlay, opacity)
        elif blend_mode == "color_dodge":
            result = cls._color_dodge_blend(base, overlay, opacity)
        elif blend_mode == "linear_dodge":
            result = cls._linear_dodge_blend(base, overlay, opacity)
        elif blend_mode == "overlay":
            result = cls._overlay_blend(base, overlay, opacity)
        elif blend_mode == "soft_light":
            result = cls._soft_light_blend(base, overlay, opacity)
        elif blend_mode == "hard_light":
            result = cls._hard_light_blend(base, overlay, opacity)
        elif blend_mode == "linear_light":
            result = cls._linear_light_blend(base, overlay, opacity)
        elif blend_mode == "vivid_light":
            result = cls._vivid_light_blend(base, overlay, opacity)
        elif blend_mode == "pin_light":
            result = cls._pin_light_blend(base, overlay, opacity)
        elif blend_mode == "hard_mix":
            result = cls._hard_mix_blend(base, overlay, opacity)
        elif blend_mode == "difference":
            result = cls._difference_blend(base, overlay, opacity)
        elif blend_mode == "exclusion":
            result = cls._exclusion_blend(base, overlay, opacity)
        elif blend_mode == "subtract":
            result = cls._subtract_blend(base, overlay, opacity)
        elif blend_mode == "divide":
            result = cls._divide_blend(base, overlay, opacity)
        elif blend_mode == "hue":
            result = cls._hue_blend(base, overlay, opacity)
        elif blend_mode == "saturation":
            result = cls._saturation_blend(base, overlay, opacity)
        elif blend_mode == "color":
            result = cls._color_blend(base, overlay, opacity)
        elif blend_mode == "luminosity":
            result = cls._luminosity_blend(base, overlay, opacity)

        # center 模式下，仅在叠加图层覆盖区域应用混合，避免未覆盖区域被影响
        if blend_region_mask is not None:
            blend_region_mask = blend_region_mask.to(base.device)
            result = base * (1.0 - blend_region_mask) + result * blend_region_mask
        
        return result

    @staticmethod
    def _stretch_overlay_to_base(overlay: torch.Tensor, base_height: int, base_width: int) -> torch.Tensor:
        """将叠加图层拉伸到基础图层尺寸"""
        overlay_nchw = overlay.permute(0, 3, 1, 2)
        resized = F.interpolate(
            overlay_nchw,
            size=(base_height, base_width),
            mode="bilinear",
            align_corners=False,
        )
        return resized.permute(0, 2, 3, 1)

    @staticmethod
    def _center_fit_overlay(
        overlay: torch.Tensor, base_height: int, base_width: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        居中放置叠加图层：
        - 比基础图层大时从中心裁切
        - 比基础图层小时居中填充
        """
        batch, overlay_height, overlay_width, channels = overlay.shape
        centered_overlay = torch.zeros(
            (batch, base_height, base_width, channels),
            device=overlay.device,
            dtype=overlay.dtype,
        )
        blend_region_mask = torch.zeros(
            (batch, base_height, base_width, 1),
            device=overlay.device,
            dtype=overlay.dtype,
        )

        copy_height = min(base_height, overlay_height)
        copy_width = min(base_width, overlay_width)

        src_y = max((overlay_height - base_height) // 2, 0)
        src_x = max((overlay_width - base_width) // 2, 0)
        dst_y = max((base_height - overlay_height) // 2, 0)
        dst_x = max((base_width - overlay_width) // 2, 0)

        centered_overlay[
            :, dst_y : dst_y + copy_height, dst_x : dst_x + copy_width, :
        ] = overlay[:, src_y : src_y + copy_height, src_x : src_x + copy_width, :]
        blend_region_mask[
            :, dst_y : dst_y + copy_height, dst_x : dst_x + copy_width, :
        ] = 1.0

        return centered_overlay, blend_region_mask
    
    # 以下是各种混合模式的实现
    
    @staticmethod
    def _normal_blend(base, overlay, opacity):
        # 正常模式：直接叠加
        return base * (1 - opacity) + overlay * opacity
    
    @staticmethod
    def _dissolve_blend(base, overlay, opacity):
        # 溶解模式：随机丢弃像素
        random_mask = torch.rand_like(base[:, :, :, 0:1]) < opacity
        result = torch.where(random_mask, overlay, base)
        return result
    
    @staticmethod
    def _darken_blend(base, overlay, opacity):
        # 变暗：取最小值
        blended = torch.minimum(base, overlay)
        return base * (1 - opacity) + blended * opacity
    
    @staticmethod
    def _multiply_blend(base, overlay, opacity):
        # 正片叠底：相乘
        blended = base * overlay
        return base * (1 - opacity) + blended * opacity
    
    @staticmethod
    def _color_burn_blend(base, overlay, opacity):
        # 颜色加深：反相基础色除以叠加色再反相
        blended = torch.zeros_like(base)
        mask = overlay > 0.0
        blended = torch.where(mask, 1.0 - torch.minimum(torch.ones_like(base), (1.0 - base) / overlay), torch.zeros_like(base))
        return base * (1 - opacity) + blended * opacity
    
    @staticmethod
    def _linear_burn_blend(base, overlay, opacity):
        # 线性加深：相加后减1
        blended = torch.maximum(base + overlay - 1.0, torch.zeros_like(base))
        return base * (1 - opacity) + blended * opacity
    
    @staticmethod
    def _add_blend(base, overlay, opacity):
        # 相加：直接相加并裁剪
        blended = torch.minimum(base + overlay, torch.ones_like(base))
        return base * (1 - opacity) + blended * opacity
    
    @staticmethod
    def _lighten_blend(base, overlay, opacity):
        # 变亮：取最大值
        blended = torch.maximum(base, overlay)
        return base * (1 - opacity) + blended * opacity
    
    @staticmethod
    def _screen_blend(base, overlay, opacity):
        # 滤色：反相乘后再反相
        blended = 1.0 - (1.0 - base) * (1.0 - overlay)
        return base * (1 - opacity) + blended * opacity
    
    @staticmethod
    def _color_dodge_blend(base, overlay, opacity):
        # 颜色减淡：基础色除以反相叠加色
        blended = torch.zeros_like(base)
        mask = overlay < 1.0
        blended = torch.where(mask, torch.minimum(torch.ones_like(base), base / (1.0 - overlay)), torch.ones_like(base))
        return base * (1 - opacity) + blended * opacity
    
    @staticmethod
    def _linear_dodge_blend(base, overlay, opacity):
        # 线性减淡：相加
        blended = torch.minimum(base + overlay, torch.ones_like(base))
        return base * (1 - opacity) + blended * opacity
    
    @staticmethod
    def _overlay_blend(base, overlay, opacity):
        # 叠加：结合正片叠底和滤色
        mask = base > 0.5
        blended = torch.zeros_like(base)
        blended = torch.where(mask, 1.0 - 2.0 * (1.0 - base) * (1.0 - overlay), 2.0 * base * overlay)
        return base * (1 - opacity) + blended * opacity
    
    @staticmethod
    def _soft_light_blend(base, overlay, opacity):
        # 柔光：柔和的光照效果
        blended = torch.zeros_like(base)
        mask = overlay > 0.5
        blended = torch.where(
            mask,
            base + (2 * overlay - 1) * (torch.sqrt(base) - base),
            base - (1 - 2 * overlay) * base * (1 - base)
        )
        return base * (1 - opacity) + blended * opacity
    
    @staticmethod
    def _hard_light_blend(base, overlay, opacity):
        # 强光：强烈的光照效果（与叠加相反）
        mask = overlay > 0.5
        blended = torch.zeros_like(base)
        blended = torch.where(mask, 1.0 - 2.0 * (1.0 - overlay) * (1.0 - base), 2.0 * overlay * base)
        return base * (1 - opacity) + blended * opacity
    
    @staticmethod
    def _linear_light_blend(base, overlay, opacity):
        # 线性光：根据叠加色决定线性减淡或线性加深
        mask = overlay > 0.5
        blended = torch.zeros_like(base)
        # 线性减淡部分
        dodge = torch.minimum(base + (2.0 * overlay - 1.0), torch.ones_like(base))
        # 线性加深部分
        burn = torch.maximum(base + 2.0 * overlay - 1.0, torch.zeros_like(base))
        blended = torch.where(mask, dodge, burn)
        return base * (1 - opacity) + blended * opacity
    
    @staticmethod
    def _vivid_light_blend(base, overlay, opacity):
        # 亮光：根据叠加色决定颜色减淡或颜色加深
        mask = overlay > 0.5
        blended = torch.zeros_like(base)
        
        # 颜色减淡部分
        dodge_overlay = (overlay - 0.5) * 2.0
        dodge_mask = dodge_overlay < 1.0
        dodge = torch.where(
            dodge_mask,
            torch.minimum(torch.ones_like(base), base / (1.0 - dodge_overlay)),
            torch.ones_like(base)
        )
        
        # 颜色加深部分
        burn_overlay = overlay * 2.0
        burn_mask = burn_overlay > 0.0
        burn = torch.where(
            burn_mask,
            1.0 - torch.minimum(torch.ones_like(base), (1.0 - base) / burn_overlay),
            torch.zeros_like(base)
        )
        
        blended = torch.where(mask, dodge, burn)
        return base * (1 - opacity) + blended * opacity
    
    @staticmethod
    def _pin_light_blend(base, overlay, opacity):
        # 点光：根据叠加色决定变亮或变暗
        mask = overlay > 0.5
        blended = torch.zeros_like(base)
        
        # 变亮部分
        lighten = torch.maximum(base, (overlay - 0.5) * 2.0)
        
        # 变暗部分
        darken = torch.minimum(base, overlay * 2.0)
        
        blended = torch.where(mask, lighten, darken)
        return base * (1 - opacity) + blended * opacity
    
    @staticmethod
    def _hard_mix_blend(base, overlay, opacity):
        # 实色混合：根据基础色和叠加色的和决定是0还是1
        temp = base + overlay
        blended = torch.where(temp > 1.0, torch.ones_like(base), torch.zeros_like(base))
        return base * (1 - opacity) + blended * opacity
    
    @staticmethod
    def _difference_blend(base, overlay, opacity):
        # 差值：取绝对差
        blended = torch.abs(base - overlay)
        return base * (1 - opacity) + blended * opacity
    
    @staticmethod
    def _exclusion_blend(base, overlay, opacity):
        # 排除：类似于差值，但对中间调的影响较小
        blended = base + overlay - 2.0 * base * overlay
        return base * (1 - opacity) + blended * opacity
    
    @staticmethod
    def _subtract_blend(base, overlay, opacity):
        # 减去：基础色减去叠加色
        blended = torch.maximum(base - overlay, torch.zeros_like(base))
        return base * (1 - opacity) + blended * opacity
    
    @staticmethod
    def _divide_blend(base, overlay, opacity):
        # 划分：基础色除以叠加色
        # 避免除以零
        safe_overlay = torch.maximum(overlay, torch.ones_like(overlay) * 1e-5)
        blended = torch.minimum(base / safe_overlay, torch.ones_like(base))
        return base * (1 - opacity) + blended * opacity
    
    @classmethod
    def _hue_blend(cls, base, overlay, opacity):
        # 色相：使用叠加层的色相，基础层的饱和度和亮度
        # 处理RGB通道
        base_rgb = base[:, :, :, :3]
        overlay_rgb = overlay[:, :, :, :3]
        
        # 转换为HSL
        base_hsl = cls._rgb_to_hsl(base_rgb)
        overlay_hsl = cls._rgb_to_hsl(overlay_rgb)
        
        # 使用叠加层的色相
        result_hsl = base_hsl.clone()
        result_hsl[:, :, :, 0] = overlay_hsl[:, :, :, 0]
        
        # 转换回RGB
        result_rgb = cls._hsl_to_rgb(result_hsl)
        
        # 创建结果
        result = base.clone()
        result[:, :, :, :3] = base_rgb * (1 - opacity) + result_rgb * opacity
        
        return result
    
    @classmethod
    def _saturation_blend(cls, base, overlay, opacity):
        # 饱和度：使用叠加层的饱和度，基础层的色相和亮度
        # 处理RGB通道
        base_rgb = base[:, :, :, :3]
        overlay_rgb = overlay[:, :, :, :3]
        
        # 转换为HSL
        base_hsl = cls._rgb_to_hsl(base_rgb)
        overlay_hsl = cls._rgb_to_hsl(overlay_rgb)
        
        # 使用叠加层的饱和度
        result_hsl = base_hsl.clone()
        result_hsl[:, :, :, 1] = overlay_hsl[:, :, :, 1]
        
        # 转换回RGB
        result_rgb = cls._hsl_to_rgb(result_hsl)
        
        # 创建结果
        result = base.clone()
        result[:, :, :, :3] = base_rgb * (1 - opacity) + result_rgb * opacity
        
        return result
    
    @classmethod
    def _color_blend(cls, base, overlay, opacity):
        # 颜色：使用叠加层的色相和饱和度，基础层的亮度
        # 处理RGB通道
        base_rgb = base[:, :, :, :3]
        overlay_rgb = overlay[:, :, :, :3]
        
        # 转换为HSL
        base_hsl = cls._rgb_to_hsl(base_rgb)
        overlay_hsl = cls._rgb_to_hsl(overlay_rgb)
        
        # 使用叠加层的色相和饱和度
        result_hsl = base_hsl.clone()
        result_hsl[:, :, :, 0] = overlay_hsl[:, :, :, 0]
        result_hsl[:, :, :, 1] = overlay_hsl[:, :, :, 1]
        
        # 转换回RGB
        result_rgb = cls._hsl_to_rgb(result_hsl)
        
        # 创建结果
        result = base.clone()
        result[:, :, :, :3] = base_rgb * (1 - opacity) + result_rgb * opacity
        
        return result
    
    @classmethod
    def _luminosity_blend(cls, base, overlay, opacity):
        # 明度：使用叠加层的亮度，基础层的色相和饱和度
        # 处理RGB通道
        base_rgb = base[:, :, :, :3]
        overlay_rgb = overlay[:, :, :, :3]
        
        # 转换为HSL
        base_hsl = cls._rgb_to_hsl(base_rgb)
        overlay_hsl = cls._rgb_to_hsl(overlay_rgb)
        
        # 使用叠加层的亮度
        result_hsl = base_hsl.clone()
        result_hsl[:, :, :, 2] = overlay_hsl[:, :, :, 2]
        
        # 转换回RGB
        result_rgb = cls._hsl_to_rgb(result_hsl)
        
        # 创建结果
        result = base.clone()
        result[:, :, :, :3] = base_rgb * (1 - opacity) + result_rgb * opacity
        
        return result
    
    @staticmethod
    def _rgb_to_hsl(rgb):
        """将RGB转换为HSL"""
        # 确保输入在0-1范围内
        rgb = torch.clamp(rgb, 0, 1)
        
        r, g, b = rgb[:, :, :, 0], rgb[:, :, :, 1], rgb[:, :, :, 2]
        
        max_val = torch.maximum(torch.maximum(r, g), b)
        min_val = torch.minimum(torch.minimum(r, g), b)
        diff = max_val - min_val
        
        # 计算亮度
        l = (max_val + min_val) / 2.0
        
        # 计算饱和度
        s = torch.zeros_like(l)
        mask = diff != 0
        s = torch.where(mask, diff / (2.0 - max_val - min_val), s)
        s = torch.where(mask & (l < 0.5), diff / (max_val + min_val), s)
        
        # 计算色相
        h = torch.zeros_like(l)
        
        # 红色主导
        mask_r = (max_val == r) & (diff != 0)
        h = torch.where(mask_r, ((g - b) / diff) % 6, h)
        
        # 绿色主导
        mask_g = (max_val == g) & (diff != 0)
        h = torch.where(mask_g, (b - r) / diff + 2, h)
        
        # 蓝色主导
        mask_b = (max_val == b) & (diff != 0)
        h = torch.where(mask_b, (r - g) / diff + 4, h)
        
        h = h / 6.0
        
        return torch.stack([h, s, l], dim=-1)
    
    @staticmethod
    def _hsl_to_rgb(hsl):
        """将HSL转换为RGB"""
        h, s, l = hsl[:, :, :, 0], hsl[:, :, :, 1], hsl[:, :, :, 2]
        
        # 确保色相在0-1范围内
        h = h % 1.0
        
        c = (1 - torch.abs(2 * l - 1)) * s
        x = c * (1 - torch.abs((h * 6) % 2 - 1))
        m = l - c / 2
        
        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)
        
        # 根据色相区间计算RGB
        mask1 = (h >= 0) & (h < 1/6)
        r = torch.where(mask1, c, r)
        g = torch.where(mask1, x, g)
        
        mask2 = (h >= 1/6) & (h < 2/6)
        r = torch.where(mask2, x, r)
        g = torch.where(mask2, c, g)
        
        mask3 = (h >= 2/6) & (h < 3/6)
        g = torch.where(mask3, c, g)
        b = torch.where(mask3, x, b)
        
        mask4 = (h >= 3/6) & (h < 4/6)
        g = torch.where(mask4, x, g)
        b = torch.where(mask4, c, b)
        
        mask5 = (h >= 4/6) & (h < 5/6)
        r = torch.where(mask5, x, r)
        b = torch.where(mask5, c, b)
        
        mask6 = (h >= 5/6) & (h <= 1)
        r = torch.where(mask6, c, r)
        b = torch.where(mask6, x, b)
        
        r = r + m
        g = g + m
        b = b + m
        
        return torch.stack([r, g, b], dim=-1)
