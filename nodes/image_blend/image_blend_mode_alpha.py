from comfy_api.latest import io
import numpy as np
import torch
from PIL import Image, ImageOps


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
        invert_mask: bool,
        overlay_mask: torch.Tensor | None = None,
    ) -> io.NodeOutput:
        # 归一化输入
        base_image = torch.clamp(base_image, 0.0, 1.0).to(torch.float32)
        overlay_image = torch.clamp(overlay_image, 0.0, 1.0).to(torch.float32)
        # 检查并转换 RGBA 图像为 RGB
        base_image = cls._convert_rgba_to_rgb(base_image)
        overlay_image = cls._convert_rgba_to_rgb(overlay_image)
        
        # 处理叠加图层
        blended = cls._apply_blend(base_image, overlay_image, blend_mode, opacity)
        
        # 如果提供了遮罩，则应用遮罩
        if overlay_mask is not None:
            # 获取批次大小
            base_batch_size = base_image.shape[0]
            mask_batch_size = overlay_mask.shape[0]
            
            # 创建输出图像列表
            output_images = []
            
            for b in range(base_batch_size):
                # 获取当前批次的图像
                current_base = base_image[b]
                current_blended = blended[b]
                
                # 确定使用哪个遮罩（如果遮罩数量少于图像数量，则循环使用）
                mask_index = b % mask_batch_size
                current_mask = overlay_mask[mask_index]
                
                # 如果需要反转遮罩
                if invert_mask:
                    current_mask = 1.0 - current_mask
                
                # 将遮罩调整为与图像相同的尺寸
                if current_mask.shape[:2] != current_base.shape[:2]:
                    # 将遮罩转换为PIL格式
                    mask_np = (current_mask.detach().cpu().numpy() * 255).astype(np.uint8)
                    mask_pil = Image.fromarray(mask_np).convert("L")
                    
                    # 调整大小
                    mask_pil = mask_pil.resize((current_base.shape[1], current_base.shape[0]), Image.Resampling.LANCZOS)
                    
                    # 转换回numpy格式并确保在正确的设备上
                    mask_np = np.array(mask_pil).astype(np.float32) / 255.0
                    current_mask = torch.from_numpy(mask_np).to(current_base.device)
                
                # 确保遮罩在正确的设备上
                current_mask = current_mask.to(current_base.device)
                
                # 扩展遮罩维度以匹配图像通道
                current_mask = current_mask.unsqueeze(-1).expand_as(current_base)
                
                # 应用遮罩混合
                masked_result = current_base * (1.0 - current_mask) + current_blended * current_mask
                output_images.append(masked_result)
            
            # 合并批次
            result = torch.stack(output_images)
        else:
            result = blended
        
        return io.NodeOutput(result)
    
    @staticmethod
    def _convert_rgba_to_rgb(image):
        """将RGBA图像转换为RGB图像"""
        # 检查图像是否为RGBA格式（4通道）
        if image.shape[3] == 4:
            # 提取RGB通道
            rgb_image = image[:, :, :, :3]
            
            # 获取Alpha通道
            alpha_channel = image[:, :, :, 3:4]
            
            # 使用Alpha通道混合RGB与白色背景
            white_bg = torch.ones_like(rgb_image)
            rgb_image = rgb_image * alpha_channel + white_bg * (1 - alpha_channel)
            
            return rgb_image
        else:
            # 如果已经是RGB格式，直接返回
            return image
    
    @classmethod
    def _apply_blend(cls, base, overlay, blend_mode, opacity):
        # 确保两个图像具有相同的批次大小
        base_batch_size = base.shape[0]
        overlay_batch_size = overlay.shape[0]
        
        # 处理批量不匹配：使用最大批次数量并复制较少批次的数据
        if base_batch_size != overlay_batch_size:
            max_batch_size = max(base_batch_size, overlay_batch_size)
            
            # 复制基础图像以匹配最大批次数量
            if base_batch_size < max_batch_size:
                repeat_factor = max_batch_size // base_batch_size
                remainder = max_batch_size % base_batch_size
                base_repeated = base.repeat(repeat_factor, 1, 1, 1)
                if remainder > 0:
                    base_remainder = base[:remainder]
                    base = torch.cat([base_repeated, base_remainder], dim=0)
                else:
                    base = base_repeated
            
            # 复制叠加图像以匹配最大批次数量
            if overlay_batch_size < max_batch_size:
                repeat_factor = max_batch_size // overlay_batch_size
                remainder = max_batch_size % overlay_batch_size
                overlay_repeated = overlay.repeat(repeat_factor, 1, 1, 1)
                if remainder > 0:
                    overlay_remainder = overlay[:remainder]
                    overlay = torch.cat([overlay_repeated, overlay_remainder], dim=0)
                else:
                    overlay = overlay_repeated
        
        # 确保两个图像具有相同的尺寸
        base_height, base_width = base.shape[1:3]
        overlay_height, overlay_width = overlay.shape[1:3]
        
        if base_height != overlay_height or base_width != overlay_width:
            # 调整叠加图层的大小以匹配基础图层
            resized_overlay = []
            
            for i in range(overlay.shape[0]):
                # 将张量转换为PIL图像
                if overlay.shape[3] == 3:  # RGB
                    img = Image.fromarray((overlay[i].detach().cpu().numpy() * 255).astype(np.uint8))
                else:  # RGBA
                    img = Image.fromarray((overlay[i].detach().cpu().numpy() * 255).astype(np.uint8), 'RGBA')
                
                # 调整大小
                img = img.resize((base_width, base_height), Image.Resampling.LANCZOS)
                
                # 转换回张量并确保在正确的设备上
                img_np = np.array(img).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_np).to(base.device)  # 添加 .to(base.device)
                resized_overlay.append(img_tensor)
            
            overlay = torch.stack(resized_overlay)
        
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
        
        return result
    
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
