import torch
import numpy as np
from PIL import Image, ImageOps


class Solid:
    """
    根据输入的颜色和尺寸生成纯色图像
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset_size": (["custom", "512×512 (1:1)", "768×768 (1:1)", "1024×1024 (1:1)", "1408×1408 (1:1)",
                                "768×512 (3:2)", "1728×1152 (3:2)",
                                "1024×768 (4:3)", "1664×1216 (4:3)",
                                "832×480 (16:9)", "1280×720 (16:9)", "1920×1088 (16:9)",
                                "2176×960 (21:9)"],
                              {"default": "custom"}),
                "flip_size": ("BOOLEAN", {"default": False}),
                "width": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 8}),
                "color": ("COLOR", {"default": "#FFFFFF"})
            },
            "optional": {
                "alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "invert": ("BOOLEAN", {"default": False}),
                "mask_opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "reference_images": ("IMAGE", )
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "solid"
    CATEGORY = "1hewNodes/adobe"

    def solid(self, preset_size, flip_size, width, height, color, alpha=1.0, invert=False, mask_opacity=1.0, reference_images=None):
        images = []
        masks = []

        if reference_images is not None:
            # 处理批量参考图像
            for reference_image in reference_images:
                # 从参考图像获取尺寸
                h, w, _ = reference_image.shape
                img_width = w
                img_height = h
        else:
            # 处理预设尺寸或自定义尺寸
            if preset_size != "custom":
                # 从预设尺寸中提取宽度和高度（去掉比例部分）
                dimensions = preset_size.split(" ")[0].split("×")
                img_width = int(dimensions[0])
                img_height = int(dimensions[1])

                # 如果选择了反转尺寸，交换宽高
                if flip_size:
                    img_width, img_height = img_height, img_width
            else:
                img_width = width
                img_height = height

            # 为了兼容批量处理，这里将单个尺寸的情况也当作一个批次处理
            num_images = 1
            reference_images = [None] * num_images

        # 解析颜色值
        if color.startswith("#"):
            color = color[1:]
        r = int(color[0:2], 16) / 255.0
        g = int(color[2:4], 16) / 255.0
        b = int(color[4:6], 16) / 255.0

        # 如果需要反转颜色
        if invert:
            r = 1.0 - r
            g = 1.0 - g
            b = 1.0 - b

        for reference_image in reference_images:
            if reference_image is not None:
                # 从参考图像获取尺寸
                h, w, _ = reference_image.shape
                img_width = w
                img_height = h

            # 创建纯色图像
            image = np.zeros((img_height, img_width, 3), dtype=np.float32)
            image[:, :, 0] = r
            image[:, :, 1] = g
            image[:, :, 2] = b
            # 应用 alpha 调整亮度
            image = image * alpha

            # 创建透明度蒙版
            mask = np.ones((img_height, img_width), dtype=np.float32) * mask_opacity

            # 转换为ComfyUI需要的格式 (批次, 高度, 宽度, 通道)
            image = torch.from_numpy(image).unsqueeze(0)
            mask = torch.from_numpy(mask).unsqueeze(0)

            images.append(image)
            masks.append(mask)

        # 合并所有图像和蒙版
        final_images = torch.cat(images, dim=0)
        final_masks = torch.cat(masks, dim=0)

        return (final_images, final_masks)


class LumaMatte:
    """
    亮度蒙版 - 支持批量处理图像
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "mask": ("MASK",)
            },
            "optional": {
                "invert_mask": ("BOOLEAN", {"default": False, "label": "反转遮罩"}),
                "add_background": ("BOOLEAN", {"default": True, "label": "添加背景"}),
                "background_color": ("STRING", {"default": "1.0", "label": "背景颜色 (灰度/HEX/RGB)"})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "luma_matte"

    CATEGORY = "1hewNodes/adobe"


    def luma_matte(self, images, mask, invert_mask=False, add_background=True, background_color="1.0"):
        # 获取图像尺寸
        batch_size, height, width, channels = images.shape
        mask_batch_size = mask.shape[0]
        
        # 解析背景颜色
        bg_color = self._parse_color(background_color)
        
        # 创建输出图像
        output_images = []
        
        for b in range(batch_size):
            # 将图像转换为PIL格式
            if images.is_cuda:
                img_np = (images[b].cpu().numpy() * 255).astype(np.uint8)
            else:
                img_np = (images[b].numpy() * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)

            # 确定使用哪个遮罩（如果遮罩数量少于图像数量，则循环使用）
            mask_index = b % mask_batch_size
            
            # 将遮罩转换为PIL格式
            if mask.is_cuda:
                mask_np = (mask[mask_index].cpu().numpy() * 255).astype(np.uint8)
            else:
                mask_np = (mask[mask_index].numpy() * 255).astype(np.uint8)
            mask_pil = Image.fromarray(mask_np)            

            # 调整遮罩大小以匹配图像
            if img_pil.size != mask_pil.size:
                mask_pil = mask_pil.resize(img_pil.size, Image.Resampling.LANCZOS)

            # 如果需要反转遮罩
            if invert_mask:
                mask_pil = ImageOps.invert(mask_pil)

            if add_background:
                # 使用解析后的背景颜色
                background = Image.new('RGB', img_pil.size, bg_color)
                background.paste(img_pil, (0, 0), mask_pil)

                # 转换回numpy格式
                background_np = np.array(background).astype(np.float32) / 255.0
                output_images.append(torch.from_numpy(background_np))
            else:
                # 创建透明图像
                transparent = Image.new('RGBA', img_pil.size)
                transparent.paste(img_pil, (0, 0), mask_pil)

                # 转换回numpy格式，保留所有4个通道（包括alpha）
                transparent_np = np.array(transparent).astype(np.float32) / 255.0
                output_images.append(torch.from_numpy(transparent_np))

        # 合并批次
        output_tensor = torch.stack(output_images)
        
        return (output_tensor,)
    
    def _parse_color(self, color_str):
        """解析不同格式的颜色输入"""
        color_str = color_str.strip()
        
        # 尝试解析为灰度值 (0.0-1.0)
        try:
            gray_value = float(color_str)
            if 0.0 <= gray_value <= 1.0:
                # 灰度值转换为RGB
                gray_int = int(gray_value * 255)
                return (gray_int, gray_int, gray_int)
        except ValueError:
            pass
        
        # 尝试解析为十六进制颜色 (#RRGGBB 或 RRGGBB)
        if color_str.startswith('#'):
            hex_color = color_str[1:]
        else:
            hex_color = color_str
            
        if len(hex_color) == 6:
            try:
                r = int(hex_color[0:2], 16)
                g = int(hex_color[2:4], 16)
                b = int(hex_color[4:6], 16)
                return (r, g, b)
            except ValueError:
                pass
        
        # 尝试解析为RGB格式 (R,G,B)
        if color_str.startswith('(') and color_str.endswith(')'):
            try:
                rgb = color_str[1:-1].split(',')
                if len(rgb) == 3:
                    r = int(rgb[0].strip())
                    g = int(rgb[1].strip())
                    b = int(rgb[2].strip())
                    return (r, g, b)
            except ValueError:
                pass
        
        # 默认返回白色
        return (255, 255, 255)

class BlendModes:
    """
    图层叠加模式 - 支持基础图层输入，控制叠加模式和透明度强度
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "overlay_image": ("IMAGE",),
                "base_image": ("IMAGE",),
                "blend_mode": (["normal", "dissolve", "darken", "multiply", "color burn", "linear burn", 
                                "add", "lighten", "screen", "color dodge", "linear dodge",
                                "overlay", "soft light", "hard light", "linear light", "vivid light", "pin light", "hard mix",
                                "difference", "exclusion",  "subtract", "divide", 
                                "hue", "saturation", "color", "luminosity",
                                 ], {"default": "normal"}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})
            },
            "optional": {
                "overlay_mask": ("MASK",),
                "invert_mask": ("BOOLEAN", {"default": False, "label": "反转遮罩"})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "blend_modes"
    CATEGORY = "1hewNodes/adobe"

    def blend_modes(self, base_image, overlay_image, blend_mode, opacity, overlay_mask=None, invert_mask=False):
        # 初始化结果为基础图层
        result = base_image.clone()
        
        # 检查并转换 RGBA 图像为 RGB
        base_image = self._convert_rgba_to_rgb(base_image)
        overlay_image = self._convert_rgba_to_rgb(overlay_image)
        
        # 处理叠加图层
        blended = self._apply_blend(result, overlay_image, blend_mode, opacity)
        
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
                    if overlay_mask.is_cuda:
                        mask_np = (current_mask.cpu().numpy() * 255).astype(np.uint8)
                    else:
                        mask_np = (current_mask.numpy() * 255).astype(np.uint8)
                    mask_pil = Image.fromarray(mask_np)
                    
                    # 调整大小
                    mask_pil = mask_pil.resize((current_base.shape[1], current_base.shape[0]), Image.Resampling.LANCZOS)
                    
                    # 转换回numpy格式
                    mask_np = np.array(mask_pil).astype(np.float32) / 255.0
                    current_mask = torch.from_numpy(mask_np)
                
                # 扩展遮罩维度以匹配图像通道
                current_mask = current_mask.unsqueeze(-1).expand_as(current_base)
                
                # 应用遮罩混合
                masked_result = current_base * (1.0 - current_mask) + current_blended * current_mask
                output_images.append(masked_result)
            
            # 合并批次
            result = torch.stack(output_images)
        else:
            result = blended
        
        return (result,)
    
    def _convert_rgba_to_rgb(self, image):
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
    
    def _apply_blend(self, base, overlay, blend_mode, opacity):
        # 确保两个图像具有相同的批次大小
        base_batch_size = base.shape[0]
        overlay_batch_size = overlay.shape[0]
        
        # 如果批次大小不同，则调整叠加图层以匹配基础图层
        if base_batch_size != overlay_batch_size:
            if overlay_batch_size == 1:
                # 如果叠加层只有一个图像，则复制它以匹配基础层的批次大小
                overlay = overlay.repeat(base_batch_size, 1, 1, 1)
            else:
                # 否则，循环使用叠加层的图像
                indices = [i % overlay_batch_size for i in range(base_batch_size)]
                overlay = overlay[indices]
        
        # 确保两个图像具有相同的尺寸
        base_height, base_width = base.shape[1:3]
        overlay_height, overlay_width = overlay.shape[1:3]
        
        if base_height != overlay_height or base_width != overlay_width:
            # 调整叠加图层的大小以匹配基础图层
            resized_overlay = []
            
            for i in range(overlay.shape[0]):
                # 将张量转换为PIL图像
                if overlay.shape[3] == 3:  # RGB
                    img = Image.fromarray((overlay[i].cpu().numpy() * 255).astype(np.uint8))
                else:  # RGBA
                    img = Image.fromarray((overlay[i].cpu().numpy() * 255).astype(np.uint8), 'RGBA')
                
                # 调整大小
                img = img.resize((base_width, base_height), Image.Resampling.LANCZOS)
                
                # 转换回张量
                img_np = np.array(img).astype(np.float32) / 255.0
                resized_overlay.append(torch.from_numpy(img_np))
            
            overlay = torch.stack(resized_overlay)
        
        # 应用混合模式
        result = base.clone()
        
        # 根据不同的混合模式应用不同的操作
        if blend_mode == "normal":
            # 正常模式：直接叠加
            result = self._normal_blend(base, overlay, opacity)
        elif blend_mode == "dissolve":
            result = self._dissolve_blend(base, overlay, opacity)
        elif blend_mode == "darken":
            result = self._darken_blend(base, overlay, opacity)
        elif blend_mode == "multiply":
            result = self._multiply_blend(base, overlay, opacity)
        elif blend_mode == "color burn":
            result = self._color_burn_blend(base, overlay, opacity)
        elif blend_mode == "linear burn":
            result = self._linear_burn_blend(base, overlay, opacity)
        elif blend_mode == "add":
            result = self._add_blend(base, overlay, opacity)
        elif blend_mode == "lighten":
            result = self._lighten_blend(base, overlay, opacity)
        elif blend_mode == "screen":
            result = self._screen_blend(base, overlay, opacity)
        elif blend_mode == "color dodge":
            result = self._color_dodge_blend(base, overlay, opacity)
        elif blend_mode == "linear dodge":
            result = self._linear_dodge_blend(base, overlay, opacity)
        elif blend_mode == "overlay":
            result = self._overlay_blend(base, overlay, opacity)
        elif blend_mode == "soft light":
            result = self._soft_light_blend(base, overlay, opacity)
        elif blend_mode == "hard light":
            result = self._hard_light_blend(base, overlay, opacity)
        elif blend_mode == "linear light":
            result = self._linear_light_blend(base, overlay, opacity)
        elif blend_mode == "vivid light":
            result = self._vivid_light_blend(base, overlay, opacity)
        elif blend_mode == "pin light":
            result = self._pin_light_blend(base, overlay, opacity)
        elif blend_mode == "hard mix":
            result = self._hard_mix_blend(base, overlay, opacity)
        elif blend_mode == "difference":
            result = self._difference_blend(base, overlay, opacity)
        elif blend_mode == "exclusion":
            result = self._exclusion_blend(base, overlay, opacity)
        elif blend_mode == "subtract":
            result = self._subtract_blend(base, overlay, opacity)
        elif blend_mode == "divide":
            result = self._divide_blend(base, overlay, opacity)
        elif blend_mode == "hue":
            result = self._hue_blend(base, overlay, opacity)
        elif blend_mode == "saturation":
            result = self._saturation_blend(base, overlay, opacity)
        elif blend_mode == "color":
            result = self._color_blend(base, overlay, opacity)
        elif blend_mode == "luminosity":
            result = self._luminosity_blend(base, overlay, opacity)
        
        return result
    
    # 以下是各种混合模式的实现
    
    def _normal_blend(self, base, overlay, opacity):
        # 正常模式：直接叠加
        return base * (1 - opacity) + overlay * opacity
    
    def _dissolve_blend(self, base, overlay, opacity):
        # 溶解模式：随机丢弃像素
        random_mask = torch.rand_like(base[:, :, :, 0:1]) < opacity
        result = torch.where(random_mask, overlay, base)
        return result
    
    def _darken_blend(self, base, overlay, opacity):
        # 变暗：取最小值
        blended = torch.minimum(base, overlay)
        return base * (1 - opacity) + blended * opacity
    
    def _multiply_blend(self, base, overlay, opacity):
        # 正片叠底：相乘
        blended = base * overlay
        return base * (1 - opacity) + blended * opacity
    
    def _color_burn_blend(self, base, overlay, opacity):
        # 颜色加深：反相基础色除以叠加色再反相
        blended = torch.zeros_like(base)
        mask = overlay > 0.0
        blended = torch.where(mask, 1.0 - torch.minimum(torch.ones_like(base), (1.0 - base) / overlay), torch.zeros_like(base))
        return base * (1 - opacity) + blended * opacity
    
    def _linear_burn_blend(self, base, overlay, opacity):
        # 线性加深：相加后减1
        blended = torch.maximum(base + overlay - 1.0, torch.zeros_like(base))
        return base * (1 - opacity) + blended * opacity
    
    def _add_blend(self, base, overlay, opacity):
        # 相加：直接相加并裁剪
        blended = torch.minimum(base + overlay, torch.ones_like(base))
        return base * (1 - opacity) + blended * opacity
    
    def _lighten_blend(self, base, overlay, opacity):
        # 变亮：取最大值
        blended = torch.maximum(base, overlay)
        return base * (1 - opacity) + blended * opacity
    
    def _screen_blend(self, base, overlay, opacity):
        # 滤色：反相乘后再反相
        blended = 1.0 - (1.0 - base) * (1.0 - overlay)
        return base * (1 - opacity) + blended * opacity
    
    def _color_dodge_blend(self, base, overlay, opacity):
        # 颜色减淡：基础色除以反相叠加色
        blended = torch.zeros_like(base)
        mask = overlay < 1.0
        blended = torch.where(mask, torch.minimum(torch.ones_like(base), base / (1.0 - overlay)), torch.ones_like(base))
        return base * (1 - opacity) + blended * opacity
    
    def _linear_dodge_blend(self, base, overlay, opacity):
        # 线性减淡：相加
        blended = torch.minimum(base + overlay, torch.ones_like(base))
        return base * (1 - opacity) + blended * opacity
    
    def _overlay_blend(self, base, overlay, opacity):
        # 叠加：结合正片叠底和滤色
        mask = base > 0.5
        blended = torch.zeros_like(base)
        blended = torch.where(mask, 1.0 - 2.0 * (1.0 - base) * (1.0 - overlay), 2.0 * base * overlay)
        return base * (1 - opacity) + blended * opacity
    
    def _soft_light_blend(self, base, overlay, opacity):
        # 柔光：柔和的光照效果
        blended = torch.zeros_like(base)
        mask = overlay > 0.5
        blended = torch.where(
            mask,
            base + (2 * overlay - 1) * (torch.sqrt(base) - base),
            base - (1 - 2 * overlay) * base * (1 - base)
        )
        return base * (1 - opacity) + blended * opacity
    
    def _hard_light_blend(self, base, overlay, opacity):
        # 强光：强烈的光照效果（与叠加相反）
        mask = overlay > 0.5
        blended = torch.zeros_like(base)
        blended = torch.where(mask, 1.0 - 2.0 * (1.0 - overlay) * (1.0 - base), 2.0 * overlay * base)
        return base * (1 - opacity) + blended * opacity
    
    def _linear_light_blend(self, base, overlay, opacity):
        # 线性光：根据叠加色决定线性减淡或线性加深
        mask = overlay > 0.5
        blended = torch.zeros_like(base)
        # 线性减淡部分
        dodge = torch.minimum(base + (2.0 * overlay - 1.0), torch.ones_like(base))
        # 线性加深部分
        burn = torch.maximum(base + 2.0 * overlay - 1.0, torch.zeros_like(base))
        blended = torch.where(mask, dodge, burn)
        return base * (1 - opacity) + blended * opacity
    
    def _vivid_light_blend(self, base, overlay, opacity):
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
    
    def _pin_light_blend(self, base, overlay, opacity):
        # 点光：根据叠加色决定变亮或变暗
        mask = overlay > 0.5
        blended = torch.zeros_like(base)
        
        # 变亮部分
        lighten = torch.maximum(base, (overlay - 0.5) * 2.0)
        
        # 变暗部分
        darken = torch.minimum(base, overlay * 2.0)
        
        blended = torch.where(mask, lighten, darken)
        return base * (1 - opacity) + blended * opacity
    
    def _hard_mix_blend(self, base, overlay, opacity):
        # 实色混合：根据基础色和叠加色的和决定是0还是1
        temp = base + overlay
        blended = torch.where(temp > 1.0, torch.ones_like(base), torch.zeros_like(base))
        return base * (1 - opacity) + blended * opacity
    
    def _difference_blend(self, base, overlay, opacity):
        # 差值：取绝对差
        blended = torch.abs(base - overlay)
        return base * (1 - opacity) + blended * opacity
    
    def _exclusion_blend(self, base, overlay, opacity):
        # 排除：类似于差值，但对中间调的影响较小
        blended = base + overlay - 2.0 * base * overlay
        return base * (1 - opacity) + blended * opacity
    
    def _subtract_blend(self, base, overlay, opacity):
        # 减去：基础色减去叠加色
        blended = torch.maximum(base - overlay, torch.zeros_like(base))
        return base * (1 - opacity) + blended * opacity
    
    def _divide_blend(self, base, overlay, opacity):
        # 划分：基础色除以叠加色
        # 避免除以零
        safe_overlay = torch.maximum(overlay, torch.ones_like(overlay) * 1e-5)
        blended = torch.minimum(base / safe_overlay, torch.ones_like(base))
        return base * (1 - opacity) + blended * opacity
    
    def _hue_blend(self, base, overlay, opacity):
        # 色相：使用叠加层的色相，基础层的饱和度和亮度
        # 处理RGB通道
        base_rgb = base[:, :, :, :3]
        overlay_rgb = overlay[:, :, :, :3]
        
        # 转换为HSL
        base_hsl = self._rgb_to_hsl(base_rgb)
        overlay_hsl = self._rgb_to_hsl(overlay_rgb)
        
        # 使用叠加层的色相
        result_hsl = base_hsl.clone()
        result_hsl[:, :, :, 0] = overlay_hsl[:, :, :, 0]
        
        # 转换回RGB
        result_rgb = self._hsl_to_rgb(result_hsl)
        
        # 创建结果
        result = base.clone()
        result[:, :, :, :3] = base_rgb * (1 - opacity) + result_rgb * opacity
        
        return result
    
    def _saturation_blend(self, base, overlay, opacity):
        # 饱和度：使用叠加层的饱和度，基础层的色相和亮度
        # 处理RGB通道
        base_rgb = base[:, :, :, :3]
        overlay_rgb = overlay[:, :, :, :3]
        
        # 转换为HSL
        base_hsl = self._rgb_to_hsl(base_rgb)
        overlay_hsl = self._rgb_to_hsl(overlay_rgb)
        
        # 使用叠加层的饱和度
        result_hsl = base_hsl.clone()
        result_hsl[:, :, :, 1] = overlay_hsl[:, :, :, 1]
        
        # 转换回RGB
        result_rgb = self._hsl_to_rgb(result_hsl)
        
        # 创建结果
        result = base.clone()
        result[:, :, :, :3] = base_rgb * (1 - opacity) + result_rgb * opacity
        
        return result
    
    def _color_blend(self, base, overlay, opacity):
        # 颜色：使用叠加层的色相和饱和度，基础层的亮度
        # 处理RGB通道
        base_rgb = base[:, :, :, :3]
        overlay_rgb = overlay[:, :, :, :3]
        
        # 转换为HSL
        base_hsl = self._rgb_to_hsl(base_rgb)
        overlay_hsl = self._rgb_to_hsl(overlay_rgb)
        
        # 使用叠加层的色相和饱和度
        result_hsl = base_hsl.clone()
        result_hsl[:, :, :, 0] = overlay_hsl[:, :, :, 0]
        result_hsl[:, :, :, 1] = overlay_hsl[:, :, :, 1]
        
        # 转换回RGB
        result_rgb = self._hsl_to_rgb(result_hsl)
        
        # 创建结果
        result = base.clone()
        result[:, :, :, :3] = base_rgb * (1 - opacity) + result_rgb * opacity
        
        return result
    
    def _luminosity_blend(self, base, overlay, opacity):
        # 明度：使用叠加层的亮度，基础层的色相和饱和度
        # 处理RGB通道
        base_rgb = base[:, :, :, :3]
        overlay_rgb = overlay[:, :, :, :3]
        
        # 转换为HSL
        base_hsl = self._rgb_to_hsl(base_rgb)
        overlay_hsl = self._rgb_to_hsl(overlay_rgb)
        
        # 使用叠加层的亮度
        result_hsl = base_hsl.clone()
        result_hsl[:, :, :, 2] = overlay_hsl[:, :, :, 2]
        
        # 转换回RGB
        result_rgb = self._hsl_to_rgb(result_hsl)
        
        # 创建结果
        result = base.clone()
        result[:, :, :, :3] = base_rgb * (1 - opacity) + result_rgb * opacity
        
        return result


NODE_CLASS_MAPPINGS = {
    "Solid": Solid,
    "LumaMatte": LumaMatte,
    "BlendModes": BlendModes,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Solid": "Solid",
    "LumaMatte": "Luma Matte",
    "BlendModes": "Blend Modes",
}
