import torch
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageColor, ImageFilter
import math
import colorsys


class ImageLumaMatte:
    """
    亮度蒙版 - 完全支持批量处理图像和遮罩，增强的背景颜色支持
    
    批量处理逻辑：
    - 当图像和遮罩数量不同时，按最大数量输出，较少的批次会循环复制
    - 例如：5张图片 + 2个遮罩 = 输出5张处理结果（遮罩按[1,2,1,2,1]循环使用）
    - 例如：2张图片 + 5个遮罩 = 输出5张处理结果（图片按[1,2,1,2,1]循环使用）
    
    支持多种颜色格式：灰度值、HEX、RGB、颜色名称、edge（边缘色）、average（平均色）
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",)
            },
            "optional": {
                "invert_mask": ("BOOLEAN", {"default": False}),
                "feather": ("INT", {"default": 0, "min": 0, "max": 50, "step": 1}),
                "background_add": ("BOOLEAN", {"default": True}),
                "background_color": ("STRING", {"default": "1.0"}),
                "out_alpha": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "image_luma_matte"
    CATEGORY = "1hewNodes/image/blend"

    def image_luma_matte(self, image, mask, invert_mask=False, feather=0, background_add=True, 
                           background_color="1.0", out_alpha=False):
        # 获取图像和遮罩的批次尺寸
        image_batch_size = image.shape[0]
        mask_batch_size = mask.shape[0]
        
        # 确定最大批次数量，输出将按照最大批次数量处理
        max_batch_size = max(image_batch_size, mask_batch_size)
        
        # 创建输出图像
        output_images = []
        
        for b in range(max_batch_size):
            # 确定使用哪个图像和遮罩（循环使用较少的批次）
            image_index = b % image_batch_size
            mask_index = b % mask_batch_size
            
            # 将图像转换为PIL格式
            if image.is_cuda:
                img_np = (image[image_index].cpu().numpy() * 255).astype(np.uint8)
            else:
                img_np = (image[image_index].numpy() * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
            
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

            # 羽化边缘处理
            if feather > 0:
                mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=feather))

            # 解析背景颜色
            bg_color = self._parse_color_advanced(background_color, img_pil, mask_pil)

            if background_add:
                # 创建背景图像
                if bg_color == 'average':
                    # 计算遮罩内的平均颜色
                    bg_color = self._get_average_color(img_pil, mask_pil)
                elif bg_color == 'edge':
                    # 获取图像边缘的平均颜色
                    bg_color = self._get_edge_color(img_pil)
                
                # 使用解析后的背景颜色创建背景
                background = Image.new('RGB', img_pil.size, bg_color)
                result = background.copy()
                result.paste(img_pil, (0, 0), mask_pil)

                if out_alpha:
                    # 转换为RGBA并保留alpha通道
                    result = result.convert('RGBA')
                    # 将遮罩应用到alpha通道
                    alpha_channel = mask_pil.copy()
                    result.putalpha(alpha_channel)
                    # 转换回numpy格式，保留所有4个通道
                    result_np = np.array(result).astype(np.float32) / 255.0
                else:
                    # 转换回numpy格式
                    result_np = np.array(result).astype(np.float32) / 255.0
                
                output_images.append(torch.from_numpy(result_np))
            else:
                # 创建透明图像
                if out_alpha:
                    transparent = Image.new('RGBA', img_pil.size, (0, 0, 0, 0))
                    transparent.paste(img_pil, (0, 0), mask_pil)
                    # 转换回numpy格式，保留所有4个通道
                    transparent_np = np.array(transparent).astype(np.float32) / 255.0
                else:
                    # 创建黑色背景的RGB图像
                    transparent = Image.new('RGB', img_pil.size, (0, 0, 0))
                    transparent.paste(img_pil, (0, 0), mask_pil)
                    transparent_np = np.array(transparent).astype(np.float32) / 255.0

                output_images.append(torch.from_numpy(transparent_np))

        # 合并批次
        output_tensor = torch.stack(output_images)
        
        return (output_tensor,)
    
    def _parse_color_advanced(self, color_str, img_pil=None, mask_pil=None):
        """
        高级颜色解析，支持多种格式：
        - 灰度值: "0.5", "1.0"
        - HEX: "#FF0000", "FF0000"
        - RGB: "255,0,0", "1.0,0.0,0.0", "(255,128,64)"
        - 颜色名称: "red", "blue", "white"
        - 特殊值: "edge", "average"
        """
        if not color_str:
            return (0, 0, 0)
        
        # 检查特殊值
        color_lower = color_str.lower().strip()
        
        # 检查是否为 average 或其缩写
        if color_lower in ['average', 'avg', 'a', 'av', 'aver']:
            return 'average'
        
        # 检查是否为 edge 或其缩写
        if color_lower in ['edge', 'e', 'ed']:
            return 'edge'
        
        # 移除括号（如果存在）
        color_str = color_str.strip()
        if color_str.startswith('(') and color_str.endswith(')'):
            color_str = color_str[1:-1].strip()
        
        # 尝试解析为灰度值 (0.0-1.0)
        try:
            gray = float(color_str)
            if 0.0 <= gray <= 1.0:
                gray_int = int(gray * 255)
                return (gray_int, gray_int, gray_int)
            elif gray > 1.0 and gray <= 255:
                # 可能是0-255范围的灰度值
                gray_int = int(gray)
                return (gray_int, gray_int, gray_int)
        except ValueError:
            pass
        
        # 尝试解析为 RGB 格式
        if ',' in color_str:
            try:
                parts = [part.strip() for part in color_str.split(',')]
                if len(parts) >= 3:
                    r, g, b = [float(parts[i]) for i in range(3)]
                    # 判断是否为 0-1 范围
                    if max(r, g, b) <= 1.0:
                        return (int(r * 255), int(g * 255), int(b * 255))
                    else:
                        return (int(r), int(g), int(b))
            except (ValueError, IndexError):
                pass
        
        # 尝试解析为十六进制颜色
        hex_color = color_str
        if hex_color.startswith('#'):
            hex_color = hex_color[1:]
        
        if len(hex_color) == 6:
            try:
                r = int(hex_color[0:2], 16)
                g = int(hex_color[2:4], 16)
                b = int(hex_color[4:6], 16)
                return (r, g, b)
            except ValueError:
                pass
        elif len(hex_color) == 3:
            try:
                r = int(hex_color[0], 16) * 17  # 扩展单个十六进制数字
                g = int(hex_color[1], 16) * 17
                b = int(hex_color[2], 16) * 17
                return (r, g, b)
            except ValueError:
                pass
        
        # 尝试解析为颜色名称
        try:
            return ImageColor.getrgb(color_str)
        except ValueError:
            pass
        
        # 默认返回白色
        return (255, 255, 255)
    
    def _get_average_color(self, img_pil, mask_pil):
        """计算遮罩内的平均颜色"""
        img_array = np.array(img_pil.convert('RGB'))
        mask_array = np.array(mask_pil) / 255.0
        
        # 创建扩展的遮罩数组以匹配图像维度
        mask_expanded = np.expand_dims(mask_array, axis=2)
        mask_expanded = np.repeat(mask_expanded, 3, axis=2)
        
        # 计算遮罩内的像素总和和像素数量
        masked_pixels = img_array * mask_expanded
        pixel_count = np.sum(mask_array)
        
        if pixel_count > 0:
            # 计算平均颜色
            avg_color = np.sum(masked_pixels, axis=(0, 1)) / pixel_count
            return tuple(int(c) for c in avg_color)
        else:
            # 如果遮罩内没有像素，返回图像整体平均颜色
            return tuple(int(c) for c in np.mean(img_array, axis=(0, 1)))
    
    def _get_edge_color(self, img_pil, side='all'):
        """获取图像边缘的平均颜色"""
        width, height = img_pil.size
        img_rgb = img_pil.convert('RGB')
        
        if side == 'left':
            edge = img_rgb.crop((0, 0, 1, height))
        elif side == 'right':
            edge = img_rgb.crop((width-1, 0, width, height))
        elif side == 'top':
            edge = img_rgb.crop((0, 0, width, 1))
        elif side == 'bottom':
            edge = img_rgb.crop((0, height-1, width, height))
        else:  # 'all' - 所有边缘
            # 获取所有边缘像素
            top = np.array(img_rgb.crop((0, 0, width, 1)))
            bottom = np.array(img_rgb.crop((0, height-1, width, height)))
            left = np.array(img_rgb.crop((0, 0, 1, height)))
            right = np.array(img_rgb.crop((width-1, 0, width, height)))
            
            # 合并所有边缘像素并计算平均值
            all_edges = np.vstack([
                top.reshape(-1, 3), 
                bottom.reshape(-1, 3), 
                left.reshape(-1, 3), 
                right.reshape(-1, 3)
            ])
            return tuple(np.mean(all_edges, axis=0).astype(int))
        
        # 计算单边的平均颜色
        edge_array = np.array(edge)
        return tuple(np.mean(edge_array.reshape(-1, 3), axis=0).astype(int))


class ImageBlendModesByAlpha:
    """
    图层叠加模式 - 支持基础图层输入，控制叠加模式和透明度强度
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "overlay_image": ("IMAGE",),
                "base_image": ("IMAGE",),
                "blend_mode": (["normal", "dissolve", "darken", "multiply", "color_burn", "linear_burn", 
                                "add", "lighten", "screen", "color_dodge", "linear_dodge",
                                "overlay", "soft_light", "hard_light", "linear_light", "vivid_light", "pin_light", "hard_mix",
                                "difference", "exclusion",  "subtract", "divide", 
                                "hue", "saturation", "color", "luminosity",
                                 ], {"default": "normal"}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})
            },
            "optional": {
                "overlay_mask": ("MASK",),
                "invert_mask": ("BOOLEAN", {"default": False})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "blend_modes"
    CATEGORY = "1hewNodes/image/blend"

    def blend_modes(self, base_image, overlay_image, blend_mode, opacity, overlay_mask=None, invert_mask=False):
        # 检查并转换 RGBA 图像为 RGB
        base_image = self._convert_rgba_to_rgb(base_image)
        overlay_image = self._convert_rgba_to_rgb(overlay_image)
        
        # 处理叠加图层
        blended = self._apply_blend(base_image, overlay_image, blend_mode, opacity)
        
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
                    img = Image.fromarray((overlay[i].cpu().numpy() * 255).astype(np.uint8))
                else:  # RGBA
                    img = Image.fromarray((overlay[i].cpu().numpy() * 255).astype(np.uint8), 'RGBA')
                
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
            result = self._normal_blend(base, overlay, opacity)
        elif blend_mode == "dissolve":
            result = self._dissolve_blend(base, overlay, opacity)
        elif blend_mode == "darken":
            result = self._darken_blend(base, overlay, opacity)
        elif blend_mode == "multiply":
            result = self._multiply_blend(base, overlay, opacity)
        elif blend_mode == "color_burn":
            result = self._color_burn_blend(base, overlay, opacity)
        elif blend_mode == "linear_burn":
            result = self._linear_burn_blend(base, overlay, opacity)
        elif blend_mode == "add":
            result = self._add_blend(base, overlay, opacity)
        elif blend_mode == "lighten":
            result = self._lighten_blend(base, overlay, opacity)
        elif blend_mode == "screen":
            result = self._screen_blend(base, overlay, opacity)
        elif blend_mode == "color_dodge":
            result = self._color_dodge_blend(base, overlay, opacity)
        elif blend_mode == "linear_dodge":
            result = self._linear_dodge_blend(base, overlay, opacity)
        elif blend_mode == "overlay":
            result = self._overlay_blend(base, overlay, opacity)
        elif blend_mode == "soft_light":
            result = self._soft_light_blend(base, overlay, opacity)
        elif blend_mode == "hard_light":
            result = self._hard_light_blend(base, overlay, opacity)
        elif blend_mode == "linear_light":
            result = self._linear_light_blend(base, overlay, opacity)
        elif blend_mode == "vivid_light":
            result = self._vivid_light_blend(base, overlay, opacity)
        elif blend_mode == "pin_light":
            result = self._pin_light_blend(base, overlay, opacity)
        elif blend_mode == "hard_mix":
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
    
    def _rgb_to_hsl(self, rgb):
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
    
    def _hsl_to_rgb(self, hsl):
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


class ImageBlendModesByCSS:
    """
    CSS 图层叠加模式 - 基于 Pilgram 库实现的 CSS 混合模式
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "overlay_image": ("IMAGE",),
                "base_image": ("IMAGE",),
                "blend_mode": (["normal", "multiply", "screen", "overlay", "darken", "lighten", 
                                "color_dodge", "color_burn", "hard_light", "soft_light", 
                                "difference", "exclusion", "hue", "saturation", "color", "luminosity"], 
                               {"default": "normal"}),
                "blend_percentage": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})
            },
            "optional": {
                "overlay_mask": ("MASK",),
                "invert_mask": ("BOOLEAN", {"default": False})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "image_blend_modes_by_css"
    CATEGORY = "1hewNodes/image/blend"

    def image_blend_modes_by_css(self, base_image, overlay_image, blend_mode, blend_percentage, overlay_mask=None, invert_mask=False):
        # 检查并安装 pilgram 库
        if not self._check_pilgram():
            raise ImportError("无法导入 pilgram 库，请确保已安装。可以使用 pip install pilgram 安装。")
        
        import pilgram.css.blending as blending
        
        # 初始化结果为基础图层
        result = base_image.clone()
        
        # 检查并转换 RGBA 图像为 RGB
        base_image = self._convert_rgba_to_rgb(base_image)
        overlay_image = self._convert_rgba_to_rgb(overlay_image)
        
        # 获取批次大小
        base_batch_size = base_image.shape[0]
        overlay_batch_size = overlay_image.shape[0]
        
        # 创建输出图像列表
        output_images = []
        
        # 处理每个批次的图像
        for b in range(base_batch_size):
            # 获取当前批次的基础图像
            current_base = base_image[b]
            
            # 确定使用哪个叠加图像（如果叠加图像数量少于基础图像数量，则循环使用）
            overlay_index = b % overlay_batch_size
            current_overlay = overlay_image[overlay_index]
            
            # 将张量转换为PIL图像
            base_pil = self._tensor_to_pil(current_base)
            overlay_pil = self._tensor_to_pil(current_overlay)
            
            # 确保两个图像具有相同的尺寸
            if base_pil.size != overlay_pil.size:
                overlay_pil = overlay_pil.resize(base_pil.size, Image.Resampling.LANCZOS)
            
            # 应用混合模式
            blended_pil = self._apply_css_blend(base_pil, overlay_pil, blend_mode, blending)
            
            # 应用混合百分比
            if blend_percentage < 1.0:
                # 创建不透明度蒙版
                opacity_mask = Image.new("L", base_pil.size, int(blend_percentage * 255))
                # 反转蒙版
                opacity_mask = ImageOps.invert(opacity_mask)
                # 合成图像
                blended_pil = Image.composite(base_pil, blended_pil, opacity_mask)
            
            # 将混合后的PIL图像转换为张量
            current_blended = self._pil_to_tensor(blended_pil).to(current_base.device)
            
            # 如果提供了遮罩，则应用遮罩
            if overlay_mask is not None:
                # 确定使用哪个遮罩（如果遮罩数量少于图像数量，则循环使用）
                mask_batch_size = overlay_mask.shape[0]
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
            else:
                output_images.append(current_blended)
        
        # 合并批次
        result = torch.stack(output_images)
        
        return (result,)
    
    def _check_pilgram(self):
        """检查是否已安装 pilgram 库"""
        try:
            import pilgram
            return True
        except ImportError:
            try:
                import pip
                pip.main(['install', 'pilgram'])
                import pilgram
                return True
            except:
                return False
    
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
    
    def _tensor_to_pil(self, tensor):
        """将张量转换为PIL图像"""
        # 确保张量在CPU上
        if tensor.is_cuda:
            tensor = tensor.cpu()
        
        # 转换为numpy数组
        np_array = (tensor.numpy() * 255).astype(np.uint8)
        
        # 创建PIL图像
        if np_array.shape[2] == 3:
            return Image.fromarray(np_array, 'RGB')
        elif np_array.shape[2] == 4:
            return Image.fromarray(np_array, 'RGBA')
        else:
            raise ValueError(f"不支持的通道数: {np_array.shape[2]}")
    
    def _pil_to_tensor(self, pil_image):
        """将PIL图像转换为张量"""
        # 确保图像是RGB模式
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # 转换为numpy数组
        np_array = np.array(pil_image).astype(np.float32) / 255.0
        
        # 转换为张量
        return torch.from_numpy(np_array)
    
    def _apply_css_blend(self, base_pil, overlay_pil, blend_mode, blending):
        """应用CSS混合模式"""
        # 将CSS混合模式名称转换为pilgram函数名
        mode_mapping = {
            "normal": "normal",
            "multiply": "multiply",
            "screen": "screen",
            "overlay": "overlay",
            "darken": "darken",
            "lighten": "lighten",
            "color_dodge": "color_dodge",
            "color_burn": "color_burn",
            "hard_light": "hard_light",
            "soft_light": "soft_light",
            "difference": "difference",
            "exclusion": "exclusion",
            "hue": "hue",
            "saturation": "saturation",
            "color": "color",
            "luminosity": "luminosity"
        }
        
        # 获取对应的混合函数
        blend_func_name = mode_mapping.get(blend_mode, "normal")
        blend_func = getattr(blending, blend_func_name)
        
        # 应用混合
        try:
            result = blend_func(base_pil, overlay_pil)
            return result
        except Exception as e:
            print(f"混合模式 {blend_mode} 应用失败: {str(e)}")
            # 如果混合失败，返回原始图像
            return base_pil


NODE_CLASS_MAPPINGS = {
    "ImageLumaMatte": ImageLumaMatte,
    "ImageBlendModesByAlpha": ImageBlendModesByAlpha,
    "ImageBlendModesByCSS": ImageBlendModesByCSS,
}

# 节点显示名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageLumaMatte": "Image Luma Matte",
    "ImageBlendModesByAlpha": "Image Blend Modes by Alpha",
    "ImageBlendModesByCSS": "Image Blend Modes by CSS",
}