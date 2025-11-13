import torch
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont, ImageFilter, ImageEnhance, ImageColor
import torch.nn.functional as F
import os
import math
import cv2
from skimage.measure import label, regionprops
import comfy.utils
import re


class ImageGetSize:
    """
    获取图像的宽度和高度信息
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "get_image_size"
    CATEGORY = "1hewNodes/image"
    
    def get_image_size(self, image):
        """
        获取图像的宽度和高度
        
        Args:
            image: 输入的图像张量，格式为 (batch, height, width, channels)
            
        Returns:
            tuple: (width, height) 图像的宽度和高度
        """
        # 确保输入是正确的维度
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        # 获取图像尺寸 (batch, height, width, channels)
        batch_size, height, width, channels = image.shape
        
        # 返回宽度和高度（整数类型）
        return (int(width), int(height))


class ImageSolidFluxKontext:
    """
    根据 FluxKontext 尺寸预设生成纯色图像
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset_size": (["672×1568 [1:2.33] (3:7)", "688×1504 [1:2.19]", "720×1456 [1:2.00] (1:2)",
                                "752×1392 [1:1.85]", "800×1328 [1:1.66]", "832×1248 [1:1.50] (2:3)",
                                "880×1184 [1:1.35]", "944×1104 [1:1.17]", "1024×1024 [1:1.00] (1:1)",
                                "1104×944 [1.17:1]", "1184×880 [1.35:1]", "1248×832 [1.50:1] (3:2)",
                                "1328×800 [1.66:1]", "1392×752 [1.85:1]", "1456×720 [2.00:1] (2:1)",
                                "1504×688 [2.19:1]", "1568×672 [2.33:1] (7:3)"],
                              {"default": "1024×1024 [1:1.00] (1:1)"}),
                "color": ("STRING", {"default": "1.0"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "image_solid_flux_kontext"
    CATEGORY = "1hewNodes/image"

    def _parse_color(self, color_str):
        """解析不同格式的颜色输入"""
        color_str = color_str.strip()
        
        # 尝试解析为灰度值 (0.0-1.0)
        try:
            gray_value = float(color_str)
            if 0.0 <= gray_value <= 1.0:
                # 灰度值转换为RGB
                return (gray_value, gray_value, gray_value)
        except ValueError:
            pass
        
        # 尝试解析为十六进制颜色 (#RRGGBB 或 RRGGBB)
        if color_str.startswith('#'):
            hex_color = color_str[1:]
        else:
            hex_color = color_str
            
        if len(hex_color) == 6:
            try:
                r = int(hex_color[0:2], 16) / 255.0
                g = int(hex_color[2:4], 16) / 255.0
                b = int(hex_color[4:6], 16) / 255.0
                return (r, g, b)
            except ValueError:
                pass
        
        # 尝试解析为RGB格式 (R,G,B)
        try:
            rgb = color_str.split(',')
            if len(rgb) == 3:
                r = float(rgb[0].strip())
                g = float(rgb[1].strip())
                b = float(rgb[2].strip())
                # 如果值大于1，假设是0-255范围，转换为0-1范围
                if r > 1.0 or g > 1.0 or b > 1.0:
                    r /= 255.0
                    g /= 255.0
                    b /= 255.0
                return (r, g, b)
        except ValueError:
            pass
        
        # 默认返回白色
        return (1.0, 1.0, 1.0)

    def image_solid_flux_kontext(self, preset_size, color):
        # 从预设尺寸中提取宽度和高度（去掉比例部分）
        dimensions = preset_size.split(" ")[0].split("×")
        img_width = int(dimensions[0])
        img_height = int(dimensions[1])

        # 解析颜色值
        r, g, b = self._parse_color(color)

        # 创建纯色图像
        image = np.zeros((img_height, img_width, 3), dtype=np.float32)
        image[:, :, 0] = r
        image[:, :, 1] = g
        image[:, :, 2] = b

        # 创建透明度蒙版
        mask = np.ones((img_height, img_width), dtype=np.float32)

        # 转换为ComfyUI需要的格式 (批次, 高度, 宽度, 通道)
        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return (image, mask)


class ImageSolidQwenImage:
    """
    根据 QwenImage 尺寸预设生成纯色图像
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset_size": (["928×1664 [1:1.79]", "1056×1584 [1:1.50] (2:3)", "1140×1472 [1:1.29]", 
                                "1328×1328 [1:1.00] (1:1)", "1472×1140 [1.29:1]", "1584×1056 [1.50:1] (3:2)", "1664×928 [1.79:1]"],
                              {"default": "1328×1328 [1:1.00] (1:1)"}),
                "color": ("STRING", {"default": "1.0"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "image_solid_qwen_image"
    CATEGORY = "1hewNodes/image"

    def _parse_color(self, color_str):
        """解析不同格式的颜色输入"""
        color_str = color_str.strip()
        
        # 尝试解析为灰度值 (0.0-1.0)
        try:
            gray_value = float(color_str)
            if 0.0 <= gray_value <= 1.0:
                # 灰度值转换为RGB
                return (gray_value, gray_value, gray_value)
        except ValueError:
            pass
        
        # 尝试解析为十六进制颜色 (#RRGGBB 或 RRGGBB)
        if color_str.startswith('#'):
            hex_color = color_str[1:]
        else:
            hex_color = color_str
            
        if len(hex_color) == 6:
            try:
                r = int(hex_color[0:2], 16) / 255.0
                g = int(hex_color[2:4], 16) / 255.0
                b = int(hex_color[4:6], 16) / 255.0
                return (r, g, b)
            except ValueError:
                pass
        
        # 尝试解析为RGB格式 (R,G,B)
        try:
            rgb = color_str.split(',')
            if len(rgb) == 3:
                r = float(rgb[0].strip())
                g = float(rgb[1].strip())
                b = float(rgb[2].strip())
                # 如果值大于1，假设是0-255范围，转换为0-1范围
                if r > 1.0 or g > 1.0 or b > 1.0:
                    r /= 255.0
                    g /= 255.0
                    b /= 255.0
                return (r, g, b)
        except ValueError:
            pass
        
        # 默认返回白色
        return (1.0, 1.0, 1.0)

    def image_solid_qwen_image(self, preset_size, color):
        # 从预设尺寸中提取宽度和高度（去掉比例部分）
        dimensions = preset_size.split("[")[0].split("×")
        img_width = int(dimensions[0])
        img_height = int(dimensions[1])

        # 解析颜色值
        r, g, b = self._parse_color(color)

        # 创建纯色图像
        image = np.zeros((img_height, img_width, 3), dtype=np.float32)
        image[:, :, 0] = r
        image[:, :, 1] = g
        image[:, :, 2] = b

        # 创建透明度蒙版
        mask = np.ones((img_height, img_width), dtype=np.float32)

        # 转换为ComfyUI需要的格式 (批次, 高度, 宽度, 通道)
        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return (image, mask)


class ImageSolid:
    """
    根据输入的颜色和尺寸生成纯色图像
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset_size": (["custom", "512×512 [1:1.00] (1:1)", "768×768 [1:1.00] (1:1)", "1024×1024 [1:1.00] (1:1)", "1328×1328 [1:1.00] (1:1)", "1408×1408 [1:1.00] (1:1)",
                                "768×1024 [1:1.33] (3:4)", "1140×1472 [1:1.29]", "1216×1664 [1:1.37]",
                                "512×768 [1:1.50] (2:3)", "832×1248 [1:1.50] (2:3)", "1056×1584 [1:1.50] (2:3)", "1152×1728 [1:1.50] (2:3)",
                                "480×832 [1:1.73]", "720×1280 [1:1.78] (9:16)", "928×1664 [1:1.79]", "1080×1920 [1:1.78] (9:16)", "1088×1920 [1:1.76]",
                                "672×1568 [1:2.33] (3:7)", "960×2176 [1:2.27]",
                                "1024×768 [1.33:1] (4:3)", "1472×1140 [1.29:1]", "1664×1216 [1.37:1]",
                                "768×512 [1.50:1] (3:2)", "1248×832 [1.50:1] (3:2)", "1584×1056 [1.50:1] (3:2)", "1728×1152 [1.50:1] (3:2)",
                                "832×480 [1.73:1]", "1280×720 [1.78:1] (16:9)", "1664×928 [1.79:1]", "1920×1080 [1.78:1] (16:9)", "1920×1088 [1.76:1]",
                                "1568×672 [2.33:1] (7:3)", "2176×960 [2.27:1]"],
                              {"default": "custom"}),
                "width": ("INT", {"default": 1024, "min": 1, "max": 8192, "step": 1}),
                "height": ("INT", {"default": 1024, "min": 1, "max": 8192, "step": 1}),
            },
            "optional": {
                "reference_image": ("IMAGE", ),
                "color": ("STRING", {"default": "1.0"}),
                "alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "invert": ("BOOLEAN", {"default": False}),
                "mask_opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "divisible_by": ("INT", {"default": 8, "min": 1, "max": 1024, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "image_solid"
    CATEGORY = "1hewNodes/image"

    def parse_color(self, color_str):
        """解析不同格式的颜色输入，支持多种颜色格式"""
        if not color_str:
            return (0, 0, 0)
        
        # 移除括号（如果存在）
        color_str = color_str.strip()
        if color_str.startswith('(') and color_str.endswith(')'):
            color_str = color_str[1:-1].strip()
        
        # 支持单字母颜色缩写
        color_shortcuts = {
            'r': 'red', 'g': 'green', 'b': 'blue', 'c': 'cyan', 
            'm': 'magenta', 'y': 'yellow', 'k': 'black', 'w': 'white'
        }
        
        # 检查是否为单字母缩写
        if len(color_str) == 1 and color_str.lower() in color_shortcuts:
            color_str = color_shortcuts[color_str.lower()]
        
        # 尝试解析为灰度值 (0.0-1.0)
        try:
            gray = float(color_str)
            if 0.0 <= gray <= 1.0:
                return (int(gray * 255), int(gray * 255), int(gray * 255))
        except ValueError:
            pass
        
        # 尝试解析为 RGB 格式 (如 "0.5,0.7,0.9" 或 "128,192,255")
        if ',' in color_str:
            try:
                # 分割并清理每个部分
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
        
        # 尝试解析为十六进制或颜色名称
        try:
            return ImageColor.getrgb(color_str)
        except ValueError:
            # 默认返回白色
            return (255, 255, 255)

    def image_solid(self, preset_size, width, height, divisible_by, color="1.0", alpha=1.0, invert=False, mask_opacity=1.0, reference_image=None):
        images = []
        masks = []

        if reference_image is not None:
            # 处理批量参考图像
            for ref_img in reference_image:
                # 从参考图像获取尺寸
                h, w, _ = ref_img.shape
                img_width = w
                img_height = h
        else:
            # 处理预设尺寸或自定义尺寸
            if preset_size != "custom":
                # 从预设尺寸中提取宽度和高度（去掉比例部分）
                dimensions = preset_size.split(" ")[0].split("×")
                img_width = int(dimensions[0])
                img_height = int(dimensions[1])
            else:
                img_width = width
                img_height = height

            # 确保尺寸能被 divisible_by 整除
            if divisible_by > 1:
                img_width = math.ceil(img_width / divisible_by) * divisible_by
                img_height = math.ceil(img_height / divisible_by) * divisible_by

            # 为了兼容批量处理，这里将单个尺寸的情况也当作一个批次处理
            num_images = 1
            reference_image = [None] * num_images

        # 使用新的 parse_color 方法解析颜色
        rgb_color = self.parse_color(color)
        r = rgb_color[0] / 255.0
        g = rgb_color[1] / 255.0
        b = rgb_color[2] / 255.0

        # 如果需要反转颜色
        if invert:
            r = 1.0 - r
            g = 1.0 - g
            b = 1.0 - b

        for ref_img in reference_image:
            if ref_img is not None:
                # 从参考图像获取尺寸
                h, w, _ = ref_img.shape
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


class ImageResizeFluxKontext:
    """
    Flux预设图像缩放器 - 支持预设分辨率和自动最优选择
    """
    
    # 将PRESET_RESOLUTIONS移到类内部
    PRESET_RESOLUTIONS = [
        ("672×1568 [1:2.33] (3:7)", 672, 1568),
        ("688×1504 [1:2.19]", 688, 1504),
        ("720×1456 [1:2.00] (1:2)", 720, 1456),
        ("752×1392 [1:1.85]", 752, 1392),
        ("800×1328 [1:1.66]", 800, 1328),
        ("832×1248 [1:1.50] (2:3)", 832, 1248),
        ("880×1184 [1:1.35]", 880, 1184),
        ("944×1104 [1:1.17]", 944, 1104),
        ("1024×1024 [1:1.00] (1:1)", 1024, 1024),
        ("1104×944 [1.17:1]", 1104, 944),
        ("1184×880 [1.35:1]", 1184, 880),
        ("1248×832 [1.50:1] (3:2)", 1248, 832),
        ("1328×800 [1.66:1]", 1328, 800),
        ("1392×752 [1.85:1]", 1392, 752),
        ("1456×720 [2.00:1] (2:1)", 1456, 720),
        ("1504×688 [2.19:1]", 1504, 688),
        ("1568×672 [2.33:1] (7:3)", 1568, 672),
    ]
    
    @classmethod
    def INPUT_TYPES(cls):
        preset_options = ["auto"] + [preset[0] for preset in cls.PRESET_RESOLUTIONS]
        return {
            "required": {
                "preset_size": (preset_options, {"default": "auto"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "image_resize_flux_kontext"
    CATEGORY = "1hewNodes/image"

    def image_resize_flux_kontext(self, preset_size, image=None, mask=None):
        # 确定目标尺寸
        if preset_size == "auto":
            if image is not None:
                # 根据输入图像的宽高比选择最合适的尺寸（类似FluxKontextImageScale的逻辑）
                input_width = image.shape[2]
                input_height = image.shape[1]
                aspect_ratio = input_width / input_height
                
                # 找到最接近的宽高比
                _, width, height = min((abs(aspect_ratio - w / h), w, h) for _, w, h in self.PRESET_RESOLUTIONS)
            elif mask is not None:
                # 根据输入mask的宽高比选择最合适的尺寸
                input_width = mask.shape[2]
                input_height = mask.shape[1]
                aspect_ratio = input_width / input_height
                
                # 找到最接近的宽高比
                _, width, height = min((abs(aspect_ratio - w / h), w, h) for _, w, h in self.PRESET_RESOLUTIONS)
            else:
                # 如果没有输入图像和mask，使用默认尺寸
                width, height = 1024, 1024
        else:
            # 查找对应的预设尺寸
            for preset in self.PRESET_RESOLUTIONS:
                if preset[0] == preset_size:
                    width, height = preset[1], preset[2]
                    break
            else:
                # 如果没找到，使用默认尺寸
                width, height = 1024, 1024
        
        # 处理图像
        if image is not None:
            # 缩放图像到目标尺寸
            scaled_image = comfy.utils.common_upscale(image.movedim(-1, 1), width, height, "lanczos", "center").movedim(1, -1)
        else:
            # 创建纯色图像（黑色）
            scaled_image = torch.zeros((1, height, width, 3), dtype=torch.float32)
        
        # 处理遮罩
        if mask is not None:
            # 确保mask是正确的格式
            if len(mask.shape) == 3:
                # mask形状为 (batch, height, width)
                batch_size, mask_height, mask_width = mask.shape
            elif len(mask.shape) == 4:
                # mask形状为 (batch, channel, height, width)
                batch_size, channels, mask_height, mask_width = mask.shape
                if channels == 1:
                    mask = mask.squeeze(1)  # 移除通道维度
                else:
                    mask = mask[:, 0, :, :]  # 取第一个通道
            
            # 使用torch.nn.functional.interpolate来缩放mask
            
            # 添加batch和channel维度进行插值
            mask_for_resize = mask.unsqueeze(1)  # 添加channel维度
            scaled_mask = F.interpolate(
                mask_for_resize, 
                size=(height, width), 
                mode='bilinear', 
                align_corners=False
            )
            scaled_mask = scaled_mask.squeeze(1)  # 移除channel维度
        else:
            # 创建全白遮罩
            scaled_mask = torch.ones((1, height, width), dtype=torch.float32)
        
        return (scaled_image, scaled_mask)


class ImageResizeQwenImage:
    """
    Qwen图像预设缩放器 - 支持Qwen视觉模型优化的预设分辨率和自动最优选择
    """
    
    # Qwen图像预设分辨率
    PRESET_RESOLUTIONS = [
        ("928×1664 [1:1.79]", 928, 1664),
        ("1056×1584 [1:1.50] (2:3)", 1056, 1584),
        ("1140×1472 [1:1.29]", 1140, 1472),
        ("1328×1328 [1:1.00] (1:1)", 1328, 1328),
        ("1472×1140 [1.29:1]", 1472, 1140),
        ("1584×1056 [1.50:1] (3:2)", 1584, 1056),
        ("1664×928 [1.79:1]", 1664, 928),
    ]
    
    @classmethod
    def INPUT_TYPES(cls):
        preset_options = ["auto"] + [preset[0] for preset in cls.PRESET_RESOLUTIONS]
        return {
            "required": {
                "preset_size": (preset_options, {"default": "auto"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "image_resize_qwen_image"
    CATEGORY = "1hewNodes/image"

    def image_resize_qwen_image(self, preset_size, image=None, mask=None):
        # 确定目标尺寸
        if preset_size == "auto":
            if image is not None:
                # 根据输入图像的宽高比选择最合适的尺寸
                input_width = image.shape[2]
                input_height = image.shape[1]
                aspect_ratio = input_width / input_height
                
                # 找到最接近的宽高比
                _, width, height = min((abs(aspect_ratio - w / h), w, h) for _, w, h in self.PRESET_RESOLUTIONS)
            elif mask is not None:
                # 根据输入mask的宽高比选择最合适的尺寸
                input_width = mask.shape[2]
                input_height = mask.shape[1]
                aspect_ratio = input_width / input_height
                
                # 找到最接近的宽高比
                _, width, height = min((abs(aspect_ratio - w / h), w, h) for _, w, h in self.PRESET_RESOLUTIONS)
            else:
                # 如果没有输入图像和mask，使用默认尺寸
                width, height = 1328, 1328
        else:
            # 查找对应的预设尺寸
            for preset in self.PRESET_RESOLUTIONS:
                if preset[0] == preset_size:
                    width, height = preset[1], preset[2]
                    break
            else:
                # 如果没找到，使用默认尺寸
                width, height = 1328, 1328
        
        # 处理图像
        if image is not None:
            # 缩放图像到目标尺寸
            scaled_image = comfy.utils.common_upscale(image.movedim(-1, 1), width, height, "lanczos", "center").movedim(1, -1)
        else:
            # 创建纯色图像（黑色）
            scaled_image = torch.zeros((1, height, width, 3), dtype=torch.float32)
        
        # 处理遮罩
        if mask is not None:
            # 确保mask是正确的格式
            if len(mask.shape) == 3:
                # mask形状为 (batch, height, width)
                batch_size, mask_height, mask_width = mask.shape
            elif len(mask.shape) == 4:
                # mask形状为 (batch, channel, height, width)
                batch_size, channels, mask_height, mask_width = mask.shape
                if channels == 1:
                    mask = mask.squeeze(1)  # 移除通道维度
                else:
                    mask = mask[:, 0, :, :]  # 取第一个通道
            
            # 使用torch.nn.functional.interpolate来缩放mask
            
            # 添加batch和channel维度进行插值
            mask_for_resize = mask.unsqueeze(1)  # 添加channel维度
            scaled_mask = F.interpolate(
                mask_for_resize, 
                size=(height, width), 
                mode='bilinear', 
                align_corners=False
            )
            scaled_mask = scaled_mask.squeeze(1)  # 移除channel维度
        else:
            # 创建全白遮罩
            scaled_mask = torch.ones((1, height, width), dtype=torch.float32)
        
        return (scaled_image, scaled_mask)


class ImageResizeUniversal:
    """
    图像通用缩放器 - 支持多种纵横比和缩放模式，完整的 mask 处理逻辑
    
    主要功能：
    1. mask 输出端在没有输入时也能正常输出
    2. pad 模式下原图区域为白色，填充区域为黑色
    3. 其他模式下 mask 尺寸与图片一致
    """
    
    NODE_NAME = "ImageResizeUniversal"
    
    @classmethod
    def INPUT_TYPES(cls):
        ratio_list = ['origin', 'custom', '1:1', '3:2', '4:3', '16:9', '21:9', '2:3', '3:4', '9:16', '9:21']
        fit_mode = ['crop', 'pad', 'stretch']
        method_mode = ['nearest', 'bilinear', 'lanczos', 'bicubic', 'hamming', 'box']
        scale_to_list = ['None', 'longest', 'shortest', 'width', 'height', 'length_to_sq_area']
        return {
            "required": {
                "preset_ratio": (ratio_list,),
                "proportional_width": ("INT", {"default": 1, "min": 1, "max": 1e8, "step": 1}),
                "proportional_height": ("INT", {"default": 1, "min": 1, "max": 1e8, "step": 1}),
                "method": (method_mode, {"default": 'lanczos'}),
                "scale_to_side": (scale_to_list, {"default": 'None'}),
                "scale_to_length": ("INT", {"default": 1024, "min": 4, "max": 1e8, "step": 1}),
                "fit": (fit_mode, {"default": "crop"}),
                "pad_color": ("STRING", {"default": "1.0"}),
                "divisible_by": ("INT", {"default": 8, "min": 1, "max": 1024, "step": 1}),
            },
            "optional": {
                "image": ("IMAGE",),  
                "mask": ("MASK",),  
                "get_image_size": ("IMAGE",), 
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = 'image_resize'
    CATEGORY = '1hewNodes/image'

    def image_resize(self, preset_ratio, proportional_width, proportional_height,
                     fit, method, divisible_by, scale_to_side, scale_to_length,
                     pad_color,
                     image=None, mask=None, get_image_size=None
                     ):
        # 初始化变量
        orig_images = []
        orig_masks = []
        orig_width = 0
        orig_height = 0
        target_width = 0
        target_height = 0
        ratio = 1.0
        ret_images = []
        ret_masks = []
        
        # 处理输入图像
        if image is not None:
            for i in image:
                i = torch.unsqueeze(i, 0)
                orig_images.append(i)
            orig_width, orig_height = self.tensor2pil(orig_images[0]).size
            
        # 处理输入遮罩
        if mask is not None:
            if mask.dim() == 2:
                mask = torch.unsqueeze(mask, 0)
            for m in mask:
                m = torch.unsqueeze(m, 0)
                if not self.is_valid_mask(m) and m.shape == torch.Size([1, 64, 64]):
                    self.log(f"警告: {self.NODE_NAME} 输入遮罩为空，已忽略。", message_type='warning')
                else:
                    orig_masks.append(m)

            if len(orig_masks) > 0:
                _width, _height = self.tensor2pil(orig_masks[0]).size
                if (orig_width > 0 and orig_width != _width) or (orig_height > 0 and orig_height != _height):
                    self.log(f"错误: {self.NODE_NAME} 执行失败，因为遮罩与图像尺寸不匹配。", message_type='error')
                    return (None, None)
                elif orig_width + orig_height == 0:
                    orig_width = _width
                    orig_height = _height

        # 检查是否有有效输入
        if orig_width + orig_height == 0:
            self.log(f"错误: {self.NODE_NAME} 执行失败，至少需要输入图像或遮罩。", message_type='error')
            return (None, None)

        # 确定目标尺寸
        if get_image_size is not None:
            # 从输入图像获取尺寸
            size_img = self.tensor2pil(get_image_size[0])
            target_width, target_height = size_img.size
            self.log(f"使用输入图像尺寸: {target_width}x{target_height}", message_type='info')
        else:
            # 根据纵横比计算目标尺寸
            if preset_ratio == 'origin':
                ratio = orig_width / orig_height
            elif preset_ratio == 'custom':
                ratio = proportional_width / proportional_height
            else:
                s = preset_ratio.split(":")
                ratio = int(s[0]) / int(s[1])

            # 根据不同缩放模式计算目标宽度和高度
            if ratio > 1:  # 宽大于高
                if scale_to_side == 'longest':
                    target_width = scale_to_length
                    target_height = int(target_width / ratio)
                elif scale_to_side == 'shortest':
                    target_height = scale_to_length
                    target_width = int(target_height * ratio)
                elif scale_to_side == 'width':
                    target_width = scale_to_length
                    target_height = int(target_width / ratio)
                elif scale_to_side == 'height':
                    target_height = scale_to_length
                    target_width = int(target_height * ratio)
                elif scale_to_side == 'length_to_sq_area':
                    target_width = math.sqrt(ratio) * scale_to_length
                    target_height = target_width / ratio
                    target_width = int(target_width)
                    target_height = int(target_height)
                else:  # 'None'
                    target_width = orig_width
                    target_height = int(target_width / ratio)
            else:  # 高大于或等于宽
                if scale_to_side == 'longest':
                    target_height = scale_to_length
                    target_width = int(target_height * ratio)
                elif scale_to_side == 'shortest':
                    target_width = scale_to_length
                    target_height = int(target_width / ratio)
                elif scale_to_side == 'width':
                    target_width = scale_to_length
                    target_height = int(target_width / ratio)
                elif scale_to_side == 'height':
                    target_height = scale_to_length
                    target_width = int(target_height * ratio)
                elif scale_to_side == 'length_to_sq_area':
                    target_width = math.sqrt(ratio) * scale_to_length
                    target_height = target_width / ratio
                    target_width = int(target_width)
                    target_height = int(target_height)
                else:  # 'None'
                    target_height = orig_height
                    target_width = int(target_height * ratio)

        # 确保尺寸能被 divisible_by 整除
        if divisible_by > 1:
            target_width = self.num_round_up_to_multiple(target_width, divisible_by)
            target_height = self.num_round_up_to_multiple(target_height, divisible_by)

        # 设置缩放采样方法
        resize_sampler = Image.LANCZOS
        if method == "bicubic":
            resize_sampler = Image.BICUBIC
        elif method == "hamming":
            resize_sampler = Image.HAMMING
        elif method == "bilinear":
            resize_sampler = Image.BILINEAR
        elif method == "box":
            resize_sampler = Image.BOX
        elif method == "nearest":
            resize_sampler = Image.NEAREST

        # 处理图像缩放
        if len(orig_images) > 0:
            for i in orig_images:
                _image = self.tensor2pil(i).convert('RGB')
                _image = self.fit_resize_image(_image, target_width, target_height, fit, resize_sampler, pad_color)
                ret_images.append(self.pil2tensor(_image))
                
        # 处理遮罩缩放逻辑
        if len(orig_masks) > 0:
            # 有输入 mask 的情况
            for m in orig_masks:
                _mask = self.tensor2pil(m).convert('L')
                _mask = self.fit_resize_mask(_mask, target_width, target_height, fit, resize_sampler, orig_width, orig_height)
                ret_masks.append(self.image2mask(_mask))
        else:
            # 没有输入 mask 时，根据图像数量生成对应的 mask
            if len(orig_images) > 0:
                for _ in orig_images:
                    _mask = self.generate_default_mask(target_width, target_height, fit, orig_width, orig_height)
                    ret_masks.append(self.image2mask(_mask))
            else:
                # 只有尺寸信息，生成一个默认 mask
                _mask = self.generate_default_mask(target_width, target_height, fit, orig_width, orig_height)
                ret_masks.append(self.image2mask(_mask))
                
        # 返回结果 - 确保总是返回 mask
        if len(ret_images) > 0 and len(ret_masks) > 0:
            self.log(f"{self.NODE_NAME} 已处理 {len(ret_images)} 张图像和 {len(ret_masks)} 张遮罩。", message_type='finish')
            return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0))
        elif len(ret_images) > 0 and len(ret_masks) == 0:
            # 这种情况通常不会发生，但保留兼容性
            self.log(f"{self.NODE_NAME} 已处理 {len(ret_images)} 张图像。", message_type='finish')
            return (torch.cat(ret_images, dim=0), None)
        elif len(ret_images) == 0 and len(ret_masks) > 0:
            self.log(f"{self.NODE_NAME} 已处理 {len(ret_masks)} 张遮罩。", message_type='finish')
            return (None, torch.cat(ret_masks, dim=0))
        else:
            self.log(f"错误: {self.NODE_NAME} 跳过，因为没有找到可用的图像或遮罩。", message_type='error')
            return (None, None)

    def generate_default_mask(self, target_width, target_height, fit_mode, orig_width, orig_height):
        """
        生成默认 mask
        - pad 模式：原图区域为白色，填充区域为黑色
        - 其他模式：全白 mask
        """
        if fit_mode == 'pad':
            # 计算原图在目标尺寸中的位置和大小
            orig_ratio = orig_width / orig_height
            target_ratio = target_width / target_height
            
            # 创建黑色背景
            mask = Image.new('L', (target_width, target_height), 0)
            
            if orig_ratio > target_ratio:
                # 原图更宽，按宽度缩放
                new_width = target_width
                new_height = int(new_width / orig_ratio)
                pad_top = (target_height - new_height) // 2
                
                # 在原图区域填充白色
                white_region = Image.new('L', (new_width, new_height), 255)
                mask.paste(white_region, (0, pad_top))
            else:
                # 原图更高，按高度缩放
                new_height = target_height
                new_width = int(new_height * orig_ratio)
                pad_left = (target_width - new_width) // 2
                
                # 在原图区域填充白色
                white_region = Image.new('L', (new_width, new_height), 255)
                mask.paste(white_region, (pad_left, 0))
                
            return mask
        else:
            # stretch 和 crop 模式：返回全白 mask
            return Image.new('L', (target_width, target_height), 255)

    def fit_resize_mask(self, mask, target_width, target_height, fit_mode, resize_sampler, orig_width, orig_height):
        """
        根据不同适应模式调整 mask 大小
        """
        if fit_mode == 'pad':
            # pad 模式下的特殊处理
            orig_ratio = orig_width / orig_height
            target_ratio = target_width / target_height
            
            # 创建黑色背景
            result_mask = Image.new('L', (target_width, target_height), 0)
            
            if orig_ratio > target_ratio:
                # 原图更宽，按宽度缩放
                new_width = target_width
                new_height = int(new_width / orig_ratio)
                resized_mask = mask.resize((new_width, new_height), resize_sampler)
                pad_top = (target_height - new_height) // 2
                result_mask.paste(resized_mask, (0, pad_top))
            else:
                # 原图更高，按高度缩放
                new_height = target_height
                new_width = int(new_height * orig_ratio)
                resized_mask = mask.resize((new_width, new_height), resize_sampler)
                pad_left = (target_width - new_width) // 2
                result_mask.paste(resized_mask, (pad_left, 0))
                
            return result_mask
        else:
            # stretch 和 crop 模式：直接使用原有逻辑
            return self.fit_resize_image(mask, target_width, target_height, fit_mode, resize_sampler).convert('L')

    def parse_color(self, color_str):
        """解析不同格式的颜色输入"""
        if not color_str:
            return (0, 0, 0)
        
        # 检查是否为 edge 或 e (不区分大小写)
        if color_str.lower() in ['edge', 'e']:
            return 'edge'
        
        # 移除括号（如果存在）
        color_str = color_str.strip()
        if color_str.startswith('(') and color_str.endswith(')'):
            color_str = color_str[1:-1].strip()
        
        # 单字母颜色缩写映射
        single_letter_colors = {
            'r': 'red',
            'g': 'green', 
            'b': 'blue',
            'c': 'cyan',
            'm': 'magenta',
            'y': 'yellow',
            'k': 'black',
            'w': 'white',
            'o': 'orange',
            'p': 'purple',
            'n': 'brown',
            's': 'silver',
            'l': 'lime',
            'i': 'indigo',
            'v': 'violet',
            't': 'turquoise',
            'a': 'aqua',
            'f': 'fuchsia',
            'h': 'hotpink',
            'd': 'darkblue'
        }
        
        # 检查是否为单字母颜色缩写
        if len(color_str) == 1 and color_str.lower() in single_letter_colors:
            color_str = single_letter_colors[color_str.lower()]
        
        # 尝试解析为灰度值
        try:
            gray = float(color_str)
            return (int(gray * 255), int(gray * 255), int(gray * 255))
        except ValueError:
            pass
        
        # 尝试解析为 RGB 格式 (如 "0.5,0.7,0.9" 或 "128,192,255")
        if ',' in color_str:
            try:
                # 分割并清理每个部分
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
        
        # 尝试解析为十六进制或颜色名称
        try:
            return ImageColor.getrgb(color_str)
        except ValueError:
            # 默认返回黑色
            return (0, 0, 0)

    def get_edge_color(self, img, side):
        """获取图像边缘的平均颜色"""
        width, height = img.size
        img = img.convert('RGB')
        
        if side == 'left':
            edge = img.crop((0, 0, 1, height))
        elif side == 'right':
            edge = img.crop((width-1, 0, width, height))
        elif side == 'top':
            edge = img.crop((0, 0, width, 1))
        elif side == 'bottom':
            edge = img.crop((0, height-1, width, height))
        else:
            # 所有边缘
            top = np.array(img.crop((0, 0, width, 1)))
            bottom = np.array(img.crop((0, height-1, width, height)))
            left = np.array(img.crop((0, 0, 1, height)))
            right = np.array(img.crop((width-1, 0, width, height)))
            
            # 合并所有边缘像素并计算平均值
            all_edges = np.vstack([top.reshape(-1, 3), bottom.reshape(-1, 3), 
                                left.reshape(-1, 3), right.reshape(-1, 3)])
            return tuple(np.mean(all_edges, axis=0).astype(int))
        
        # 计算平均颜色
        edge_array = np.array(edge)
        return tuple(np.mean(edge_array.reshape(-1, 3), axis=0).astype(int))

    def fit_resize_image(self, img, target_width, target_height, fit_mode, resize_sampler, background_color=None):
        """根据不同适应模式调整图像大小"""
        # 解析背景颜色
        bg_color = self.parse_color(background_color) if background_color else (0, 0, 0)
        
        # 获取原始尺寸
        orig_width, orig_height = img.size
        
        # 计算宽高比
        orig_ratio = orig_width / orig_height
        target_ratio = target_width / target_height
        
        # 根据适应模式处理图像
        if fit_mode == 'stretch':  # 直接拉伸
            # 直接拉伸到目标尺寸
            return img.resize((target_width, target_height), resize_sampler)
        
        elif fit_mode == 'crop':  # 裁剪
            # 调整大小保持比例，然后裁剪
            if orig_ratio > target_ratio:
                # 原图更宽，按高度缩放后裁剪宽度
                new_height = target_height
                new_width = int(new_height * orig_ratio)
                img = img.resize((new_width, new_height), resize_sampler)
                # 居中裁剪
                left = (new_width - target_width) // 2
                img = img.crop((left, 0, left + target_width, new_height))
            else:
                # 原图更高，按宽度缩放后裁剪高度
                new_width = target_width
                new_height = int(new_width / orig_ratio)
                img = img.resize((new_width, new_height), resize_sampler)
                # 居中裁剪
                top = (new_height - target_height) // 2
                img = img.crop((0, top, new_width, top + target_height))
            return img
        
        elif fit_mode == 'pad':  # 填充
            # 调整大小保持比例，然后填充
            if orig_ratio > target_ratio:
                # 原图更宽，按宽度缩放后填充高度
                new_width = target_width
                new_height = int(new_width / orig_ratio)
                img = img.resize((new_width, new_height), resize_sampler)
                
                # 计算需要填充的像素数
                pad_top = (target_height - new_height) // 2
                pad_bottom = target_height - new_height - pad_top
                
                # 处理边缘颜色填充
                if bg_color == 'edge':
                    top_color = self.get_edge_color(img, 'top')
                    bottom_color = self.get_edge_color(img, 'bottom')
                    
                    # 创建填充图像
                    padded = Image.new('RGB', (target_width, target_height), (0, 0, 0))
                    # 填充顶部
                    if pad_top > 0:
                        top_pad = Image.new('RGB', (target_width, pad_top), top_color)
                        padded.paste(top_pad, (0, 0))
                    # 粘贴原图
                    padded.paste(img, (0, pad_top))
                    # 填充底部
                    if pad_bottom > 0:
                        bottom_pad = Image.new('RGB', (target_width, pad_bottom), bottom_color)
                        padded.paste(bottom_pad, (0, pad_top + new_height))
                else:
                    # 使用指定颜色填充
                    padded = Image.new('RGB', (target_width, target_height), bg_color)
                    padded.paste(img, (0, pad_top))
            else:
                # 原图更高，按高度缩放后填充宽度
                new_height = target_height
                new_width = int(new_height * orig_ratio)
                img = img.resize((new_width, new_height), resize_sampler)
                
                # 计算需要填充的像素数
                pad_left = (target_width - new_width) // 2
                pad_right = target_width - new_width - pad_left
                
                # 处理边缘颜色填充
                if bg_color == 'edge':
                    left_color = self.get_edge_color(img, 'left')
                    right_color = self.get_edge_color(img, 'right')
                    
                    # 创建填充图像
                    padded = Image.new('RGB', (target_width, target_height), (0, 0, 0))
                    # 填充左侧
                    if pad_left > 0:
                        left_pad = Image.new('RGB', (pad_left, target_height), left_color)
                        padded.paste(left_pad, (0, 0))
                    # 粘贴原图
                    padded.paste(img, (pad_left, 0))
                    # 填充右侧
                    if pad_right > 0:
                        right_pad = Image.new('RGB', (pad_right, target_height), right_color)
                        padded.paste(right_pad, (pad_left + new_width, 0))
                else:
                    # 使用指定颜色填充
                    padded = Image.new('RGB', (target_width, target_height), bg_color)
                    padded.paste(img, (pad_left, 0))
            
            return padded
        
        # 默认情况下直接调整大小
        return img.resize((target_width, target_height), resize_sampler)

    @staticmethod
    def tensor2pil(image):
        """将张量转换为PIL图像"""
        return Image.fromarray(np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

    @staticmethod
    def pil2tensor(image):
        """将PIL图像转换为张量"""
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

    @staticmethod
    def image2mask(image):
        """将图像转换为遮罩"""
        return torch.from_numpy(np.array(image.convert("L")).astype(np.float32) / 255.0).unsqueeze(0)

    @staticmethod
    def is_valid_mask(mask):
        """检查遮罩是否有效"""
        if mask is None:
            return False
        return torch.min(mask) < 0.9999

    @staticmethod
    def num_round_up_to_multiple(num, multiple):
        """将数字向上取整到指定倍数"""
        return math.ceil(num / multiple) * multiple

    @staticmethod
    def log(message, message_type='info'):
        """输出日志信息"""
        print(f"[{message_type.upper()}] {message}")


class ImageRotateWithMask:
    """
    图像旋转节点 - 支持任意角度旋转和多种填充模式
    当传入mask时，mask与image进行相同的变换操作
    支持以mask白色区域为中心进行旋转
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "angle": ("FLOAT", {"default": 0.0, "min": -3600.0, "max": 3600.0, "step": 0.01}),
                "fill_mode": (["color", "edge_extend", "mirror"], {"default": "color"}),
                "fill_color": ("STRING", {"default": "0.0"}),
                "expand": ("BOOLEAN", {"default": True}),
                "use_mask_center": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "rotate_image"
    CATEGORY = "1hewNodes/image"

    def rotate_image(self, image, angle, expand=True, fill_mode="color", use_mask_center=False, mask=None, fill_color="0.0"):
        """
        旋转图像
        
        Args:
            image: 输入图像张量
            angle: 旋转角度（度）
            expand: 是否扩展画布以包含完整旋转后的图像
            fill_mode: 填充模式（color, edge_extend, mirror）
            use_mask_center: 是否以mask的白色区域为中心进行旋转
            mask: 可选的遮罩，与image进行相同的变换操作
            fill_color: 填充颜色（仅在color模式下使用）
        """
        # 修正角度方向
        angle = -angle
        
        batch_size = image.shape[0]
        output_images = []
        output_masks = []
        
        for i in range(batch_size):
            # 转换为PIL图像
            img_tensor = image[i]
            img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            
            # 处理mask（如果提供）
            if mask is not None:
                mask_tensor = mask[i % mask.shape[0]]
                mask_np = (mask_tensor.cpu().numpy() * 255).astype(np.uint8)
                mask_pil = Image.fromarray(mask_np).convert("L")
                
                # 调整mask尺寸以匹配图像
                if pil_img.size != mask_pil.size:
                    mask_pil = mask_pil.resize(pil_img.size, Image.NEAREST)
            else:
                mask_pil = None
            
            # 获取图像尺寸和旋转中心
            width, height = pil_img.size
            
            # 根据use_mask_center参数确定旋转中心
            if use_mask_center and mask_pil is not None:
                center = self._calculate_mask_center(mask_pil)
            else:
                center = (width // 2, height // 2)
            
            # 处理填充模式
            if fill_mode == "color":
                # 解析填充颜色
                fill_rgb = self._parse_color_advanced(fill_color, img_tensor)
                fill_color_rgba = tuple(fill_rgb)
                
                # 执行旋转
                rotated_img = pil_img.rotate(
                    angle, 
                    resample=Image.BILINEAR, 
                    expand=expand, 
                    center=center, 
                    fillcolor=fill_color_rgba
                )
                
                # 对mask执行相同的旋转操作
                if mask_pil is not None:
                    rotated_mask = mask_pil.rotate(
                        angle,
                        resample=Image.BILINEAR,
                        expand=expand,
                        center=center,
                        fillcolor=0  # mask填充区域为黑色
                    )
                else:
                    # 创建旋转区域遮罩
                    rotated_mask = self._create_rotation_mask(pil_img.size, angle, center, expand)
                
            elif fill_mode in ["edge_extend", "mirror"]:
                # 使用高级填充模式
                rotated_img, rotated_mask = self._rotate_with_advanced_fill(
                    pil_img, mask_pil, angle, center, expand, fill_mode
                )
            else:
                # 默认使用color模式
                rotated_img = pil_img.rotate(
                    angle, 
                    resample=Image.BILINEAR, 
                    expand=expand, 
                    center=center, 
                    fillcolor=(0, 0, 0)
                )
                
                # 对mask执行相同的旋转操作
                if mask_pil is not None:
                    rotated_mask = mask_pil.rotate(
                        angle,
                        resample=Image.BILINEAR,
                        expand=expand,
                        center=center,
                        fillcolor=0
                    )
                else:
                    rotated_mask = self._create_rotation_mask(pil_img.size, angle, center, expand)
            
            # 确保图像为RGB模式
            if rotated_img.mode == "RGBA":
                # 创建白色背景
                background = Image.new("RGB", rotated_img.size, (255, 255, 255))
                background.paste(rotated_img, mask=rotated_img.split()[-1])
                rotated_img = background
            elif rotated_img.mode != "RGB":
                rotated_img = rotated_img.convert("RGB")
            
            # 转换图像为张量
            img_np = np.array(rotated_img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np)
            output_images.append(img_tensor)
            
            # 处理遮罩
            if rotated_mask is not None:
                mask_np = np.array(rotated_mask).astype(np.float32) / 255.0
                mask_tensor = torch.from_numpy(mask_np)
            else:
                # 创建全白遮罩（表示整个图像都是有效的）
                mask_tensor = torch.ones((img_tensor.shape[0], img_tensor.shape[1]), dtype=torch.float32)
            
            output_masks.append(mask_tensor)
        
        # 堆叠为批次
        result_images = torch.stack(output_images, dim=0)
        result_masks = torch.stack(output_masks, dim=0)
        
        return (result_images, result_masks)
    
    def _calculate_mask_center(self, mask_pil):
        """
        计算mask白色区域的加权重心
        
        Args:
            mask_pil: PIL格式的mask图像
            
        Returns:
            tuple: (center_x, center_y) 重心坐标
        """
        # 转换为numpy数组
        mask_array = np.array(mask_pil, dtype=np.float32)
        
        # 获取图像尺寸
        height, width = mask_array.shape
        
        # 计算总权重（所有白色像素的强度总和）
        total_weight = np.sum(mask_array)
        
        # 如果没有白色区域，返回图像中心
        if total_weight == 0:
            return (width // 2, height // 2)
        
        # 创建坐标网格
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        
        # 计算加权重心
        center_x = np.sum(x_coords * mask_array) / total_weight
        center_y = np.sum(y_coords * mask_array) / total_weight
        
        # 返回整数坐标
        return (int(center_x), int(center_y))
    
    def _create_rotation_mask(self, original_size, angle, center, expand):
        """
        创建旋转区域遮罩，标识哪些区域是原始图像，哪些是填充区域
        """
        width, height = original_size
        
        # 创建原始图像的遮罩（全白）
        mask = Image.new("L", (width, height), 255)
        
        # 旋转遮罩
        rotated_mask = mask.rotate(
            angle,
            resample=Image.BILINEAR,
            expand=expand,
            center=center,
            fillcolor=0  # 填充区域为黑色
        )
        
        return rotated_mask
    
    def _create_rotation_mask_cv(self, original_size, angle, center, expand, target_size):
        """
        使用OpenCV创建与旋转图像尺寸一致的遮罩
        """
        width, height = original_size
        target_width, target_height = target_size
        
        # 创建原始图像的遮罩（全白）
        mask_cv = np.ones((height, width), dtype=np.uint8) * 255
        
        # 计算旋转矩阵
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # 如果需要扩展，调整旋转矩阵
        if expand:
            rotation_matrix[0, 2] += (target_width / 2) - center[0]
            rotation_matrix[1, 2] += (target_height / 2) - center[1]
        
        # 执行旋转
        rotated_mask_cv = cv2.warpAffine(
            mask_cv,
            rotation_matrix,
            (target_width, target_height),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        # 转换为PIL格式
        rotated_mask = Image.fromarray(rotated_mask_cv, mode='L')
        
        return rotated_mask
    
    def _rotate_with_advanced_fill(self, pil_img, mask_pil, angle, center, expand, fill_mode):
        """
        使用高级填充模式进行旋转，同时处理mask
        """
        # 转换为OpenCV格式
        img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        height, width = img_cv.shape[:2]
        
        # 计算旋转矩阵
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # 计算新的边界框尺寸
        if expand:
            cos_val = abs(rotation_matrix[0, 0])
            sin_val = abs(rotation_matrix[0, 1])
            new_width = int((height * sin_val) + (width * cos_val))
            new_height = int((height * cos_val) + (width * sin_val))
            
            # 调整旋转矩阵以居中图像
            rotation_matrix[0, 2] += (new_width / 2) - center[0]
            rotation_matrix[1, 2] += (new_height / 2) - center[1]
        else:
            new_width, new_height = width, height
        
        # 设置边界填充模式
        if fill_mode == "edge_extend":
            border_mode = cv2.BORDER_REPLICATE
        elif fill_mode == "mirror":
            border_mode = cv2.BORDER_REFLECT
        else:
            border_mode = cv2.BORDER_CONSTANT
        
        # 执行图像旋转
        rotated_cv = cv2.warpAffine(
            img_cv, 
            rotation_matrix, 
            (new_width, new_height),
            borderMode=border_mode
        )
        
        # 转换回PIL格式
        rotated_pil = Image.fromarray(cv2.cvtColor(rotated_cv, cv2.COLOR_BGR2RGB))
        
        # 处理mask旋转
        if mask_pil is not None:
            # 将mask转换为OpenCV格式
            mask_cv = np.array(mask_pil)
            
            # 对mask执行相同的旋转操作
            rotated_mask_cv = cv2.warpAffine(
                mask_cv,
                rotation_matrix,
                (new_width, new_height),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            
            # 转换回PIL格式
            rotated_mask = Image.fromarray(rotated_mask_cv, mode='L')
        else:
            # 创建与旋转图像尺寸一致的遮罩
            rotated_mask = self._create_rotation_mask_cv(pil_img.size, angle, center, expand, (new_width, new_height))
        
        return rotated_pil, rotated_mask
    
    def _parse_color_advanced(self, color_str, img_tensor=None):
        """
        高级颜色解析，参考ImageEdgeCropPad的实现
        支持多种格式：灰度值、HEX、RGB、颜色名称、特殊值
        """
        if not color_str:
            return (0, 0, 0)
        
        # 检查特殊值
        color_lower = color_str.lower().strip()
        
        # 检查是否为 average 或其缩写
        if color_lower in ['average', 'avg', 'a', 'av', 'aver']:
            if img_tensor is not None:
                return self._get_average_color_tensor(img_tensor)
            return (128, 128, 128)  # 默认灰色
        
        # 检查是否为 edge 或其缩写
        if color_lower in ['edge', 'e', 'ed']:
            if img_tensor is not None:
                return self._get_edge_color_tensor(img_tensor)
            return (128, 128, 128)  # 默认灰色
        
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
        except ValueError:
            pass
        
        # 尝试解析为HEX格式
        if color_str.startswith('#'):
            color_str = color_str[1:]
        
        if len(color_str) == 6 and all(c in '0123456789ABCDEFabcdef' for c in color_str):
            try:
                r = int(color_str[0:2], 16)
                g = int(color_str[2:4], 16)
                b = int(color_str[4:6], 16)
                return (r, g, b)
            except ValueError:
                pass
        
        # 尝试解析为RGB格式 "255,0,0" 或 "1.0,0.0,0.0"
        try:
            rgb_values = color_str.split(',')
            if len(rgb_values) == 3:
                r, g, b = [float(v.strip()) for v in rgb_values]
                
                # 检查是否为0-1范围的浮点数
                if all(0.0 <= v <= 1.0 for v in [r, g, b]):
                    return (int(r * 255), int(g * 255), int(b * 255))
                # 检查是否为0-255范围的整数
                elif all(0 <= v <= 255 for v in [r, g, b]):
                    return (int(r), int(g), int(b))
        except ValueError:
            pass
        
        # 颜色名称映射
        color_names = {
            'black': (0, 0, 0),
            'white': (255, 255, 255),
            'red': (255, 0, 0),
            'green': (0, 255, 0),
            'blue': (0, 0, 255),
            'yellow': (255, 255, 0),
            'cyan': (0, 255, 255),
            'magenta': (255, 0, 255),
            'gray': (128, 128, 128),
            'grey': (128, 128, 128),
        }
        
        if color_lower in color_names:
            return color_names[color_lower]
        
        # 默认返回黑色
        return (0, 0, 0)
    
    def _get_average_color_tensor(self, img_tensor):
        """
        获取图像张量的平均颜色
        """
        # img_tensor shape: [H, W, C]
        mean_color = img_tensor.mean(dim=(0, 1))  # [C]
        mean_color_255 = (mean_color * 255).int().tolist()
        return tuple(mean_color_255)
    
    def _get_edge_color_tensor(self, img_tensor):
        """
        获取图像张量边缘的平均颜色
        """
        # img_tensor shape: [H, W, C]
        h, w, c = img_tensor.shape
        
        # 收集边缘像素
        edge_pixels = []
        
        # 上边缘和下边缘
        edge_pixels.append(img_tensor[0, :, :])  # 上边缘
        edge_pixels.append(img_tensor[-1, :, :])  # 下边缘
        
        # 左边缘和右边缘（排除角落避免重复）
        if h > 2:
            edge_pixels.append(img_tensor[1:-1, 0, :])  # 左边缘
            edge_pixels.append(img_tensor[1:-1, -1, :])  # 右边缘
        
        # 合并所有边缘像素
        all_edge_pixels = torch.cat([p.reshape(-1, c) for p in edge_pixels], dim=0)
        
        # 计算平均颜色
        mean_color = all_edge_pixels.mean(dim=0)
        mean_color_255 = (mean_color * 255).int().tolist()
        return tuple(mean_color_255)


class ImageEditStitch:
    """
        图像编辑缝合 - 将参考图像和编辑图像拼接在一起，支持上下左右四种拼接方式
        优化版本：当match_edit_size为false时，保持reference_image的原始比例
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_image": ("IMAGE",),
                "edit_image": ("IMAGE",),
                "edit_image_position": (["top", "bottom", "left", "right"], {"default": "right"}),
                "match_edit_size": ("BOOLEAN", {"default": False}),
                "spacing": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "spacing_color": ("STRING", {"default": "1.0"}),
                "pad_color": ("STRING", {"default": "1.0"})
            },
            "optional": {
                "edit_mask": ("MASK",)
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("image", "mask", "split_mask")
    FUNCTION = "image_edit_stitch"
    CATEGORY = "1hewNodes/image"

    def image_edit_stitch(
        self,
        reference_image,
        edit_image,
        edit_mask=None,
        edit_image_position='right',
        match_edit_size=True,
        spacing=0,
        spacing_color="1.0",
        pad_color="1.0",
    ):
        # 检查输入
        if reference_image is None and edit_image is None:
            # 如果两个图像都为空，创建默认图像
            default_image = torch.ones((1, 512, 512, 3), dtype=torch.float32)
            default_mask = torch.ones((1, 512, 512), dtype=torch.float32)
            return default_image, default_mask, default_mask

        # 如果只有一个图像存在，直接返回该图像
        if reference_image is None:
            # 规范遮罩形状，并按图像批次广播
            bs = edit_image.shape[0]
            edit_mask = self._ensure_mask_3d(edit_mask)
            if edit_mask is None:
                edit_mask = torch.ones(
                    (bs, edit_image.shape[1], edit_image.shape[2]),
                    dtype=torch.float32,
                )
            else:
                edit_mask = self._broadcast_mask(edit_mask, bs)
            # 创建分离遮罩（全黑，表示全部是编辑区域）
            split_mask = torch.zeros_like(edit_mask)
            return edit_image, edit_mask, split_mask

        if edit_image is None:
            # 创建与参考图像相同尺寸的空白图像
            edit_image = torch.zeros_like(reference_image)
            # 创建全白遮罩，批次与参考图像一致
            bs = reference_image.shape[0]
            white_mask = torch.ones(
                (bs, reference_image.shape[1], reference_image.shape[2]),
                dtype=torch.float32,
            )
            # 创建分离遮罩（全白，表示全部是参考区域）
            split_mask = torch.ones_like(white_mask)
            return reference_image, white_mask, split_mask

        # 确保编辑遮罩存在，如果不存在则创建全白遮罩
        edit_mask = self._ensure_mask_3d(edit_mask)
        if edit_mask is None:
            edit_mask = torch.ones(
                (edit_image.shape[0], edit_image.shape[1], edit_image.shape[2]),
                dtype=torch.float32,
            )

        # 统一批量尺寸（广播到最大批次）
        bs = max(reference_image.shape[0], edit_image.shape[0], edit_mask.shape[0])
        reference_image = self._broadcast_image(reference_image, bs)
        edit_image = self._broadcast_image(edit_image, bs)
        edit_mask = self._broadcast_mask(edit_mask, bs)

        # 获取图像尺寸
        ref_batch, ref_height, ref_width, _ = reference_image.shape
        edit_batch, edit_height, edit_width, _ = edit_image.shape

        # 遮罩尺寸与编辑图像对齐（最近邻），避免拼接时高度/宽度不一致
        edit_mask = self._resize_mask_to_image(edit_mask, edit_image)

        # 处理尺寸调整逻辑（批量按样本处理）
        if match_edit_size:
            if ref_height != edit_height or ref_width != edit_width:
                reference_image = self._resize_with_padding(
                    reference_image, edit_width, edit_height, pad_color
                )
                ref_batch, ref_height, ref_width, _ = reference_image.shape
        else:
            reference_image = self._resize_keeping_aspect_ratio(
                reference_image, edit_image, edit_image_position
            )
            ref_batch, ref_height, ref_width, _ = reference_image.shape

        # 间隔条颜色解析（丰富字符串 -> 0..1 RGB）
        try:
            ctx = reference_image[0] if reference_image is not None else None
        except Exception:
            ctx = None
        rgb255 = self._parse_color_advanced(spacing_color, ctx)
        space_rgb = (
            rgb255[0] / 255.0,
            rgb255[1] / 255.0,
            rgb255[2] / 255.0,
        )
        spacing_color_tensor = torch.tensor(
            space_rgb, dtype=torch.float32
        ).view(1, 1, 1, 3)
        
        # 根据编辑图像位置拼接图像
        if edit_image_position == "right":
            # 参考图像在左，编辑图像在右
            if spacing > 0:
                # 创建垂直间距条
                spacing_strip = spacing_color_tensor.expand(bs, ref_height, spacing, 3)
                combined_image = torch.cat([
                    reference_image,
                    spacing_strip,
                    edit_image
                ], dim=2)  # 水平拼接

                # 拼接遮罩（参考区域为0，间距区域为0，编辑区域保持原样）
                zero_mask_ref = torch.zeros((bs, ref_height, ref_width), dtype=torch.float32)
                zero_mask_spacing = torch.zeros((bs, ref_height, spacing), dtype=torch.float32)
                combined_mask = torch.cat([zero_mask_ref, zero_mask_spacing, edit_mask], dim=2)
                
                # 创建分离遮罩（参考区域为黑色，间距区域为黑色，编辑区域为白色）
                split_mask_left = torch.zeros((bs, ref_height, ref_width), dtype=torch.float32)
                split_mask_spacing = torch.zeros((bs, ref_height, spacing), dtype=torch.float32)
                split_mask_right = torch.ones((bs, edit_height, edit_width), dtype=torch.float32)
                split_mask = torch.cat([split_mask_left, split_mask_spacing, split_mask_right], dim=2)
            else:
                combined_image = torch.cat([
                    reference_image,
                    edit_image
                ], dim=2)  # 水平拼接
                
                # 拼接遮罩（参考区域为0，编辑区域保持原样）
                zero_mask = torch.zeros((bs, ref_height, ref_width), dtype=torch.float32)
                combined_mask = torch.cat([zero_mask, edit_mask], dim=2)
                
                # 创建分离遮罩（参考区域为黑色，编辑区域为白色）
                split_mask_left = torch.zeros((bs, ref_height, ref_width), dtype=torch.float32)
                split_mask_right = torch.ones((bs, edit_height, edit_width), dtype=torch.float32)
                split_mask = torch.cat([split_mask_left, split_mask_right], dim=2)

        elif edit_image_position == "left":
            # 编辑图像在左，参考图像在右
            if spacing > 0:
                # 创建垂直间距条
                spacing_strip = spacing_color_tensor.expand(bs, edit_height, spacing, 3)
                combined_image = torch.cat([
                    edit_image,
                    spacing_strip,
                    reference_image
                ], dim=2)  # 水平拼接
                
                # 拼接遮罩（编辑区域保持原样，间距区域为0，参考区域为0）
                zero_mask_spacing = torch.zeros((bs, edit_height, spacing), dtype=torch.float32)
                zero_mask_ref = torch.zeros((bs, ref_height, ref_width), dtype=torch.float32)
                combined_mask = torch.cat([edit_mask, zero_mask_spacing, zero_mask_ref], dim=2)
                
                # 创建分离遮罩（编辑区域为白色，间距区域为黑色，参考区域为黑色）
                split_mask_left = torch.ones((bs, edit_height, edit_width), dtype=torch.float32)
                split_mask_spacing = torch.zeros((bs, edit_height, spacing), dtype=torch.float32)
                split_mask_right = torch.zeros((bs, ref_height, ref_width), dtype=torch.float32)
                split_mask = torch.cat([split_mask_left, split_mask_spacing, split_mask_right], dim=2)
            else:
                combined_image = torch.cat([
                    edit_image,
                    reference_image
                ], dim=2)  # 水平拼接
                
                # 拼接遮罩（编辑区域保持原样，参考区域为0）
                zero_mask = torch.zeros((bs, ref_height, ref_width), dtype=torch.float32)
                combined_mask = torch.cat([edit_mask, zero_mask], dim=2)
                
                # 创建分离遮罩（编辑区域为白色，参考区域为黑色）
                split_mask_left = torch.ones((bs, edit_height, edit_width), dtype=torch.float32)
                split_mask_right = torch.zeros((bs, ref_height, ref_width), dtype=torch.float32)
                split_mask = torch.cat([split_mask_left, split_mask_right], dim=2)

        elif edit_image_position == "bottom":
            # 参考图像在上，编辑图像在下
            if spacing > 0:
                # 创建水平间距条
                spacing_strip = spacing_color_tensor.expand(bs, spacing, ref_width, 3)
                combined_image = torch.cat([
                    reference_image,
                    spacing_strip,
                    edit_image
                ], dim=1)  # 垂直拼接
                
                # 拼接遮罩（参考区域为0，间距区域为0，编辑区域保持原样）
                zero_mask_ref = torch.zeros((bs, ref_height, ref_width), dtype=torch.float32)
                zero_mask_spacing = torch.zeros((bs, spacing, ref_width), dtype=torch.float32)
                combined_mask = torch.cat([zero_mask_ref, zero_mask_spacing, edit_mask], dim=1)
                
                # 创建分离遮罩（参考区域为黑色，间距区域为黑色，编辑区域为白色）
                split_mask_top = torch.zeros((bs, ref_height, ref_width), dtype=torch.float32)
                split_mask_spacing = torch.zeros((bs, spacing, ref_width), dtype=torch.float32)
                split_mask_bottom = torch.ones((bs, edit_height, edit_width), dtype=torch.float32)
                split_mask = torch.cat([split_mask_top, split_mask_spacing, split_mask_bottom], dim=1)
            else:
                combined_image = torch.cat([
                    reference_image,
                    edit_image
                ], dim=1)  # 垂直拼接
                
                # 拼接遮罩（参考区域为0，编辑区域保持原样）
                zero_mask = torch.zeros((bs, ref_height, ref_width), dtype=torch.float32)
                combined_mask = torch.cat([zero_mask, edit_mask], dim=1)
                
                # 创建分离遮罩（参考区域为黑色，编辑区域为白色）
                split_mask_top = torch.zeros((bs, ref_height, ref_width), dtype=torch.float32)
                split_mask_bottom = torch.ones((bs, edit_height, edit_width), dtype=torch.float32)
                split_mask = torch.cat([split_mask_top, split_mask_bottom], dim=1)

        elif edit_image_position == "top":
            # 编辑图像在上，参考图像在下
            if spacing > 0:
                # 创建水平间距条
                spacing_strip = spacing_color_tensor.expand(bs, spacing, edit_width, 3)
                combined_image = torch.cat([
                    edit_image,
                    spacing_strip,
                    reference_image
                ], dim=1)  # 垂直拼接
                
                # 拼接遮罩（编辑区域保持原样，间距区域为0，参考区域为0）
                zero_mask_spacing = torch.zeros((bs, spacing, edit_width), dtype=torch.float32)
                zero_mask_ref = torch.zeros((bs, ref_height, ref_width), dtype=torch.float32)
                combined_mask = torch.cat([edit_mask, zero_mask_spacing, zero_mask_ref], dim=1)
                
                # 创建分离遮罩（编辑区域为白色，间距区域为黑色，参考区域为黑色）
                split_mask_top = torch.ones((bs, edit_height, edit_width), dtype=torch.float32)
                split_mask_spacing = torch.zeros((bs, spacing, edit_width), dtype=torch.float32)
                split_mask_bottom = torch.zeros((bs, ref_height, ref_width), dtype=torch.float32)
                split_mask = torch.cat([split_mask_top, split_mask_spacing, split_mask_bottom], dim=1)
            else:
                combined_image = torch.cat([
                    edit_image,
                    reference_image
                ], dim=1)  # 垂直拼接
                
                # 拼接遮罩（编辑区域保持原样，参考区域为0）
                zero_mask = torch.zeros((bs, ref_height, ref_width), dtype=torch.float32)
                combined_mask = torch.cat([edit_mask, zero_mask], dim=1)
                
                # 创建分离遮罩（编辑区域为白色，参考区域为黑色）
                split_mask_top = torch.ones((bs, edit_height, edit_width), dtype=torch.float32)
                split_mask_bottom = torch.zeros((bs, ref_height, ref_width), dtype=torch.float32)
                split_mask = torch.cat([split_mask_top, split_mask_bottom], dim=1)

        return combined_image, combined_mask, split_mask

    def _resize_with_padding(self, image, target_width, target_height, pad_color):
        """
        原有的resize逻辑：使用padding填充到目标尺寸（pad_color 控制颜色）
        """
        bs = image.shape[0]
        out = []
        for i in range(bs):
            # 针对每个样本按上下文计算填充颜色，支持 average/edge
            fill_rgb = self._parse_color_advanced(pad_color, image[i])
            img_np = (image[i].cpu().numpy() * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)

            img_width, img_height = img_pil.size
            img_aspect = img_width / max(img_height, 1)
            target_aspect = target_width / max(target_height, 1)

            if img_aspect > target_aspect:
                new_width = target_width
                new_height = int(target_width / img_aspect)
            else:
                new_height = target_height
                new_width = int(target_height * img_aspect)

            img_pil = img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)

            new_img_pil = Image.new("RGB", (target_width, target_height), fill_rgb)
            paste_x = (target_width - new_width) // 2
            paste_y = (target_height - new_height) // 2
            new_img_pil.paste(img_pil, (paste_x, paste_y))

            arr = np.array(new_img_pil).astype(np.float32) / 255.0
            out.append(torch.from_numpy(arr))

        return torch.stack(out, dim=0)

    def _parse_color_string(self, color_str):
        """解析任意颜色字符串为 0..1 RGB，复用 ImageSolid.parse_color。"""
        try:
            parser = ImageSolid()
            r, g, b = parser.parse_color(color_str)
            return (r / 255.0, g / 255.0, b / 255.0)
        except Exception:
            return (1.0, 1.0, 1.0)

    def _parse_color_advanced(self, color_str, img_tensor=None):
        """
        高级颜色解析：支持灰度、HEX、RGB、颜色名以及特殊值。
        - 灰度："0.0".."1.0"
        - HEX："#RRGGBB" 或 "RRGGBB"
        - RGB："r,g,b"，可为 0..1 或 0..255
        - 名称：black/white/red/green/blue/yellow/cyan/magenta/gray/grey
        - 特殊：average/edge（需要参考图像张量）
        """
        if not color_str:
            return (0, 0, 0)

        color_lower = color_str.lower().strip()

        # 特殊值：平均色与边缘平均色
        if color_lower in ["average", "avg", "a", "av", "aver"]:
            if img_tensor is not None:
                return self._get_average_color_tensor(img_tensor)
            return (128, 128, 128)
        if color_lower in ["edge", "e", "ed"]:
            if img_tensor is not None:
                return self._get_edge_color_tensor(img_tensor)
            return (128, 128, 128)

        # 去除包裹括号
        text = color_str.strip()
        if text.startswith("(") and text.endswith(")"):
            text = text[1:-1].strip()

        # 灰度值（0..1）
        try:
            gray = float(text)
            if 0.0 <= gray <= 1.0:
                g8 = int(gray * 255)
                return (g8, g8, g8)
        except ValueError:
            pass

        # HEX 解析
        hex_text = text[1:] if text.startswith("#") else text
        if (len(hex_text) == 6 and
                all(c in "0123456789ABCDEFabcdef" for c in hex_text)):
            try:
                r = int(hex_text[0:2], 16)
                g = int(hex_text[2:4], 16)
                b = int(hex_text[4:6], 16)
                return (r, g, b)
            except ValueError:
                pass

        # RGB（"r,g,b"），支持 0..1 或 0..255
        try:
            parts = [p.strip() for p in text.split(",")]
            if len(parts) == 3:
                r_f, g_f, b_f = [float(p) for p in parts]
                if all(0.0 <= v <= 1.0 for v in [r_f, g_f, b_f]):
                    return (int(r_f * 255), int(g_f * 255), int(b_f * 255))
                if all(0 <= v <= 255 for v in [r_f, g_f, b_f]):
                    return (int(r_f), int(g_f), int(b_f))
        except ValueError:
            pass

        # 颜色名称
        name_map = {
            "black": (0, 0, 0),
            "white": (255, 255, 255),
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "yellow": (255, 255, 0),
            "cyan": (0, 255, 255),
            "magenta": (255, 0, 255),
            "gray": (128, 128, 128),
            "grey": (128, 128, 128),
        }
        if color_lower in name_map:
            return name_map[color_lower]

        return (0, 0, 0)

    def _get_average_color_tensor(self, img_tensor):
        """返回图像张量的平均颜色（0..255 RGB 元组）。"""
        mean_color = img_tensor.mean(dim=(0, 1))
        mean_255 = (mean_color * 255).int().tolist()
        return tuple(mean_255)

    def _get_edge_color_tensor(self, img_tensor):
        """返回图像张量边缘像素的平均颜色（0..255）。"""
        h, w, c = img_tensor.shape
        edge_pixels = []
        edge_pixels.append(img_tensor[0, :, :])
        edge_pixels.append(img_tensor[-1, :, :])
        if h > 2:
            edge_pixels.append(img_tensor[1:-1, 0, :])
            edge_pixels.append(img_tensor[1:-1, -1, :])
        all_edge = torch.cat([p.reshape(-1, c) for p in edge_pixels], dim=0)
        mean_color = all_edge.mean(dim=0)
        mean_255 = (mean_color * 255).int().tolist()
        return tuple(mean_255)

    def _resize_keeping_aspect_ratio(self, reference_image, edit_image, edit_image_position):
        """
        新的resize逻辑：保持reference_image的原始比例，根据拼接方向调整尺寸
        """
        bs = max(reference_image.shape[0], edit_image.shape[0])
        out = []
        for i in range(bs):
            # 单样本 (H, W, C)
            ref_arr = (reference_image[i].cpu().numpy() * 255).astype(np.uint8)
            ref_pil = Image.fromarray(ref_arr)

            edit_h = int(edit_image[i].shape[0])
            edit_w = int(edit_image[i].shape[1])
            ref_h = int(reference_image[i].shape[0])
            ref_w = int(reference_image[i].shape[1])

            if edit_image_position in ["left", "right"]:
                target_height = max(edit_h, 1)
                aspect_ratio = ref_w / max(ref_h, 1)
                target_width = max(int(round(target_height * aspect_ratio)), 1)
            else:
                target_width = max(edit_w, 1)
                aspect_ratio = ref_h / max(ref_w, 1)
                target_height = max(int(round(target_width * aspect_ratio)), 1)

            ref_pil = ref_pil.resize((target_width, target_height), Image.Resampling.LANCZOS)
            arr = np.array(ref_pil).astype(np.float32) / 255.0
            out.append(torch.from_numpy(arr))

        return torch.stack(out, dim=0)

    def _broadcast_image(self, img, batch_size):
        b = img.shape[0]
        if b == batch_size:
            return img
        if b == 1:
            return img.repeat(batch_size, 1, 1, 1)
        reps = int(math.ceil(batch_size / b))
        tiled = img.repeat(reps, 1, 1, 1)[:batch_size]
        return tiled

    def _broadcast_mask(self, mask, batch_size):
        b = mask.shape[0]
        if b == batch_size:
            return mask
        if b == 1:
            return mask.repeat(batch_size, 1, 1)
        reps = int(math.ceil(batch_size / b))
        tiled = mask.repeat(reps, 1, 1)[:batch_size]
        return tiled

    def _ensure_mask_3d(self, mask):
        """确保遮罩为 (batch, height, width) 形状。
        - (batch, 1, height, width) -> (batch, height, width)
        - (height, width) -> (1, height, width)
        其他情况原样返回。
        """
        if mask is None:
            return None
        if mask.dim() == 4:
            if mask.shape[1] == 1:
                return mask.squeeze(1)
            # 取第一个通道作为遮罩
            return mask[:, 0, :, :]
        if mask.dim() == 2:
            return mask.unsqueeze(0)
        return mask

    def _resize_mask_to_image(self, mask, image):
        """将遮罩尺寸重采样到与图像一致 (H, W)。
        使用最近邻以保持二值或离散值。
        """
        if mask is None or image is None:
            return mask
        target_h = int(image.shape[1])
        target_w = int(image.shape[2])
        if int(mask.shape[1]) == target_h and int(mask.shape[2]) == target_w:
            return mask
        mask4 = mask.unsqueeze(1)
        resized4 = F.interpolate(
            mask4, size=(target_h, target_w), mode="nearest"
        )
        return resized4.squeeze(1)


class ImageThreeStitch:
    """
    ImageThreeStitch：三图拼接。先合并 image_2 与 image_3，再按 direction
    将该组合贴到 image_1 的一侧。

    - direction: top/bottom/left/right
      - top/bottom: 2 与 3 先水平并排，再与 1 垂直拼接
      - left/right: 2 与 3 先垂直堆叠，再与 1 水平拼接
    - match_image_size: True 时沿目标轴等比缩放；False 时用 pad_color 居中填充
    - spacing_width: 同一数值同时用于 2+3 的拼接与 (2,3) 与 1 的拼接
    - spacing_color / pad_color: 字符串颜色，解析规则与 ImageSolid.parse_color 相同
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_1": ("IMAGE", {"forceInput": True}),
                "image_2": ("IMAGE", {"forceInput": True}),
                "image_3": ("IMAGE", {"forceInput": True}),
                "direction": (["top", "bottom", "left", "right"], {"default": "left"}),
                "match_image_size": ("BOOLEAN", {"default": True}),
                "spacing_width": ("INT", {"default": 10, "min": 0, "max": 1000, "step": 1}),
                "spacing_color": ("STRING", {"default": "1.0"}),
                "pad_color": ("STRING", {"default": "1.0"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "stitch_attach"
    CATEGORY = "1hewNodes/image"
    DESCRIPTION = "合并 (2,3) 后按方向贴到 1 的一侧。"

    def stitch_attach(
        self,
        image_1,
        image_2,
        image_3,
        direction,
        match_image_size,
        spacing_width,
        spacing_color,
        pad_color,
    ):
        a = torch.clamp(image_1, min=0.0, max=1.0).to(torch.float32)
        b = torch.clamp(image_2, min=0.0, max=1.0).to(torch.float32)
        c = torch.clamp(image_3, min=0.0, max=1.0).to(torch.float32)

        # 对齐批量维度
        bs = max(a.shape[0], b.shape[0], c.shape[0])
        a = self._broadcast_image(a, bs)
        b = self._broadcast_image(b, bs)
        c = self._broadcast_image(c, bs)

        space_rgb = self._parse_color_string(spacing_color)
        pad_rgb = self._parse_color_string(pad_color)

        if direction in ("top", "bottom"):
            pair = self._combine_horizontal(
                b, c, match_image_size, spacing_width, space_rgb, pad_rgb, bs
            )
            out = self._stack_vertical(
                a,
                pair,
                direction,
                match_image_size,
                spacing_width,
                space_rgb,
                pad_rgb,
                bs,
            )
        else:
            pair = self._combine_vertical(
                b, c, match_image_size, spacing_width, space_rgb, pad_rgb, bs
            )
            out = self._stack_horizontal(
                a,
                pair,
                direction,
                match_image_size,
                spacing_width,
                space_rgb,
                pad_rgb,
                bs,
            )

        return (torch.clamp(out, min=0.0, max=1.0).to(torch.float32),)

    # --- 合并 2 与 3 ---
    def _combine_horizontal(
        self, img2, img3, match, spacing_width, space_rgb, pad_rgb, batch_size
    ):
        _, h2, w2, _ = img2.shape
        _, h3, w3, _ = img3.shape
        # 合并阶段：
        # - match=True 按较大高度等比缩放；
        # - match=False 统一到较大高度，使用居中填充保持原图尺寸不变。
        if match:
            target_h = max(h2, h3)
            img2 = self._resize_keep_ratio(img2, None, target_h)
            img3 = self._resize_keep_ratio(img3, None, target_h)
        else:
            target_h = max(h2, h3)
            img2 = self._pad_to_rgb(img2, target_h, w2, pad_rgb)
            img3 = self._pad_to_rgb(img3, target_h, w3, pad_rgb)

        _, hh2, ww2, _ = img2.shape
        _, hh3, ww3, _ = img3.shape
        strip = self._make_strip(
            hh2, spacing_width, space_rgb, axis="v", dtype=img2.dtype,
            device=img2.device, batch_size=batch_size,
        )

        return torch.cat([img2, strip, img3], dim=2)

    def _combine_vertical(
        self, img2, img3, match, spacing_width, space_rgb, pad_rgb, batch_size
    ):
        _, h2, w2, _ = img2.shape
        _, h3, w3, _ = img3.shape
        # 合并阶段：
        # - match=True 按较大宽度等比缩放；
        # - match=False 统一到较大宽度，使用居中填充保持原图尺寸不变。
        if match:
            target_w = max(w2, w3)
            img2 = self._resize_keep_ratio(img2, target_w, None)
            img3 = self._resize_keep_ratio(img3, target_w, None)
        else:
            target_w = max(w2, w3)
            img2 = self._pad_to_rgb(img2, h2, target_w, pad_rgb)
            img3 = self._pad_to_rgb(img3, h3, target_w, pad_rgb)

        _, hh2, ww2, _ = img2.shape
        _, hh3, ww3, _ = img3.shape
        strip = self._make_strip(
            spacing_width, ww2, space_rgb, axis="h", dtype=img2.dtype,
            device=img2.device, batch_size=batch_size,
        )

        return torch.cat([img2, strip, img3], dim=1)

    # --- 将 (2,3) 组合贴到 1 ---
    def _stack_vertical(
        self,
        img1,
        pair,
        direction,
        match,
        spacing_width,
        space_rgb,
        pad_rgb,
        batch_size,
    ):
        _, h1, w1, _ = img1.shape
        _, hp, wp, _ = pair.shape
        # 外部贴合：以 image_1 的宽度为基准
        target_w = w1

        if match:
            img1 = self._resize_keep_ratio(img1, target_w, None)
            pair = self._resize_keep_ratio(pair, target_w, None)
        else:
            # 不匹配尺寸：统一到较大的宽度，使用居中填充（不裁剪）
            unified_w = max(w1, wp)
            img1 = self._pad_to_rgb(img1, h1, unified_w, pad_rgb)
            pair = self._pad_to_rgb(pair, hp, unified_w, pad_rgb)

        _, h1, w1, _ = img1.shape
        _, hp, wp, _ = pair.shape
        # 间隔条宽度与最终统一宽度一致
        strip = self._make_strip(spacing_width, w1 if match else unified_w, space_rgb, axis="h", dtype=img1.dtype, device=img1.device, batch_size=batch_size)

        if direction == "bottom":
            return torch.cat([img1, strip, pair], dim=1)
        return torch.cat([pair, strip, img1], dim=1)

    def _stack_horizontal(
        self,
        img1,
        pair,
        direction,
        match,
        spacing_width,
        space_rgb,
        pad_rgb,
        batch_size,
    ):
        _, h1, w1, _ = img1.shape
        _, hp, wp, _ = pair.shape
        # 外部贴合：以 image_1 的高度为基准
        target_h = h1

        if match:
            img1 = self._resize_keep_ratio(img1, None, target_h)
            pair = self._resize_keep_ratio(pair, None, target_h)
        else:
            # 不匹配尺寸：统一到较大的高度，使用居中填充（不裁剪）
            unified_h = max(h1, hp)
            img1 = self._pad_to_rgb(img1, unified_h, w1, pad_rgb)
            pair = self._pad_to_rgb(pair, unified_h, wp, pad_rgb)

        _, h1, w1, _ = img1.shape
        _, hp, wp, _ = pair.shape
        # 间隔条高度与最终统一高度一致
        strip = self._make_strip(h1 if match else unified_h, spacing_width, space_rgb, axis="v", dtype=img1.dtype, device=img1.device, batch_size=batch_size)

        if direction == "right":
            return torch.cat([img1, strip, pair], dim=2)
        return torch.cat([pair, strip, img1], dim=2)

    # --- 工具函数 ---
    def _parse_color_string(self, color_str):
        """解析任意颜色字符串为 0..1 RGB，复用 ImageSolid.parse_color。"""
        try:
            from .image import ImageSolid  # 延迟导入避免循环依赖
            parser = ImageSolid()
            rgb = parser.parse_color(color_str)
            r = rgb[0] / 255.0
            g = rgb[1] / 255.0
            b = rgb[2] / 255.0
            return (r, g, b)
        except Exception:
            return (1.0, 1.0, 1.0)

    def _resize_keep_ratio(self, img, target_w, target_h):
        """按目标宽或高等比缩放，另一维自适应；不拉伸。"""
        _, h, w, _ = img.shape
        if target_w is None:
            scale = target_h / max(h, 1)
            new_w = max(int(round(w * scale)), 1)
            new_h = target_h
        elif target_h is None:
            scale = target_w / max(w, 1)
            new_h = max(int(round(h * scale)), 1)
            new_w = target_w
        else:
            new_w, new_h = target_w, target_h

        nchw = img.permute(0, 3, 1, 2)
        out = F.interpolate(
            nchw, size=(new_h, new_w), mode="bicubic", align_corners=False
        ).permute(0, 2, 3, 1)
        return out

    def _pad_to_rgb(self, img, target_h, target_w, fill_rgb):
        """居中填充到指定尺寸，使用 RGB 填充颜色。"""
        b, h, w, c = img.shape
        out = torch.zeros(
            (b, target_h, target_w, c), dtype=img.dtype, device=img.device
        )
        fill_t = torch.tensor(fill_rgb, dtype=img.dtype, device=img.device)
        out[:] = fill_t
        top = max((target_h - h) // 2, 0)
        left = max((target_w - w) // 2, 0)
        out[:, top : top + h, left : left + w, :] = img
        return out

    def _pad_or_crop_to_rgb(self, img, target_h, target_w, fill_rgb):
        """居中对齐到指定尺寸：
        - 若源尺寸小于目标尺寸，则填充居中；
        - 若源尺寸大于目标尺寸，则居中裁剪到目标尺寸；
        使用 RGB 填充颜色处理需要填充的区域。
        """
        b, h, w, c = img.shape
        out = torch.zeros(
            (b, target_h, target_w, c), dtype=img.dtype, device=img.device
        )
        fill_t = torch.tensor(fill_rgb, dtype=img.dtype, device=img.device)
        out[:] = fill_t

        # 计算源裁剪窗口（居中裁剪）
        src_top = max((h - target_h) // 2, 0)
        src_left = max((w - target_w) // 2, 0)
        copy_h = min(h, target_h)
        copy_w = min(w, target_w)

        # 计算目标粘贴位置（居中填充）
        dst_top = max((target_h - h) // 2, 0)
        dst_left = max((target_w - w) // 2, 0)

        out[:, dst_top : dst_top + copy_h, dst_left : dst_left + copy_w, :] = (
            img[:, src_top : src_top + copy_h, src_left : src_left + copy_w, :]
        )
        return out

    def _make_strip(self, h, w, fill_rgb, axis, dtype, device, batch_size=1):
        """生成间隔条：axis='v' 垂直条，axis='h' 水平条。"""
        if w <= 0:
            return torch.zeros((batch_size, h, 0, 3), dtype=dtype, device=device)
        if axis == "v":
            out = torch.zeros((batch_size, h, w, 3), dtype=dtype, device=device)
            out[:] = torch.tensor(fill_rgb, dtype=dtype, device=device)
            return out
        out = torch.zeros((batch_size, w, h, 3), dtype=dtype, device=device)
        out[:] = torch.tensor(fill_rgb, dtype=dtype, device=device)
        return out.permute(0, 2, 1, 3)

    def _broadcast_image(self, img, batch_size):
        b = img.shape[0]
        if b == batch_size:
            return img
        if b == 1:
            return img.repeat(batch_size, 1, 1, 1)
        reps = int(math.ceil(batch_size / b))
        tiled = img.repeat(reps, 1, 1, 1)[:batch_size]
        return tiled


class ImageAddLabel:
    """
    为图像添加标签文本 - 支持比例缩放的标签
    标签大小会根据图像尺寸自动调整，确保同比例不同尺寸的图片在缩放后标签大小保持一致
    支持批量图像和批量标签，支持动态引用输入值
    支持 -- 分隔符功能，当存在只包含连字符的行时，-- 之间的内容作为完整标签
    自动选择最佳缩放模式，根据标签方向智能优化
    """

    @classmethod
    def INPUT_TYPES(s):
        # 获取字体目录中的所有字体文件
        font_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fonts")
        font_files = []
        if os.path.exists(font_dir):
            for file in os.listdir(font_dir):
                if file.lower().endswith(('.ttf', '.otf')):
                    font_files.append(file)

        if not font_files:
            font_files = ["FreeMono.ttf"]  # 默认字体

        return {
            "required": {
                "image": ("IMAGE",),
                "height_pad": ("INT", {"default": 24, "min": 0, "max": 1024}),
                "font_size": ("INT", {"default": 36, "min": 1, "max": 256}),
                "invert_color": ("BOOLEAN", {"default": True}),
                "font": (font_files, {"default": "Alibaba-PuHuiTi-Regular.otf"}),
                "text": ("STRING", {"default": "", "multiline": True, "placeholder": "-- splits override separator\nelse use newline."}),
                "direction": (["top", "bottom", "left", "right"], {"default": "top"})
            },
            "optional": {
                "input1": ("STRING", {"default": ""}),
                "input2": ("STRING", {"default": ""})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "image_add_label"
    CATEGORY = "1hewNodes/image"

    def auto_select_scale_mode(self, image_width, image_height, direction="top"):
        """
        自动选择最佳的缩放模式
        根据图像的宽高比、尺寸特征和标签放置方向来智能选择
        """
        aspect_ratio = image_width / image_height
        
        # 根据 direction 优先选择合适的缩放模式
        if direction in ["top", "bottom"]:
            # 顶部/底部标签：标签高度相对于图像高度的比例应该保持一致
            # 优先使用 height 模式，确保标签高度占图像高度的比例在等比例缩放时保持一致
            
            if aspect_ratio > 3.0:
                # 极宽图像：使用高度模式，避免标签过大
                return "height"
            elif aspect_ratio < 0.3:
                # 极高图像：可能需要考虑面积，避免标签过小
                return "area" if min(image_width, image_height) < 512 else "height"
            else:
                # 大多数情况使用高度模式
                return "height"
                
        elif direction in ["left", "right"]:
            # 左侧/右侧标签：标签宽度相对于图像宽度的比例应该保持一致
            # 优先使用 width 模式，确保标签宽度占图像宽度的比例在等比例缩放时保持一致
            
            if aspect_ratio < 0.33:
                # 极高图像：使用宽度模式，避免标签过大
                return "width"
            elif aspect_ratio > 3.0:
                # 极宽图像：可能需要考虑面积，避免标签过小
                return "area" if min(image_width, image_height) < 512 else "width"
            else:
                # 大多数情况使用宽度模式
                return "width"
        
        # 回退到原有逻辑（用于其他未知的 direction 值）
        if 0.8 <= aspect_ratio <= 1.25:
            # 接近正方形的图像 (宽高比在 0.8-1.25 之间)
            return "area"
        elif aspect_ratio > 2.0:
            # 极宽图像
            return "height"
        elif aspect_ratio < 0.5:
            # 极高图像
            return "width"
        else:
            # 默认使用面积模式
            return "area"

    def calculate_scale_factor(self, image_width, image_height, base_font_size, scale_mode=None, direction="top"):
        """
        根据不同的缩放模式计算缩放因子
        如果未指定 scale_mode，则自动选择最佳模式
        """
        if scale_mode is None:
            scale_mode = self.auto_select_scale_mode(image_width, image_height, direction)
        
        # 使用1024作为基准分辨率
        base_resolution = 1024
        
        if scale_mode == "area":
            # 基于面积的缩放 - 推荐模式，确保同比例图像标签大小一致
            base_area = base_resolution * base_resolution
            current_area = image_width * image_height
            scale_factor = math.sqrt(current_area / base_area)
        elif scale_mode == "width":
            # 基于宽度的缩放
            scale_factor = image_width / base_resolution
        elif scale_mode == "height":
            # 基于高度的缩放
            scale_factor = image_height / base_resolution
        elif scale_mode == "min_side":
            # 基于最短边的缩放
            min_side = min(image_width, image_height)
            scale_factor = min_side / base_resolution
        elif scale_mode == "max_side":
            # 基于最长边的缩放
            max_side = max(image_width, image_height)
            scale_factor = max_side / base_resolution
        else:
            scale_factor = 1.0
            
        return scale_factor

    def parse_text_with_inputs(self, text, input1=None, input2=None, batch_index=None, total_batches=None):
        """
        解析文本中的输入引用，支持变量和简单数学运算
        """
        
        parsed_text = text
        
        # 替换 {input1} 引用
        if input1 is not None and input1 != "":
            parsed_text = parsed_text.replace("{input1}", str(input1))
        
        # 替换 {input2} 引用
        if input2 is not None and input2 != "":
            parsed_text = parsed_text.replace("{input2}", str(input2))
        
        # 处理索引相关变量和运算 - 批量标注时生效
        if batch_index is not None and total_batches is not None:
            # 定义变量值
            variables = {
                'index': batch_index,      # 从0开始
                'idx': batch_index,        # 从0开始
                'range': batch_index       # 从0开始
            }
            
            # 使用正则表达式匹配 {变量名} 或 {变量名+数字} 或 {变量名-数字} 等表达式
            pattern = r'\{((?:index|idx|range)(?:[+\-*/]\d+)?)\}'
            
            def replace_expression(match):
                expression = match.group(1)
                try:
                    # 解析表达式
                    if '+' in expression:
                        var_name, operand = expression.split('+')
                        result = variables[var_name.strip()] + int(operand.strip())
                    elif '-' in expression:
                        var_name, operand = expression.split('-')
                        result = variables[var_name.strip()] - int(operand.strip())
                    elif '*' in expression:
                        var_name, operand = expression.split('*')
                        result = variables[var_name.strip()] * int(operand.strip())
                    elif '/' in expression:
                        var_name, operand = expression.split('/')
                        result = variables[var_name.strip()] // int(operand.strip())  # 整数除法
                    else:
                        # 纯变量名
                        result = variables[expression.strip()]
                    
                    # 对于 range 变量，保持补零格式
                    if expression.strip() == 'range' or (expression.startswith('range') and result >= 0):
                        if total_batches >= 100:
                            return f"{result:03d}"
                        elif total_batches >= 10:
                            return f"{result:02d}"
                        else:
                            return str(result)
                    else:
                        # index 和 idx 默认不补零
                        return str(result)
                        
                except (ValueError, KeyError, ZeroDivisionError):
                    # 如果解析失败，返回原始表达式
                    return match.group(0)
            
            parsed_text = re.sub(pattern, replace_expression, parsed_text)
            
        return parsed_text

    def parse_text_list(self, text):
        """
        解析文本列表，支持连字符分割和换行分割
        当有只包含连字符的行时，只按 -- 进行分割，其他分割方式失效
        否则按照换行符(\n) 分割
        """
        
        if not text.strip():
            return [""]
        
        # 检查是否有只包含连字符的行
        lines = text.split('\n')
        has_dash_separator = any(line.strip() and all(c == '-' for c in line.strip()) for line in lines)
        
        if has_dash_separator:
            # 按连字符分割，其他分割方式失效（包括换行符）
            sections = re.split(r'^\s*-+\s*$', text, flags=re.MULTILINE)
            all_lists = []
            
            for section in sections:
                section = section.strip()
                if not section:
                    continue
                    
                # 当有连字符分割时，每个段落作为一个完整项目，保留内部换行
                # 移除引号
                if (section.startswith('"') and section.endswith('"')) or (section.startswith("'") and section.endswith("'")):
                    section = section[1:-1]
                if section:
                    all_lists.append(str(section))
            
            return all_lists if all_lists else [""]
        else:
            # 按传统方式分割（换行符）
            text_lines = text.strip().split('\n')
            # 过滤空行
            text_lines = [line.strip() for line in text_lines if line.strip()]
            return text_lines if text_lines else [""]

    def _calculate_text_size(self, text, font_obj):
        """
        计算文本的尺寸，支持多行文本
        使用固定行高确保一致性
        """
        # 获取固定行高与顶部偏移（带缓存）
        fixed_line_height, text_top_offset = self._get_line_metrics(font_obj)

        # 创建绘制对象测量宽度
        temp_img = Image.new("RGB", (1, 1))
        temp_draw = ImageDraw.Draw(temp_img)

        # 分割文本为多行
        lines = text.split('\n')
        max_width = 0

        # 计算每行的实际宽度，但使用固定行高
        line_heights = []
        for line in lines:
            try:
                text_bbox = temp_draw.textbbox((0, 0), line, font=font_obj)
                line_width = text_bbox[2] - text_bbox[0]
            except AttributeError:
                line_width, _ = temp_draw.textsize(line, font=font_obj)

            max_width = max(max_width, line_width)
            line_heights.append(fixed_line_height)  # 使用固定行高

        total_height = fixed_line_height * len(lines)  # 总高度 = 固定行高 × 行数

        return max_width, total_height, text_top_offset, line_heights

    def _get_line_metrics(self, font_obj):
        """获取字体固定行高与顶部偏移，并做缓存。"""
        cache = getattr(self, "_font_metrics_cache", None)
        if cache is None:
            cache = {}
            setattr(self, "_font_metrics_cache", cache)

        key = id(font_obj)
        if key in cache:
            return cache[key]

        # 优先使用字体度量
        try:
            ascent, descent = font_obj.getmetrics()
            fixed_line_height = ascent + descent
            text_top_offset = 0
            cache[key] = (fixed_line_height, text_top_offset)
            return cache[key]
        except AttributeError:
            pass

        # 回退：用标准字符测量高度
        temp_img = Image.new("RGB", (1, 1))
        temp_draw = ImageDraw.Draw(temp_img)
        standard_chars = (
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"  # noqa: E501
            "!@#$%^&*()_+-=[]{}|;':,.<>?/~`"  # noqa: E501
            "中文测试样本汉字字符高度宽度计算标准参考"  # noqa: E501
        )
        try:
            text_bbox = temp_draw.textbbox((0, 0), standard_chars, font=font_obj)
            fixed_line_height = text_bbox[3] - text_bbox[1]
            text_top_offset = -text_bbox[1]
        except AttributeError:
            _, fixed_line_height = temp_draw.textsize(standard_chars, font=font_obj)
            text_top_offset = 0

        cache[key] = (fixed_line_height, text_top_offset)
        return cache[key]

    def _wrap_text_to_width(self, text, font_obj, max_width):
        """
        根据最大像素宽度自动换行，兼顾中英文与无空格文本。

        按行处理：
        - 优先按空格分词保持单词完整；
        - 对无空格或超长连续字符回退为逐字符包裹。
        """
        if max_width <= 0:
            return text

        # 包裹缓存：同一字体与宽度重复文本直接复用
        wrap_cache = getattr(self, "_wrap_cache", None)
        if wrap_cache is None:
            wrap_cache = {}
            setattr(self, "_wrap_cache", wrap_cache)

        font_id = id(font_obj)
        wrap_key = (text, font_id, int(max_width))
        if wrap_key in wrap_cache:
            return wrap_cache[wrap_key]

        temp_img = Image.new("RGB", (1, 1))
        draw = ImageDraw.Draw(temp_img)

        # 宽度测量缓存，减少重复 textbbox 调用
        measure_cache = getattr(self, "_measure_cache", None)
        if measure_cache is None:
            measure_cache = {}
            setattr(self, "_measure_cache", measure_cache)
        if font_id not in measure_cache:
            measure_cache[font_id] = {}

        def measure(s):
            cache = measure_cache[font_id]
            if s in cache:
                return cache[s]
            try:
                bbox = draw.textbbox((0, 0), s, font=font_obj)
                w = bbox[2] - bbox[0]
            except AttributeError:
                w, _ = draw.textsize(s, font=font_obj)
            cache[s] = w
            return w

        def max_fit_index(line):
            """二分查找可容纳的最大子串长度。"""
            lo, hi = 0, len(line)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if measure(line[:mid]) <= max_width:
                    lo = mid
                else:
                    hi = mid - 1
            return lo

        wrapped_lines = []
        for raw_line in text.split("\n"):
            if not raw_line:
                wrapped_lines.append("")
                continue

            # 如果整行已在宽度内，直接添加
            if measure(raw_line) <= max_width:
                wrapped_lines.append(raw_line)
                continue

            # 优先基于空格分词
            if " " in raw_line or "\t" in raw_line:
                tokens = re.findall(r"\S+\s*", raw_line)
                current = ""
                for token in tokens:
                    candidate = current + token
                    if measure(candidate) <= max_width:
                        current = candidate
                    else:
                        if current.strip():
                            wrapped_lines.append(current.rstrip())
                            current = token.lstrip()
                            # 若新行上的 token 仍超宽，立即按字符切分
                            if measure(current) > max_width:
                                idx = max_fit_index(current)
                                if idx > 0:
                                    wrapped_lines.append(current[:idx])
                                    current = current[idx:]
                                # 如果仍有剩余且再次超宽，循环处理
                                while current and measure(current) > max_width:
                                    idx = max_fit_index(current)
                                    if idx == 0:
                                        # 极端情况，至少放一个字符
                                        wrapped_lines.append(current[0])
                                        current = current[1:]
                                    else:
                                        wrapped_lines.append(current[:idx])
                                        current = current[idx:]
                        else:
                            # 单个token过长，回退字符级拆分
                            rest = token
                            while rest:
                                idx = max_fit_index(rest)
                                if idx == 0:
                                    # 至少输出一个字符，避免死循环
                                    if current:
                                        wrapped_lines.append(current)
                                        current = ""
                                    wrapped_lines.append(rest[0])
                                    rest = rest[1:]
                                else:
                                    cand = current + rest[:idx]
                                    if measure(cand) <= max_width:
                                        current = cand
                                    else:
                                        if current:
                                            wrapped_lines.append(current)
                                        wrapped_lines.append(rest[:idx])
                                        current = ""
                                    rest = rest[idx:]
                if current:
                    wrapped_lines.append(current)
            else:
                # 无空格文本，逐字符包裹（适合中文等情况）
                line = raw_line
                while line:
                    idx = max_fit_index(line)
                    if idx == 0:
                        # 至少输出一个字符
                        wrapped_lines.append(line[0])
                        line = line[1:]
                    else:
                        wrapped_lines.append(line[:idx])
                        line = line[idx:]
        wrapped = "\n".join(wrapped_lines)
        wrap_cache[wrap_key] = wrapped
        return wrapped

    def _draw_multiline_text(self, draw, text, x, y, font_obj, font_color, line_heights):
        """
        绘制多行文本，使用固定行高
        """
        lines = text.split('\n')
        current_y = y
        
        for i, line in enumerate(lines):
            draw.text((x, current_y), line, fill=font_color, font=font_obj)
            if i < len(line_heights):
                current_y += line_heights[i]  # 使用固定行高间距

    def _load_font(self, font, font_size):
        """
        加载字体对象
        """
        try:
            font_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fonts", font)
            if not os.path.exists(font_path):
                system_font_dirs = [
                    "C:/Windows/Fonts",
                    "/usr/share/fonts",
                    "/System/Library/Fonts"
                ]
                for font_dir in system_font_dirs:
                    if os.path.exists(os.path.join(font_dir, font)):
                        font_path = os.path.join(font_dir, font)
                        break
            font_obj = ImageFont.truetype(font_path, font_size)
        except Exception as e:
            print(f"无法加载字体 {font}: {e}，使用默认字体")
            font_obj = ImageFont.load_default()
        return font_obj

    def image_add_label(self, image, height_pad, font_size, invert_color, font, text, direction, input1=None, input2=None):
        # 设置颜色，根据invert_color决定黑白配色
        if invert_color:
            font_color = "black"
            label_color = "white"
        else:
            font_color = "white"
            label_color = "black"

        result = []
        total_batches = len(image)
        
        # 缓存字体对象和文本尺寸
        font_cache = {}
        text_size_cache = {}
        scale_factor_cache = {}
        
        # 预处理所有文本，获取每张图片对应的文本
        all_current_texts = []
        all_scale_factors = []
        all_font_sizes = []
        all_height_pads = []
        all_selected_modes = []
        
        for i in range(total_batches):
            # 为每个批次解析文本，包含 {range} 替换
            parsed_text = self.parse_text_with_inputs(text, input1, input2, i, total_batches)
            
            # 解析文本列表并获取当前文本
            text_lines = self.parse_text_list(parsed_text)
            current_text = text_lines[i % len(text_lines)] if text_lines else ""
            all_current_texts.append(current_text)
            
            # 获取当前图像尺寸并计算缩放因子
            img_data = 255. * image[i].cpu().numpy()
            img_pil = Image.fromarray(np.clip(img_data, 0, 255).astype(np.uint8))
            width, height = img_pil.size
            
            # 使用缓存的缩放因子计算
            size_key = (width, height, direction)
            if size_key not in scale_factor_cache:
                selected_mode = self.auto_select_scale_mode(width, height, direction)
                scale_factor = self.calculate_scale_factor(width, height, font_size, selected_mode, direction)
                scale_factor_cache[size_key] = (selected_mode, scale_factor)
            else:
                selected_mode, scale_factor = scale_factor_cache[size_key]
                
            all_scale_factors.append(scale_factor)
            all_selected_modes.append(selected_mode)
            
            # 计算缩放后的字体大小和高度填充
            scaled_font_size = max(8, int(font_size * scale_factor))
            scaled_height_pad = max(4, int(height_pad * scale_factor))
            
            all_font_sizes.append(scaled_font_size)
            all_height_pads.append(scaled_height_pad)
        
        # 批量转换图像为PIL格式，减少重复转换
        pil_images = []
        for i, img in enumerate(image):
            img_data = 255. * img.cpu().numpy()
            img_pil = Image.fromarray(np.clip(img_data, 0, 255).astype(np.uint8))
            pil_images.append(img_pil)
        
        # 处理每张图片
        for i, img_pil in enumerate(pil_images):
            current_text = all_current_texts[i]
            scaled_font_size = all_font_sizes[i]
            scaled_height_pad = all_height_pads[i]
            
            # 使用缓存的字体对象
            font_key = (font, scaled_font_size)
            if font_key not in font_cache:
                font_cache[font_key] = self._load_font(font, scaled_font_size)
            font_obj = font_cache[font_key]
            
            # 计算内容最大宽度并执行自动换行
            width, orig_height = img_pil.size
            scale_factor = all_scale_factors[i]
            text_margin = max(10, int(10 * scale_factor))
            if direction in ["top", "bottom"]:
                max_content_width = max(1, width - 2 * text_margin)
            else:
                max_content_width = max(1, orig_height - 2 * text_margin)

            # 复用包裹缓存，避免重复计算
            wrap_cache = getattr(self, "_wrap_cache", None)
            if wrap_cache is None:
                wrap_cache = {}
                setattr(self, "_wrap_cache", wrap_cache)
            wrap_key = (current_text, id(font_obj), int(max_content_width))
            if wrap_key in wrap_cache:
                wrapped_text = wrap_cache[wrap_key]
            else:
                wrapped_text = self._wrap_text_to_width(
                    current_text, font_obj, max_content_width
                )

            # 使用缓存的文本尺寸计算（基于换行后文本）
            text_key = (wrapped_text, font_key)
            if text_key not in text_size_cache:
                text_size_cache[text_key] = self._calculate_text_size(wrapped_text, font_obj)
            text_width, text_height, text_top_offset, line_heights = text_size_cache[text_key]
            
            min_padding = max(scaled_height_pad, 4)
            label_height = text_height + min_padding

            # 创建标签区域
            if direction in ["top", "bottom"]:
                # 创建标签图像
                label_img = Image.new("RGB", (width, label_height), label_color)
                draw = ImageDraw.Draw(label_img)

                # 计算文本位置 - 左对齐，空出比例化的边距
                text_x = text_margin
                text_y = min_padding // 2 + text_top_offset

                # 绘制多行文本
                self._draw_multiline_text(draw, wrapped_text, text_x, text_y, font_obj, font_color, line_heights)

                # 合并图像和标签
                if direction == "top":
                    new_img = Image.new("RGB", (width, orig_height + label_height))
                    new_img.paste(label_img, (0, 0))
                    new_img.paste(img_pil, (0, label_height))
                else:  # bottom
                    new_img = Image.new("RGB", (width, orig_height + label_height))
                    new_img.paste(img_pil, (0, 0))
                    new_img.paste(label_img, (0, orig_height))
                    
            else:  # left or right
                if direction == "left":
                    temp_label_img = Image.new("RGB", (orig_height, label_height), label_color)
                    draw = ImageDraw.Draw(temp_label_img)

                    # 计算文本位置 - 左对齐，空出比例化的边距
                    text_x = text_margin
                    text_y = min_padding // 2 + text_top_offset

                    # 绘制多行文本（保持一致性）
                    self._draw_multiline_text(draw, wrapped_text, text_x, text_y, font_obj, font_color, line_heights)

                    # 旋转标签图像逆时针90度
                    label_img = temp_label_img.rotate(90, expand=True)

                    # 合并图像和标签
                    new_img = Image.new("RGB", (width + label_height, orig_height))
                    new_img.paste(label_img, (0, 0))
                    new_img.paste(img_pil, (label_height, 0))

                else:  # right
                    temp_label_img = Image.new("RGB", (orig_height, label_height), label_color)
                    draw = ImageDraw.Draw(temp_label_img)

                    # 计算文本位置 - 左对齐，空出比例化的边距
                    text_x = text_margin
                    text_y = min_padding // 2 + text_top_offset

                    # 绘制多行文本（保持一致性）
                    self._draw_multiline_text(draw, wrapped_text, text_x, text_y, font_obj, font_color, line_heights)

                    # 旋转标签图像顺时针90度（即逆时针270度）
                    label_img = temp_label_img.rotate(270, expand=True)

                    # 合并图像和标签
                    new_img = Image.new("RGB", (width + label_height, orig_height))
                    new_img.paste(img_pil, (0, 0))
                    new_img.paste(label_img, (width, 0))
    
            # 转换回tensor
            img_np = np.array(new_img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np)[None,]
            result.append(img_tensor)
    
        return (torch.cat(result, dim=0),)


class ImagePlot:
    """
    支持单张图像和批量图片收集，将输入按指定布局排列显示
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "layout": (["horizontal", "vertical", "grid"], {"default": "horizontal"}),
                "spacing": ("INT", {"default": 10, "min": 0, "max": 100}),
                "grid_columns": ("INT", {"default": 2, "min": 1, "max": 100}),
                "background_color": ("STRING", {"default": "1.0"})
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "image_plot"
    CATEGORY = "1hewNodes/image"
    
    def image_plot(self, image, layout, spacing, grid_columns, background_color):
        """
        主处理函数，自动检测输入类型并选择相应的处理方式
        """
        # 自动检测输入类型并选择处理方式
        if self._is_video_collection(image):
            return self._process_video_collection(image, layout, spacing, grid_columns, background_color)
        else:
            return self._process_standard_plot(image, layout, spacing, grid_columns, background_color)
    
    def _is_video_collection(self, image):
        """自动检测是否为视频收集数据"""
        # 只有当输入为列表格式时才认为是视频收集数据
        if isinstance(image, list):
            return True
        # 移除基于帧数的检测，避免误判
        return False
    
    def _process_standard_plot(self, image, layout, spacing, grid_columns, background_color):
        """标准图像拼接处理"""
        # 解析背景颜色
        bg_color = self._parse_color(background_color)
        
        # 获取图像数量
        num_images = image.shape[0]
        
        # 将所有图像转换为PIL格式
        pil_images = []
        for i in range(num_images):
            img = 255. * image[i].cpu().numpy()
            pil_img = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
            pil_images.append(pil_img)
        
        # 根据布局处理图像
        result_img = self._combine_images(pil_images, layout, spacing, grid_columns, bg_color)
        
        # 转换回tensor
        result_np = np.array(result_img).astype(np.float32) / 255.0
        result_tensor = torch.from_numpy(result_np)[None,]
        
        return (result_tensor,)
    
    def _process_video_collection(self, image, layout, spacing, grid_columns, background_color):
        """视频收集处理，支持多批次图像的时间序列显示"""
        try:
            # 处理输入数据
            batch_list = self._extract_batch_list(image)
            if not batch_list:
                print("警告: 没有找到有效的批量图片")
                return (torch.zeros(1, 256, 256, 3),)
            
            print(f"接收到{len(batch_list)}个批量图片组")
            
            # 获取所有批次的最大帧数
            max_frames = max(batch.shape[0] for batch in batch_list)
            print(f"最大帧数: {max_frames}")
            
            # 为每一帧创建并列显示
            combined_frames = []
            for frame_idx in range(max_frames):
                frame_images = []
                
                # 从每个批次中提取对应帧（循环使用如果帧数不足）
                for batch in batch_list:
                    actual_idx = frame_idx % batch.shape[0]
                    frame_images.append(batch[actual_idx])
                
                # 合并当前帧的所有图片
                combined_frame = self._combine_frame_images(
                    frame_images, layout, spacing, grid_columns, background_color
                )
                combined_frames.append(combined_frame)
            
            # 合并所有帧
            result_tensor = torch.stack(combined_frames, dim=0)
            print(f"输出形状: {result_tensor.shape}")
            
            return (result_tensor,)
            
        except Exception as e:
            print(f"视频收集显示错误: {str(e)}")
            return (torch.zeros(1, 256, 256, 3),)
    
    def _extract_batch_list(self, video_collection):
        """从video_collection中提取批量图片列表"""
        if isinstance(video_collection, list):
            return [item for item in video_collection if isinstance(item, torch.Tensor)]
        elif isinstance(video_collection, torch.Tensor):
            return [video_collection]
        else:
            print(f"未知的输入类型: {type(video_collection)}")
            return []
    
    def _combine_frame_images(self, frame_images, layout, spacing, grid_columns, background_color):
        """合并单帧的多个图片"""
        if not frame_images:
            return torch.zeros(256, 256, 3)
        
        # 统一图片尺寸
        normalized_images = self._normalize_image_sizes(frame_images)
        
        # 转换为PIL图像
        pil_images = []
        for img_tensor in normalized_images:
            img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            pil_images.append(pil_img)
        
        # 解析背景颜色
        bg_color = self._parse_color(background_color)
        
        # 合并图像
        result_img = self._combine_images(pil_images, layout, spacing, grid_columns, bg_color)
        
        # 转换回tensor
        result_np = np.array(result_img).astype(np.float32) / 255.0
        return torch.from_numpy(result_np)
    
    def _combine_images(self, pil_images, layout, spacing, grid_columns, bg_color):
        """统一的图像合并逻辑"""
        if not pil_images:
            return Image.new('RGB', (256, 256), bg_color)
        
        # 获取所有图像的尺寸
        widths = [img.width for img in pil_images]
        heights = [img.height for img in pil_images]
        num_images = len(pil_images)
        
        if layout == "horizontal":
            # 水平排列
            total_width = sum(widths) + spacing * (num_images - 1)
            max_height = max(heights)
            result_img = Image.new("RGB", (total_width, max_height), bg_color)
            
            x_offset = 0
            for img in pil_images:
                y_offset = (max_height - img.height) // 2
                result_img.paste(img, (x_offset, y_offset))
                x_offset += img.width + spacing
                
        elif layout == "vertical":
            # 垂直排列
            max_width = max(widths)
            total_height = sum(heights) + spacing * (num_images - 1)
            result_img = Image.new("RGB", (max_width, total_height), bg_color)
            
            y_offset = 0
            for img in pil_images:
                x_offset = (max_width - img.width) // 2
                result_img.paste(img, (x_offset, y_offset))
                y_offset += img.height + spacing
                
        else:  # "grid" - 网格模式
            # 使用grid_columns参数确定网格尺寸
            cols = grid_columns
            rows = math.ceil(num_images / cols)
            
            # 计算每行每列的最大尺寸
            max_width_per_col = []
            for col in range(cols):
                col_images = [pil_images[i] for i in range(num_images) if i % cols == col]
                max_width_per_col.append(max([img.width for img in col_images]) if col_images else 0)
                
            max_height_per_row = []
            for row in range(rows):
                row_images = [pil_images[i] for i in range(num_images) if i // cols == row]
                max_height_per_row.append(max([img.height for img in row_images]) if row_images else 0)
            
            # 计算总宽度和总高度
            total_width = sum(max_width_per_col) + spacing * (cols - 1)
            total_height = sum(max_height_per_row) + spacing * (rows - 1)
            
            result_img = Image.new("RGB", (total_width, total_height), bg_color)
            
            # 放置图像
            for i, img in enumerate(pil_images[:rows * cols]):
                row = i // cols
                col = i % cols
                
                # 计算当前位置的x和y偏移
                x_offset = sum(max_width_per_col[:col]) + spacing * col
                y_offset = sum(max_height_per_row[:row]) + spacing * row
                
                # 在当前单元格内居中
                x_center = (max_width_per_col[col] - img.width) // 2
                y_center = (max_height_per_row[row] - img.height) // 2
                
                result_img.paste(img, (x_offset + x_center, y_offset + y_center))
        
        return result_img
    
    def _normalize_image_sizes(self, images):
        """统一图片尺寸到最小公共尺寸"""
        if not images:
            return []
        
        # 找到最小尺寸
        min_height = min(img.shape[0] for img in images)
        min_width = min(img.shape[1] for img in images)
        
        normalized = []
        for img in images:
            if img.shape[0] != min_height or img.shape[1] != min_width:
                # 调整尺寸
                img = img.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
                img = F.interpolate(img, size=(min_height, min_width), mode='bilinear', align_corners=False)
                img = img.squeeze(0).permute(1, 2, 0)  # [H, W, C]
            normalized.append(img)
        
        return normalized
    
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
        try:
            rgb = color_str.split(',')
            if len(rgb) == 3:
                r = int(rgb[0].strip())
                g = int(rgb[1].strip())
                b = int(rgb[2].strip())
                return (r, g, b)
        except ValueError:
            pass
        
        # 默认返回白色
        return (255, 255, 255)


class ImageStrokeByMask:
    """
    图像遮罩描边节点
    对输入图像的指定遮罩区域进行描边处理
    """
    
    def __init__(self):
        self.NODE_NAME = 'ImageStrokeByMask'
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # 输入图像
                "mask": ("MASK",),   # 输入遮罩
                "stroke_width": ("INT", {"default": 20, "min": 0, "max": 1000, "step": 1}),  # 描边宽度
                "stroke_color": ("STRING", {"default": "1.0"}),  # 描边颜色
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "stroke_by_mask"
    CATEGORY = "1hewNodes/image"
    
    def tensor_to_pil(self, tensor):
        """将tensor转换为PIL图像，自动处理alpha通道"""
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        if tensor.dim() == 3:
            if tensor.shape[2] == 3:
                # RGB图像
                np_img = (tensor.numpy() * 255).astype(np.uint8)
                return Image.fromarray(np_img, 'RGB')
            elif tensor.shape[2] == 4:
                # RGBA图像，去除alpha通道
                tensor = tensor[:, :, :3]  # 只保留RGB通道
                np_img = (tensor.numpy() * 255).astype(np.uint8)
                return Image.fromarray(np_img, 'RGB')
            elif tensor.shape[0] == 1:
                # 批次维度的遮罩 [1, H, W]
                tensor = tensor.squeeze(0)
                np_img = (tensor.numpy() * 255).astype(np.uint8)
                return Image.fromarray(np_img, 'L')
            else:
                # 可能是 [H, W, 1] 格式的遮罩
                if tensor.shape[2] == 1:
                    tensor = tensor.squeeze(2)
                    np_img = (tensor.numpy() * 255).astype(np.uint8)
                    return Image.fromarray(np_img, 'L')
                else:
                    raise ValueError(f"不支持的3维tensor形状: {tensor.shape}")
        elif tensor.dim() == 2:
            # 灰度图像/遮罩
            np_img = (tensor.numpy() * 255).astype(np.uint8)
            return Image.fromarray(np_img, 'L')
        else:
            raise ValueError(f"不支持的tensor维度: {tensor.shape}")
    
    def pil_to_tensor(self, pil_img):
        """将PIL图像转换为tensor"""
        if pil_img.mode == 'RGB':
            np_img = np.array(pil_img).astype(np.float32) / 255.0
            return torch.from_numpy(np_img).unsqueeze(0)
        elif pil_img.mode == 'L':
            np_img = np.array(pil_img).astype(np.float32) / 255.0
            return torch.from_numpy(np_img).unsqueeze(0)  # 添加批次维度 [1, H, W]
        else:
            raise ValueError(f"不支持的PIL图像模式: {pil_img.mode}")
    
    def parse_color(self, color_str):
        """解析颜色字符串，支持多种格式"""
        color_str = color_str.strip()
        
        # 移除括号（如果存在）
        if color_str.startswith('(') and color_str.endswith(')'):
            color_str = color_str[1:-1].strip()
        
        # 尝试解析为浮点数（灰度值）
        try:
            gray_value = float(color_str)
            if 0.0 <= gray_value <= 1.0:
                # 0-1范围的灰度值
                gray_int = int(gray_value * 255)
                return (gray_int, gray_int, gray_int)
            elif gray_value > 1.0 and gray_value <= 255:
                # 可能是0-255范围的灰度值
                gray_int = int(gray_value)
                gray_int = max(0, min(255, gray_int))
                return (gray_int, gray_int, gray_int)
        except ValueError:
            pass
        
        # 尝试解析为 RGB 格式 (如 "0.5,0.7,0.9" 或 "128,192,255")
        if ',' in color_str:
            try:
                # 分割并清理每个部分
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
        
        # 尝试使用 PIL.ImageColor 解析（支持 HEX、RGB、颜色名称等）
        try:
            return ImageColor.getrgb(color_str)
        except ValueError:
            pass
        
        # 颜色映射
        color_map = {
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "yellow": (255, 255, 0),
            "cyan": (0, 255, 255),
            "magenta": (255, 0, 255),
            "white": (255, 255, 255),
            "black": (0, 0, 0)
        }
        
        return color_map.get(color_str.lower(), (255, 255, 255))
    
    def create_stroke_mask(self, mask_pil, stroke_width):
        """创建描边遮罩 - 简化逻辑：直接对原始遮罩膨胀创建描边区域"""
        # 将PIL遮罩转换为numpy数组
        mask_np = np.array(mask_pil)
        
        # 使用形态学操作创建描边
        stroke_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (stroke_width*2+1, stroke_width*2+1))
        
        # 膨胀原始遮罩创建描边外边界
        dilated_mask = cv2.dilate(mask_np, stroke_kernel, iterations=1)
        
        # 描边区域 = 膨胀后的遮罩 - 原始遮罩
        stroke_mask = dilated_mask - mask_np
        
        # 确保值在0-255范围内
        stroke_mask = np.clip(stroke_mask, 0, 255)
        
        # 返回描边遮罩
        stroke_mask_pil = Image.fromarray(stroke_mask.astype(np.uint8), 'L')
        
        return stroke_mask_pil
    
    def apply_stroke_mask(self, image, mask, stroke_width, stroke_color):
        """应用描边遮罩效果 - 支持批处理，简化逻辑"""
        # 获取批次大小
        batch_size = image.shape[0]
        mask_batch_size = mask.shape[0]
        
        # 确定最大批次大小
        max_batch_size = max(batch_size, mask_batch_size)
        
        output_images = []
        output_masks = []
        
        for i in range(max_batch_size):
            # 使用循环索引获取对应的输入
            img_idx = i % batch_size
            mask_idx = i % mask_batch_size
            
            # 转换当前批次的输入
            current_image = image[img_idx:img_idx+1]  # 保持4D格式
            current_mask = mask[mask_idx:mask_idx+1]   # 保持3D格式
            
            image_pil = self.tensor_to_pil(current_image)
            mask_pil = self.tensor_to_pil(current_mask)
            
            # 确保图像和遮罩尺寸一致
            if image_pil.size != mask_pil.size:
                mask_pil = mask_pil.resize(image_pil.size, Image.LANCZOS)
            
            # 解析描边颜色
            stroke_rgb = self.parse_color(stroke_color)
            
            # 创建描边遮罩
            stroke_mask_pil = self.create_stroke_mask(mask_pil, stroke_width)
            
            # 创建输出图像（黑色背景）
            output_image = Image.new('RGB', image_pil.size, (0, 0, 0))
            
            # 先填充描边区域为指定颜色
            stroke_color_image = Image.new('RGB', image_pil.size, stroke_rgb)
            output_image.paste(stroke_color_image, mask=stroke_mask_pil)
            
            # 再粘贴原始图像内容到原始遮罩区域
            output_image.paste(image_pil, mask=mask_pil)
            
            # 创建输出遮罩（先填充描边，然后在描边mask基础上直接填充，确保满fill）
            stroke_mask_np = np.array(stroke_mask_pil)
            mask_np = np.array(mask_pil)
            
            # 先创建描边遮罩作为基础
            output_mask_np = stroke_mask_np.copy()
            
            # 在描边mask基础上直接填充原始mask区域为满值（255）
            output_mask_np[mask_np > 0] = 255
            
            output_mask_pil = Image.fromarray(output_mask_np.astype(np.uint8), 'L')
            
            # 转换回tensor并添加到列表
            output_image_tensor = self.pil_to_tensor(output_image)
            output_mask_tensor = self.pil_to_tensor(output_mask_pil)
            
            output_images.append(output_image_tensor)
            output_masks.append(output_mask_tensor)
        
        # 合并批次
        final_images = torch.cat(output_images, dim=0)
        final_masks = torch.cat(output_masks, dim=0)
        
        return (final_images, final_masks)
    
    def stroke_by_mask(self, image, mask, stroke_width, stroke_color):
        """主函数：对图像进行遮罩描边处理"""
        return self.apply_stroke_mask(image, mask, stroke_width, stroke_color)


class ImageBBoxOverlayByMask:
    """
    基于遮罩的图像边界框叠加节点 - 根据遮罩生成检测框并以描边或填充形式叠加到图像上
    支持独立模式和合并模式
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "bbox_color": (["red", "green", "blue", "yellow", "cyan", "magenta", "white", "black"], {"default": "green"}),
                "stroke_width": ("INT", {"default": 4, "min": 1, "max": 100, "step": 1}),
                "fill": ("BOOLEAN", {"default": True}),
                "padding": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "output_mode": (["separate", "merge"], {"default": "separate"})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "overlay_bbox"
    CATEGORY = "1hewNodes/image"

    def overlay_bbox(self, image, mask, bbox_color="green", stroke_width=3, fill=False, padding=0, output_mode="separate"):
        # 确保输入是正确的维度
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
            
        batch_size = image.shape[0]
        mask_batch_size = mask.shape[0]
        
        # 颜色映射
        color_map = {
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "yellow": (255, 255, 0),
            "cyan": (0, 255, 255),
            "magenta": (255, 0, 255),
            "white": (255, 255, 255),
            "black": (0, 0, 0)
        }
        
        bbox_rgb = color_map.get(bbox_color, (255, 0, 0))
        
        output_images = []
        
        for b in range(batch_size):
            # 获取当前批次的图像
            current_image = image[b]
            
            # 转换图像为PIL格式
            if current_image.is_cuda:
                img_np = (current_image.cpu().numpy() * 255).astype(np.uint8)
            else:
                img_np = (current_image.numpy() * 255).astype(np.uint8)
            
            img_pil = Image.fromarray(img_np, 'RGB')
            draw = ImageDraw.Draw(img_pil)
            
            if output_mode == "merge":
                # 合并模式：将所有mask合并为一个边界框
                combined_mask = None
                for m in range(mask_batch_size):
                    mask_idx = m % mask_batch_size
                    current_mask = mask[mask_idx]
                    
                    # 转换遮罩为numpy格式
                    if current_mask.is_cuda:
                        mask_np = (current_mask.cpu().numpy() * 255).astype(np.uint8)
                    else:
                        mask_np = (current_mask.numpy() * 255).astype(np.uint8)
                    
                    if combined_mask is None:
                        combined_mask = mask_np
                    else:
                        # 合并mask（取最大值）
                        combined_mask = np.maximum(combined_mask, mask_np)
                
                # 调整合并后的遮罩尺寸
                mask_pil = Image.fromarray(combined_mask, 'L')
                if mask_pil.size != img_pil.size:
                    mask_pil = mask_pil.resize(img_pil.size, Image.Resampling.LANCZOS)
                
                # 获取合并后的边界框
                bbox = self.get_single_bbox_from_mask(mask_pil, padding)
                if bbox is not None:
                    x_min, y_min, x_max, y_max = bbox
                    self.draw_bbox(draw, x_min, y_min, x_max, y_max, bbox_rgb, stroke_width, fill)
                    print(f"合并边界框: ({x_min}, {y_min}, {x_max}, {y_max})")
            else:
                # 独立模式：为每个独立的mask区域生成单独的边界框
                all_bboxes = []
                for m in range(mask_batch_size):
                    mask_idx = m % mask_batch_size
                    current_mask = mask[mask_idx]
                    
                    # 转换遮罩为numpy格式
                    if current_mask.is_cuda:
                        mask_np = (current_mask.cpu().numpy() * 255).astype(np.uint8)
                    else:
                        mask_np = (current_mask.numpy() * 255).astype(np.uint8)
                    
                    mask_pil = Image.fromarray(mask_np, 'L')
                    
                    # 调整遮罩尺寸以匹配图像
                    if mask_pil.size != img_pil.size:
                        mask_pil = mask_pil.resize(img_pil.size, Image.Resampling.LANCZOS)
                    
                    # 获取当前mask的所有独立区域的边界框
                    bboxes = self.get_multiple_bboxes_from_mask(mask_pil, padding)
                    all_bboxes.extend(bboxes)
                
                # 绘制所有边界框
                for i, bbox in enumerate(all_bboxes):
                    x_min, y_min, x_max, y_max = bbox
                    self.draw_bbox(draw, x_min, y_min, x_max, y_max, bbox_rgb, stroke_width, fill)
                    print(f"独立边界框 {i+1}: ({x_min}, {y_min}, {x_max}, {y_max})")
            
            # 转换回tensor
            result_np = np.array(img_pil).astype(np.float32) / 255.0
            result_tensor = torch.from_numpy(result_np)
            output_images.append(result_tensor)
        
        # 合并批次
        output_tensor = torch.stack(output_images)
        
        return (output_tensor,)
    
    def draw_bbox(self, draw, x_min, y_min, x_max, y_max, color, stroke_width, fill):
        """绘制边界框，支持填充和描边模式"""
        if fill:
            # 填充模式：绘制填充的矩形
            draw.rectangle(
                [x_min, y_min, x_max, y_max],
                fill=color
            )
        else:
            # 描边模式：仅绘制边框
            draw.rectangle(
                [x_min, y_min, x_max, y_max],
                outline=color,
                width=stroke_width
            )
    
    def get_single_bbox_from_mask(self, mask_pil, padding=0):
        """从遮罩中提取单个边界框坐标（合并所有白色区域）"""
        mask_np = np.array(mask_pil)
        
        # 创建二值遮罩（mask值大于128认为是有效区域）
        binary_mask = mask_np > 128
        
        # 找到有效像素的坐标
        y_coords, x_coords = np.where(binary_mask)
        
        if len(y_coords) == 0 or len(x_coords) == 0:
            return None
        
        # 获取边界框坐标
        x_min = int(np.min(x_coords))
        x_max = int(np.max(x_coords))
        y_min = int(np.min(y_coords))
        y_max = int(np.max(y_coords))
        
        # 添加填充
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(mask_pil.width - 1, x_max + padding)
        y_max = min(mask_pil.height - 1, y_max + padding)
        
        return (x_min, y_min, x_max, y_max)
    
    def get_multiple_bboxes_from_mask(self, mask_pil, padding=0):
        """从遮罩中提取多个独立区域的边界框坐标"""
        mask_np = np.array(mask_pil)
        
        # 创建二值遮罩
        binary_mask = mask_np > 128
        
        if not np.any(binary_mask):
            return []
        
        # 使用连通组件分析找到独立的区域
        labeled_mask = label(binary_mask)
        regions = regionprops(labeled_mask)
        
        bboxes = []
        for region in regions:
            # 获取区域的边界框
            min_row, min_col, max_row, max_col = region.bbox
            
            # 添加填充
            x_min = max(0, min_col - padding)
            y_min = max(0, min_row - padding)
            x_max = min(mask_pil.width - 1, max_col - 1 + padding)
            y_max = min(mask_pil.height - 1, max_row - 1 + padding)
            
            bboxes.append((x_min, y_min, x_max, y_max))
        
        return bboxes


NODE_CLASS_MAPPINGS = {
    "1hew_ImageGetSize": ImageGetSize,
    "1hew_ImageSolidFluxKontext": ImageSolidFluxKontext,
    "1hew_ImageSolidQwenImage": ImageSolidQwenImage,
    "1hew_ImageSolid": ImageSolid,
    "1hew_ImageResizeFluxKontext": ImageResizeFluxKontext,
    "1hew_ImageResizeQwenImage": ImageResizeQwenImage,
    "1hew_ImageResizeUniversal": ImageResizeUniversal,
    "1hew_ImageRotateWithMask": ImageRotateWithMask,
    "1hew_ImageEditStitch": ImageEditStitch,
    "1hew_ImageThreeStitch": ImageThreeStitch,
    "1hew_ImageAddLabel": ImageAddLabel,
    "1hew_ImagePlot": ImagePlot,
    "1hew_ImageStrokeByMask": ImageStrokeByMask,
    "1hew_ImageBBoxOverlayByMask": ImageBBoxOverlayByMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "1hew_ImageGetSize": "Image Get Size",
    "1hew_ImageSolidFluxKontext": "Image Solid FluxKontext",
    "1hew_ImageSolidQwenImage": "Image Solid QwenImage",
    "1hew_ImageSolid": "Image Solid",
    "1hew_ImageResizeFluxKontext": "Image Resize FluxKontext",
    "1hew_ImageResizeQwenImage": "Image Resize QwenImage",
    "1hew_ImageResizeUniversal": "Image Resize Universal",
    "1hew_ImageRotateWithMask": "Image Rotate with Mask",
    "1hew_ImageEditStitch": "Image Edit Stitch",
    "1hew_ImageThreeStitch": "Image Three Stitch",
    "1hew_ImageAddLabel": "Image Add Label",
    "1hew_ImagePlot": "Image Plot",
    "1hew_ImageStrokeByMask": "Image Stroke by Mask",
    "1hew_ImageBBoxOverlayByMask": "Image BBox Overlay by Mask",
}