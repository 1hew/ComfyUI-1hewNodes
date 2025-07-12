import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
import math
from fractions import Fraction

class ImageCropSquare:
    """
    图像方形裁剪器 - 根据遮罩裁切图像为方形，支持放大系数和填充颜色
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "scale_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.01}),
                "apply_mask": ("BOOLEAN", {"default": False}),
                "extra_padding": ("INT", {"default": 0, "min": 0, "max": 512, "step": 1}),
                "fill_color": ("STRING", {"default": "1.0"}),
                "divisible_by": ("INT", {"default": 8, "min": 1, "max": 1024, "step": 1})
            },
            "optional": {
                "mask": ("MASK",)
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "image_crop_square"
    CATEGORY = "1hewNodes/image/crop"

    def image_crop_square(self, image, scale_factor=1.0, fill_color="1.0", apply_mask=False, extra_padding=0, divisible_by=8, mask=None):
        # 获取图像尺寸
        batch_size, height, width, channels = image.shape

        # 创建输出图像列表
        output_images = []

        for b in range(batch_size):
            # 将图像转换为PIL格式
            if image.is_cuda:
                img_np = (image[b].cpu().numpy() * 255).astype(np.uint8)
                # 处理可选的mask参数
                if mask is not None:
                    mask_np = (mask[b % mask.shape[0]].cpu().numpy() * 255).astype(np.uint8)
                else:
                    # 创建空遮罩
                    mask_np = np.zeros((height, width), dtype=np.uint8)
            else:
                img_np = (image[b].numpy() * 255).astype(np.uint8)
                # 处理可选的mask参数
                if mask is not None:
                    mask_np = (mask[b % mask.shape[0]].numpy() * 255).astype(np.uint8)
                else:
                    # 创建空遮罩
                    mask_np = np.zeros((height, width), dtype=np.uint8)

            img_pil = Image.fromarray(img_np)
            mask_pil = Image.fromarray(mask_np).convert("L")

            # 调整遮罩大小以匹配图像 - 使用填充而非缩放
            if img_pil.size != mask_pil.size:
                # 创建一个与图像相同大小的空白遮罩
                new_mask = Image.new("L", img_pil.size, 0)

                # 计算居中位置
                paste_x = max(0, (img_pil.width - mask_pil.width) // 2)
                paste_y = max(0, (img_pil.height - mask_pil.height) // 2)

                # 将原始遮罩粘贴到中心位置
                new_mask.paste(mask_pil, (paste_x, paste_y))
                mask_pil = new_mask

            # 找到遮罩中非零区域的边界框
            bbox = self.get_bbox(mask_pil)

            # 如果没有找到有效区域（遮罩为空），执行居中方形裁剪
            if bbox is None:
                # 计算最大方形尺寸
                square_size = min(img_pil.width, img_pil.height)
                
                # 应用放大系数
                scaled_size = int(square_size * scale_factor)
                
                # 计算居中裁剪的坐标
                center_x = img_pil.width // 2
                center_y = img_pil.height // 2
                
                crop_x1 = center_x - scaled_size // 2
                crop_y1 = center_y - scaled_size // 2
                crop_x2 = crop_x1 + scaled_size
                crop_y2 = crop_y1 + scaled_size
                
                # 确保裁剪坐标在图像范围内
                crop_x1 = max(0, crop_x1)
                crop_y1 = max(0, crop_y1)
                crop_x2 = min(img_pil.width, crop_x2)
                crop_y2 = min(img_pil.height, crop_y2)
                
                # 最终尺寸（包含额外边距）
                final_size = scaled_size + extra_padding * 2
                
                # 确保最终尺寸是divisible_by的倍数
                final_size = (final_size // divisible_by) * divisible_by
                if final_size <= 0:
                    final_size = divisible_by
                
                # 解析填充颜色
                bg_color = self.parse_color(fill_color)
                
                # 创建方形画布
                square_img = Image.new("RGB", (final_size, final_size), bg_color)
                
                # 裁剪原图
                cropped_region = img_pil.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                
                # 计算粘贴位置（居中）
                paste_x = (final_size - cropped_region.width) // 2
                paste_y = (final_size - cropped_region.height) // 2
                
                # 粘贴到方形画布
                square_img.paste(cropped_region, (paste_x, paste_y))
                
                # 转换回tensor
                square_img_np = np.array(square_img).astype(np.float32) / 255.0
                output_images.append(torch.from_numpy(square_img_np))
                continue

            # 处理有效遮罩的情况
            # 计算边界框的宽度和高度
            bbox_width = bbox[2] - bbox[0]
            bbox_height = bbox[3] - bbox[1]
            
            # 计算中心点
            center_x = (bbox[0] + bbox[2]) // 2
            center_y = (bbox[1] + bbox[3]) // 2
            
            # 计算方形边长 (取最大值)
            square_size = max(bbox_width, bbox_height)
            
            # 应用放大系数
            scaled_size = int(square_size * scale_factor)
            
            # 计算方形边界框的左上角和右下角坐标 (仅应用scale_factor)
            scaled_x1 = center_x - scaled_size // 2
            scaled_y1 = center_y - scaled_size // 2
            scaled_x2 = scaled_x1 + scaled_size
            scaled_y2 = scaled_y1 + scaled_size
            
            # 最终尺寸 (包含额外边距)
            final_size = scaled_size + extra_padding * 2
            
            # 确保最终尺寸是divisible_by的倍数
            final_size = (final_size // divisible_by) * divisible_by
            if final_size <= 0:
                final_size = divisible_by
            
            # 最终边界框坐标
            square_x1 = center_x - final_size // 2
            square_y1 = center_y - final_size // 2
            square_x2 = square_x1 + final_size
            square_y2 = square_y1 + final_size
            
            # 处理填充颜色
            if fill_color.lower() in ["e", "edge"]:
                if apply_mask:
                    # 当应用遮罩时，获取遮罩区域的平均颜色
                    mask_area_color = self.get_mask_area_color(img_pil, mask_pil, bbox)
                    # 创建方形画布 - 使用遮罩区域的平均颜色
                    square_img = Image.new("RGB", (final_size, final_size), mask_area_color)
                else:
                    # 当不应用遮罩时，获取四个边的平均颜色
                    edge_colors = self.get_four_edge_colors(img_pil, scaled_x1, scaled_y1, scaled_x2, scaled_y2)
                    # 创建方形画布 - 使用四边颜色填充
                    square_img = Image.new("RGB", (final_size, final_size), (255, 255, 255))
                    # 填充四个边缘区域
                    self.fill_edges_with_colors(square_img, edge_colors, extra_padding)
            else:
                # 解析填充颜色
                bg_color = self.parse_color(fill_color)
                # 创建方形画布
                square_img = Image.new("RGB", (final_size, final_size), bg_color)
            
            # 计算粘贴位置 - 考虑额外边距
            paste_x = max(0, -scaled_x1) + extra_padding
            paste_y = max(0, -scaled_y1) + extra_padding
            
            # 计算从原图裁剪的区域 - 只考虑scale_factor
            crop_x1 = max(0, scaled_x1)
            crop_y1 = max(0, scaled_y1)
            crop_x2 = min(img_pil.width, scaled_x2)
            crop_y2 = min(img_pil.height, scaled_y2)
            
            # 裁剪原图并粘贴到方形画布上
            if crop_x1 < crop_x2 and crop_y1 < crop_y2:
                cropped_region = img_pil.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                
                # 如果需要应用遮罩抠图
                if apply_mask:
                    # 裁剪遮罩
                    cropped_mask = mask_pil.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                    # 将遮罩应用到裁剪区域
                    if fill_color.lower() in ["e", "edge"]:
                        # 使用遮罩区域的平均颜色作为背景
                        bg_img = Image.new("RGB", cropped_region.size, mask_area_color)
                    else:
                        # 使用指定的填充颜色作为背景
                        bg_img = Image.new("RGB", cropped_region.size, self.parse_color(fill_color))
                    
                    # 合成图像
                    cropped_region = Image.composite(cropped_region, bg_img, cropped_mask)
                
                square_img.paste(cropped_region, (paste_x, paste_y))
            
            # 转换回tensor
            square_img_np = np.array(square_img).astype(np.float32) / 255.0
            output_images.append(torch.from_numpy(square_img_np))

        # 合并批次
        if output_images:
            output_image_tensor = torch.stack(output_images)
            return (output_image_tensor,)
        else:
            # 如果没有有效的输出图像，返回原始图像
            return (image,)

    def get_bbox(self, mask_pil):
        # 将遮罩转换为numpy数组
        mask_np = np.array(mask_pil)

        # 找到非零区域的坐标
        rows = np.any(mask_np > 10, axis=1)
        cols = np.any(mask_np > 10, axis=0)

        # 如果没有找到非零区域，返回None
        if not np.any(rows) or not np.any(cols):
            return None

        # 获取边界框坐标
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        # 返回边界框 (left, top, right, bottom)
        return (x_min, y_min, x_max + 1, y_max + 1)
    
    def get_mask_area_color(self, img, mask, bbox):
        """获取遮罩区域的平均颜色"""
        # 裁剪到边界框区域
        img_crop = img.crop(bbox)
        mask_crop = mask.crop(bbox)
        
        # 将图像和遮罩转换为numpy数组
        img_np = np.array(img_crop)
        mask_np = np.array(mask_crop)
        
        # 找到遮罩中非零区域
        mask_indices = mask_np > 10
        
        # 如果没有有效区域，返回白色
        if not np.any(mask_indices):
            return (255, 255, 255)
        
        # 获取遮罩区域内的像素
        masked_pixels = img_np[mask_indices]
        
        # 计算平均颜色
        r_mean = int(np.mean(masked_pixels[:, 0]))
        g_mean = int(np.mean(masked_pixels[:, 1]))
        b_mean = int(np.mean(masked_pixels[:, 2]))
        
        return (r_mean, g_mean, b_mean)
    
    def get_four_edge_colors(self, img, x1, y1, x2, y2):
        """获取图像四个边缘的平均颜色"""
        width, height = img.size
        
        # 确保坐标在图像范围内
        x1 = max(0, min(x1, width-1))
        y1 = max(0, min(y1, height-1))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        
        # 收集四个边缘的像素
        top_pixels = []
        bottom_pixels = []
        left_pixels = []
        right_pixels = []
        
        # 上边缘
        if y1 >= 0 and y1 < height:
            for x in range(max(0, x1), min(width, x2)):
                if 0 <= x < width:
                    top_pixels.append(img.getpixel((x, y1)))
        
        # 下边缘
        if y2-1 >= 0 and y2-1 < height:
            for x in range(max(0, x1), min(width, x2)):
                if 0 <= x < width:
                    bottom_pixels.append(img.getpixel((x, y2-1)))
        
        # 左边缘
        if x1 >= 0 and x1 < width:
            for y in range(max(0, y1), min(height, y2)):
                if 0 <= y < height:
                    left_pixels.append(img.getpixel((x1, y)))
        
        # 右边缘
        if x2-1 >= 0 and x2-1 < width:
            for y in range(max(0, y1), min(height, y2)):
                if 0 <= y < height:
                    right_pixels.append(img.getpixel((x2-1, y)))
        
        # 计算每个边缘的平均颜色
        top_color = self.calculate_average_color(top_pixels)
        bottom_color = self.calculate_average_color(bottom_pixels)
        left_color = self.calculate_average_color(left_pixels)
        right_color = self.calculate_average_color(right_pixels)
        
        return {
            'top': top_color,
            'bottom': bottom_color,
            'left': left_color,
            'right': right_color
        }
    
    def calculate_average_color(self, pixels):
        """计算像素列表的平均颜色"""
        if not pixels:
            return (255, 255, 255)
        
        r_sum = sum(p[0] for p in pixels)
        g_sum = sum(p[1] for p in pixels)
        b_sum = sum(p[2] for p in pixels)
        
        pixel_count = len(pixels)
        return (r_sum // pixel_count, g_sum // pixel_count, b_sum // pixel_count)
    
    def fill_edges_with_colors(self, img, edge_colors, padding):
        """使用四个边缘颜色填充图像的边缘区域"""
        if padding <= 0:
            return
        
        width, height = img.size
        draw = ImageDraw.Draw(img)
        
        # 填充上边缘
        draw.rectangle([0, 0, width, padding], fill=edge_colors['top'])
        
        # 填充下边缘
        draw.rectangle([0, height-padding, width, height], fill=edge_colors['bottom'])
        
        # 填充左边缘 (不包括已经填充的角落)
        draw.rectangle([0, padding, padding, height-padding], fill=edge_colors['left'])
        
        # 填充右边缘 (不包括已经填充的角落)
        draw.rectangle([width-padding, padding, width, height-padding], fill=edge_colors['right'])
    
    def get_edge_colors(self, img, x1, y1, x2, y2):
        """获取图像边缘的平均颜色 (兼容旧代码)"""
        width, height = img.size
        
        # 确保坐标在图像范围内
        x1 = max(0, min(x1, width-1))
        y1 = max(0, min(y1, height-1))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        
        # 收集边缘像素
        edge_pixels = []
        
        # 上边缘
        if y1 >= 0 and y1 < height:
            for x in range(max(0, x1), min(width, x2)):
                if 0 <= x < width:
                    edge_pixels.append(img.getpixel((x, y1)))
        
        # 下边缘
        if y2-1 >= 0 and y2-1 < height:
            for x in range(max(0, x1), min(width, x2)):
                if 0 <= x < width:
                    edge_pixels.append(img.getpixel((x, y2-1)))
        
        # 左边缘
        if x1 >= 0 and x1 < width:
            for y in range(max(0, y1), min(height, y2)):
                if 0 <= y < height:
                    edge_pixels.append(img.getpixel((x1, y)))
        
        # 右边缘
        if x2-1 >= 0 and x2-1 < width:
            for y in range(max(0, y1), min(height, y2)):
                if 0 <= y < height:
                    edge_pixels.append(img.getpixel((x2-1, y)))
        
        # 如果没有有效的边缘像素，返回白色
        if not edge_pixels:
            return (255, 255, 255)
        
        # 计算平均颜色
        r_sum = sum(p[0] for p in edge_pixels)
        g_sum = sum(p[1] for p in edge_pixels)
        b_sum = sum(p[2] for p in edge_pixels)
        
        pixel_count = len(edge_pixels)
        return (r_sum // pixel_count, g_sum // pixel_count, b_sum // pixel_count)
    
    def parse_color(self, color_str):
        """解析颜色字符串，支持灰度值、HEX和RGB格式"""
        # 尝试作为灰度值解析
        try:
            gray_value = float(color_str)
            gray_int = int(gray_value * 255)
            return (gray_int, gray_int, gray_int)
        except ValueError:
            pass
        
        # 尝试作为HEX解析
        if color_str.startswith('#'):
            # 移除 # 符号
            color_str = color_str[1:]
            
            # 处理不同长度的HEX
            if len(color_str) == 3:  # 短格式 #RGB
                r = int(color_str[0] + color_str[0], 16)
                g = int(color_str[1] + color_str[1], 16)
                b = int(color_str[2] + color_str[2], 16)
                return (r, g, b)
            elif len(color_str) == 6:  # 标准格式 #RRGGBB
                r = int(color_str[0:2], 16)
                g = int(color_str[2:4], 16)
                b = int(color_str[4:6], 16)
                return (r, g, b)
        
        # 尝试作为RGB格式解析 "255,0,0"
        try:
            rgb_values = color_str.split(',')
            if len(rgb_values) == 3:
                r = int(rgb_values[0].strip())
                g = int(rgb_values[1].strip())
                b = int(rgb_values[2].strip())
                return (r, g, b)
        except ValueError:
            pass
        
        # 默认返回白色
        return (255, 255, 255)


class ImageCropEdge:
    """
    图像裁剪边缘 - 支持同时裁剪四边或单独设置每边裁剪量
    """
    
    # 内置最大分辨率常量
    MAX_RESOLUTION = 8192

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "left_amount": ("FLOAT", {
                    "default": 0,
                    "min": 0.0,
                    "max": cls.MAX_RESOLUTION,
                    "step": 0.01
                }),
                "right_amount": ("FLOAT", {
                    "default": 0,
                    "min": 0.0,
                    "max": cls.MAX_RESOLUTION,
                    "step": 0.01
                }),
                "top_amount": ("FLOAT", {
                    "default": 0,
                    "min": 0.0,
                    "max": cls.MAX_RESOLUTION,
                    "step": 0.01
                }),
                "bottom_amount": ("FLOAT", {
                    "default": 0,
                    "min": 0.0,
                    "max": cls.MAX_RESOLUTION,
                    "step": 0.01
                }),
                "uniform_amount": ("FLOAT", {
                    "default": 0,
                    "min": 0.0,
                    "max": cls.MAX_RESOLUTION,
                    "step": 0.01
                }),
                "divisible_by": ("INT", {
                    "default": 8, 
                    "min": 1, 
                    "max": 1024, 
                    "step": 1
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "crop"
    CATEGORY = "1hewNodes/image/crop"

    @staticmethod
    def num_round_up_to_multiple(num, multiple):
        """将数字向上取整到指定倍数"""
        return (num + multiple - 1) // multiple * multiple

    def crop(self, image, uniform_amount, left_amount, right_amount, top_amount, bottom_amount, divisible_by=8):
        """执行裁剪操作"""
        _, height, width, _ = image.shape

        # 处理参数值：0-1为百分比，>=1为像素值
        def process_value(value, dimension):
            if value > 0:
                if value < 1:  # 百分比模式 (0-1)
                    return int(dimension * value)
                else:  # 像素模式 (>=1)
                    return int(value)
            return 0

        # 当uniform_amount大于0时，覆盖left_amount, right_amount, top_amount, bottom_amount的值
        if uniform_amount > 0:
            if uniform_amount < 1:  # 百分比模式 (0-1)
                # 修改：百分比接近1时应该裁剪更多（几乎裁剪完整个图像）
                # 当uniform_amount为0.5时裁剪掉25%，当uniform_amount为0.9时裁剪掉45%
                crop_percent = uniform_amount / 2
                uniform_left = int(width * crop_percent)
                uniform_right = int(width * crop_percent)
                uniform_top = int(height * crop_percent)
                uniform_bottom = int(height * crop_percent)
            else:  # 像素模式 (>=1)
                uniform_left = int(uniform_amount)
                uniform_right = int(uniform_amount)
                uniform_top = int(uniform_amount)
                uniform_bottom = int(uniform_amount)
            left = uniform_left
            right = uniform_right
            top = uniform_top
            bottom = uniform_bottom
        else:
            # 处理各边的值 - 保持原有逻辑
            left = process_value(left_amount, width)
            right = process_value(right_amount, width)
            top = process_value(top_amount, height)
            bottom = process_value(bottom_amount, height)

        # 确保值为divisible_by的倍数
        left = left // divisible_by * divisible_by
        right = right // divisible_by * divisible_by
        top = top // divisible_by * divisible_by
        bottom = bottom // divisible_by * divisible_by

        # 如果所有裁剪值为0，直接返回原图
        if left == 0 and right == 0 and bottom == 0 and top == 0:
            return (image,)

        # 计算新的边界
        inset_left = left
        inset_right = width - right
        inset_top = top
        inset_bottom = height - bottom

        # 确保最终尺寸是divisible_by的倍数
        new_width = inset_right - inset_left
        new_height = inset_bottom - inset_top
        
        # 调整边界以确保最终尺寸是divisible_by的倍数
        target_width = new_width // divisible_by * divisible_by
        target_height = new_height // divisible_by * divisible_by
        
        # 如果调整后尺寸变小，则从边界处减少裁剪量
        if target_width < new_width:
            width_diff = new_width - target_width
            inset_right -= width_diff // 2
            inset_left += (width_diff - width_diff // 2)
        
        if target_height < new_height:
            height_diff = new_height - target_height
            inset_bottom -= height_diff // 2
            inset_top += (height_diff - height_diff // 2)

        # 验证裁剪尺寸是否有效
        if inset_top >= inset_bottom:
            raise ValueError(
                f"无效的裁剪尺寸：顶部({inset_top})超过或等于底部({inset_bottom})")
        if inset_left >= inset_right:
            raise ValueError(
                f"无效的裁剪尺寸：左侧({inset_left})超过或等于右侧({inset_right})")

        # 执行裁剪
        print(f'裁剪图像 {width}x{height}，左侧裁剪至 {inset_left}，右侧裁剪至 {inset_right}，' +
              f'顶部裁剪至 {inset_top}，底部裁剪至 {inset_bottom}')
        
        # 最终尺寸检查
        final_width = inset_right - inset_left
        final_height = inset_bottom - inset_top
        if final_width % divisible_by != 0 or final_height % divisible_by != 0:
            print(f"警告：裁剪后尺寸 {final_width}x{final_height} 不是 {divisible_by} 的倍数")
        
        image = image[:, inset_top:inset_bottom, inset_left:inset_right, :]
        return (image,)
        

class ImageCropWithBBoxMask:
    """
    图像裁切器 - 根据遮罩裁切图像，并返回边界框遮罩信息以便后续粘贴回原位置
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "preset_ratio": (["mask_ratio", "image_ratio", "1:1", "3:2", "4:3", "16:9", "21:9", "2:3", "3:4", "9:16", "9:21"], {"default": "mask_ratio"}),
                "scale_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "divisible_by": ("INT", {"default": 8, "min": 1, "max": 64, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("cropped_image", "bbox_mask", "cropped_mask")
    FUNCTION = "image_crop_with_bbox_mask"
    CATEGORY = "1hewNodes/image/crop"

    @staticmethod
    def gcd_of_two(a, b):
        """计算两个数的最大公约数"""
        while b:
            a, b = b, a % b
        return a
    
    @staticmethod
    def get_simplified_ratio(width, height):
        """计算图像的最简比例"""
        gcd = ImageCropWithBBoxMask.gcd_of_two(width, height)
        return width // gcd, height // gcd
    
    def get_bbox(self, mask_pil):
        """从遮罩中获取边界框"""
        try:
            mask_np = np.array(mask_pil)
            if mask_np.size == 0:
                return None
                
            rows = np.any(mask_np > 10, axis=1)
            cols = np.any(mask_np > 10, axis=0)

            if not np.any(rows) or not np.any(cols):
                return None

            y_indices = np.where(rows)[0]
            x_indices = np.where(cols)[0]
            
            if len(y_indices) == 0 or len(x_indices) == 0:
                return None
                
            y_min, y_max = int(y_indices[0]), int(y_indices[-1])
            x_min, x_max = int(x_indices[0]), int(x_indices[-1])

            return (x_min, y_min, x_max + 1, y_max + 1)
        except Exception as e:
            print(f"Error in get_bbox: {e}")
            return None
    
    def parse_ratio_exact(self, preset_ratio, img_width, img_height, mask_width, mask_height):
        """精确解析比例设置"""
        if preset_ratio == "mask_ratio":

            if mask_height > 0:
                # 计算mask的原始比例
                original_ratio = mask_width / mask_height
                
                # 简化为最简分数
                simplified_w, simplified_h = self.get_simplified_ratio(mask_width, mask_height)
                
                # 如果简化后的分子分母过大，则调整为接近的标准比例
                if simplified_w > 21 or simplified_h > 21:
                    # 寻找最接近的标准比例
                    standard_ratios = [
                        (1, 1), (3, 2), (4, 3), (16, 9), (21, 9),
                        (2, 3), (3, 4), (9, 16), (9, 21)
                    ]
                    
                    best_ratio = (1, 1)
                    min_diff = float('inf')
                    
                    for w, h in standard_ratios:
                        ratio_diff = abs(w/h - original_ratio)
                        if ratio_diff < min_diff:
                            min_diff = ratio_diff
                            best_ratio = (w, h)
                    
                    return Fraction(best_ratio[0], best_ratio[1])
                else:
                    return Fraction(simplified_w, simplified_h)
            else:
                return Fraction(1, 1)
        elif preset_ratio == "image_ratio":
            simplified_w, simplified_h = self.get_simplified_ratio(img_width, img_height)
            return Fraction(simplified_w, simplified_h)
        else:
            ratio_map = {
                "1:1": Fraction(1, 1), "3:2": Fraction(3, 2), "4:3": Fraction(4, 3), 
                "16:9": Fraction(16, 9), "21:9": Fraction(21, 9),
                "2:3": Fraction(2, 3), "3:4": Fraction(3, 4), 
                "9:16": Fraction(9, 16), "9:21": Fraction(9, 21)
            }
            return ratio_map.get(preset_ratio, Fraction(1, 1))
    
    def calculate_exact_dimensions(self, target_ratio, divisible_by, base_width=None, base_height=None):
        """计算精确符合比例和divisible_by约束的尺寸"""
        ratio_w = target_ratio.numerator
        ratio_h = target_ratio.denominator
        
        if base_width is not None:
            # 基于宽度计算
            unit = math.ceil(base_width / (ratio_w * divisible_by))
            final_width = unit * ratio_w * divisible_by
            final_height = unit * ratio_h * divisible_by
        elif base_height is not None:
            # 基于高度计算
            unit = math.ceil(base_height / (ratio_h * divisible_by))
            final_width = unit * ratio_w * divisible_by
            final_height = unit * ratio_h * divisible_by
        else:
            # 默认最小尺寸
            final_width = ratio_w * divisible_by
            final_height = ratio_h * divisible_by
        
        return int(final_width), int(final_height)
    
    def calculate_padding_size(self, mask_bbox, target_ratio, divisible_by):
        """计算基于mask bbox的padding尺寸以满足目标比例和divisible_by"""
        x_min, y_min, x_max, y_max = mask_bbox
        mask_width = x_max - x_min
        mask_height = y_max - y_min
        mask_center_x = (x_min + x_max) / 2.0
        mask_center_y = (y_min + y_max) / 2.0
        
        # 方案1：基于mask宽度计算精确尺寸
        width_1, height_1 = self.calculate_exact_dimensions(target_ratio, divisible_by, base_width=mask_width)
        
        # 方案2：基于mask高度计算精确尺寸
        width_2, height_2 = self.calculate_exact_dimensions(target_ratio, divisible_by, base_height=mask_height)
        
        # 选择能包含mask的最小尺寸方案
        if width_1 >= mask_width and height_1 >= mask_height:
            if width_2 >= mask_width and height_2 >= mask_height:
                # 两个方案都可行，选择面积更小的
                if width_1 * height_1 <= width_2 * height_2:
                    final_width, final_height = width_1, height_1
                else:
                    final_width, final_height = width_2, height_2
            else:
                final_width, final_height = width_1, height_1
        elif width_2 >= mask_width and height_2 >= mask_height:
            final_width, final_height = width_2, height_2
        else:
            # 两个方案都不够，选择面积更大的
            if width_1 * height_1 >= width_2 * height_2:
                final_width, final_height = width_1, height_1
            else:
                final_width, final_height = width_2, height_2
        
        return final_width, final_height, mask_center_x, mask_center_y
    
    def calculate_max_size_in_image(self, center_x, center_y, target_ratio, img_width, img_height, divisible_by):
        """计算在图像内以给定中心点能达到的最大尺寸"""
        ratio_w = target_ratio.numerator
        ratio_h = target_ratio.denominator
        
        # 计算从中心点到边界的距离
        max_half_width = min(center_x, img_width - center_x)
        max_half_height = min(center_y, img_height - center_y)
        
        # 基于边界约束计算最大可能的单位数
        max_unit_from_width = int((max_half_width * 2) / (ratio_w * divisible_by))
        max_unit_from_height = int((max_half_height * 2) / (ratio_h * divisible_by))
        
        # 选择较小的单位数以确保不超出边界
        max_unit = min(max_unit_from_width, max_unit_from_height)
        max_unit = max(1, max_unit)  # 确保至少为1
        
        final_width = max_unit * ratio_w * divisible_by
        final_height = max_unit * ratio_h * divisible_by
        
        return final_width, final_height
    
    def calculate_interpolated_size(self, padding_size, max_size, scale_strength, target_ratio, divisible_by):
        """计算插值后的尺寸"""
        padding_width, padding_height = padding_size
        max_width, max_height = max_size
        
        # 线性插值
        interp_width = padding_width + (max_width - padding_width) * scale_strength
        interp_height = padding_height + (max_height - padding_height) * scale_strength
        
        # 基于插值结果计算精确尺寸
        ratio_w = target_ratio.numerator
        ratio_h = target_ratio.denominator
        
        # 计算单位数（基于面积插值）
        padding_area = padding_width * padding_height
        max_area = max_width * max_height
        interp_area = padding_area + (max_area - padding_area) * scale_strength
        
        # 根据目标比例和面积计算尺寸
        unit_area = ratio_w * ratio_h * divisible_by * divisible_by
        target_unit = max(1, int(math.sqrt(interp_area / unit_area)))
        
        final_width = target_unit * ratio_w * divisible_by
        final_height = target_unit * ratio_h * divisible_by
        
        return final_width, final_height
    
    def can_fit_in_image(self, center_x, center_y, width, height, img_width, img_height):
        """检查给定尺寸是否能在图像内以指定中心点放置"""
        half_w = width / 2
        half_h = height / 2
        
        return (center_x - half_w >= 0 and center_x + half_w <= img_width and
                center_y - half_h >= 0 and center_y + half_h <= img_height)
    
    def crop_from_center(self, center_x, center_y, target_ratio, img_width, img_height, divisible_by):
        """以mask中心为基准裁剪原图以符合比例和divisible_by需求"""
        # 计算能在图像内放置的最大尺寸
        max_width, max_height = self.calculate_max_size_in_image(
            center_x, center_y, target_ratio, img_width, img_height, divisible_by
        )
        
        return max_width, max_height
    
    def calculate_crop_region(self, center_x, center_y, target_width, target_height, img_width, img_height):
        """计算裁剪区域，确保不超出图像边界"""
        # 计算初始裁剪区域
        crop_x1 = int(center_x - target_width // 2)
        crop_y1 = int(center_y - target_height // 2)
        crop_x2 = crop_x1 + target_width
        crop_y2 = crop_y1 + target_height
        
        # 边界调整
        if crop_x1 < 0:
            crop_x2 -= crop_x1
            crop_x1 = 0
        if crop_y1 < 0:
            crop_y2 -= crop_y1
            crop_y1 = 0
        if crop_x2 > img_width:
            crop_x1 -= (crop_x2 - img_width)
            crop_x2 = img_width
        if crop_y2 > img_height:
            crop_y1 -= (crop_y2 - img_height)
            crop_y2 = img_height
        
        # 最终边界检查
        crop_x1 = max(0, crop_x1)
        crop_y1 = max(0, crop_y1)
        crop_x2 = min(img_width, crop_x2)
        crop_y2 = min(img_height, crop_y2)
        
        return crop_x1, crop_y1, crop_x2, crop_y2

    def image_crop_with_bbox_mask(self, image, mask, preset_ratio="mask_ratio", scale_strength=0.0, divisible_by=8):
        batch_size, img_height, img_width, channels = image.shape
        mask_batch_size = mask.shape[0]
        
        # 确定最终的批次大小
        final_batch_size = max(batch_size, mask_batch_size)
        
        # 扩展图像和遮罩以匹配最终批次大小
        if batch_size < final_batch_size:
            repeat_times = final_batch_size // batch_size
            remainder = final_batch_size % batch_size
            expanded_images = []
            for _ in range(repeat_times):
                expanded_images.append(image)
            if remainder > 0:
                expanded_images.append(image[:remainder])
            image = torch.cat(expanded_images, dim=0)
        
        if mask_batch_size < final_batch_size:
            repeat_times = final_batch_size // mask_batch_size
            remainder = final_batch_size % mask_batch_size
            expanded_masks = []
            for _ in range(repeat_times):
                expanded_masks.append(mask)
            if remainder > 0:
                expanded_masks.append(mask[:remainder])
            mask = torch.cat(expanded_masks, dim=0)
        
        cropped_images = []
        bbox_masks = []
        cropped_masks = []
        
        for b in range(final_batch_size):
            try:
                # 转换为numpy和PIL
                if image.is_cuda:
                    img_np = (image[b].cpu().numpy() * 255).astype(np.uint8)
                    mask_np = (mask[b].cpu().numpy() * 255).astype(np.uint8)
                else:
                    img_np = (image[b].numpy() * 255).astype(np.uint8)
                    mask_np = (mask[b].numpy() * 255).astype(np.uint8)
                
                # 确保数组形状正确
                if len(img_np.shape) != 3 or img_np.shape[2] != 3:
                    print(f"Warning: Unexpected image shape {img_np.shape}, skipping batch {b}")
                    continue
                    
                if len(mask_np.shape) != 2:
                    print(f"Warning: Unexpected mask shape {mask_np.shape}, skipping batch {b}")
                    continue
                
                img_pil = Image.fromarray(img_np)
                mask_pil = Image.fromarray(mask_np).convert("L")
            except Exception as e:
                print(f"Error processing batch {b}: {e}")
                # 使用原始图像作为fallback
                cropped_images.append(image[b])
                cropped_masks.append(mask[b])
                bbox_mask = torch.ones((img_height, img_width), dtype=torch.float32, device=image.device)
                bbox_masks.append(bbox_mask)
                continue
            
            # 调整遮罩大小以匹配图像
            if img_pil.size != mask_pil.size:
                new_mask = Image.new("L", img_pil.size, 0)
                paste_x = max(0, (img_pil.width - mask_pil.width) // 2)
                paste_y = max(0, (img_pil.height - mask_pil.height) // 2)
                new_mask.paste(mask_pil, (paste_x, paste_y))
                mask_pil = new_mask
            
            # 获取遮罩边界框
            bbox = self.get_bbox(mask_pil)
            if bbox is None:
                # 如果没有有效遮罩，使用整个图像
                cropped_images.append(image[b])
                cropped_masks.append(mask[b])
                bbox_mask = torch.ones((img_height, img_width), dtype=torch.float32, device=image.device)
                bbox_masks.append(bbox_mask)
                continue
            
            x_min, y_min, x_max, y_max = bbox
            mask_width = x_max - x_min
            mask_height = y_max - y_min
            mask_center_x = (x_min + x_max) / 2.0
            mask_center_y = (y_min + y_max) / 2.0
            
            try:
                # 解析目标比例
                target_ratio = self.parse_ratio_exact(
                    preset_ratio, img_width, img_height, mask_width, mask_height
                )
                
                # 核心逻辑：根据scale_strength决定裁剪策略
                if scale_strength == 0.0:
                    # scale_strength=0: 基于mask bbox进行padding以满足比例和divisible_by
                    target_width, target_height, center_x, center_y = self.calculate_padding_size(
                        bbox, target_ratio, divisible_by
                    )
                    
                    # 检查是否能在图像内放置
                    if not self.can_fit_in_image(center_x, center_y, target_width, target_height, img_width, img_height):
                        # 如果padding后的尺寸超出图像，则以mask中心进行原图裁剪
                        print(f"Padding size {target_width}x{target_height} exceeds image bounds, cropping from center")
                        target_width, target_height = self.crop_from_center(
                            mask_center_x, mask_center_y, target_ratio, img_width, img_height, divisible_by
                        )
                        center_x, center_y = mask_center_x, mask_center_y
                    
                elif scale_strength == 1.0:
                    # scale_strength=1: 获取最大符合条件的区域
                    target_width, target_height = self.calculate_max_size_in_image(
                        mask_center_x, mask_center_y, target_ratio, img_width, img_height, divisible_by
                    )
                    center_x, center_y = mask_center_x, mask_center_y
                    
                else:
                    # scale_strength在0-1之间：在padding尺寸和最大尺寸之间插值
                    # 计算padding尺寸
                    padding_width, padding_height, _, _ = self.calculate_padding_size(
                        bbox, target_ratio, divisible_by
                    )
                    
                    # 计算最大尺寸
                    max_width, max_height = self.calculate_max_size_in_image(
                        mask_center_x, mask_center_y, target_ratio, img_width, img_height, divisible_by
                    )
                    
                    # 检查padding尺寸是否能在图像内放置
                    if self.can_fit_in_image(mask_center_x, mask_center_y, padding_width, padding_height, img_width, img_height):
                        # padding尺寸可行，进行插值
                        target_width, target_height = self.calculate_interpolated_size(
                            (padding_width, padding_height), (max_width, max_height), 
                            scale_strength, target_ratio, divisible_by
                        )
                    else:
                        # padding尺寸不可行，使用裁剪模式的插值
                        crop_width, crop_height = self.crop_from_center(
                            mask_center_x, mask_center_y, target_ratio, img_width, img_height, divisible_by
                        )
                        target_width, target_height = self.calculate_interpolated_size(
                            (crop_width, crop_height), (max_width, max_height), 
                            scale_strength, target_ratio, divisible_by
                        )
                    
                    center_x, center_y = mask_center_x, mask_center_y
                
            except Exception as e:
                print(f"Error calculating target size: {e}")
                # 使用默认尺寸
                target_width, target_height = self.calculate_exact_dimensions(
                    target_ratio, divisible_by, base_width=mask_width
                )
                center_x, center_y = mask_center_x, mask_center_y
            
            # 计算裁剪区域（基于计算出的中心点，不超出原图）
            crop_x1, crop_y1, crop_x2, crop_y2 = self.calculate_crop_region(
                center_x, center_y, target_width, target_height, img_width, img_height
            )
            
            # 确保裁剪区域有效
            if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                print(f"Warning: Invalid crop region, using full image")
                crop_x1, crop_y1 = 0, 0
                crop_x2, crop_y2 = img_width, img_height
            
            # 创建bbox_mask（记录裁剪区域在原图中的位置）
            bbox_mask = torch.zeros((img_height, img_width), dtype=torch.float32, device=image.device)
            if crop_x1 < crop_x2 and crop_y1 < crop_y2:
                bbox_mask[crop_y1:crop_y2, crop_x1:crop_x2] = 1.0
            
            # 裁剪图像和mask
            cropped_img = image[b][crop_y1:crop_y2, crop_x1:crop_x2]
            cropped_msk = mask[b][crop_y1:crop_y2, crop_x1:crop_x2]
            
            cropped_images.append(cropped_img)
            bbox_masks.append(bbox_mask)
            cropped_masks.append(cropped_msk)
        
        # 处理尺寸不一致的情况
        if len(cropped_images) > 0:
            # 找到最大的有效尺寸
            max_height = max(img.shape[0] for img in cropped_images)
            max_width = max(img.shape[1] for img in cropped_images)
            
            # 确保尺寸不超过原图
            max_height = min(max_height, img_height)
            max_width = min(max_width, img_width)
            
            padded_images = []
            padded_masks = []
            
            for img, msk in zip(cropped_images, cropped_masks):
                h, w = img.shape[:2]
                if h != max_height or w != max_width:
                    # 创建统一尺寸的张量
                    padded_img = torch.zeros((max_height, max_width, img.shape[2]), dtype=img.dtype, device=img.device)
                    padded_msk = torch.zeros((max_height, max_width), dtype=msk.dtype, device=msk.device)
                    
                    # 居中放置
                    start_y = (max_height - h) // 2
                    start_x = (max_width - w) // 2
                    
                    padded_img[start_y:start_y+h, start_x:start_x+w] = img
                    padded_msk[start_y:start_y+h, start_x:start_x+w] = msk
                    
                    padded_images.append(padded_img)
                    padded_masks.append(padded_msk)
                else:
                    padded_images.append(img)
                    padded_masks.append(msk)
            
            cropped_images = padded_images
            cropped_masks = padded_masks
        
        # 转换为批次张量
        if len(cropped_images) == 0:
            print("Warning: No valid crops found, returning original data")
            bbox_masks = torch.ones((final_batch_size, img_height, img_width), dtype=torch.float32, device=image.device)
            return (image, bbox_masks, mask)
            
        try:
            cropped_images = torch.stack(cropped_images, dim=0)
            bbox_masks = torch.stack(bbox_masks, dim=0)
            cropped_masks = torch.stack(cropped_masks, dim=0)
        except Exception as e:
            print(f"Error stacking tensors: {e}")
            print(f"Cropped images shapes: {[img.shape for img in cropped_images]}")
            bbox_masks = torch.ones((final_batch_size, img_height, img_width), dtype=torch.float32, device=image.device)
            return (image, bbox_masks, mask)
        
        return (cropped_images, bbox_masks, cropped_masks)


class ImageCropByMaskAlpha:
    """
    图像遮罩裁剪 - 根据边界框遮罩信息批量裁剪图像
    支持两种输出模式：完整区域或仅白色区域（带alpha通道）
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "output_mode": (["bbox_rgb", "mask_rgba"], {"default": "bbox_rgb"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("cropped_image", "cropped_mask")
    FUNCTION = "image_crop_by_mask_alpha"
    CATEGORY = "1hewNodes/image/crop"

    def image_crop_by_mask_alpha(self, image, mask, output_mode="bbox_rgb"):
        batch_size, height, width, channels = image.shape
        mask_batch_size = mask.shape[0]
        
        # 修改：使用最大批次数进行循环
        max_batch = max(batch_size, mask_batch_size)
        output_images = []
        output_masks = []
        
        for b in range(max_batch):
            # 获取对应的图像和遮罩索引
            img_idx = b % batch_size
            mask_idx = b % mask_batch_size
            
            if image.is_cuda:
                img_np = (image[img_idx].cpu().numpy() * 255).astype(np.uint8)
                mask_np = (mask[mask_idx].cpu().numpy() * 255).astype(np.uint8)
            else:
                img_np = (image[img_idx].numpy() * 255).astype(np.uint8)
                mask_np = (mask[mask_idx].numpy() * 255).astype(np.uint8)
            
            # 新增：根据输出模式处理图像通道
            if output_mode == "bbox_rgb" and img_np.shape[2] == 4:
                # bbox_rgb模式下，如果输入是4通道，只取RGB通道
                img_np = img_np[:, :, :3]
            
            img_pil = Image.fromarray(img_np)
            mask_pil = Image.fromarray(mask_np).convert("L")
            
            # 从遮罩中获取边界框
            bbox = self.get_bbox_from_mask(mask_pil)
            
            if bbox is None:
                # 如果没有找到有效区域，返回原始图像和遮罩
                if output_mode == "bbox_rgb":
                    # 确保bbox_rgb模式下输出3通道
                    if image[img_idx].shape[2] == 4:
                        rgb_image = image[img_idx][:, :, :3]
                        output_images.append(rgb_image)
                    else:
                        output_images.append(image[img_idx])
                else:
                    # 为mask_rgba模式创建带alpha通道的图像
                    if img_np.shape[2] == 3:
                        img_rgba = np.concatenate([img_np, np.ones_like(img_np[:,:,0:1]) * 255], axis=2)
                    else:
                        img_rgba = img_np
                    img_rgba_tensor = torch.from_numpy(img_rgba.astype(np.float32) / 255.0)
                    output_images.append(img_rgba_tensor)
                output_masks.append(mask[mask_idx])
                continue
            
            x_min, y_min, x_max, y_max = bbox
            
            # 确保边界框不超出图像范围
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(img_pil.width, x_max)
            y_max = min(img_pil.height, y_max)
            
            # 裁剪图像和遮罩
            cropped_img = img_pil.crop((x_min, y_min, x_max, y_max))
            cropped_mask = mask_pil.crop((x_min, y_min, x_max, y_max))
            
            # 将裁剪后的mask转换为tensor
            cropped_mask_np = np.array(cropped_mask).astype(np.float32) / 255.0
            cropped_mask_tensor = torch.from_numpy(cropped_mask_np)
            output_masks.append(cropped_mask_tensor)
            
            if output_mode == "bbox_rgb":
                # 模式1：输出完整的裁剪区域（RGB格式）
                cropped_img_np = np.array(cropped_img).astype(np.float32) / 255.0
                output_images.append(torch.from_numpy(cropped_img_np))
                
            elif output_mode == "mask_rgba":
                # 模式2：仅输出白色区域，带alpha通道
                # 转换为RGBA模式
                cropped_img_rgba = cropped_img.convert("RGBA")
                
                # 将遮罩应用为alpha通道
                # 获取RGBA数据
                img_data = np.array(cropped_img_rgba)
                mask_data = np.array(cropped_mask)
                
                # 将遮罩应用到alpha通道
                img_data[:, :, 3] = mask_data
                
                # 创建带alpha通道的图像
                result_img = Image.fromarray(img_data, "RGBA")
                
                # 转换回tensor（包含alpha通道）
                result_img_np = np.array(result_img).astype(np.float32) / 255.0
                output_images.append(torch.from_numpy(result_img_np))
        
        # 合并批次
        if output_images and output_masks:
            # 处理图像
            image_sizes = [img.shape for img in output_images]
            if len(set(image_sizes)) == 1:
                output_image_tensor = torch.stack(output_images)
            else:
                # 如果尺寸不同，填充到相同尺寸
                output_images = self.pad_to_same_size(output_images)
                output_image_tensor = torch.stack(output_images)
            
            # 处理遮罩
            mask_sizes = [mask.shape for mask in output_masks]
            if len(set(mask_sizes)) == 1:
                output_mask_tensor = torch.stack(output_masks)
            else:
                # 如果尺寸不同，填充到相同尺寸
                output_masks = self.pad_masks_to_same_size(output_masks)
                output_mask_tensor = torch.stack(output_masks)
            
            return (output_image_tensor, output_mask_tensor)
        else:
            # 如果没有有效输出，返回原始数据
            return (image, mask)
    
    def get_bbox_from_mask(self, mask_pil):
        """从遮罩中获取边界框"""
        mask_np = np.array(mask_pil)
        rows = np.any(mask_np > 10, axis=1)
        cols = np.any(mask_np > 10, axis=0)

        if not np.any(rows) or not np.any(cols):
            return None

        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        return (x_min, y_min, x_max + 1, y_max + 1)
    
    def pad_to_same_size(self, images):
        """将所有图像填充到相同尺寸"""
        max_height = max(img.shape[0] for img in images)
        max_width = max(img.shape[1] for img in images)
        max_channels = max(img.shape[2] for img in images)
        
        padded_images = []
        
        for img in images:
            h, w, c = img.shape
            pad_h = max_height - h
            pad_w = max_width - w
            pad_c = max_channels - c
            
            # 填充空间维度
            padded_img = torch.nn.functional.pad(img, (0, pad_c, 0, pad_w, 0, pad_h), value=0)
            
            padded_images.append(padded_img)
        
        return padded_images
    
    def pad_masks_to_same_size(self, masks):
        """将所有遮罩填充到相同尺寸"""
        max_height = max(mask.shape[0] for mask in masks)
        max_width = max(mask.shape[1] for mask in masks)
        
        padded_masks = []
        
        for mask in masks:
            h, w = mask.shape
            pad_h = max_height - h
            pad_w = max_width - w
            
            # 填充空间维度
            padded_mask = torch.nn.functional.pad(mask, (0, pad_w, 0, pad_h), value=0)
            
            padded_masks.append(padded_mask)
        
        return padded_masks


class ImagePasteByBBoxMask:
    """
    图像遮罩粘贴器 - 将处理后的裁剪图像根据边界框遮罩粘贴回原始图像的位置
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image": ("IMAGE",),
                "cropped_image": ("IMAGE",),
                "bbox_mask": ("MASK",),
                "blend_mode": (
                    ["normal", "multiply", "screen", "overlay", "soft_light", "difference"],
                    {"default": "normal"}
                ),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "cropped_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "image_paste_by_bbox_mask"
    CATEGORY = "1hewNodes/image/crop"

    def image_paste_by_bbox_mask(self, base_image, cropped_image, bbox_mask, blend_mode="normal", opacity=1.0, cropped_mask=None):
        # 获取各输入的批次大小
        base_batch_size = base_image.shape[0]
        cropped_batch_size = cropped_image.shape[0]
        bbox_mask_batch_size = bbox_mask.shape[0]
        
        # 获取遮罩批次大小（如果存在）
        mask_batch_size = cropped_mask.shape[0] if cropped_mask is not None else 1
        
        # 确定最大批次大小
        max_batch_size = max(base_batch_size, cropped_batch_size, bbox_mask_batch_size, mask_batch_size)
        
        # 创建输出图像列表
        output_images = []
        
        for b in range(max_batch_size):
            # 使用循环索引获取对应的输入
            base_idx = b % base_batch_size
            cropped_idx = b % cropped_batch_size
            bbox_idx = b % bbox_mask_batch_size
            mask_idx = b % mask_batch_size if cropped_mask is not None else 0
            
            # 将图像转换为PIL格式
            if base_image.is_cuda:
                base_np = (base_image[base_idx].cpu().numpy() * 255).astype(np.uint8)
            else:
                base_np = (base_image[base_idx].numpy() * 255).astype(np.uint8)
            
            if cropped_image.is_cuda:
                cropped_np = (cropped_image[cropped_idx].cpu().numpy() * 255).astype(np.uint8)
            else:
                cropped_np = (cropped_image[cropped_idx].numpy() * 255).astype(np.uint8)
            
            if bbox_mask.is_cuda:
                bbox_np = (bbox_mask[bbox_idx].cpu().numpy() * 255).astype(np.uint8)
            else:
                bbox_np = (bbox_mask[bbox_idx].numpy() * 255).astype(np.uint8)
            
            base_pil = Image.fromarray(base_np)
            bbox_pil = Image.fromarray(bbox_np).convert("L")
            
            # 检查裁剪图像是否有alpha通道
            if cropped_np.shape[2] == 4:  # RGBA
                cropped_pil = Image.fromarray(cropped_np, "RGBA")
            else:  # RGB
                cropped_pil = Image.fromarray(cropped_np)
            
            # 从边界框遮罩获取边界框
            bbox = self.get_bbox_from_mask(bbox_pil)
            
            if bbox is None:
                # 如果没有找到有效位置，返回原始图像
                output_images.append(base_image[base_idx])
                continue
            
            # 处理裁剪遮罩
            mask_pil = None
            if cropped_mask is not None:
                if cropped_mask.is_cuda:
                    mask_np = (cropped_mask[mask_idx].cpu().numpy() * 255).astype(np.uint8)
                else:
                    mask_np = (cropped_mask[mask_idx].numpy() * 255).astype(np.uint8)
                mask_pil = Image.fromarray(mask_np).convert("L")
                
                # 调整遮罩大小以匹配处理后的图像
                if mask_pil.size != cropped_pil.size:
                    mask_pil = mask_pil.resize(cropped_pil.size, Image.LANCZOS)
            elif cropped_pil.mode == "RGBA":
                # 如果裁剪图像有alpha通道，使用它作为遮罩
                mask_pil = cropped_pil.split()[-1]  # 获取alpha通道
            
            # 执行粘贴操作
            result_pil = self.paste_image(base_pil, cropped_pil, bbox, blend_mode, opacity, mask_pil)
            
            # 转换回tensor
            result_np = np.array(result_pil).astype(np.float32) / 255.0
            output_images.append(torch.from_numpy(result_np))
        
        # 合并批次
        output_image_tensor = torch.stack(output_images)
        return (output_image_tensor,)
    
    def get_bbox_from_mask(self, mask_pil):
        """从遮罩中获取边界框"""
        mask_np = np.array(mask_pil)
        rows = np.any(mask_np > 10, axis=1)
        cols = np.any(mask_np > 10, axis=0)

        if not np.any(rows) or not np.any(cols):
            return None

        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        return (x_min, y_min, x_max + 1, y_max + 1)
    
    def paste_image(self, base_pil, cropped_pil, bbox, blend_mode, opacity, mask_pil=None):
        """将裁剪图像粘贴到基础图像上"""
        x_min, y_min, x_max, y_max = bbox
        
        # 调整裁剪图像大小以匹配边界框
        target_width = x_max - x_min
        target_height = y_max - y_min
        
        if cropped_pil.size != (target_width, target_height):
            cropped_pil = cropped_pil.resize((target_width, target_height), Image.LANCZOS)
            if mask_pil is not None:
                mask_pil = mask_pil.resize((target_width, target_height), Image.LANCZOS)
        
        # 创建结果图像的副本
        result_pil = base_pil.copy()
        
        if blend_mode == "normal":
            # 普通混合模式
            if mask_pil is not None:
                # 应用不透明度到遮罩
                if opacity < 1.0:
                    mask_array = np.array(mask_pil).astype(np.float32)
                    mask_array = (mask_array * opacity).astype(np.uint8)
                    mask_pil = Image.fromarray(mask_array)
                
                result_pil.paste(cropped_pil, (x_min, y_min), mask_pil)
            else:
                # 如果没有遮罩但有不透明度
                if opacity < 1.0:
                    # 创建基于不透明度的遮罩
                    alpha_mask = Image.new("L", cropped_pil.size, int(255 * opacity))
                    result_pil.paste(cropped_pil, (x_min, y_min), alpha_mask)
                else:
                    result_pil.paste(cropped_pil, (x_min, y_min))
        else:
            # 其他混合模式
            # 提取要混合的区域
            base_region = base_pil.crop((x_min, y_min, x_max, y_max))
            
            # 应用混合模式
            blended_region = self.apply_blend_mode(base_region, cropped_pil, blend_mode)
            
            # 应用不透明度和遮罩
            if mask_pil is not None or opacity < 1.0:
                # 创建最终遮罩
                final_mask = mask_pil if mask_pil is not None else Image.new("L", cropped_pil.size, 255)
                
                if opacity < 1.0:
                    mask_array = np.array(final_mask).astype(np.float32)
                    mask_array = (mask_array * opacity).astype(np.uint8)
                    final_mask = Image.fromarray(mask_array)
                
                result_pil.paste(blended_region, (x_min, y_min), final_mask)
            else:
                result_pil.paste(blended_region, (x_min, y_min))
        
        return result_pil
    
    def apply_blend_mode(self, base, overlay, mode):
        """应用混合模式"""
        base_array = np.array(base).astype(np.float32) / 255.0
        overlay_array = np.array(overlay).astype(np.float32) / 255.0
        
        if mode == "multiply":
            result = base_array * overlay_array
        elif mode == "screen":
            result = 1 - (1 - base_array) * (1 - overlay_array)
        elif mode == "overlay":
            result = np.where(
                base_array < 0.5,
                2 * base_array * overlay_array,
                1 - 2 * (1 - base_array) * (1 - overlay_array)
            )
        elif mode == "soft_light":
            result = np.where(
                overlay_array < 0.5,
                2 * base_array * overlay_array + base_array**2 * (1 - 2 * overlay_array),
                2 * base_array * (1 - overlay_array) + np.sqrt(base_array) * (2 * overlay_array - 1)
            )
        elif mode == "difference":
            result = np.abs(base_array - overlay_array)
        else:  # normal
            result = overlay_array
        
        # 确保结果在有效范围内
        result = np.clip(result, 0, 1)
        
        # 转换回PIL图像
        result_array = (result * 255).astype(np.uint8)
        return Image.fromarray(result_array)


NODE_CLASS_MAPPINGS = {
    "ImageCropSquare": ImageCropSquare,
    "ImageCropEdge": ImageCropEdge,
    "ImageCropWithBBoxMask": ImageCropWithBBoxMask,
    "ImageCropByMaskAlpha": ImageCropByMaskAlpha,
    "ImagePasteByBBoxMask": ImagePasteByBBoxMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageCropSquare": "Image Crop Square",
    "ImageCropEdge": "Image Crop Edge",
    "ImageCropWithBBoxMask": "Image Crop with BBox Mask",
    "ImageCropByMaskAlpha": "Image Crop by Mask Alpha",
    "ImagePasteByBBoxMask": "Image Paste by BBox Mask",
}