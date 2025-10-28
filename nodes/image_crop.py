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
                "preset_ratio": (["mask", "image", "1:1", "3:2", "4:3", "16:9", "21:9", "2:3", "3:4", "9:16", "9:21"], {"default": "mask"}),
                "scale_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "divisible_by": ("INT", {"default": 8, "min": 1, "max": 1024, "step": 1}),
                "crop_to_side": (["None", "longest", "shortest", "width", "height"], {"default": "None"}),
                "crop_to_length": ("INT", {"default": 1024, "min": 8, "max": 8192, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("cropped_image", "bbox_mask", "cropped_mask")
    FUNCTION = "image_crop_with_bbox_mask"
    CATEGORY = "1hewNodes/image/crop"

    def get_bbox(self, mask_pil):
        """获取遮罩的边界框"""
        bbox = mask_pil.getbbox()
        if bbox is None:
            return None
        return bbox

    def determine_ratio_orientation(self, preset_ratio, img_width, img_height, mask_width, mask_height):
        """判断比例的方向（横图、竖图、方图）"""
        if preset_ratio == "mask":
            ratio = mask_width / mask_height
        elif preset_ratio == "image":
            ratio = img_width / img_height
        else:
            # 解析标准比例格式 "width:height"
            try:
                parts = preset_ratio.split(":")
                if len(parts) == 2:
                    w, h = float(parts[0]), float(parts[1])
                    ratio = w / h
                else:
                    ratio = mask_width / mask_height
            except:
                ratio = mask_width / mask_height
        
        if abs(ratio - 1.0) < 0.01:  # 方图
            return "square", ratio
        elif ratio > 1.0:  # 横图
            return "landscape", ratio
        else:  # 竖图
            return "portrait", ratio

    def map_crop_side_to_dimension(self, crop_to_side, orientation):
        """根据crop_to_side和图像方向映射到具体的维度"""
        if crop_to_side == "None":
            return None
        elif crop_to_side == "width":
            return "width"
        elif crop_to_side == "height":
            return "height"
        elif crop_to_side == "longest":
            if orientation == "landscape":
                return "width"
            elif orientation == "portrait":
                return "height"
            else:  # square
                return "width"  # 方图时默认选择width
        elif crop_to_side == "shortest":
            if orientation == "landscape":
                return "height"
            elif orientation == "portrait":
                return "width"
            else:  # square
                return "height"  # 方图时默认选择height
        return None

    def preprocess_mask_dimensions_flexible(self, mask_width, mask_height, divisible_by):
        """灵活预处理 mask 尺寸，提供更大的选择范围"""
        original_ratio = mask_width / mask_height
        
        # 生成多种可能的调整方案，不局限于最小误差
        candidates = []
        
        # 基础调整方案
        for width_factor in [0.8, 0.9, 1.0, 1.1, 1.2]:  # 宽度调整范围
            for height_factor in [0.8, 0.9, 1.0, 1.1, 1.2]:  # 高度调整范围
                adjusted_width = max(divisible_by, round(mask_width * width_factor))
                adjusted_height = max(divisible_by, round(mask_height * height_factor))
                
                # 调整到 divisible_by 的倍数
                adjusted_width = ((adjusted_width + divisible_by - 1) // divisible_by) * divisible_by
                adjusted_height = ((adjusted_height + divisible_by - 1) // divisible_by) * divisible_by
                
                ratio = adjusted_width / adjusted_height
                error = abs(ratio - original_ratio)
                
                # 只要误差在合理范围内就加入候选
                if error < 0.3:  # 30% 的误差范围，比较宽松
                    candidates.append((adjusted_width, adjusted_height, ratio, error))
        
        # 去重并排序
        unique_candidates = []
        seen = set()
        for w, h, r, e in candidates:
            if (w, h) not in seen:
                seen.add((w, h))
                unique_candidates.append((w, h, r, e))
        
        # 按面积排序，提供从小到大的选择范围
        unique_candidates.sort(key=lambda x: x[0] * x[1])
        
        print(f"Mask尺寸灵活预处理: {mask_width}x{mask_height}")
        print(f"原始比例: {original_ratio:.6f}")
        print(f"生成候选方案数量: {len(unique_candidates)}")
        
        return unique_candidates

    def generate_flexible_valid_sizes(self, candidates, divisible_by, min_size=8, max_size=4096):
        """基于候选方案生成灵活的有效尺寸范围"""
        all_valid_sizes = set()
        
        for base_width, base_height, ratio, error in candidates:
            # 为每个候选方案生成尺寸序列
            multiplier = 1
            while True:
                width = base_width * multiplier
                height = base_height * multiplier
                
                if width > max_size or height > max_size:
                    break
                    
                if width >= min_size and height >= min_size:
                    all_valid_sizes.add((width, height))
                
                multiplier += 1
        
        # 转换为排序列表
        valid_sizes = sorted(list(all_valid_sizes), key=lambda x: x[0] * x[1])
        
        print(f"灵活方案生成的总有效尺寸数量: {len(valid_sizes)}")
        
        return valid_sizes

    def parse_ratio_exact(self, preset_ratio, img_width, img_height, mask_width, mask_height, divisible_by=8):
        """解析比例设置，返回精确的宽高比值"""
        if preset_ratio == "mask":
            return mask_width / mask_height
        elif preset_ratio == "image":
            return img_width / img_height
        else:
            # 解析标准比例格式 "width:height"
            try:
                parts = preset_ratio.split(":")
                if len(parts) == 2:
                    w, h = float(parts[0]), float(parts[1])
                    return w / h
            except:
                pass
            # 默认返回mask比例
            return mask_width / mask_height

    def generate_flexible_mask_sizes(self, mask_width, mask_height, divisible_by, min_size=8, max_size=4096):
        """专门为 mask 生成灵活的有效尺寸"""
        print(f"使用灵活的 mask 处理: {mask_width}x{mask_height}")
        
        # 使用灵活的预处理方式
        candidates = self.preprocess_mask_dimensions_flexible(mask_width, mask_height, divisible_by)
        
        if not candidates:
            print("警告: 无法生成有效的候选方案，回退到原始尺寸")
            # 回退方案
            adjusted_width = ((mask_width + divisible_by - 1) // divisible_by) * divisible_by
            adjusted_height = ((mask_height + divisible_by - 1) // divisible_by) * divisible_by
            candidates = [(adjusted_width, adjusted_height, adjusted_width/adjusted_height, 0)]
        
        # 生成灵活的有效尺寸
        valid_sizes = self.generate_flexible_valid_sizes(candidates, divisible_by, min_size, max_size)
        
        if not valid_sizes:
            print("警告: 无法生成有效尺寸，使用默认尺寸")
            return [(divisible_by, divisible_by)]
        
        print(f"最终生成有效尺寸数量: {len(valid_sizes)}")
        print(f"尺寸范围: {valid_sizes[0]} 到 {valid_sizes[-1]}")
        
        return valid_sizes

    def generate_valid_sizes(self, target_ratio, divisible_by, min_size=8, max_size=4096):
        """生成所有满足比例和divisible_by要求的有效尺寸"""
        valid_sizes = set()  # 使用set避免重复
        
        # 解析目标比例为分数形式
        if isinstance(target_ratio, str) and ":" in target_ratio:
            parts = target_ratio.split(":")
            ratio_w, ratio_h = int(parts[0]), int(parts[1])
        else:
            # 对于预处理后的 mask，直接使用简化的分数转换
            ratio_w, ratio_h = self._simple_float_to_fraction(target_ratio, divisible_by)
        
        # 计算最大公约数，简化比例
        from math import gcd
        g = gcd(ratio_w, ratio_h)
        ratio_w //= g
        ratio_h //= g
        
        print(f"简化后的比例: {ratio_w}:{ratio_h}")
        
        # 由于比例已经预处理过，基础尺寸就是比例本身乘以 divisible_by
        base_width = ratio_w * divisible_by
        base_height = ratio_h * divisible_by
        
        print(f"基础尺寸: {base_width}x{base_height}")
        
        # 生成所有有效尺寸
        multiplier = 1
        while True:
            width = base_width * multiplier
            height = base_height * multiplier
            
            if width > max_size or height > max_size:
                break
                
            if width >= min_size and height >= min_size:
                valid_sizes.add((width, height))
            
            multiplier += 1
        
        # 转换为排序列表
        valid_sizes = sorted(list(valid_sizes), key=lambda x: x[0] * x[1])
        
        return valid_sizes
    
    def _simple_float_to_fraction(self, ratio, divisible_by):
        """为预处理后的比例提供简单的分数转换"""
        # 尝试不同的分母，找到最接近的整数比例
        for denominator in range(1, 101):  # 限制在合理范围内
            numerator = round(ratio * denominator)
            if numerator > 0:
                actual_ratio = numerator / denominator
                if abs(actual_ratio - ratio) < 0.001:  # 精度足够
                    return numerator, denominator
        
        # 如果找不到合适的，使用默认方法
        from fractions import Fraction
        frac = Fraction(ratio).limit_denominator(100)
        return frac.numerator, frac.denominator
    
    def find_valid_size_range(self, valid_sizes, mask_bbox, img_width, img_height, preset_ratio="mask"):
        """在有效尺寸中找到满足约束的范围"""
        x_min, y_min, x_max, y_max = mask_bbox
        mask_width = x_max - x_min
        mask_height = y_max - y_min
        mask_center_x = (x_min + x_max) / 2.0
        mask_center_y = (y_min + y_max) / 2.0
        
        min_valid_size = None
        max_valid_size = None
        
        # 根据模式选择不同的检查策略
        use_flexible_bounds = (preset_ratio == "mask")
        
        for width, height in valid_sizes:
            # 检查是否能包含mask
            can_contain_mask = width >= mask_width and height >= mask_height
            
            if not can_contain_mask:
                continue
            
            if use_flexible_bounds:
                # mask模式：使用灵活的边界检查，允许更大的裁剪范围
                # 计算以mask为中心时的裁剪区域
                crop_x1 = mask_center_x - width / 2
                crop_y1 = mask_center_y - height / 2
                crop_x2 = crop_x1 + width
                crop_y2 = crop_y1 + height
                
                # 检查是否可以通过调整来适应图像边界
                can_fit_with_adjustment = True
                
                # 水平方向检查
                if crop_x1 < 0:
                    required_shift = -crop_x1
                    new_crop_x2 = crop_x2 + required_shift
                    if new_crop_x2 > img_width:
                        can_fit_with_adjustment = False
                elif crop_x2 > img_width:
                    required_shift = crop_x2 - img_width
                    new_crop_x1 = crop_x1 - required_shift
                    if new_crop_x1 < 0:
                        can_fit_with_adjustment = False
                
                # 垂直方向检查
                if can_fit_with_adjustment:
                    if crop_y1 < 0:
                        required_shift = -crop_y1
                        new_crop_y2 = crop_y2 + required_shift
                        if new_crop_y2 > img_height:
                            can_fit_with_adjustment = False
                    elif crop_y2 > img_height:
                        required_shift = crop_y2 - img_height
                        new_crop_y1 = crop_y1 - required_shift
                        if new_crop_y1 < 0:
                            can_fit_with_adjustment = False
                
                # 最终检查：裁剪尺寸不能超过图像尺寸
                if width > img_width or height > img_height:
                    can_fit_with_adjustment = False
                
                if can_fit_with_adjustment:
                    if min_valid_size is None:
                        min_valid_size = (width, height)
                    max_valid_size = (width, height)
            else:
                # 其他模式：使用严格的边界检查
                if width <= img_width and height <= img_height:
                    if min_valid_size is None:
                        min_valid_size = (width, height)
                    max_valid_size = (width, height)
        
        return min_valid_size, max_valid_size

    def calculate_target_size_from_range(self, min_size, max_size, scale_strength, valid_sizes):
        """根据scale_strength在有效尺寸范围内进行映射"""
        if min_size is None or max_size is None or not valid_sizes:
            return None
        
        # 找到min_size和max_size在valid_sizes中的索引
        min_index = None
        max_index = None
        
        for i, size in enumerate(valid_sizes):
            if size == min_size:
                min_index = i
            if size == max_size:
                max_index = i
        
        if min_index is None or max_index is None:
            return None
        
        # 如果min和max是同一个尺寸，直接返回
        if min_index == max_index:
            return min_size
        
        # 根据scale_strength在索引范围内进行插值
        target_index = min_index + (max_index - min_index) * scale_strength
        target_index = int(round(target_index))
        
        # 确保索引在有效范围内
        target_index = max(min_index, min(max_index, target_index))
        
        return valid_sizes[target_index]

    def find_target_size_by_length(self, valid_sizes, target_dimension, target_length):
        """根据指定的维度和长度找到最合适的尺寸"""
        best_size = None
        best_value = 0
        
        for width, height in valid_sizes:
            if target_dimension == "width":
                current_value = width
            else:  # height
                current_value = height
            
            # 找到小于等于目标长度的最大值
            if current_value <= target_length and current_value > best_value:
                best_value = current_value
                best_size = (width, height)
        
        # 如果没找到合适的，返回最小的有效尺寸
        if best_size is None and valid_sizes:
            best_size = valid_sizes[0]
            print(f"警告: 未找到满足长度 {target_length} 的尺寸，使用最小有效尺寸 {best_size}")
        
        return best_size

    def calculate_flexible_crop_region(self, center_x, center_y, target_width, target_height, img_width, img_height):
        """使用灵活边界计算裁剪区域"""
        target_width = int(target_width)
        target_height = int(target_height)
        
        # 初始位置（居中）
        crop_x1 = int(center_x - target_width / 2)
        crop_y1 = int(center_y - target_height / 2)
        crop_x2 = crop_x1 + target_width
        crop_y2 = crop_y1 + target_height
        
        # 灵活边界调整
        if crop_x1 < 0:
            shift = -crop_x1
            crop_x1 += shift
            crop_x2 += shift
        elif crop_x2 > img_width:
            shift = crop_x2 - img_width
            crop_x1 -= shift
            crop_x2 -= shift
        
        if crop_y1 < 0:
            shift = -crop_y1
            crop_y1 += shift
            crop_y2 += shift
        elif crop_y2 > img_height:
            shift = crop_y2 - img_height
            crop_y1 -= shift
            crop_y2 -= shift
        
        # 最终边界检查，确保不超出图像范围
        crop_x1 = max(0, min(crop_x1, img_width - target_width))
        crop_y1 = max(0, min(crop_y1, img_height - target_height))
        crop_x2 = crop_x1 + target_width
        crop_y2 = crop_y1 + target_height
        
        return crop_x1, crop_y1, crop_x2, crop_y2

    def image_crop_with_bbox_mask(self, image, mask, preset_ratio="mask", scale_strength=0.0, divisible_by=8, crop_to_side="None", crop_to_length=512):
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
                if len(img_np.shape) != 3 or img_np.shape[2] not in [3, 4]:
                    print(f"Warning: Unexpected image shape {img_np.shape}, using original image")
                    cropped_images.append(image[b])
                    cropped_masks.append(mask[b])
                    bbox_mask = torch.ones((img_height, img_width), dtype=torch.float32, device=image.device)
                    bbox_masks.append(bbox_mask)
                    continue
                
                img_pil = Image.fromarray(img_np)
                mask_pil = Image.fromarray(mask_np, mode='L')
                
                # 获取遮罩边界框
                bbox = self.get_bbox(mask_pil)
                if bbox is None:
                    print(f"Warning: Empty mask for batch {b}, using original image")
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
                
                print(f"\n批次 {b}: 图像尺寸 {img_width}x{img_height}, Mask边界框 {bbox}, Mask尺寸 {mask_width}x{mask_height}")
                
                # 判断比例方向
                orientation, target_ratio = self.determine_ratio_orientation(preset_ratio, img_width, img_height, mask_width, mask_height)
                print(f"比例方向: {orientation}, 目标比例: {target_ratio:.6f}")
                
                # 生成有效尺寸
                if preset_ratio == "mask":
                    valid_sizes = self.generate_flexible_mask_sizes(mask_width, mask_height, divisible_by)
                else:
                    valid_sizes = self.generate_valid_sizes(target_ratio, divisible_by)
                
                if not valid_sizes:
                    print(f"Warning: No valid sizes found for batch {b}, using original image")
                    cropped_images.append(image[b])
                    cropped_masks.append(mask[b])
                    bbox_mask = torch.ones((img_height, img_width), dtype=torch.float32, device=image.device)
                    bbox_masks.append(bbox_mask)
                    continue
                
                # 找到有效尺寸范围
                min_valid_size, max_valid_size = self.find_valid_size_range(valid_sizes, bbox, img_width, img_height, preset_ratio)
                
                if min_valid_size is None or max_valid_size is None:
                    print(f"Warning: No valid size range found for batch {b}, using original image")
                    cropped_images.append(image[b])
                    cropped_masks.append(mask[b])
                    bbox_mask = torch.ones((img_height, img_width), dtype=torch.float32, device=image.device)
                    bbox_masks.append(bbox_mask)
                    continue
                
                # 确定目标尺寸
                if crop_to_side == "None":
                    # 使用原有的scale_strength逻辑
                    target_size = self.calculate_target_size_from_range(min_valid_size, max_valid_size, scale_strength, valid_sizes)
                else:
                    # 使用新的精确尺寸控制逻辑
                    target_dimension = self.map_crop_side_to_dimension(crop_to_side, orientation)
                    if target_dimension:
                        # 过滤有效尺寸，只保留在范围内的
                        filtered_sizes = []
                        for size in valid_sizes:
                            if size >= min_valid_size and size <= max_valid_size:
                                filtered_sizes.append(size)
                        
                        if filtered_sizes:
                            target_size = self.find_target_size_by_length(filtered_sizes, target_dimension, crop_to_length)
                            print(f"使用精确尺寸控制: {crop_to_side} -> {target_dimension}, 目标长度: {crop_to_length}, 选择尺寸: {target_size}")
                        else:
                            target_size = min_valid_size
                            print(f"警告: 过滤后无有效尺寸，使用最小尺寸: {target_size}")
                    else:
                        target_size = min_valid_size
                        print(f"警告: 无法映射crop_to_side，使用最小尺寸: {target_size}")
                
                if target_size is None:
                    print(f"Warning: Could not determine target size for batch {b}, using original image")
                    cropped_images.append(image[b])
                    cropped_masks.append(mask[b])
                    bbox_mask = torch.ones((img_height, img_width), dtype=torch.float32, device=image.device)
                    bbox_masks.append(bbox_mask)
                    continue
                
                target_width, target_height = target_size
                print(f"最终目标尺寸: {target_width}x{target_height}")
                
                # 计算裁剪区域
                crop_x1, crop_y1, crop_x2, crop_y2 = self.calculate_flexible_crop_region(
                    mask_center_x, mask_center_y, target_width, target_height, img_width, img_height
                )
                
                print(f"裁剪区域: ({crop_x1}, {crop_y1}) 到 ({crop_x2}, {crop_y2})")
                
                # 执行裁剪
                cropped_img_pil = img_pil.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                cropped_mask_pil = mask_pil.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                
                # 转换回tensor
                cropped_img_np = np.array(cropped_img_pil).astype(np.float32) / 255.0
                cropped_mask_np = np.array(cropped_mask_pil).astype(np.float32) / 255.0
                
                # 确保维度正确
                if len(cropped_img_np.shape) == 2:
                    cropped_img_np = np.stack([cropped_img_np] * 3, axis=-1)
                elif cropped_img_np.shape[2] == 4:
                    cropped_img_np = cropped_img_np[:, :, :3]
                
                cropped_img_tensor = torch.from_numpy(cropped_img_np).to(image.device)
                cropped_mask_tensor = torch.from_numpy(cropped_mask_np).to(mask.device)
                
                # 创建边界框遮罩
                bbox_mask = torch.zeros((img_height, img_width), dtype=torch.float32, device=image.device)
                bbox_mask[crop_y1:crop_y2, crop_x1:crop_x2] = 1.0
                
                cropped_images.append(cropped_img_tensor)
                cropped_masks.append(cropped_mask_tensor)
                bbox_masks.append(bbox_mask)
                
            except Exception as e:
                print(f"Error processing batch {b}: {str(e)}")
                # 出错时返回原始图像
                cropped_images.append(image[b])
                cropped_masks.append(mask[b])
                bbox_mask = torch.ones((img_height, img_width), dtype=torch.float32, device=image.device)
                bbox_masks.append(bbox_mask)
        
        # 堆叠所有结果
        cropped_images = torch.stack(cropped_images)
        bbox_masks = torch.stack(bbox_masks)
        cropped_masks = torch.stack(cropped_masks)
        
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
                "paste_image": ("IMAGE",),
                "base_image": ("IMAGE",),
                "bbox_mask": ("MASK",),
                "position_x": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
                "position_y": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.01}),
                "rotation": ("FLOAT", {"default": 0.0, "min": -3600.0, "max": 3600.0, "step": 0.01}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "apply_paste_mask": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "paste_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "image_paste_by_bbox_mask"
    CATEGORY = "1hewNodes/image/crop"

    def image_paste_by_bbox_mask(self, paste_image, base_image, bbox_mask, position_x=0, 
                                 position_y=0, scale=1.0, rotation=0.0, opacity=1.0, apply_paste_mask=False, paste_mask=None):
        # 获取各输入的批次大小
        base_batch_size = base_image.shape[0]
        paste_batch_size = paste_image.shape[0]
        bbox_mask_batch_size = bbox_mask.shape[0]
        
        # 获取遮罩批次大小（如果存在）
        mask_batch_size = paste_mask.shape[0] if paste_mask is not None else 1
        
        # 确定最大批次大小
        max_batch_size = max(base_batch_size, paste_batch_size, bbox_mask_batch_size, mask_batch_size)
        
        # 创建输出图像和遮罩列表
        output_images = []
        output_masks = []
        
        for b in range(max_batch_size):
            # 使用循环索引获取对应的输入
            base_idx = b % base_batch_size
            paste_idx = b % paste_batch_size
            bbox_idx = b % bbox_mask_batch_size
            mask_idx = b % mask_batch_size if paste_mask is not None else 0
            
            # 将图像转换为PIL格式
            if base_image.is_cuda:
                base_np = (base_image[base_idx].cpu().numpy() * 255).astype(np.uint8)
            else:
                base_np = (base_image[base_idx].numpy() * 255).astype(np.uint8)
            
            if paste_image.is_cuda:
                paste_np = (paste_image[paste_idx].cpu().numpy() * 255).astype(np.uint8)
            else:
                paste_np = (paste_image[paste_idx].numpy() * 255).astype(np.uint8)
            
            if bbox_mask.is_cuda:
                bbox_np = (bbox_mask[bbox_idx].cpu().numpy() * 255).astype(np.uint8)
            else:
                bbox_np = (bbox_mask[bbox_idx].numpy() * 255).astype(np.uint8)
            
            base_pil = Image.fromarray(base_np)
            bbox_pil = Image.fromarray(bbox_np).convert("L")
            
            # 检查粘贴图像是否有alpha通道
            if paste_np.shape[2] == 4:  # RGBA
                paste_pil = Image.fromarray(paste_np, "RGBA")
            else:  # RGB
                paste_pil = Image.fromarray(paste_np)
            
            # 从边界框遮罩获取边界框
            bbox = self.get_bbox_from_mask(bbox_pil)
            
            if bbox is None:
                # 如果没有找到有效位置，返回原始图像和空遮罩
                output_images.append(base_image[base_idx])
                # 创建与base_image尺寸相同的空遮罩（全黑）
                empty_mask = np.zeros((base_np.shape[0], base_np.shape[1]), dtype=np.float32)
                output_masks.append(torch.from_numpy(empty_mask))
                continue
            
            # 处理粘贴遮罩
            mask_pil = None
            if paste_mask is not None:
                if paste_mask.is_cuda:
                    mask_np = (paste_mask[mask_idx].cpu().numpy() * 255).astype(np.uint8)
                else:
                    mask_np = (paste_mask[mask_idx].numpy() * 255).astype(np.uint8)
                mask_pil = Image.fromarray(mask_np).convert("L")
            elif paste_pil.mode == "RGBA":
                # 如果粘贴图像有alpha通道，使用它作为遮罩
                mask_pil = paste_pil.split()[-1]  # 获取alpha通道
            
            # 执行粘贴变换
            result_pil, result_mask_pil = self.paste_image_with_transform(
                base_pil, paste_pil, bbox, position_x, position_y, scale, rotation, mask_pil, opacity, apply_paste_mask
            )
            
            # 转换回tensor
            result_np = np.array(result_pil).astype(np.float32) / 255.0
            output_images.append(torch.from_numpy(result_np))
            
            # 转换遮罩为tensor
            result_mask_np = np.array(result_mask_pil).astype(np.float32) / 255.0
            output_masks.append(torch.from_numpy(result_mask_np))
        
        # 合并批次
        output_image_tensor = torch.stack(output_images)
        output_mask_tensor = torch.stack(output_masks)
        return (output_image_tensor, output_mask_tensor)
    
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
    
    def paste_image_with_transform(self, base_pil, paste_pil, bbox, position_x, position_y, scale, rotation, mask_pil=None, opacity=1.0, apply_paste_mask=False):

        """将粘贴图像应用变换后粘贴到基础图像上，并返回处理区域的遮罩"""
        x_min, y_min, x_max, y_max = bbox
        
        # 计算原始边界框的中心点和尺寸
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        bbox_center_x = x_min + bbox_width // 2
        bbox_center_y = y_min + bbox_height // 2
        
        # 根据 apply_paste_mask 参数决定使用什么尺寸作为智能缩放的参考
        if apply_paste_mask and mask_pil is not None:
            # apply_paste_mask=True 且有paste_mask时：使用paste_mask处理后的图像尺寸作为参考
            # 首先确保paste_image和paste_mask尺寸一致
            if paste_pil.size != mask_pil.size:
                mask_pil = mask_pil.resize(paste_pil.size, Image.LANCZOS)
            
            # 获取mask的边界框来确定有效内容区域
            mask_bbox = mask_pil.getbbox()
            if mask_bbox is None:
                # 如果mask完全为空，使用整个图像尺寸作为参考
                reference_width, reference_height = paste_pil.size
                effective_paste_pil = paste_pil
                effective_mask_pil = mask_pil
            else:
                # 裁剪到mask的有效区域，使用裁剪后的尺寸作为参考
                effective_paste_pil = paste_pil.crop(mask_bbox)
                effective_mask_pil = mask_pil.crop(mask_bbox)
                reference_width, reference_height = effective_paste_pil.size
        else:
            # apply_paste_mask=False 或 paste_mask未接入时：始终使用paste_image原始尺寸作为参考
            reference_width, reference_height = paste_pil.size
            effective_paste_pil = paste_pil
            if mask_pil is not None:
                # 确保mask尺寸一致
                if paste_pil.size != mask_pil.size:
                    mask_pil = mask_pil.resize(paste_pil.size, Image.LANCZOS)
                effective_mask_pil = mask_pil
            else:
                # 当未使用paste_mask时，创建纯白mask
                effective_mask_pil = Image.new('L', paste_pil.size, 255)
        
        # 使用参考尺寸进行智能缩放计算（等比例缩放，最大限度匹配bbox）
        reference_ratio = reference_width / reference_height
        bbox_ratio = bbox_width / bbox_height
        
        if reference_ratio > bbox_ratio:
            # 参考图像更宽，以bbox宽度为准
            fitted_width = bbox_width
            fitted_height = int(bbox_width / reference_ratio)
        else:
            # 参考图像更高，以bbox高度为准
            fitted_height = bbox_height
            fitted_width = int(bbox_height * reference_ratio)
        
        # 再应用scale进行额外缩放
        new_width = int(fitted_width * scale)
        new_height = int(fitted_height * scale)
        
        # 调整粘贴图像大小
        if effective_paste_pil.size != (new_width, new_height):
            paste_pil = effective_paste_pil.resize((new_width, new_height), Image.LANCZOS)
            mask_pil = effective_mask_pil.resize((new_width, new_height), Image.LANCZOS)
        else:
            paste_pil = effective_paste_pil
            mask_pil = effective_mask_pil
        
        # 应用旋转变换
        if rotation != 0.0:
            paste_pil, mask_pil = self.apply_rotation(paste_pil, rotation, mask_pil)
            # 旋转后重新获取尺寸
            new_width, new_height = paste_pil.size
        
        # 应用透明度控制
        if opacity < 1.0:
            # 确保粘贴图像有alpha通道
            if paste_pil.mode != "RGBA":
                paste_pil = paste_pil.convert("RGBA")
            
            # 调整alpha通道以实现透明度控制
            alpha = paste_pil.split()[-1]
            alpha = alpha.point(lambda p: int(p * opacity))
            paste_pil.putalpha(alpha)
            
            # 如果有遮罩，也需要相应调整
            if mask_pil is not None:
                mask_pil = mask_pil.point(lambda p: int(p * opacity))
            else:
                # 使用调整后的alpha通道作为遮罩
                mask_pil = alpha
        
        # 计算新的粘贴位置（考虑偏移，position_y反转）
        new_x = bbox_center_x - new_width // 2 + position_x
        new_y = bbox_center_y - new_height // 2 - position_y
        
        # 确保粘贴位置在基础图像范围内
        base_width, base_height = base_pil.size
        
        # 计算实际可粘贴的区域
        paste_x = max(0, new_x)
        paste_y = max(0, new_y)
        paste_x_end = min(base_width, new_x + new_width)
        paste_y_end = min(base_height, new_y + new_height)
        
        # 如果完全超出边界，返回原图
        if paste_x >= paste_x_end or paste_y >= paste_y_end:
            return base_pil
        
        # 计算需要裁剪的粘贴图像区域
        crop_x = paste_x - new_x
        crop_y = paste_y - new_y
        crop_x_end = crop_x + (paste_x_end - paste_x)
        crop_y_end = crop_y + (paste_y_end - paste_y)
        
        # 裁剪粘贴图像和遮罩
        if crop_x > 0 or crop_y > 0 or crop_x_end < new_width or crop_y_end < new_height:
            paste_pil = paste_pil.crop((crop_x, crop_y, crop_x_end, crop_y_end))
            if mask_pil is not None:
                mask_pil = mask_pil.crop((crop_x, crop_y, crop_x_end, crop_y_end))
        
        # 创建结果图像的副本
        result_pil = base_pil.copy()
        
        # 创建与base_image尺寸相同的输出遮罩（全黑背景）
        base_width, base_height = base_pil.size
        output_mask_pil = Image.new('L', (base_width, base_height), 0)
        
        # 粘贴图像（考虑透明度）
        if opacity < 1.0 or mask_pil is not None:
            # 使用alpha合成模式粘贴
            if paste_pil.mode != "RGBA":
                paste_pil = paste_pil.convert("RGBA")
            result_pil.paste(paste_pil, (paste_x, paste_y), mask_pil)
            
            # 在输出遮罩上标记处理区域
            if mask_pil is not None:
                # 使用实际的粘贴遮罩
                output_mask_pil.paste(mask_pil, (paste_x, paste_y))
            else:
                # 使用白色矩形标记整个粘贴区域
                paste_area_mask = Image.new('L', (paste_x_end - paste_x, paste_y_end - paste_y), 255)
                output_mask_pil.paste(paste_area_mask, (paste_x, paste_y))
        else:
            result_pil.paste(paste_pil, (paste_x, paste_y))
            # 在输出遮罩上标记处理区域（白色矩形）
            paste_area_mask = Image.new('L', (paste_x_end - paste_x, paste_y_end - paste_y), 255)
            output_mask_pil.paste(paste_area_mask, (paste_x, paste_y))
        
        return result_pil, output_mask_pil
    
    def apply_rotation(self, paste_pil, rotation_angle, mask_pil=None):
        """应用旋转变换到粘贴图像和遮罩"""
        # 修正旋转方向：负数为顺时针旋转（符合用户习惯）
        actual_angle = -rotation_angle
        
        # 旋转粘贴图像
        if paste_pil.mode == "RGBA":
            # 对于RGBA图像，使用透明背景旋转
            rotated_paste = paste_pil.rotate(actual_angle, expand=True, fillcolor=(0, 0, 0, 0))
        else:
            # 对于RGB图像，先转换为RGBA避免填充背景色问题
            paste_rgba = paste_pil.convert("RGBA")
            # 创建alpha遮罩（原图区域为不透明）
            alpha_mask = Image.new("L", paste_pil.size, 255)
            paste_rgba.putalpha(alpha_mask)
            # 使用透明背景旋转
            rotated_paste = paste_rgba.rotate(actual_angle, expand=True, fillcolor=(0, 0, 0, 0))
        
        # 旋转遮罩（如果存在）
        rotated_mask = None
        if mask_pil is not None:
            rotated_mask = mask_pil.rotate(actual_angle, expand=True, fillcolor=0)
        elif rotated_paste.mode == "RGBA":
            # 如果没有提供遮罩但图像有alpha通道，使用alpha通道作为遮罩
            rotated_mask = rotated_paste.split()[-1]
        
        return rotated_paste, rotated_mask
    



class ImageEdgeCropPad:
    """
    图像边缘裁剪填充 - 支持向内裁剪和向外填充
    负数值：向内裁剪
    正数值：向外填充（pad）
    支持多种颜色格式的填充颜色和边缘填充模式
    输出 mask：裁剪或填充的区域为白色，原图区域为黑色
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
                    "min": -cls.MAX_RESOLUTION,
                    "max": cls.MAX_RESOLUTION,
                    "step": 0.01
                }),
                "right_amount": ("FLOAT", {
                    "default": 0,
                    "min": -cls.MAX_RESOLUTION,
                    "max": cls.MAX_RESOLUTION,
                    "step": 0.01
                }),
                "top_amount": ("FLOAT", {
                    "default": 0,
                    "min": -cls.MAX_RESOLUTION,
                    "max": cls.MAX_RESOLUTION,
                    "step": 0.01
                }),
                "bottom_amount": ("FLOAT", {
                    "default": 0,
                    "min": -cls.MAX_RESOLUTION,
                    "max": cls.MAX_RESOLUTION,
                    "step": 0.01
                }),
                "uniform_amount": ("FLOAT", {
                    "default": 0,
                    "min": -cls.MAX_RESOLUTION,
                    "max": cls.MAX_RESOLUTION,
                    "step": 0.01
                }),
                "pad_color": ("STRING", {"default": "0.0"}),
                "divisible_by": ("INT", {
                    "default": 8, 
                    "min": 1, 
                    "max": 1024, 
                    "step": 1
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "crop_or_pad"
    CATEGORY = "1hewNodes/image/crop"

    def crop_or_pad(self, image, uniform_amount, left_amount, right_amount, top_amount, bottom_amount, 
                   pad_color="0.0", divisible_by=8):
        """执行裁剪或填充操作"""
        batch_size, height, width, channels = image.shape
        
        # 处理参数值：0-1为百分比，>=1为像素值
        def process_value(value, dimension):
            if value != 0:
                if abs(value) < 1:  # 百分比模式 (0-1)
                    return int(dimension * value)
                else:  # 像素模式 (>=1)
                    return int(value)
            return 0

        # 当uniform_amount不为0时，覆盖其他参数的值
        if uniform_amount != 0:
            if abs(uniform_amount) < 1:  # 百分比模式
                if uniform_amount < 0:  # 负数：向内裁剪
                    crop_percent = abs(uniform_amount) / 2
                    uniform_left = -int(width * crop_percent)
                    uniform_right = -int(width * crop_percent)
                    uniform_top = -int(height * crop_percent)
                    uniform_bottom = -int(height * crop_percent)
                else:  # 正数：向外填充
                    pad_percent = uniform_amount / 2
                    uniform_left = int(width * pad_percent)
                    uniform_right = int(width * pad_percent)
                    uniform_top = int(height * pad_percent)
                    uniform_bottom = int(height * pad_percent)
            else:  # 像素模式
                uniform_left = int(uniform_amount)
                uniform_right = int(uniform_amount)
                uniform_top = int(uniform_amount)
                uniform_bottom = int(uniform_amount)
            
            left = uniform_left
            right = uniform_right
            top = uniform_top
            bottom = uniform_bottom
        else:
            # 处理各边的值
            left = process_value(left_amount, width)
            right = process_value(right_amount, width)
            top = process_value(top_amount, height)
            bottom = process_value(bottom_amount, height)

        # 确保值为divisible_by的倍数
        left = (abs(left) // divisible_by * divisible_by) * (1 if left >= 0 else -1)
        right = (abs(right) // divisible_by * divisible_by) * (1 if right >= 0 else -1)
        top = (abs(top) // divisible_by * divisible_by) * (1 if top >= 0 else -1)
        bottom = (abs(bottom) // divisible_by * divisible_by) * (1 if bottom >= 0 else -1)

        # 如果所有值为0，直接返回原图和全黑mask
        if left == 0 and right == 0 and bottom == 0 and top == 0:
            mask = torch.zeros((batch_size, height, width), dtype=torch.float32, device=image.device)
            return (image, mask)

        # 处理批量图像
        output_images = []
        output_masks = []
        
        for b in range(batch_size):
            # 将图像转换为tensor格式进行处理
            img_tensor = image[b:b+1]  # 保持批次维度
            
            # 执行裁剪或填充
            result_tensor, result_mask = self._crop_or_pad_tensor(img_tensor, left, right, top, bottom, pad_color)
            
            output_images.append(result_tensor.squeeze(0))
            output_masks.append(result_mask.squeeze(0))

        # 合并批次
        output_tensor = torch.stack(output_images)
        output_mask = torch.stack(output_masks)
        
        return (output_tensor, output_mask)

    def _crop_or_pad_tensor(self, img_tensor, left, right, top, bottom, pad_color):
        """
        对图像tensor执行裁剪或填充操作，同时生成对应的mask
        """
        B, H, W, C = img_tensor.shape
        original_H, original_W = H, W  # 保存原始尺寸
        
        # 计算裁剪和填充的数量
        crop_left = max(0, -left)
        crop_right = max(0, -right)
        crop_top = max(0, -top)
        crop_bottom = max(0, -bottom)
        
        pad_left = max(0, left)
        pad_right = max(0, right)
        pad_top = max(0, top)
        pad_bottom = max(0, bottom)
        
        # 创建原始尺寸的mask，初始为全黑（0）
        original_mask = torch.zeros((B, original_H, original_W), dtype=torch.float32, device=img_tensor.device)
        
        # 先执行裁剪
        if crop_left > 0 or crop_right > 0 or crop_top > 0 or crop_bottom > 0:
            # 计算裁剪边界
            crop_x1 = crop_left
            crop_y1 = crop_top
            crop_x2 = W - crop_right
            crop_y2 = H - crop_bottom
            
            # 验证裁剪尺寸
            if crop_x1 >= crop_x2 or crop_y1 >= crop_y2:
                raise ValueError(f"裁剪尺寸无效：裁剪后图像尺寸为 {crop_x2-crop_x1}x{crop_y2-crop_y1}")
            
            # 在原始尺寸的mask中标记被裁剪的区域为白色（1）
            for b in range(B):
                # 标记被裁剪的区域
                if crop_top > 0:
                    original_mask[b, :crop_top, :] = 1.0  # 顶部裁剪区域
                if crop_bottom > 0:
                    original_mask[b, original_H-crop_bottom:, :] = 1.0  # 底部裁剪区域
                if crop_left > 0:
                    original_mask[b, :, :crop_left] = 1.0  # 左侧裁剪区域
                if crop_right > 0:
                    original_mask[b, :, original_W-crop_right:] = 1.0  # 右侧裁剪区域
            
            # 执行图像裁剪
            img_tensor = img_tensor[:, crop_y1:crop_y2, crop_x1:crop_x2, :]
            B, H, W, C = img_tensor.shape
            print(f'裁剪图像：左{crop_left}，右{crop_right}，上{crop_top}，下{crop_bottom}')
        
        # 再执行填充
        if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
            # 计算新的尺寸
            new_height = H + pad_top + pad_bottom
            new_width = W + pad_left + pad_right
            
            # 创建输出tensor和mask
            out_tensor = torch.zeros((B, new_height, new_width, C), dtype=img_tensor.dtype, device=img_tensor.device)
            out_mask = torch.ones((B, new_height, new_width), dtype=torch.float32, device=img_tensor.device)
            
            # 检查是否为 edge 模式的颜色参数
            color_lower = pad_color.lower().strip()
            if color_lower in ['edge', 'e', 'ed']:
                # 使用 ImagePadKJ 风格的边缘填充模式
                for b in range(B):
                    # 先放置原图像
                    out_tensor[b, pad_top:pad_top+H, pad_left:pad_left+W, :] = img_tensor[b]
                    # 原图区域设为黑色（0）
                    out_mask[b, pad_top:pad_top+H, pad_left:pad_left+W] = 0.0
                    
                    # 获取各边缘的平均颜色
                    top_edge = img_tensor[b, 0, :, :]  # [W, C]
                    bottom_edge = img_tensor[b, H-1, :, :]  # [W, C]
                    left_edge = img_tensor[b, :, 0, :]  # [H, C]
                    right_edge = img_tensor[b, :, W-1, :]  # [H, C]
                    
                    top_color = top_edge.mean(dim=0)  # [C]
                    bottom_color = bottom_edge.mean(dim=0)  # [C]
                    left_color = left_edge.mean(dim=0)  # [C]
                    right_color = right_edge.mean(dim=0)  # [C]
                    
                    # 填充各区域
                    if pad_top > 0:
                        out_tensor[b, :pad_top, :, :] = top_color.unsqueeze(0).unsqueeze(0)
                    if pad_bottom > 0:
                        out_tensor[b, pad_top+H:, :, :] = bottom_color.unsqueeze(0).unsqueeze(0)
                    if pad_left > 0:
                        out_tensor[b, :, :pad_left, :] = left_color.unsqueeze(0).unsqueeze(0)
                    if pad_right > 0:
                        out_tensor[b, :, pad_left+W:, :] = right_color.unsqueeze(0).unsqueeze(0)
            else:
                # 普通颜色填充模式
                fill_color = self._parse_color_advanced(pad_color, img_tensor[0])
                bg_color = torch.tensor(fill_color, dtype=img_tensor.dtype, device=img_tensor.device) / 255.0
                
                # 填充背景颜色
                for b in range(B):
                    out_tensor[b, :, :, :] = bg_color.unsqueeze(0).unsqueeze(0)
                    out_tensor[b, pad_top:pad_top+H, pad_left:pad_left+W, :] = img_tensor[b]
                    # 原图区域设为黑色（0）
                    out_mask[b, pad_top:pad_top+H, pad_left:pad_left+W] = 0.0
            
            img_tensor = out_tensor
            mask_tensor = out_mask
            print(f'填充图像：左{pad_left}，右{pad_right}，上{pad_top}，下{pad_bottom}，颜色{pad_color}')
            
            return img_tensor, mask_tensor
        else:
            # 只有裁剪操作或无操作的情况
            # 返回裁剪后的图像和原始尺寸的mask
            return img_tensor, original_mask

    def _parse_color_advanced(self, color_str, img_tensor=None):
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
        
        # 默认返回黑色
        return (0, 0, 0)
    
    def _get_average_color_tensor(self, img_tensor):
        """计算tensor图像的平均颜色"""
        # img_tensor shape: [H, W, C]
        avg_color = torch.mean(img_tensor, dim=(0, 1))  # 在H和W维度上求平均
        avg_color_255 = (avg_color * 255).int().tolist()
        return tuple(avg_color_255)
    
    def _get_edge_color_tensor(self, img_tensor):
        """获取tensor图像边缘的平均颜色"""
        H, W, C = img_tensor.shape
        
        # 获取所有边缘像素
        top_edge = img_tensor[0, :, :]  # 顶部边缘
        bottom_edge = img_tensor[H-1, :, :]  # 底部边缘
        left_edge = img_tensor[:, 0, :]  # 左侧边缘
        right_edge = img_tensor[:, W-1, :]  # 右侧边缘
        
        # 合并所有边缘像素
        all_edges = torch.cat([
            top_edge.reshape(-1, C),
            bottom_edge.reshape(-1, C),
            left_edge.reshape(-1, C),
            right_edge.reshape(-1, C)
        ], dim=0)
        
        # 计算平均颜色
        avg_color = torch.mean(all_edges, dim=0)
        avg_color_255 = (avg_color * 255).int().tolist()
        return tuple(avg_color_255)


class ImageGridSplit:
    """
    图片宫格分割器 - 将图片按指定行列分割成多个子图片
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "rows": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1}),
                "columns": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1}),
                "output_index": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "image_grid_split"
    CATEGORY = "1hewNodes/image/crop"

    def image_grid_split(self, image, rows, columns, output_index):
        """
        将图片按宫格分割
        
        Args:
            image: 输入图片张量 (batch, height, width, channels)
            rows: 分割行数
            columns: 分割列数
            output_index: 输出索引，0表示所有分割后的图片按批次输出，
                         1表示第1张，2表示第2张，以此类推（横向优先）
        
        Returns:
            分割后的图片张量
        """
        batch_size, height, width, channels = image.shape
        
        # 计算每个网格的尺寸
        grid_height = height // rows
        grid_width = width // columns
        
        # 总的网格数量
        total_grids = rows * columns
        
        # 验证输出索引
        if output_index > total_grids:
            raise ValueError(f"输出索引 {output_index} 超出范围，最大值为 {total_grids}")
        
        # 存储所有分割后的图片
        all_split_images = []
        
        for batch_idx in range(batch_size):
            current_image = image[batch_idx]
            
            # 按行列分割图片
            for row in range(rows):
                for col in range(columns):
                    # 计算当前网格的坐标
                    start_y = row * grid_height
                    end_y = start_y + grid_height
                    start_x = col * grid_width
                    end_x = start_x + grid_width
                    
                    # 裁剪当前网格
                    grid_image = current_image[start_y:end_y, start_x:end_x, :]
                    all_split_images.append(grid_image)
        
        # 根据输出索引返回结果
        if output_index == 0:
            # 返回所有分割后的图片作为批次
            result = torch.stack(all_split_images, dim=0)
        else:
            # 返回指定索引的图片
            selected_images = []
            for batch_idx in range(batch_size):
                # 计算在当前批次中的索引
                grid_idx = (output_index - 1) % total_grids
                actual_idx = batch_idx * total_grids + grid_idx
                
                if actual_idx < len(all_split_images):
                    selected_images.append(all_split_images[actual_idx])
                else:
                    # 如果索引超出范围，使用第一个图片
                    selected_images.append(all_split_images[0])
            
            result = torch.stack(selected_images, dim=0)
        
        return (result,)

    @staticmethod
    def tensor_to_pil(tensor):
        """将张量转换为PIL图像"""
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        # 确保值在0-1范围内
        tensor = torch.clamp(tensor, 0, 1)
        
        # 转换为numpy数组
        np_image = (tensor.cpu().numpy() * 255).astype(np.uint8)
        
        return Image.fromarray(np_image)

    @staticmethod
    def pil_to_tensor(pil_image):
        """将PIL图像转换为张量"""
        np_image = np.array(pil_image).astype(np.float32) / 255.0
        
        # 确保是RGB格式
        if len(np_image.shape) == 2:
            np_image = np.stack([np_image] * 3, axis=-1)
        elif np_image.shape[-1] == 4:
            np_image = np_image[:, :, :3]
        
        tensor = torch.from_numpy(np_image)
        
        # 添加批次维度
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        
        return tensor



NODE_CLASS_MAPPINGS = {
    "ImageCropSquare": ImageCropSquare,
    "ImageCropWithBBoxMask": ImageCropWithBBoxMask,
    "ImageCropByMaskAlpha": ImageCropByMaskAlpha,
    "ImagePasteByBBoxMask": ImagePasteByBBoxMask,
    "ImageEdgeCropPad": ImageEdgeCropPad,
    "ImageGridSplit": ImageGridSplit,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageCropSquare": "Image Crop Square",
    "ImageCropWithBBoxMask": "Image Crop with BBox Mask",
    "ImageCropByMaskAlpha": "Image Crop by Mask Alpha",
    "ImagePasteByBBoxMask": "Image Paste by BBox Mask",
    "ImageEdgeCropPad": "Image Edge Crop Pad",
    "ImageGridSplit": "Image Grid Split",
}