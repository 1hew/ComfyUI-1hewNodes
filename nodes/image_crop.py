import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

class ImageCropSquare:
    """
    图像方形裁剪器 - 根据遮罩裁切图像为方形，支持放大系数和填充颜色
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "scale_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.01}),
                "apply_mask": ("BOOLEAN", {"default": False}),
                "extra_padding": ("INT", {"default": 0, "min": 0, "max": 512, "step": 1}),
                "fill_color": ("STRING", {"default": "1.0"})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "image_crop_square"
    CATEGORY = "1hewNodes/image/crop"

    def image_crop_square(self, image, mask, scale_factor=1.0, fill_color="1.0", apply_mask=False, extra_padding=0):
        # 获取图像尺寸
        batch_size, height, width, channels = image.shape

        # 创建输出图像列表
        output_images = []

        for b in range(batch_size):
            # 将图像转换为PIL格式
            if image.is_cuda:
                img_np = (image[b].cpu().numpy() * 255).astype(np.uint8)
                mask_np = (mask[b % mask.shape[0]].cpu().numpy() * 255).astype(np.uint8)
            else:
                img_np = (image[b].numpy() * 255).astype(np.uint8)
                mask_np = (mask[b % mask.shape[0]].numpy() * 255).astype(np.uint8)

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

            # 如果没有找到有效区域，返回原始图像
            if bbox is None:
                output_images.append(image[b])
                continue

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
        

class ImageCropWithBBox:
    """
    图像裁切器 - 根据遮罩裁切图像，并返回边界框信息以便后续粘贴回原位置
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "aspect_ratio": (["mask_ratio", "1:1", "3:2", "4:3", "16:9", "21:9", "2:3", "3:4", "9:16", "9:21"], {"default": "mask_ratio"}),
                "scale_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "extra_padding": ("INT", {"default": 0, "min": 0, "max": 512, "step": 1}),
                "exceed_image": ("BOOLEAN", {"default": False}),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "fill_color": ("STRING", {"default": "1.0"}),
                "divisible_by": ("INT", {"default": 8, "min": 1, "max": 1024, "step": 1})
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "DICT")
    RETURN_NAMES = ("cropped_image", "cropped_mask", "bbox_meta")
    FUNCTION = "image_crop_with_bbox"
    CATEGORY = "1hewNodes/image/crop"

    def image_crop_with_bbox(self, image, mask, invert_mask=False, extra_padding=0, aspect_ratio="mask_ratio", scale_factor=1.0, 
                            exceed_image=False, fill_color="1.0", divisible_by=8, resize_mode="none", target_width=512, target_height=512):
        # 获取图像尺寸
        batch_size, height, width, channels = image.shape
        mask_batch_size = mask.shape[0]

        # 创建输出图像和遮罩列表
        output_images = []
        output_masks = []
        output_bboxes_list = []
        original_sizes = []  # 记录原始裁剪尺寸

        for b in range(batch_size):
            # 将图像转换为PIL格式
            if image.is_cuda:
                img_np = (image[b].cpu().numpy() * 255).astype(np.uint8)
                mask_np = (mask[b % mask_batch_size].cpu().numpy() * 255).astype(np.uint8)
            else:
                img_np = (image[b].numpy() * 255).astype(np.uint8)
                mask_np = (mask[b % mask_batch_size].numpy() * 255).astype(np.uint8)

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

            # 如果需要反转遮罩
            if invert_mask:
                mask_pil = ImageOps.invert(mask_pil)

            # 找到遮罩中非零区域的边界框
            bbox = self.get_bbox(mask_pil, extra_padding)

            # 如果没有找到有效区域，返回原始图像
            if bbox is None:
                output_images.append(image[b])
                output_masks.append(mask[b % mask_batch_size])
                # 使用整个图像作为边界框
                output_bboxes_list.append({
                    "x_min": 0,
                    "y_min": 0,
                    "x_max": width,
                    "y_max": height,
                    "width": width,
                    "height": height
                })
                original_sizes.append((width, height))
                continue

            # 根据选择的比例调整边界框
            if aspect_ratio != "mask_ratio":
                bbox = self.adjust_bbox_aspect_ratio(bbox, aspect_ratio, img_pil.size, exceed_image)
            
            # 应用缩放系数
            if scale_factor != 1.0:
                bbox = self.apply_scale_factor(bbox, scale_factor, img_pil.size, exceed_image)
            
            # 调整尺寸以满足整除要求
            if divisible_by > 1:
                bbox = self.adjust_for_divisibility(bbox, divisible_by, img_pil.size, exceed_image)

            # 获取最终的边界框坐标
            x_min, y_min, x_max, y_max = bbox
            crop_width = x_max - x_min
            crop_height = y_max - y_min
            
            # 如果允许超出图像范围，创建带填充的画布
            if exceed_image and (x_min < 0 or y_min < 0 or x_max > img_pil.width or y_max > img_pil.height):
                # 解析填充颜色
                bg_color = self.parse_color(fill_color, img_pil, bbox)
                
                # 创建新的画布
                canvas_img = Image.new("RGB", (crop_width, crop_height), bg_color)
                canvas_mask = Image.new("L", (crop_width, crop_height), 0)
                
                # 计算原图在新画布上的位置
                paste_x = max(0, -x_min)
                paste_y = max(0, -y_min)
                
                # 计算从原图裁剪的区域
                crop_x1 = max(0, x_min)
                crop_y1 = max(0, y_min)
                crop_x2 = min(img_pil.width, x_max)
                crop_y2 = min(img_pil.height, y_max)
                
                # 裁剪原图并粘贴到新画布上
                if crop_x1 < crop_x2 and crop_y1 < crop_y2:
                    cropped_region = img_pil.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                    cropped_mask_region = mask_pil.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                    
                    canvas_img.paste(cropped_region, (paste_x, paste_y))
                    canvas_mask.paste(cropped_mask_region, (paste_x, paste_y))
                
                cropped_img = canvas_img
                cropped_mask = canvas_mask
            else:
                # 确保边界框不超出图像范围
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(img_pil.width, x_max)
                y_max = min(img_pil.height, y_max)
                
                # 裁切图像和遮罩
                cropped_img = img_pil.crop((x_min, y_min, x_max, y_max))
                cropped_mask = mask_pil.crop((x_min, y_min, x_max, y_max))

            # 记录原始裁剪尺寸
            original_sizes.append(cropped_img.size)
            
            # 转换回tensor
            cropped_img_np = np.array(cropped_img).astype(np.float32) / 255.0
            cropped_mask_np = np.array(cropped_mask).astype(np.float32) / 255.0

            output_images.append(torch.from_numpy(cropped_img_np))
            output_masks.append(torch.from_numpy(cropped_mask_np))
            # 将边界框转换为字典格式
            output_bboxes_list.append({
                "x_min": x_min,
                "y_min": y_min, 
                "x_max": x_max,
                "y_max": y_max,
                "width": x_max - x_min,
                "height": y_max - y_min,
                "original_width": cropped_img.size[0],
                "original_height": cropped_img.size[1]
            })
        
        # 处理批量尺寸统一
        if batch_size > 1 and resize_mode != "none":
            output_images, output_masks = self.resize_batch_images(output_images, output_masks, resize_mode, target_width, target_height, original_sizes)
        
        # 合并批次 - 只有在所有图像尺寸相同时才能使用 stack
        try:
            output_image_tensor = torch.stack(output_images)
            output_mask_tensor = torch.stack(output_masks)
        except RuntimeError as e:
            if "stack expects each tensor to be equal size" in str(e):
                # 如果尺寸不匹配，使用最大尺寸进行填充
                output_images, output_masks = self.pad_to_same_size(output_images, output_masks)
                output_image_tensor = torch.stack(output_images)
                output_mask_tensor = torch.stack(output_masks)
            else:
                raise e
        
        # 统一返回格式：单张图像返回字典，多张图像返回带索引的字典
        if batch_size == 1:
            bbox_meta = output_bboxes_list[0]
        else:
            bbox_meta = {
                "batch_size": batch_size,
                "bboxes": output_bboxes_list,
                "format": "multi_image",
                "original_sizes": original_sizes
            }
        
        return (output_image_tensor, output_mask_tensor, bbox_meta)

    def resize_batch_images(self, images, masks, resize_mode, target_width, target_height, original_sizes):
        """统一批量图像尺寸"""
        if resize_mode == "max_size":
            # 找到最大尺寸
            max_width = max(size[0] for size in original_sizes)
            max_height = max(size[1] for size in original_sizes)
            target_size = (max_width, max_height)
        elif resize_mode == "fixed_size":
            target_size = (target_width, target_height)
        else:
            return images, masks
        
        resized_images = []
        resized_masks = []
        
        for i, (img_tensor, mask_tensor) in enumerate(zip(images, masks)):
            # 转换为PIL进行resize
            img_np = (img_tensor.numpy() * 255).astype(np.uint8)
            mask_np = (mask_tensor.numpy() * 255).astype(np.uint8)
            
            img_pil = Image.fromarray(img_np)
            mask_pil = Image.fromarray(mask_np)
            
            # 调整尺寸
            img_resized = img_pil.resize(target_size, Image.LANCZOS)
            mask_resized = mask_pil.resize(target_size, Image.LANCZOS)
            
            # 转换回tensor
            img_resized_np = np.array(img_resized).astype(np.float32) / 255.0
            mask_resized_np = np.array(mask_resized).astype(np.float32) / 255.0
            
            resized_images.append(torch.from_numpy(img_resized_np))
            resized_masks.append(torch.from_numpy(mask_resized_np))
        
        return resized_images, resized_masks

    def pad_to_same_size(self, images, masks):
        """将所有图像填充到相同尺寸"""
        # 找到最大尺寸
        max_height = max(img.shape[0] for img in images)
        max_width = max(img.shape[1] for img in images)
        
        padded_images = []
        padded_masks = []
        
        for img, mask in zip(images, masks):
            h, w = img.shape[:2]
            
            # 计算填充量
            pad_h = max_height - h
            pad_w = max_width - w
            
            # 对图像进行填充 (在右下角填充)
            if len(img.shape) == 3:  # RGB图像
                padded_img = torch.nn.functional.pad(img, (0, 0, 0, pad_w, 0, pad_h), value=0)
            else:  # 灰度图像
                padded_img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), value=0)
            
            # 对遮罩进行填充
            padded_mask = torch.nn.functional.pad(mask, (0, pad_w, 0, pad_h), value=0)
            
            padded_images.append(padded_img)
            padded_masks.append(padded_mask)
        
        return padded_images, padded_masks

    def get_bbox(self, mask_pil, extra_padding=0):
        """从遮罩中获取边界框"""
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

        # 添加边距
        x_min = max(0, x_min - extra_padding)
        y_min = max(0, y_min - extra_padding)
        x_max = min(mask_pil.width - 1, x_max + extra_padding)
        y_max = min(mask_pil.height - 1, y_max + extra_padding)

        # 返回边界框 (left, top, right, bottom)
        return (x_min, y_min, x_max + 1, y_max + 1)
    
    def adjust_bbox_aspect_ratio(self, bbox, aspect_ratio, img_size, exceed_image=False):
        """根据指定的宽高比调整边界框"""
        x_min, y_min, x_max, y_max = bbox
        
        # 计算当前宽高
        width = x_max - x_min
        height = y_max - y_min
        
        # 计算中心点
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        
        # 解析目标宽高比
        if aspect_ratio == "1:1":
            target_ratio = 1/1
        elif aspect_ratio == "3:2":
            target_ratio = 3/2
        elif aspect_ratio == "4:3":
            target_ratio = 4/3
        elif aspect_ratio == "16:9":
            target_ratio = 16/9
        elif aspect_ratio == "21:9":
            target_ratio = 21/9
        elif aspect_ratio == "2:3":
            target_ratio = 2/3
        elif aspect_ratio == "3:4":
            target_ratio = 3/4
        elif aspect_ratio == "9:16":
            target_ratio = 9/16
        elif aspect_ratio == "9:21":
            target_ratio = 9/21
        else:
            # 保持原始比例
            return bbox
        
        # 计算当前比例
        current_ratio = width / height if height > 0 else 1
        
        # 调整边界框以匹配目标比例
        if current_ratio > target_ratio:
            # 当前比例更宽，需要增加高度
            new_height = width / target_ratio
            y_min = center_y - new_height / 2
            y_max = center_y + new_height / 2
        else:
            # 当前比例更高，需要增加宽度
            new_width = height * target_ratio
            x_min = center_x - new_width / 2
            x_max = center_x + new_width / 2
        
        # 如果不允许超出图像范围，则进行限制
        if not exceed_image:
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(img_size[0], x_max)
            y_max = min(img_size[1], y_max)
        
        return (int(x_min), int(y_min), int(x_max), int(y_max))
    
    def apply_scale_factor(self, bbox, scale_factor, img_size, exceed_image=False):
        """应用缩放系数到边界框，保持中心点和比例不变"""
        x_min, y_min, x_max, y_max = bbox
        
        # 计算中心点
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        
        # 计算当前宽高
        width = x_max - x_min
        height = y_max - y_min
        
        # 应用缩放系数
        new_width = width * scale_factor
        new_height = height * scale_factor
        
        # 计算新的边界框
        new_x_min = center_x - new_width / 2
        new_y_min = center_y - new_height / 2
        new_x_max = center_x + new_width / 2
        new_y_max = center_y + new_height / 2
        
        # 如果不允许超出图像范围，则进行限制
        if not exceed_image:
            new_x_min = max(0, new_x_min)
            new_y_min = max(0, new_y_min)
            new_x_max = min(img_size[0], new_x_max)
            new_y_max = min(img_size[1], new_y_max)
        
        return (int(new_x_min), int(new_y_min), int(new_x_max), int(new_y_max))
    
    def adjust_for_divisibility(self, bbox, divisible_by, img_size, exceed_image=False):
        """调整边界框使宽高可被指定整数整除"""
        x_min, y_min, x_max, y_max = bbox
        
        # 计算当前宽高
        width = x_max - x_min
        height = y_max - y_min
        
        # 计算需要调整的量，使宽高可被整除
        width_remainder = width % divisible_by
        height_remainder = height % divisible_by
        
        # 如果已经可以整除，不需要调整
        if width_remainder == 0 and height_remainder == 0:
            return bbox
        
        # 计算需要增加的宽高
        width_add = 0 if width_remainder == 0 else divisible_by - width_remainder
        height_add = 0 if height_remainder == 0 else divisible_by - height_remainder
        
        # 计算中心点
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        
        # 计算新的边界框，保持中心点不变
        new_width = width + width_add
        new_height = height + height_add
        
        new_x_min = center_x - new_width / 2
        new_y_min = center_y - new_height / 2
        new_x_max = center_x + new_width / 2
        new_y_max = center_y + new_height / 2
        
        # 如果不允许超出图像范围，则进行限制
        if not exceed_image:
            new_x_min = max(0, new_x_min)
            new_y_min = max(0, new_y_min)
            new_x_max = min(img_size[0], new_x_max)
            new_y_max = min(img_size[1], new_y_max)
            
            # 重新检查调整后的尺寸是否满足整除要求
            adjusted_width = new_x_max - new_x_min
            adjusted_height = new_y_max - new_y_min
            
            # 如果调整后不满足整除要求，则缩小尺寸
            if adjusted_width % divisible_by != 0:
                new_width = (adjusted_width // divisible_by) * divisible_by
                new_x_min = center_x - new_width / 2
                new_x_max = center_x + new_width / 2
                
                # 确保不超出图像范围
                if new_x_min < 0:
                    new_x_min = 0
                    new_x_max = new_width
                if new_x_max > img_size[0]:
                    new_x_max = img_size[0]
                    new_x_min = new_x_max - new_width
            
            if adjusted_height % divisible_by != 0:
                new_height = (adjusted_height // divisible_by) * divisible_by
                new_y_min = center_y - new_height / 2
                new_y_max = center_y + new_height / 2
                
                # 确保不超出图像范围
                if new_y_min < 0:
                    new_y_min = 0
                    new_y_max = new_height
                if new_y_max > img_size[1]:
                    new_y_max = img_size[1]
                    new_y_min = new_y_max - new_height
        
        return (int(new_x_min), int(new_y_min), int(new_x_max), int(new_y_max))
    
    def parse_color(self, color_str, img, bbox=None):
        """解析颜色字符串为RGB元组"""
        # 处理边缘颜色
        if color_str.lower() in ["e", "edge"]:
            if bbox is not None:
                # 获取边缘颜色
                return self.get_edge_color(img, bbox)
            else:
                return (255, 255, 255)  # 默认白色
        
        # 处理灰度值 (0.0-1.0)
        try:
            gray_value = float(color_str)
            if 0 <= gray_value <= 1:
                gray_int = int(gray_value * 255)
                return (gray_int, gray_int, gray_int)
        except ValueError:
            pass
        
        # 处理十六进制颜色
        if color_str.startswith("#"):
            color_str = color_str.lstrip("#")
            try:
                if len(color_str) == 6:
                    r = int(color_str[0:2], 16)
                    g = int(color_str[2:4], 16)
                    b = int(color_str[4:6], 16)
                    return (r, g, b)
                elif len(color_str) == 3:
                    r = int(color_str[0] + color_str[0], 16)
                    g = int(color_str[1] + color_str[1], 16)
                    b = int(color_str[2] + color_str[2], 16)
                    return (r, g, b)
            except ValueError:
                pass
        
        # 处理RGB格式 (r,g,b) 或 r,g,b
        if color_str.startswith("(") and color_str.endswith(")"):
            color_str = color_str.strip("()")
        
        # 处理不带括号的RGB格式
        try:
            rgb = color_str.split(",")
            if len(rgb) == 3:
                r = int(rgb[0].strip())
                g = int(rgb[1].strip())
                b = int(rgb[2].strip())
                return (r, g, b)
        except ValueError:
            pass
        
        # 默认返回白色
        return (255, 255, 255)
    
    def get_edge_color(self, img, bbox):
        """获取边界框边缘的平均颜色"""
        x_min, y_min, x_max, y_max = bbox
        width, height = img.size
        
        # 确保坐标在图像范围内
        x_min = max(0, min(x_min, width-1))
        y_min = max(0, min(y_min, height-1))
        x_max = max(0, min(x_max, width))
        y_max = max(0, min(y_max, height))
        
        # 收集边缘像素
        edge_pixels = []
        
        # 上边缘
        if 0 <= y_min < height:
            for x in range(max(0, x_min), min(width, x_max)):
                edge_pixels.append(img.getpixel((x, y_min)))
        
        # 下边缘
        if 0 <= y_max-1 < height:
            for x in range(max(0, x_min), min(width, x_max)):
                edge_pixels.append(img.getpixel((x, y_max-1)))
        
        # 左边缘
        if 0 <= x_min < width:
            for y in range(max(0, y_min+1), min(height, y_max-1)):
                edge_pixels.append(img.getpixel((x_min, y)))
        
        # 右边缘
        if 0 <= x_max-1 < width:
            for y in range(max(0, y_min+1), min(height, y_max-1)):
                edge_pixels.append(img.getpixel((x_max-1, y)))
        
        # 如果没有有效的边缘像素，返回白色
        if not edge_pixels:
            return (255, 255, 255)
        
        # 计算平均颜色
        r_sum = sum(p[0] for p in edge_pixels)
        g_sum = sum(p[1] for p in edge_pixels)
        b_sum = sum(p[2] for p in edge_pixels)
        
        pixel_count = len(edge_pixels)
        return (r_sum // pixel_count, g_sum // pixel_count, b_sum // pixel_count)


class ImageBBoxCrop:
    """
    图像检测框裁剪 - 根据边界框信息批量裁剪图像
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "bbox_meta": ("DICT",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("cropped_image",)
    FUNCTION = "image_bbox_crop"
    CATEGORY = "1hewNodes/image/crop"

    def image_bbox_crop(self, image, bbox_meta):
        # 获取图像尺寸
        batch_size, height, width, channels = image.shape

        # 创建输出图像列表
        output_images = []
        
        # 处理 bbox_meta 格式
        if isinstance(bbox_meta, dict):
            if "batch_size" in bbox_meta and "bboxes" in bbox_meta:
                # 多图像格式
                bboxes_list = bbox_meta["bboxes"]
            else:
                # 单图像格式，转换为列表
                bboxes_list = [bbox_meta]
        else:
            raise ValueError("bbox_meta must be a dictionary")
        
        for b in range(batch_size):
            # 将图像转换为PIL格式
            if image.is_cuda:
                img_np = (image[b].cpu().numpy() * 255).astype(np.uint8)
            else:
                img_np = (image[b].numpy() * 255).astype(np.uint8)
            
            img_pil = Image.fromarray(img_np)
            
            # 获取当前批次对应的边界框字典
            bbox_dict = bboxes_list[b % len(bboxes_list)]
            
            # 使用原始尺寸信息（如果可用）
            if "original_width" in bbox_dict and "original_height" in bbox_dict:
                target_width = bbox_dict["original_width"]
                target_height = bbox_dict["original_height"]
            else:
                target_width = bbox_dict["width"]
                target_height = bbox_dict["height"]
            
            x_min = bbox_dict["x_min"]
            y_min = bbox_dict["y_min"]
            x_max = bbox_dict["x_max"]
            y_max = bbox_dict["y_max"]
            
            # 确保边界框不超出图像范围
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(img_pil.width, x_max)
            y_max = min(img_pil.height, y_max)
            
            # 裁切图像
            cropped_img = img_pil.crop((x_min, y_min, x_max, y_max))
            
            # 如果需要调整到原始尺寸
            if cropped_img.size != (target_width, target_height):
                cropped_img = cropped_img.resize((target_width, target_height), Image.LANCZOS)
            
            # 转换回tensor
            cropped_img_np = np.array(cropped_img).astype(np.float32) / 255.0
            output_images.append(torch.from_numpy(cropped_img_np))
        
        # 合并批次
        if output_images:
            output_image_tensor = torch.stack(output_images)
            return (output_image_tensor,)
        else:
            # 如果没有有效的输出图像，返回原始图像
            return (image,)


class ImageBBoxPaste:
    """
    图像裁切后粘贴器 - 将处理后的裁剪图像粘贴回原始图像的位置
    支持智能批次处理：自动循环复制较少的输入以匹配最大批次大小
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image": ("IMAGE",),
                "cropped_image": ("IMAGE",),
                "bbox_meta": ("DICT",),
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
    FUNCTION = "image_bbox_paste"
    CATEGORY = "1hewNodes/image/crop"

    def image_bbox_paste(self, base_image, cropped_image, bbox_meta, blend_mode="normal", opacity=1.0, cropped_mask=None):
        # 获取各输入的批次大小
        base_batch_size = base_image.shape[0]
        cropped_batch_size = cropped_image.shape[0]
        
        # 处理 bbox_meta 格式并获取其批次大小
        if isinstance(bbox_meta, dict):
            if "batch_size" in bbox_meta and "bboxes" in bbox_meta:
                # 多图像格式
                bboxes_list = bbox_meta["bboxes"]
                bbox_batch_size = len(bboxes_list)
            else:
                # 单图像格式，转换为列表
                bboxes_list = [bbox_meta]
                bbox_batch_size = 1
        else:
            raise ValueError("bbox_meta must be a dictionary")
        
        # 获取遮罩批次大小（如果存在）
        mask_batch_size = cropped_mask.shape[0] if cropped_mask is not None else 1
        
        # 确定最大批次大小
        max_batch_size = max(base_batch_size, cropped_batch_size, bbox_batch_size, mask_batch_size)
        
        # 创建输出图像列表
        output_images = []
        
        for b in range(max_batch_size):
            # 使用循环索引获取对应的输入
            base_idx = b % base_batch_size
            cropped_idx = b % cropped_batch_size
            bbox_idx = b % bbox_batch_size
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
            
            base_pil = Image.fromarray(base_np)
            cropped_pil = Image.fromarray(cropped_np)
            
            # 获取当前批次对应的边界框字典
            bbox_dict = bboxes_list[bbox_idx]
            x_min = bbox_dict["x_min"]
            y_min = bbox_dict["y_min"]
            x_max = bbox_dict["x_max"]
            y_max = bbox_dict["y_max"]
            
            # 处理遮罩
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
            
            # 执行粘贴操作
            result_pil = self.paste_image(base_pil, cropped_pil, (x_min, y_min, x_max, y_max), blend_mode, opacity, mask_pil)
            
            # 转换回tensor
            result_np = np.array(result_pil).astype(np.float32) / 255.0
            output_images.append(torch.from_numpy(result_np))
        
        # 合并批次
        output_image_tensor = torch.stack(output_images)
        return (output_image_tensor,)

    def paste_image(self, base_img, cropped_img, bbox, blend_mode, opacity, mask=None):
        """将处理后的图像粘贴到原始图像上"""
        x_min, y_min, x_max, y_max = bbox
        crop_width = x_max - x_min
        crop_height = y_max - y_min
        
        # 如果处理后的图像尺寸与裁剪区域不匹配，调整大小
        if cropped_img.size != (crop_width, crop_height):
            cropped_img = cropped_img.resize((crop_width, crop_height), Image.LANCZOS)
        
        # 创建结果图像的副本
        result_img = base_img.copy()
        
        # 获取原始裁剪区域
        orig_crop = base_img.crop((x_min, y_min, x_max, y_max))
        
        # 应用混合模式
        if blend_mode != "normal":
            cropped_img = self.blend_images(orig_crop, cropped_img, blend_mode)
        
        # 应用不透明度
        if opacity < 1.0:
            cropped_img = Image.blend(orig_crop, cropped_img, opacity)
        
        # 粘贴处理后的图像
        result_img.paste(cropped_img, (x_min, y_min), mask)
        
        return result_img

    def blend_images(self, img1, img2, mode):
        """应用不同的混合模式"""
        img1 = img1.convert("RGB")
        img2 = img2.convert("RGB")

        if mode == "normal":
            return img2

        # 将图像转换为numpy数组以便进行混合计算
        img1_np = np.array(img1).astype(np.float32) / 255.0
        img2_np = np.array(img2).astype(np.float32) / 255.0

        if mode == "multiply":
            result_np = img1_np * img2_np
        elif mode == "screen":
            result_np = 1 - (1 - img1_np) * (1 - img2_np)
        elif mode == "overlay":
            mask = img1_np <= 0.5
            result_np = np.zeros_like(img1_np)
            result_np[mask] = 2 * img1_np[mask] * img2_np[mask]
            result_np[~mask] = 1 - 2 * (1 - img1_np[~mask]) * (1 - img2_np[~mask])
        elif mode == "soft_light":
            result_np = (1 - 2 * img2_np) * img1_np ** 2 + 2 * img2_np * img1_np
        elif mode == "difference":
            result_np = np.abs(img1_np - img2_np)
        else:
            return img2

        # 将结果转换回PIL图像
        result_np = np.clip(result_np * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(result_np)


NODE_CLASS_MAPPINGS = {
    "ImageCropSquare": ImageCropSquare,
    "ImageCropEdge": ImageCropEdge,
    "ImageCropWithBBox": ImageCropWithBBox,
    "ImageBBoxCrop": ImageBBoxCrop,
    "ImageBBoxPaste": ImageBBoxPaste,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageCropSquare": "Image Crop Square",
    "ImageCropEdge": "Image Crop Edge",
    "ImageCropWithBBox": "Image Crop With BBox",
    "ImageBBoxCrop": "Image BBox Crop",
    "ImageBBoxPaste": "Image BBox Paste",
}