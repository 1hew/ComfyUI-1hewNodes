import asyncio
from fractions import Fraction
from math import gcd
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from comfy_api.latest import io

class ImageCropWithBBoxMask(io.ComfyNode):
    """
    图像裁切器 - 根据遮罩裁切图像，并返回边界框遮罩信息以便后续粘贴回原位置
    """
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_ImageCropWithBBoxMask",
            display_name="Image Crop With BBox Mask",
            category="1hewNodes/image/crop",
            inputs=[
                io.Image.Input("image"),
                io.Mask.Input("mask"),
                io.Combo.Input("preset_ratio", options=["mask", "image", "auto", "9:16", "2:3", "3:4", "4:5", "1:1", "5:4", "4:3", "3:2", "16:9", "21:9"], default="mask"),
                io.Float.Input("scale_strength", default=0.0, min=0.0, max=1.0, step=0.01),
                io.Combo.Input("crop_to_side", options=["None", "longest", "shortest", "width", "height"], default="None"),
                io.Int.Input("crop_to_length", default=1024, min=8, max=8192, step=1),
                io.Int.Input("divisible_by", default=8, min=1, max=1024, step=1),
            ],
            outputs=[
                io.Image.Output(display_name="cropped_image"),
                io.Mask.Output(display_name="bbox_mask"),
                io.Mask.Output(display_name="cropped_mask"),
            ],
        )

    @classmethod
    async def execute(
        cls,
        image: torch.Tensor,
        mask: torch.Tensor,
        preset_ratio: str,
        scale_strength: float,
        crop_to_side: str,
        crop_to_length: int,
        divisible_by: int,
    ) -> io.NodeOutput:
        image = image.to(torch.float32).clamp(0.0, 1.0)
        mask = mask.to(torch.float32).clamp(0.0, 1.0)

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
        
        async def _proc(b):
            def _do():
                try:
                    img_np = (
                        image[b].detach().cpu().numpy() * 255
                    ).astype(np.uint8)
                    mask_np = (
                        mask[b].detach().cpu().numpy() * 255
                    ).astype(np.uint8)
                    if len(img_np.shape) != 3 or img_np.shape[2] not in [3, 4]:
                        bbox_mask = torch.ones(
                            (img_height, img_width), dtype=torch.float32, device=image.device
                        )
                        return image[b], bbox_mask, mask[b]
                    img_pil = Image.fromarray(img_np)
                    mask_pil = Image.fromarray(mask_np).convert("L")
                    bbox = cls._get_bbox(mask_pil)
                    if bbox is None:
                        bbox_mask = torch.ones(
                            (img_height, img_width), dtype=torch.float32, device=image.device
                        )
                        return image[b], bbox_mask, mask[b]
                    x_min, y_min, x_max, y_max = bbox
                    mask_width = x_max - x_min
                    mask_height = y_max - y_min
                    mask_center_x = (x_min + x_max) / 2.0
                    mask_center_y = (y_min + y_max) / 2.0
                    orientation, target_ratio = cls.determine_ratio_orientation(
                        preset_ratio, img_width, img_height, mask_width, mask_height
                    )
                    if preset_ratio == "mask":
                        valid_sizes = cls.generate_flexible_mask_sizes(
                            mask_width, mask_height, divisible_by
                        )
                    else:
                        valid_sizes = cls.generate_valid_sizes(target_ratio, divisible_by)
                    if not valid_sizes:
                        bbox_mask = torch.ones(
                            (img_height, img_width), dtype=torch.float32, device=image.device
                        )
                        return image[b], bbox_mask, mask[b]
                    min_valid_size, max_valid_size = cls.find_valid_size_range(
                        valid_sizes, bbox, img_width, img_height, preset_ratio
                    )
                    if min_valid_size is None or max_valid_size is None:
                        bbox_mask = torch.ones(
                            (img_height, img_width), dtype=torch.float32, device=image.device
                        )
                        return image[b], bbox_mask, mask[b]
                    if crop_to_side == "None":
                        target_size = cls.calculate_target_size_from_range(
                            min_valid_size, max_valid_size, scale_strength, valid_sizes
                        )
                    else:
                        target_dimension = cls.map_crop_side_to_dimension(
                            crop_to_side, orientation
                        )
                        if target_dimension:
                            filtered_sizes = [
                                s for s in valid_sizes if s >= min_valid_size and s <= max_valid_size
                            ]
                            if filtered_sizes:
                                target_size = cls.find_target_size_by_length(
                                    filtered_sizes, target_dimension, crop_to_length
                                )
                            else:
                                target_size = min_valid_size
                        else:
                            target_size = min_valid_size
                    if target_size is None:
                        bbox_mask = torch.ones(
                            (img_height, img_width), dtype=torch.float32, device=image.device
                        )
                        return image[b], bbox_mask, mask[b]
                    target_width, target_height = target_size
                    crop_x1, crop_y1, crop_x2, crop_y2 = cls.calculate_flexible_crop_region(
                        mask_center_x, mask_center_y, target_width, target_height, img_width, img_height
                    )
                    cropped_img_pil = img_pil.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                    cropped_mask_pil = mask_pil.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                    cropped_img_np = (
                        np.array(cropped_img_pil).astype(np.float32) / 255.0
                    )
                    cropped_mask_np = (
                        np.array(cropped_mask_pil).astype(np.float32) / 255.0
                    )
                    if len(cropped_img_np.shape) == 2:
                        cropped_img_np = np.stack([cropped_img_np] * 3, axis=-1)
                    elif cropped_img_np.shape[2] == 4:
                        cropped_img_np = cropped_img_np[:, :, :3]
                    cropped_img_tensor = torch.from_numpy(cropped_img_np).to(image.device)
                    cropped_mask_tensor = torch.from_numpy(cropped_mask_np).to(mask.device)
                    bbox_mask = torch.zeros((img_height, img_width), dtype=torch.float32, device=image.device)
                    bbox_mask[crop_y1:crop_y2, crop_x1:crop_x2] = 1.0
                    return cropped_img_tensor, bbox_mask, cropped_mask_tensor
                except Exception:
                    bbox_mask = torch.ones((img_height, img_width), dtype=torch.float32, device=image.device)
                    return image[b], bbox_mask, mask[b]
            return await asyncio.to_thread(_do)

        results = await asyncio.gather(*[_proc(b) for b in range(final_batch_size)])
        cropped_images = [r[0] for r in results]
        bbox_masks = [r[1] for r in results]
        cropped_masks = [r[2] for r in results]
        
        img_sizes = [img.shape for img in cropped_images]
        if len(set(img_sizes)) > 1:
            cropped_images = cls._pad_images_to_same_size(cropped_images)
        mask_sizes = [m.shape for m in cropped_masks]
        if len(set(mask_sizes)) > 1:
            cropped_masks = cls._pad_masks_to_same_size(cropped_masks)

        cropped_images = torch.stack(cropped_images).to(image.device)
        bbox_masks = torch.stack(bbox_masks).to(image.device)
        cropped_masks = torch.stack(cropped_masks).to(mask.device)

        cropped_images = cropped_images.to(torch.float32).clamp(0.0, 1.0)
        bbox_masks = bbox_masks.to(torch.float32).clamp(0.0, 1.0)
        cropped_masks = cropped_masks.to(torch.float32).clamp(0.0, 1.0)

        return io.NodeOutput(cropped_images, bbox_masks, cropped_masks)


    @staticmethod
    def _get_bbox(mask_pil):
        """获取遮罩的边界框"""
        bbox = mask_pil.getbbox()
        if bbox is None:
            return None
        return bbox

    @staticmethod
    def determine_ratio_orientation(
        preset_ratio, img_width, img_height, mask_width, mask_height
    ):
        """判断比例的方向（横图、竖图、方图）"""
        if preset_ratio == "mask":
            ratio = mask_width / mask_height
        elif preset_ratio == "image":
            ratio = img_width / img_height
        elif preset_ratio == "auto":
            img_ratio = img_width / img_height
            candidates = [
                (1, 1), (3, 2), (4, 3), (5, 4), (16, 9), (21, 9),
                (2, 3), (3, 4), (4, 5), (9, 16)
            ]
            best = min(
                candidates,
                key=lambda wh: abs(img_ratio - (wh[0] / wh[1]))
            )
            ratio = best[0] / best[1]
        else:
            try:
                parts = preset_ratio.split(":")
                if len(parts) == 2:
                    w, h = float(parts[0]), float(parts[1])
                    ratio = w / h
                else:
                    ratio = mask_width / mask_height
            except:
                ratio = mask_width / mask_height
        
        if abs(ratio - 1.0) < 0.01:
            return "square", ratio
        elif ratio > 1.0:
            return "landscape", ratio
        else:
            return "portrait", ratio

    @staticmethod
    def map_crop_side_to_dimension(crop_to_side, orientation):
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

    @staticmethod
    def preprocess_mask_dimensions_flexible(mask_width, mask_height, divisible_by):
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
        
        
        
        return unique_candidates

    @staticmethod
    def generate_flexible_valid_sizes(
        candidates, divisible_by, min_size=8, max_size=4096
    ):
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
        
        
        
        return valid_sizes

    @staticmethod
    def parse_ratio_exact(
        preset_ratio,
        img_width,
        img_height,
        mask_width,
        mask_height,
        divisible_by=8,
    ):
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

    @staticmethod
    def generate_flexible_mask_sizes(
        mask_width, mask_height, divisible_by, min_size=8, max_size=4096
    ):
        """专门为 mask 生成灵活的有效尺寸"""
        
        
        # 使用灵活的预处理方式
        candidates = ImageCropWithBBoxMask.preprocess_mask_dimensions_flexible(
            mask_width, mask_height, divisible_by
        )
        
        if not candidates:
            
            # 回退方案
            adjusted_width = ((mask_width + divisible_by - 1) // divisible_by) * divisible_by
            adjusted_height = ((mask_height + divisible_by - 1) // divisible_by) * divisible_by
            candidates = [(adjusted_width, adjusted_height, adjusted_width/adjusted_height, 0)]
        
        # 生成灵活的有效尺寸
        valid_sizes = ImageCropWithBBoxMask.generate_flexible_valid_sizes(
            candidates, divisible_by, min_size, max_size
        )
        
        if not valid_sizes:
            
            return [(divisible_by, divisible_by)]
        
        
        
        return valid_sizes

    @staticmethod
    def generate_valid_sizes(
        target_ratio, divisible_by, min_size=8, max_size=4096
    ):
        """生成所有满足比例和divisible_by要求的有效尺寸"""
        valid_sizes = set()  # 使用set避免重复
        
        # 解析目标比例为分数形式
        if isinstance(target_ratio, str) and ":" in target_ratio:
            parts = target_ratio.split(":")
            ratio_w, ratio_h = int(parts[0]), int(parts[1])
        else:
            # 对于预处理后的 mask，直接使用简化的分数转换
            ratio_w, ratio_h = ImageCropWithBBoxMask._simple_float_to_fraction(
                target_ratio, divisible_by
            )
        
        # 计算最大公约数，简化比例
        g = gcd(ratio_w, ratio_h)
        ratio_w //= g
        ratio_h //= g
        
        
        
        # 由于比例已经预处理过，基础尺寸就是比例本身乘以 divisible_by
        base_width = ratio_w * divisible_by
        base_height = ratio_h * divisible_by
        
        
        
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
    
    @staticmethod
    def _simple_float_to_fraction(ratio, divisible_by):
        """为预处理后的比例提供简单的分数转换"""
        # 尝试不同的分母，找到最接近的整数比例
        for denominator in range(1, 101):  # 限制在合理范围内
            numerator = round(ratio * denominator)
            if numerator > 0:
                actual_ratio = numerator / denominator
                if abs(actual_ratio - ratio) < 0.001:  # 精度足够
                    return numerator, denominator
        
        # 如果找不到合适的，使用默认方法
        frac = Fraction(ratio).limit_denominator(100)
        return frac.numerator, frac.denominator
    
    @staticmethod
    def find_valid_size_range(
        valid_sizes, mask_bbox, img_width, img_height, preset_ratio="mask"
    ):
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

    @staticmethod
    def calculate_target_size_from_range(
        min_size, max_size, scale_strength, valid_sizes
    ):
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

    @staticmethod
    def find_target_size_by_length(valid_sizes, target_dimension, target_length):
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
            
        
        return best_size

    @staticmethod
    def calculate_flexible_crop_region(
        center_x, center_y, target_width, target_height, img_width, img_height
    ):
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

    @staticmethod
    def _pad_images_to_same_size(images):
        max_h = max(img.shape[0] for img in images)
        max_w = max(img.shape[1] for img in images)
        max_c = max(img.shape[2] for img in images)
        padded = []
        for img in images:
            h, w, c = img.shape
            pad_h = max_h - h
            pad_w = max_w - w
            pad_c = max_c - c
            padded_img = F.pad(img, (0, pad_c, 0, pad_w, 0, pad_h), value=0)
            padded.append(padded_img)
        return padded

    @staticmethod
    def _pad_masks_to_same_size(masks):
        max_h = max(m.shape[0] for m in masks)
        max_w = max(m.shape[1] for m in masks)
        padded = []
        for m in masks:
            h, w = m.shape
            pad_h = max_h - h
            pad_w = max_w - w
            padded_mask = F.pad(m, (0, pad_w, 0, pad_h), value=0)
            padded.append(padded_mask)
        return padded
