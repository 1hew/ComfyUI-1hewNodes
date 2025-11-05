import torch
import numpy as np
from PIL import Image, ImageColor
import cv2
import math
import time
import re
from collections import defaultdict


class ImageBatchExtract:
    """
    批量图像提取节点
    支持多种提取模式：自定义索引、步长间隔、总帧数自动计算
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["index", "step", "uniform"], {"default": "step"}),
                "index": ("STRING", {"default": "0"}),
                "step": ("INT", {"default": 4, "min": 1, "max": 8192, "step": 1}),
                "uniform": ("INT", {"default": 4, "min": 0, "max": 8192, "step": 1}),
                "max_keep": ("INT", {"default": 10, "min": 0, "max": 8192, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "extract_batch"
    CATEGORY = "1hewNodes/batch"
    
    def extract_batch(self, image, mode, index="", step=1, uniform=0, max_keep=1024):
        try:
            batch_size = image.shape[0]
            print(f"[ImageBatchExtract] 输入批量图像信息: 形状={image.shape}, 总帧数={batch_size}")
            print(f"[ImageBatchExtract] 提取参数: 模式={mode}, 索引='{index}', 步长={step}, 数量={uniform}, 最大保留={max_keep}")
            
            # 根据模式确定提取索引
            extract_indices = self._get_extract_indices(batch_size, mode, index, step, uniform)
            
            if not extract_indices:
                print(f"[ImageBatchExtract] 没有有效的提取索引，返回空结果")
                empty_image = torch.empty((0,) + image.shape[1:], 
                                        dtype=image.dtype, device=image.device)
                return (empty_image,)
            
            # 提取图像
            extracted_images = []
            valid_indices = []
            
            for idx in extract_indices:
                if 0 <= idx < batch_size:
                    extracted_images.append(image[idx:idx+1])
                    valid_indices.append(idx)
                else:
                    print(f"[ImageBatchExtract] 跳过超出范围的索引: {idx} (总帧数: {batch_size})")
            
            if not extracted_images:
                print(f"[ImageBatchExtract] 所有索引都超出范围，返回空结果")
                empty_image = torch.empty((0,) + image.shape[1:], 
                                        dtype=image.dtype, device=image.device)
                return (empty_image,)
            
            # 应用最大保留限制（max_keep=0表示不限制）
            if max_keep > 0 and len(extracted_images) > max_keep:
                print(f"[ImageBatchExtract] 应用最大保留限制: {len(extracted_images)} -> {max_keep}")
                extracted_images = extracted_images[:max_keep]
                valid_indices = valid_indices[:max_keep]
            elif max_keep == 0:
                print(f"[ImageBatchExtract] max_keep=0，不限制最大保留数量，保留所有{len(extracted_images)}张图像")
            
            # 合并提取的图像
            result_images = torch.cat(extracted_images, dim=0)
            source_indices_str = ",".join(map(str, valid_indices))
            
            print(f"[ImageBatchExtract] 提取完成: 提取了{len(valid_indices)}张图像，索引=[{source_indices_str}]")
            print(f"[ImageBatchExtract] 输出形状: {result_images.shape}")
            
            return (result_images,)
            
        except Exception as e:
            print(f"[ImageBatchExtract] 错误: {str(e)}")
            # 出错时返回空结果
            empty_image = torch.empty((0,) + image.shape[1:], 
                                    dtype=image.dtype, device=image.device)
            return (empty_image,)
    
    def _get_extract_indices(self, batch_size, mode, index, step, uniform):
        """根据提取模式获取索引列表"""
        extract_indices = []
        
        try:
            if mode == "index":
                 # 自定义索引模式：为空就输出空
                 if not index.strip():
                     print(f"[ImageBatchExtract] 自定义索引为空，返回空结果")
                     return []
                 print(f"[ImageBatchExtract] 使用自定义索引模式: '{index}'")
                 extract_indices = self._parse_custom_indices(index, batch_size)
                
            elif mode == "step":
                # 步长模式：step从1开始
                if step < 1:
                    print(f"[ImageBatchExtract] 步长小于1，返回空结果")
                    return []
                print(f"[ImageBatchExtract] 使用步长模式: 步长{step}")
                extract_indices = self._calculate_step_indices(batch_size, step)
                
            elif mode == "uniform":
                # 数量模式：uniform为0输出空，1首帧，2首尾帧，依次类推
                if uniform <= 0:
                    print(f"[ImageBatchExtract] 数量为0或负数，返回空结果")
                    return []
                print(f"[ImageBatchExtract] 使用数量模式: 数量{uniform}")
                extract_indices = self._calculate_count_indices(batch_size, uniform)
            
            print(f"[ImageBatchExtract] 计算得到索引: {extract_indices}")
            return extract_indices
            
        except Exception as e:
            print(f"[ImageBatchExtract] 索引计算错误: {str(e)}")
            return []
    
    def _parse_custom_indices(self, indices_str, batch_size=None):
        """
        解析自定义索引字符串，支持负数索引
        支持格式: "1,3,5,20" 或 "1, 3, 5, 20" 或 "-1,-2,0" 或 "1，2，-1"（中文逗号）
        保持输入顺序，支持中英文逗号分割，处理空格和空内容
        """
        indices = []
        try:
            # 替换中文逗号为英文逗号，然后分割
            normalized_str = indices_str.replace('，', ',')
            parts = normalized_str.split(',')
            
            for part in parts:
                # 去除空格
                part = part.strip()
                # 跳过空内容
                if not part:
                    continue
                    
                try:
                    idx = int(part)
                    # 处理负数索引
                    if batch_size is not None and idx < 0:
                        idx = batch_size + idx
                    indices.append(idx)
                except ValueError:
                    print(f"[ImageBatchExtract] 跳过无效索引: '{part}'")
                    continue
            
            print(f"[ImageBatchExtract] 解析自定义索引: '{indices_str}' -> {indices}")
            
        except Exception as e:
            print(f"[ImageBatchExtract] 自定义索引解析错误: {str(e)}")
            indices = []
        
        return indices
    
    def _calculate_step_indices(self, batch_size, step):
        """计算步长索引，从0开始，步长从1开始"""
        indices = list(range(0, batch_size, step))
        print(f"[ImageBatchExtract] 步长计算: 总帧数={batch_size}, 步长={step} -> {indices}")
        return indices
    
    def _calculate_count_indices(self, batch_size, count):
        """
        根据数量计算索引
        count=1: 首帧 [0]
        count=2: 首尾帧 [0, batch_size-1]
        count=3: 首中尾帧 [0, middle, batch_size-1]
        依次类推
        """
        if count <= 0:
            return []
        
        if count == 1:
            # 只要首帧
            indices = [0]
        elif count == 2:
            # 首尾帧
            indices = [0, batch_size - 1] if batch_size > 1 else [0]
        elif count >= batch_size:
            # 数量大于等于总帧数，返回所有帧
            indices = list(range(batch_size))
        else:
            # 均匀分布
            step = (batch_size - 1) / (count - 1)
            indices = [int(round(i * step)) for i in range(count)]
            # 确保最后一帧是最后一个索引
            indices[-1] = batch_size - 1
            # 去重并排序
            indices = sorted(list(set(indices)))
        
        print(f"[ImageBatchExtract] 数量计算: 总帧数={batch_size}, 数量={count} -> {indices}")
        return indices
  

class ImageBatchSplit:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "take_count": ("INT", {"default": 8, "min": 1, "max": 1024, "step": 1}),
                "from_start": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image_1", "image_2")
    FUNCTION = "split_batch"
    CATEGORY = "1hewNodes/batch"
    
    def split_batch(self, image, take_count, from_start=False):
        try:
            # 获取批次大小
            batch_size = image.shape[0]
            print(f"[ImageBatchSplit] 输入图片批次信息: 形状={image.shape}, 数据类型={image.dtype}, 设备={image.device}")
            print(f"[ImageBatchSplit] 拆分参数: 总图片数={batch_size}, 取数={take_count}, 从开头切={from_start}")
            
            # 验证拆分数量
            if take_count >= batch_size:
                print(f"[ImageBatchSplit] 边界情况: 取数({take_count})大于等于总图片数({batch_size})")
                
                if from_start:
                    # 从开头切：第一部分是全部图片，第二部分为空
                    print(f"[ImageBatchSplit] from_start=True: 第一部分=全部图片，第二部分=空")
                    empty_second = torch.empty((0,) + image.shape[1:], dtype=image.dtype, device=image.device)
                    print(f"[ImageBatchSplit] 输出: 第一部分=原图片({batch_size}张), 第二部分=空张量")
                    return (image, empty_second)
                else:
                    # 从结尾切：第一部分为空，第二部分是全部图片
                    print(f"[ImageBatchSplit] from_start=False: 第一部分=空，第二部分=全部图片")
                    empty_first = torch.empty((0,) + image.shape[1:], dtype=image.dtype, device=image.device)
                    print(f"[ImageBatchSplit] 输出: 第一部分=空张量, 第二部分=原图片({batch_size}张)")
                    return (empty_first, image)
            
            # 根据参数计算拆分位置
            if from_start:
                # 从开头切：split_count是第一部分的数量
                first_count = take_count
                second_count = batch_size - take_count
                first_batch = image[:first_count]
                second_batch = image[first_count:]
                print(f"[ImageBatchSplit] from_start=True拆分完成: 总数{batch_size} -> 第一部分{first_count}张, 第二部分{second_count}张")
            else:
                # 从结尾切：split_count是第二部分的数量（原有逻辑）
                first_count = batch_size - take_count
                second_count = take_count
                first_batch = image[:first_count]
                second_batch = image[first_count:]
                print(f"[ImageBatchSplit] from_start=False拆分完成: 总数{batch_size} -> 第一部分{first_count}张, 第二部分{second_count}张")
            
            print(f"[ImageBatchSplit] 输出形状: 第一部分={first_batch.shape}, 第二部分={second_batch.shape}")
            return (first_batch, second_batch)
            
        except Exception as e:
            print(f"[ImageBatchSplit] 错误: {str(e)}")
            print(f"[ImageBatchSplit] 异常处理: 返回原图片和空张量")
            # 出错时返回原图片和空张量
            empty_batch = torch.empty((0,) + image.shape[1:], dtype=image.dtype, device=image.device)
            print(f"[ImageBatchSplit] 异常输出: 第一部分=原图片, 第二部分=空张量")
            return (image, empty_batch)


class ImageBatchGroup:
    """
    图像批次分组器 - 将批量图片按指定大小分组处理
    支持重叠帧和多种最后一组处理方式
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "batch_size": ("INT", {"default": 81, "min": 1, "max": 1024, "step": 4}),
                "overlap": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 1}),
                "last_batch_mode": (["drop_incomplete", "keep_remaining", "backtrack_last", "fill_color"], {"default": "backtrack_last"})
            },
            "optional": {
                "color": ("STRING", {"default": "1.0"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("image", "group_total", "start_index", "batch_count", "valid_count")
    OUTPUT_IS_LIST = (False, False, True, True, True)
    CATEGORY = "1hewNodes/batch"
    FUNCTION = "split_batch_sequential"
    
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
    
    def _create_white_image(self, reference_image, color_str="1.0"):
        """创建与参考图像相同尺寸的指定颜色图像"""
        # 解析颜色
        rgb_color = self.parse_color(color_str)
        r = rgb_color[0] / 255.0
        g = rgb_color[1] / 255.0
        b = rgb_color[2] / 255.0
        
        # 确保创建的图像与输入图像具有相同的维度
        if len(reference_image.shape) == 4:
            # 如果输入是4维 (batch, height, width, channels)，取第一个图像
            height, width, channels = reference_image[0].shape
            colored_image = torch.ones((1, height, width, channels), 
                                     dtype=reference_image.dtype, 
                                     device=reference_image.device)
        else:
            # 如果输入是3维 (height, width, channels)
            height, width, channels = reference_image.shape
            colored_image = torch.ones((1, height, width, channels), 
                                     dtype=reference_image.dtype, 
                                     device=reference_image.device)
        
        # 根据通道数设置颜色
        if channels == 1:
            # 灰度图像，使用 RGB 的平均值作为灰度值
            gray_value = (r + g + b) / 3.0
            colored_image[0, :, :, 0] = gray_value
        elif channels >= 3:
            # RGB 或 RGBA 图像
            colored_image[0, :, :, 0] = r
            colored_image[0, :, :, 1] = g
            colored_image[0, :, :, 2] = b
            # 如果是 RGBA，设置 alpha 通道为完全不透明
            if channels == 4:
                colored_image[0, :, :, 3] = 1.0
        
        return colored_image
    
    def _validate_parameters(self, total_images, batch_size, overlap, last_batch_mode=None):
        """验证参数有效性"""
        if total_images < 1:
            raise ValueError("输入图片数量必须大于0")
        
        if batch_size < 1:
            raise ValueError("批次大小必须大于0")
        
        if overlap < 0:
            raise ValueError("重叠帧数不能为负数")
        
        # 在 backtrack_last 模式下，允许 overlap 等于 batch_size
        if last_batch_mode == "backtrack_last":
            if overlap > batch_size:
                raise ValueError(f"重叠帧数 ({overlap}) 不能大于批次大小 ({batch_size})")
        else:
            if overlap >= batch_size:
                raise ValueError(f"重叠帧数 ({overlap}) 必须小于批次大小 ({batch_size})")
    
    def _calculate_start_indices(self, total_images, batch_size, overlap, last_batch_mode):
        """统一计算所有批次的起始索引"""
        if total_images <= batch_size:
            # 边界情况：当输入张数 <= batch_size 时的特殊处理
            if last_batch_mode == "drop_incomplete":
                # drop_incomplete 模式：如果图像数量不足一个完整批次，返回空列表
                return []
            else:
                # keep_remaining, backtrack_last, fill_color 模式：都从索引0开始
                return [0]
        
        # 计算基础步长
        step_size = batch_size - overlap
        if step_size <= 0:
            # 当 overlap >= batch_size 时的特殊处理
            if overlap == batch_size:
                step_size = max(1, (batch_size + 1) // 2)
            else:
                step_size = 1
        
        # 生成批次起始位置
        start_indices = []
        current_start = 0
        
        while current_start < total_images:
            # 对于drop_incomplete模式，检查当前批次是否完整
            if last_batch_mode == "drop_incomplete":
                # 如果当前批次不能满足完整的batch_size，则终止
                if current_start + batch_size > total_images:
                    break
            
            start_indices.append(current_start)
            current_start += step_size
            
            # 对于非backtrack_last和非drop_incomplete模式，如果当前批次已经能覆盖到最后一个图片，则无需继续
            if (last_batch_mode not in ["backtrack_last", "drop_incomplete"] and 
                len(start_indices) > 0 and 
                start_indices[-1] + batch_size >= total_images):
                break
        
        # 根据模式调整最后一批的位置
        if last_batch_mode == "backtrack_last" and len(start_indices) > 1:
            # 最后一批从末尾开始
            last_start = total_images - batch_size
            
            # 确保最后一批不会与第一批重叠（第一批必须从0开始）
            if last_start <= 0:
                # 如果只需要一批就能覆盖所有图像，保持第一批从0开始
                start_indices = [0]
            else:
                # 调整最后一批位置，但保持中间批次
                # 检查最后一批是否与现有批次重叠过多
                if last_start < start_indices[-1]:
                    # 如果最后一批位置向前移动，需要调整序列
                    # 找到第一个会与last_start重叠的批次
                    valid_indices = [0]  # 第一批总是从0开始
                    
                    for i in range(1, len(start_indices)):
                        # 检查当前批次是否与last_start批次重叠过多
                        current_end = start_indices[i] + batch_size - 1
                        last_start_end = last_start + batch_size - 1
                        
                        # 如果当前批次的结束位置 + overlap < last_start，则保留
                        if start_indices[i] + overlap <= last_start:
                            valid_indices.append(start_indices[i])
                    
                    # 添加最后一批
                    if valid_indices[-1] != last_start:
                        valid_indices.append(last_start)
                    
                    start_indices = valid_indices
                else:
                    # 最后一批位置合理，直接调整
                    start_indices[-1] = last_start
        
        return start_indices
    
    def _calculate_batch_counts(self, start_indices, total_images, batch_size, last_batch_mode):
        """根据起始索引和模式计算每批次的数量"""
        batch_counts = []
        
        # 边界情况：当输入图像数量 <= batch_size 时的特殊处理
        if total_images <= batch_size:
            if len(start_indices) == 0:
                # drop_incomplete 模式返回空列表
                return []
            elif last_batch_mode == "fill_color":
                # fill_color 模式：batch_count 使用 batch_size
                return [batch_size]
            else:
                # keep_remaining, backtrack_last 模式：batch_count 使用实际图像数量
                return [total_images]
        
        for i, start_idx in enumerate(start_indices):
            remaining = total_images - start_idx
            
            if i == len(start_indices) - 1:
                # 最后一批
                if last_batch_mode == "fill_color":
                    # 补充彩色图模式：总是保持批次大小
                    batch_counts.append(batch_size)
                elif last_batch_mode == "drop_incomplete":
                    # drop_incomplete 模式：保留的批次都是完整的
                    batch_counts.append(batch_size)
                elif last_batch_mode == "backtrack_last":
                    # backtrack_last 模式：
                    if len(start_indices) == 1:
                        # 单批次：使用剩余数量，但不超过total_images
                        batch_count = min(remaining, total_images)
                        batch_counts.append(batch_count)
                    else:
                        # 多批次：保持批次大小
                        batch_counts.append(batch_size)
                else:
                    # keep_remaining 模式：使用实际剩余数量，但不超过total_images
                    batch_count = min(remaining, total_images)
                    batch_counts.append(batch_count)
            else:
                # 非最后一批：总是使用批次大小
                batch_counts.append(batch_size)
        
        return batch_counts
    
    def _calculate_valid_counts(self, start_indices, batch_counts, overlap, last_batch_mode, total_images=None):
        """计算每批次的有效帧数"""
        valid_counts = []
        
        # 边界情况：当输入图像数量 <= batch_size 时的特殊处理
        if total_images is not None and len(start_indices) <= 1:
            if len(start_indices) == 0:
                # drop_incomplete 模式返回空列表
                return []
            else:
                # keep_remaining, backtrack_last, fill_color 模式：valid_count 都是实际图像数量
                return [total_images]
        
        for i, (start_idx, batch_count) in enumerate(zip(start_indices, batch_counts)):
            # 统一的valid_count计算逻辑，适用于所有模式
            if i == len(start_indices) - 1:
                # 最后一批：对于单批次情况，使用实际图像数量
                if (len(start_indices) == 1 and total_images is not None and 
                    last_batch_mode != "drop_incomplete"):
                    # drop_incomplete模式下，保留的批次都是完整的，使用batch_count
                    actual_images_in_batch = total_images - start_idx
                    valid_counts.append(actual_images_in_batch)
                elif last_batch_mode == "fill_color" and total_images is not None:
                    # fill_color模式最后一批：计算实际的原始图像数量
                    remaining_images = total_images - start_idx
                    actual_images_in_batch = min(remaining_images, batch_count)
                    valid_counts.append(actual_images_in_batch)
                else:
                    # 多批次情况或drop_incomplete模式：全部有效
                    valid_counts.append(batch_count)
            else:
                # 非最后一批：有效数量 = 下一批的起始位置 - 当前批的起始位置
                # 这个逻辑适用于所有模式，包括fill_color
                next_start = start_indices[i + 1]
                valid_count = next_start - start_idx
                valid_counts.append(valid_count)
        
        return valid_counts
    
    def split_batch_sequential(self, image, batch_size, overlap, last_batch_mode, color="1.0"):
        """
        顺序分割批量图片
        """
        # 验证参数
        total_images = len(image)
        self._validate_parameters(total_images, batch_size, overlap, last_batch_mode)
        
        # 保存原始图像数量
        original_total = total_images
        
        # 如果输入图片数量少于批次大小，添加指定颜色图补充
        if total_images < batch_size:
            colored_images = []
            for _ in range(batch_size - total_images):
                colored_img = self._create_white_image(image, color)
                colored_images.append(colored_img)
            # 将彩色图列表合并为一个tensor
            colored_batch = torch.cat(colored_images, dim=0)
            image = torch.cat([image, colored_batch], dim=0)
            total_images = len(image)
        
        # 使用新的统一计算方法
        start_indices = self._calculate_start_indices(total_images, batch_size, overlap, last_batch_mode)
        
        # 如果没有有效的批次（drop_incomplete模式下图像数量不足），返回空结果
        if not start_indices:
            return (image[:original_total], 0, [], [], [])
        
        batch_counts = self._calculate_batch_counts(start_indices, original_total, batch_size, last_batch_mode)
        
        # 处理 fill_color 模式的额外彩色图补充
        if last_batch_mode == "fill_color":
            max_needed = max(start_idx + batch_count for start_idx, batch_count in zip(start_indices, batch_counts))
            if max_needed > total_images:
                colored_images = []
                for _ in range(max_needed - total_images):
                    colored_img = self._create_white_image(image, color)
                    colored_images.append(colored_img)
                # 将彩色图列表合并为一个tensor
                colored_batch = torch.cat(colored_images, dim=0)
                image = torch.cat([image, colored_batch], dim=0)
        
        # 计算有效帧数
        valid_counts = self._calculate_valid_counts(start_indices, batch_counts, overlap, last_batch_mode, original_total)
        
        # 修正 fill_color 模式下最后一批的有效帧数
        if last_batch_mode == "fill_color" and len(valid_counts) > 0:
            last_start = start_indices[-1]
            actual_remaining = original_total - last_start
            if actual_remaining > 0:
                valid_counts[-1] = actual_remaining
        
        # 确定输出图像：只有fill_color模式需要输出包含彩色图的图像，其他模式直接输出原始图像
        if last_batch_mode == "fill_color":
            output_image = image  # 已经包含了彩色图填充
        else:
            # 恢复到原始输入图像（去除可能添加的彩色图填充）
            output_image = image[:original_total]
        
        return (output_image, len(start_indices), start_indices, batch_counts, valid_counts)


class ImageListAppend:
    """
    图片列表追加节点 - 将图片收集为列表格式
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",)
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_list",)
    FUNCTION = "image_list_append"
    CATEGORY = "1hewNodes/batch"
    
    def image_list_append(self, image_1, image_2):
        """
        将两个图片输入追加为列表
        """
        try:
            # 处理None值
            if image_1 is None and image_2 is None:
                return ([],)
            elif image_1 is None:
                return ([image_2],)
            elif image_2 is None:
                return ([image_1],)
            
            return self._append_to_list(image_1, image_2)
                
        except Exception as e:
            print(f"图片列表追加错误: {str(e)}")
            return ([image_1],)
    
    def _append_to_list(self, image_1, image_2):
        """
        将输入追加为列表，保持批量结构
        """
        result = []
        
        # 处理第一个输入
        if isinstance(image_1, list):
            result.extend(image_1)
        else:
            result.append(image_1)
        
        # 处理第二个输入
        if isinstance(image_2, list):
            result.extend(image_2)
        else:
            result.append(image_2)
        
        print(f"图片列表追加完成: 收集了{len(result)}个图片项目")
        return (result,)


class MaskBatchMathOps:
    """
    蒙版批量数学运算节点 - 支持批量处理所有图层的OR和AND功能
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "operation": (["or", "and"], {"default": "or"})
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "batch_mask_math_ops"
    CATEGORY = "1hewNodes/batch"

    def batch_mask_math_ops(self, mask, operation):
        # 获取批次大小
        batch_size = mask.shape[0]
        
        # 如果批次大小为1，直接返回
        if batch_size <= 1:
            return (mask,)
        
        # 创建输出蒙版
        output_mask = None
        
        # 对每个批次进行处理
        for b in range(batch_size):
            current_mask = mask[b]
            
            # 将蒙版转换为numpy数组
            if mask.is_cuda:
                mask_np = current_mask.cpu().numpy()
            else:
                mask_np = current_mask.numpy()
            
            # 初始化输出蒙版（使用第一个蒙版）
            if output_mask is None:
                output_mask = mask_np.copy()
                continue
            
            # 应用选定的操作
            if operation == "or":
                # or操作（取最大值）
                output_mask = np.maximum(output_mask, mask_np)
            elif operation == "and":
                # and操作（取最小值）
                output_mask = np.minimum(output_mask, mask_np)
        
        # 转换回tensor
        output_tensor = torch.from_numpy(output_mask).unsqueeze(0)
        
        return (output_tensor,)


class MaskBatchSplit:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "take_count": ("INT", {"default": 8, "min": 1, "max": 1024, "step": 1}),
                "from_start": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("MASK", "MASK")
    RETURN_NAMES = ("mask_1", "mask_2")
    FUNCTION = "split_batch"
    CATEGORY = "1hewNodes/batch"
    
    def split_batch(self, mask, take_count, from_start=False):
        try:
            # 获取批次大小
            batch_size = mask.shape[0]
            print(f"[MaskBatchSplit] 输入遮罩批次信息: 形状={mask.shape}, 数据类型={mask.dtype}, 设备={mask.device}")
            print(f"[MaskBatchSplit] 拆分参数: 总遮罩数={batch_size}, 取数={take_count}, 从开头切={from_start}")
            
            # 验证拆分数量
            if take_count >= batch_size:
                print(f"[MaskBatchSplit] 边界情况: 取数({take_count})大于等于总遮罩数({batch_size})")
                
                if from_start:
                    # 从开头切：第一部分是全部遮罩，第二部分为空
                    print(f"[MaskBatchSplit] from_start=True: 第一部分=全部遮罩，第二部分=空")
                    empty_second = torch.empty((0,) + mask.shape[1:], dtype=mask.dtype, device=mask.device)
                    print(f"[MaskBatchSplit] 输出: 第一部分=原遮罩({batch_size}个), 第二部分=空张量")
                    return (mask, empty_second)
                else:
                    # 从结尾切：第一部分为空，第二部分是全部遮罩
                    print(f"[MaskBatchSplit] from_start=False: 第一部分=空，第二部分=全部遮罩")
                    empty_first = torch.empty((0,) + mask.shape[1:], dtype=mask.dtype, device=mask.device)
                    print(f"[MaskBatchSplit] 输出: 第一部分=空张量, 第二部分=原遮罩({batch_size}个)")
                    return (empty_first, mask)
            
            # 根据参数计算拆分位置
            if from_start:
                # 从开头切：take_count是第一部分的数量
                first_count = take_count
                second_count = batch_size - take_count
                first_batch = mask[:first_count]
                second_batch = mask[first_count:]
                print(f"[MaskBatchSplit] from_start=True拆分完成: 总数{batch_size} -> 第一部分{first_count}个, 第二部分{second_count}个")
            else:
                # 从结尾切：take_count是第二部分的数量
                first_count = batch_size - take_count
                second_count = take_count
                first_batch = mask[:first_count]
                second_batch = mask[first_count:]
                print(f"[MaskBatchSplit] from_start=False拆分完成: 总数{batch_size} -> 第一部分{first_count}个, 第二部分{second_count}个")
            
            print(f"[MaskBatchSplit] 输出形状: 第一部分={first_batch.shape}, 第二部分={second_batch.shape}")
            return (first_batch, second_batch)
            
        except Exception as e:
            print(f"[MaskBatchSplit] 错误: {str(e)}")
            print(f"[MaskBatchSplit] 异常处理: 返回原遮罩和空张量")
            # 出错时返回原遮罩和空张量
            empty_batch = torch.empty((0,) + mask.shape[1:], dtype=mask.dtype, device=mask.device)
            print(f"[MaskBatchSplit] 异常输出: 第一部分=原遮罩, 第二部分=空张量")
            return (mask, empty_batch)


class VideoCutGroup:
    """
    VideoCutGroup - 视频硬切检测节点
    
    这是一个用于检测视频中场景切换的节点，通过分析相邻帧之间的相似度来识别硬切点。
    
    核心特性：
    - 支持两种检测模式：快速模式和精确模式
    - 快速模式：使用简化的SSIM计算，适合快速预览
    - 精确模式：使用多核模糊SSIM计算，提供更准确的检测结果
    - 灵活的阈值配置：支持单一阈值或多阈值检测
    - 智能分组：根据最小/最大帧数要求自动调整分组
    - 手动调整：支持手动添加或删除特定的切点
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold_base": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "threshold_range": ("FLOAT", {"default": 0.05, "min": 0.01, "max": 0.2, "step": 0.01}),
                "threshold_count": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1}),
                "kernel": ("STRING", {"default": "3, 7, 11", "multiline": False}),
                "min_frame_count": ("INT", {"default": 10, "min": 1, "max": 1000, "step": 1}),
                "max_frame_count": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "fast": ("BOOLEAN", {"default": False}),

                "add_frame": ("STRING", {"default": ""}),
                "delete_frame": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT")
    RETURN_NAMES = ("image", "group_total", "start_index", "batch_count")
    OUTPUT_IS_LIST = (False, False, True, True)
    FUNCTION = "execute"
    CATEGORY = "1hewNodes/batch"

    def __init__(self):
        # 动态的核配置：将在execute方法中根据用户输入设置
        self.kernel_configs = None
        
        # 固定参数
        self.enable_blur = True
        self.enable_kernel = True
        self.vote_ratio = 1.0  # 全部保留
        
        # 性能统计
        self.performance_stats = {
            'detection_time': 0,
            'total_frames': 0,
            'keyframes_found': 0
        }
    
    def parse_user_frames(self, frame_string):
        """
        解析用户输入的帧索引字符串，支持逗号分隔，智能处理中英文逗号和空格
        """
        if not frame_string or not frame_string.strip():
            return []
        
        # 替换中文逗号为英文逗号，移除多余空格
        normalized = re.sub(r'[，,]\s*', ',', frame_string.strip())
        
        # 分割并转换为整数
        frame_indices = []
        for item in normalized.split(','):
            item = item.strip()
            if item and item.isdigit():
                frame_indices.append(int(item))
        
        return sorted(list(set(frame_indices)))  # 去重并排序

    def parse_custom_kernels(self, kernel_string):
        """
        解析用户输入的kernel配置字符串，支持逗号分隔
        格式: "3,7,11" 或 "3,5,7,9,11,13,15"
        返回: [(kernel_size, sigma), ...] 格式的列表
        """
        if not kernel_string or not kernel_string.strip():
            # 如果为空，返回默认配置
            return [(3, 0.6), (7, 1.0), (11, 1.5)]
        
        # 替换中文逗号为英文逗号，移除多余空格
        normalized = re.sub(r'[，,]\s*', ',', kernel_string.strip())
        
        # 分割并转换为整数
        kernel_sizes = []
        for item in normalized.split(','):
            item = item.strip()
            if item and item.isdigit():
                size = int(item)
                # 验证kernel大小必须是奇数且大于等于3
                if size >= 3 and size % 2 == 1:
                    kernel_sizes.append(size)
        
        # 去重并排序
        kernel_sizes = sorted(list(set(kernel_sizes)))
        
        # 如果没有有效的kernel，返回默认配置
        if not kernel_sizes:
            return [(3, 0.6), (7, 1.0), (11, 1.5)]
        
        # 为每个kernel大小生成对应的sigma值
        # sigma = kernel_size * 0.2 (经验公式)
        kernel_configs = []
        for size in kernel_sizes:
            sigma = size * 0.2
            kernel_configs.append((size, sigma))
        
        return kernel_configs

    def simple_ssim(self, img1, img2, C1=0.01**2, C2=0.03**2):
        """
        简易SSIM计算方法，用于快速模式
        """
        # 如果是彩色图像，先转为灰度
        if img1.shape[-1] == 3:
            img1 = 0.299 * img1[..., 0] + 0.587 * img1[..., 1] + 0.114 * img1[..., 2]
            img2 = 0.299 * img2[..., 0] + 0.587 * img2[..., 1] + 0.114 * img2[..., 2]
        
        # 计算均值
        mu1 = img1.mean()
        mu2 = img2.mean()
        
        # 计算方差
        sigma1 = ((img1 - mu1) ** 2).mean()
        sigma2 = ((img2 - mu2) ** 2).mean()
        
        # 计算协方差
        sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
        
        # 计算SSIM
        ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2))
        
        # 保证结果在0~1之间
        return max(0.0, min(1.0, ssim.item() if hasattr(ssim, 'item') else float(ssim)))

    def preprocess_images_batch(self, images_np):
        """
        批量预处理所有图像，避免重复转换
        """
        # 确保数据格式正确
        if images_np.dtype != np.float32:
            images_np = images_np.astype(np.float32)
        
        # 如果图像值在 [0, 1] 范围内，转换为 [0, 255] 以便SSIM计算
        if images_np.max() <= 1.0:
            images_np = images_np * 255.0
        
        # 转换为灰度图像以提高检测效果
        if len(images_np.shape) == 4 and images_np.shape[3] == 3:
            # RGB转灰度：0.299*R + 0.587*G + 0.114*B
            images_np = np.dot(images_np[..., :3], [0.299, 0.587, 0.114])
        elif len(images_np.shape) == 4 and images_np.shape[3] == 1:
            # 已经是单通道，去掉最后一个维度
            images_np = images_np.squeeze(-1)
        
        return images_np

    def batch_calculate_ssim_matrix(self, processed_images):
        """
        批量计算所有相邻帧的SSIM值矩阵，使用固定的核配置和模糊模式
        """
        B = processed_images.shape[0]
        if B <= 1:
            return {}
        
        ssim_matrix = {}
        
        # 为每个核配置计算模糊SSIM
        for kernel_idx, kernel_config in enumerate(self.kernel_configs):
            kernel_size, sigma = kernel_config
            ssim_values = np.zeros(B - 1, dtype=np.float32)
            
            # 计算所有相邻帧的模糊SSIM
            for i in range(B - 1):
                ssim_val = self._blur_pixel_ssim(
                    processed_images[i], processed_images[i + 1], kernel_size, sigma
                )
                ssim_values[i] = ssim_val
            
            ssim_matrix[kernel_idx] = ssim_values
        
        return ssim_matrix

    def _blur_pixel_ssim(self, img1, img2, kernel_size, sigma):
        """模糊像素SSIM计算，优化内存使用"""
        # 应用高斯模糊
        ksize = (kernel_size, kernel_size)
        img1_blur = cv2.GaussianBlur(img1, ksize, sigma)
        img2_blur = cv2.GaussianBlur(img2, ksize, sigma)
        
        # 计算均值
        mu1 = cv2.boxFilter(img1_blur, -1, (kernel_size, kernel_size))
        mu2 = cv2.boxFilter(img2_blur, -1, (kernel_size, kernel_size))
        
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        
        # 计算方差和协方差
        sigma1_sq = cv2.boxFilter(img1_blur * img1_blur, -1, (kernel_size, kernel_size)) - mu1_sq
        sigma2_sq = cv2.boxFilter(img2_blur * img2_blur, -1, (kernel_size, kernel_size)) - mu2_sq
        sigma12 = cv2.boxFilter(img1_blur * img2_blur, -1, (kernel_size, kernel_size)) - mu1_mu2
        
        # SSIM常数
        C1 = (0.01) ** 2
        C2 = (0.03) ** 2
        
        # 计算SSIM
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return float(np.mean(ssim_map))

    def generate_dynamic_thresholds(self, threshold_base, threshold_range=0.05, threshold_count=2):
        """
        生成动态数量的阈值
        threshold_count=1: 只使用threshold_base
        threshold_count=2: 范围两端 [base-range, base+range]
        threshold_count=3: 两端+中间 [base-range, base, base+range]
        threshold_count=4+: 在范围内均匀分布
        """
        if threshold_count == 1:
            # 只使用基础阈值
            thresholds = [threshold_base]
        elif threshold_count == 2:
            # 范围两端
            thresholds = [
                threshold_base - threshold_range,  # 下限
                threshold_base + threshold_range   # 上限
            ]
        else:
            # 3个或更多：在范围内均匀分布
            min_threshold = threshold_base - threshold_range
            max_threshold = threshold_base + threshold_range
            
            if threshold_count == 3:
                # 特殊处理：两端+中间
                thresholds = [min_threshold, threshold_base, max_threshold]
            else:
                # 4个或更多：均匀分布
                step = (max_threshold - min_threshold) / (threshold_count - 1)
                thresholds = [min_threshold + i * step for i in range(threshold_count)]
        
        # 确保阈值在合理范围内
        thresholds = [max(0.0, min(1.0, t)) for t in thresholds]
        
        return sorted(thresholds)

    def optimized_single_threshold_detection(self, ssim_matrix, user_threshold, kernel_idx, 
                                           min_frame_count, max_frame_count, total_frames):
        """
        基于预计算SSIM矩阵的纯粹阈值检测方法
        修正阈值逻辑：用户阈值越大，检测越严格，画面越少
        
        注意：此方法只进行纯粹的阈值检测，不应用分组规则
        分组规则应该在所有阈值检测完成后，在最终合并阶段统一应用
        """
        B = total_frames
        if B <= 1:
            return [0]
        
        # 获取对应的SSIM值数组
        if kernel_idx not in ssim_matrix:
            return [0]
        
        ssim_values = ssim_matrix[kernel_idx]
        
        # 纯粹的阈值检测：找出所有满足阈值条件的硬切点
        threshold_cuts = [0]  # 第一组总是从0开始
        for i in range(len(ssim_values)):
            ssim_val = ssim_values[i]
            # 直接比较：(1-ssim)大于用户阈值时，认为是硬切
            if (1.0 - ssim_val) > user_threshold:
                cut_point = i + 1  # 硬切后的帧作为新组的起始
                if cut_point < B and cut_point not in threshold_cuts:
                    threshold_cuts.append(cut_point)
        
        return threshold_cuts


    def batch_detection_all_features(self, images_np, threshold_base, min_frame_count, max_frame_count, threshold_range=0.05, threshold_count=2):
        """
        批量检测所有特征组合，使用固定的核配置和参数
        """
        total_frames = len(images_np)
        
        if total_frames < 2:
            return [[0]]
        
        # 预处理图像
        processed_images = self.preprocess_images_batch(images_np)
        
        # 批量计算SSIM矩阵
        ssim_matrix = self.batch_calculate_ssim_matrix(processed_images)
        
        # 生成动态数量的阈值
        user_thresholds = self.generate_dynamic_thresholds(threshold_base, threshold_range, threshold_count)
        
        # 打印检测任务概览
        print()
        print("=== 🚀 VideoCutGroup 多核模糊模式检测 启动 ===")
        print(f"threshold: {[f'{t:.3f}' for t in user_thresholds]}")
        kernel_list = [str(k[0]) for k in self.kernel_configs]
        print(f"kernel: [{','.join(kernel_list)}]")
        print()
        
        total_groups = len(self.kernel_configs) * len(user_thresholds)
        print(f"📈 {total_groups} 组检测任务详情")
        
        # 对每个核和每个阈值进行检测
        all_detection_results = []
        group_num = 1
        
        for kernel_idx in range(len(self.kernel_configs)):
            kernel_size, sigma = self.kernel_configs[kernel_idx]
            
            for user_threshold in user_thresholds:
                # 使用优化的检测方法
                result = self.optimized_single_threshold_detection(
                    ssim_matrix, user_threshold, kernel_idx, min_frame_count, max_frame_count, total_frames
                )
                all_detection_results.append(result)
                
                # 获取阈值详细信息用于日志 - 基于实际检测结果
                threshold_details = []
                if kernel_idx in ssim_matrix and len(result) > 1:
                    ssim_values = ssim_matrix[kernel_idx]
                    # 遍历实际检测到的切点（排除起始点0）
                    for cut_point in result[1:]:  # 跳过起始点0
                        if cut_point > 0 and cut_point <= len(ssim_values):
                            # 切点对应的是前一帧与当前帧的比较
                            ssim_val = ssim_values[cut_point - 1]
                            threshold_val = 1.0 - ssim_val
                            threshold_details.append(f"{cut_point}:{threshold_val:.3f}")
                
                # 格式化日志输出（排除起始点0）
                actual_cut_points = len(result) - 1 if result and result[0] == 0 else len(result)
                print(f"🔍 第{group_num}组： threshold={user_threshold:.3f}，kernel = {kernel_size}")
                print(f"- 检测切点：{actual_cut_points} 个 [index：threshold]")
                if threshold_details:
                    print(f"- [{', '.join(threshold_details)}]")
                print()
                
                group_num += 1
        
        # 存储检测结果用于后续汇总
        self._detection_results_summary = {
            'user_thresholds': user_thresholds,
            'kernel_configs': self.kernel_configs,
            'all_detection_results': all_detection_results
        }
        
        return all_detection_results

    def unified_voting_fusion(self, all_detection_results, total_frames, min_frame_count=10, max_frame_count=0):
        """
        统一融合方法，整合所有检测结果并应用分组规则
        
        重要：在此阶段统一应用min_frame_count和max_frame_count规则，
        确保所有阈值组合都使用相同的分组策略
        """
        if not all_detection_results:
            return [0]
        
        # 打印最终检测结果汇总
        print()
        print("✅ 最终检测结果")
        if hasattr(self, '_detection_results_summary'):
            summary = self._detection_results_summary
            user_thresholds = summary['user_thresholds']
            kernel_configs = summary['kernel_configs']
            results = summary['all_detection_results']
            
            # 按组显示每个检测结果
            result_idx = 0
            for kernel_idx, (kernel_size, sigma) in enumerate(kernel_configs):
                for threshold in user_thresholds:
                    if result_idx < len(results):
                        start_indices = results[result_idx]  # 包含起始点0的完整列表
                        cut_points_count = len(start_indices)
                        print(f"[threshold={threshold:.3f}，kernel = {kernel_size}]: start index 共计 {cut_points_count} 个")
                        print(f"{start_indices}")
                        result_idx += 1
        
        # 收集所有切点（去重）
        all_cut_points = set([0])  # 起始帧
        for result in all_detection_results:
            for frame in result[1:]:  # 跳过起始帧0
                all_cut_points.add(frame)
        
        # 排序
        raw_cut_points = sorted(list(all_cut_points))
        
        # 显示合并前的结果
        print(f"合并检测结果：{len(raw_cut_points)} 个")
        print(f"{raw_cut_points}")
        print()
        
        # 应用分组规则
        final_cut_points = self._apply_final_grouping_rules(raw_cut_points, min_frame_count, max_frame_count, total_frames)
        
        # 显示最终结果
        print(f"min_frame_count={min_frame_count}, max_frame_count={max_frame_count} ，分组规则处理后：{len(final_cut_points)} 个")
        print(f"{final_cut_points}")
        print()
        
        return final_cut_points
    
    def _apply_final_grouping_rules(self, cut_points, min_frame_count, max_frame_count, total_frames):
        """
        在最终阶段应用分组规则
        """
        if not cut_points or len(cut_points) <= 1:
            return [0]
        
        # 应用min_frame_count规则：合并过近的切点
        filtered_points = [cut_points[0]]  # 保留起始点0
        
        for point in cut_points[1:]:
            if point - filtered_points[-1] >= min_frame_count:
                filtered_points.append(point)
            # 如果距离太近，跳过这个切点
        
        # 应用max_frame_count规则：拆分过长的段
        if max_frame_count > 0:
            final_points = [0]
            
            for i in range(1, len(filtered_points)):
                start = filtered_points[i-1]
                end = filtered_points[i]
                segment_length = end - start
                
                # 如果段长度超过限制，需要拆分
                if segment_length > max_frame_count:
                    # 在这个段内按max_frame_count间隔插入切点
                    current = start
                    while current + max_frame_count < end:
                        current += max_frame_count
                        final_points.append(current)
                
                # 添加原始切点
                final_points.append(end)
            
            # 处理最后一段（到视频结尾）
            if len(filtered_points) > 0:
                last_point = filtered_points[-1]
                if last_point < total_frames:
                    remaining_length = total_frames - last_point
                    if remaining_length > max_frame_count:
                        current = last_point
                        while current + max_frame_count < total_frames:
                            current += max_frame_count
                            final_points.append(current)
            
            return sorted(list(set(final_points)))
        else:
            return filtered_points

    def sequential_detection(self, images, threshold_base, min_frame_count, max_frame_count, threshold_range=0.05, threshold_count=2):
        """
        序列检测方法，根据参数配置进行视频硬切检测
        """
        # 转换图像格式
        if hasattr(images, 'cpu'):
            images_np = images.cpu().numpy()
        else:
            images_np = images
        
        # 批量检测所有特征组合
        all_detection_results = self.batch_detection_all_features(
            images_np, threshold_base, min_frame_count, max_frame_count, threshold_range, threshold_count
        )
        
        # 统一投票融合
        final_split_points = self.unified_voting_fusion(all_detection_results, len(images_np), min_frame_count, max_frame_count)
        
        return final_split_points

    def fast_mode_detection(self, images, threshold_base, min_frame_count, max_frame_count):
        """
        快速模式检测，使用简化的SSIM计算方法
        注意：这里的threshold_base已经经过1-处理，需要转换回原始阈值逻辑
        """
        # 转换图像格式
        if hasattr(images, 'cpu'):
            images_np = images.cpu().numpy()
        else:
            images_np = images
        
        B = images_np.shape[0]
        if B < 2:
            return [0]
        
        # 打印快速模式概览
        print()
        print("=== ⚡ VideoCutGroup fast 模式检测 启动 ===")
        print(f"threshold={threshold_base:.3f}，")
        
        # 计算所有相邻帧的简易SSIM
        ssim_list = []
        for i in range(B - 1):
            ssim_val = self.simple_ssim(images_np[i], images_np[i + 1])
            ssim_list.append(ssim_val)
        
        if not ssim_list:
            return [0]
        
        # 使用类似nodes.py的动态阈值计算
        ssim_max = max(ssim_list)
        ssim_mean = sum(ssim_list) / len(ssim_list)
        
        # 将用户阈值转换为检测阈值
        # 由于threshold_base经过了1-处理，这里需要转换回原始逻辑
        # 用户期望：threshold_base越大，检测越严格，画面越少
        # 转换：threshold_base -> (1 - threshold_base) -> 作为检测算法中的threshold参数
        nodes_threshold = 1.0 - threshold_base
        
        # 使用nodes.py的阈值计算逻辑
        ssim_limit = ssim_max - (ssim_max - ssim_mean) * 2 - nodes_threshold
        
        # 确保阈值在合理范围内 [0, 1]
        ssim_limit = max(0.0, min(1.0, ssim_limit))
        
        # 找到所有低于阈值的切点
        keyframes = [0]
        threshold_details = []  # 存储索引和对应的阈值值(1-ssim)
        for i, ssim_val in enumerate(ssim_list):
            if ssim_val < ssim_limit:
                keyframes.append(i + 1)
                threshold_val = 1.0 - ssim_val  # 转换为阈值值
                threshold_details.append(f"{i+1}:{threshold_val:.3f}")
        
        # 显示初始检测结果
        print(f"检测结果：{len(keyframes)} 个")
        print(f"{keyframes}")
        
        # 应用最小帧数限制，合并过近的切点
        filtered_keyframes = [keyframes[0]]
        for kf in keyframes[1:]:
            if kf - filtered_keyframes[-1] > min_frame_count:
                filtered_keyframes.append(kf)
            else:
                filtered_keyframes[-1] = kf  # 替换为更大的索引
        
        # 检查尾部：如果尾部到视频结尾的帧数 < min_frame_count，向左归并
        while len(filtered_keyframes) > 1 and (B - filtered_keyframes[-1]) < min_frame_count:
            filtered_keyframes.pop()
        
        # 应用最大帧数限制，拆分过长的段
        if max_frame_count > 0:
            final_keyframes = [0]
            for i in range(1, len(filtered_keyframes) + 1):
                start = filtered_keyframes[i - 1]
                end = filtered_keyframes[i] if i < len(filtered_keyframes) else B
                segment_length = end - start
                
                if segment_length > max_frame_count:
                    # 拆分长段
                    num_splits = math.ceil(segment_length / max_frame_count)
                    frames_per_split = segment_length // num_splits
                    
                    for j in range(num_splits):
                        if j < num_splits - 1:
                            final_keyframes.append(start + (j + 1) * frames_per_split)
                        else:
                            final_keyframes.append(end)
                else:
                    final_keyframes.append(end)
        else:
            final_keyframes = filtered_keyframes[:]
            if final_keyframes[-1] != B:
                final_keyframes.append(B)
        
        # 移除最后一个点（如果是B）
        if final_keyframes and final_keyframes[-1] == B:
            final_keyframes.pop()
        
        # 显示分组规则处理后的结果
        max_frame_text = "0" if max_frame_count == 0 else str(max_frame_count)
        print(f"min_frame_count={min_frame_count}， max_frame_count={max_frame_text} ，分组规则处理后：{len(final_keyframes)} 个")
        print(f"{final_keyframes}")
        
        return final_keyframes

    def apply_user_modifications(self, cut_points, add_frame, delete_frame, total_frames):
        """
        应用用户自定义的添加和删除帧修改
        """
        # 解析用户输入
        add_frames = self.parse_user_frames(add_frame)
        delete_frames = self.parse_user_frames(delete_frame)
        
        # 应用修改
        modified_cut_points = list(cut_points)
        
        # 记录实际添加和删除的帧
        actually_added = []
        actually_deleted = []
        
        # 添加用户指定的帧
        for frame in add_frames:
            if 0 <= frame < total_frames and frame not in modified_cut_points:
                modified_cut_points.append(frame)
                actually_added.append(frame)
        
        # 删除用户指定的帧（但保留起始帧0）
        for frame in delete_frames:
            if frame in modified_cut_points and frame != 0:
                modified_cut_points.remove(frame)
                actually_deleted.append(frame)
        
        # 打印用户设定的所有添加和删除信息（不是过滤后的）
        if add_frames:
            print(f"➕ 用户添加帧: {add_frames}")
        if delete_frames:
            print(f"➖ 用户删除帧: {delete_frames}")
        
        # 排序并返回
        final_cut_points = sorted(list(set(modified_cut_points)))
        
        if add_frames or delete_frames:
            print(f"🔧 后期增减帧处理后：{len(final_cut_points)} 个")
            print(f"{final_cut_points}")
        
        return final_cut_points

    def execute(self, image, threshold_base, threshold_range, threshold_count, min_frame_count, max_frame_count, 
                fast, add_frame, delete_frame, kernel):
        """
        主执行函数，支持自定义kernel配置和多种检测模式
        """
        try:
            # 设置动态kernel配置
            self.kernel_configs = self.parse_custom_kernels(kernel)
            # print(f"🔧 使用kernel配置: {[k for k, s in self.kernel_configs]}")
            
            B = image.shape[0]
            if B < 2:
                return (image, 1, [0], [B])
            
            # 参数验证：max_frame_count=0表示无限制
            if max_frame_count > 0 and min_frame_count >= max_frame_count:
                max_frame_count = min_frame_count + 10
            
            start_time = time.time()
            
            if fast:
                # 使用快速模式检测
                start_indices = self.fast_mode_detection(
                    image, threshold_base, min_frame_count, max_frame_count
                )
                
                detection_time = time.time() - start_time
            else:
                # 使用固定参数的多特征检测算法
                start_indices = self.sequential_detection(
                    image, threshold_base, min_frame_count, max_frame_count, threshold_range, threshold_count
                )
                
                detection_time = time.time() - start_time
            
            # 应用用户自定义修改
            original_count = len(start_indices)
            start_indices = self.apply_user_modifications(start_indices, add_frame, delete_frame, B)
            
            # 显示处理步骤已经在apply_user_modifications中完成，这里不需要重复打印
            
            # 计算每段的帧数
            batch_counts = []
            for i in range(len(start_indices)):
                if i < len(start_indices) - 1:
                    # 当前段的帧数 = 下一个起始点 - 当前起始点
                    batch_count = start_indices[i + 1] - start_indices[i]
                else:
                    # 最后一段的帧数 = 总帧数 - 当前起始点
                    batch_count = B - start_indices[i]
                batch_counts.append(batch_count)
            
            # 计算分组总数
            group_total = len(start_indices)
            
            # 计算总耗时
            total_time = time.time() - start_time
            
            # 总结性日志输出已在各自的检测方法中完成
            
            # 显示对应每组帧数
            print(f"对应每组帧数")
            print(f"{batch_counts}")
            print()
            
            print(f"任务总耗时：{total_time:.1f} 秒")
            if fast:
                print("=== ⚡ VideoCutGroup fast 模式检测 完成 ===")
            else:
                print("=== 🚀 VideoCutGroup 多核模糊模式检测 完成 ===")
            print()
            
            return (
                image[start_indices],  # 返回起始帧的图像
                group_total,
                start_indices,
                batch_counts
            )
            
        except Exception as e:
            print(f"VideoCutGroup 执行错误: {str(e)}")
            return (image, 1, [0], [image.shape[0]])


NODE_CLASS_MAPPINGS = {
    "1hew_ImageBatchExtract": ImageBatchExtract,
    "1hew_ImageBatchSplit": ImageBatchSplit,
    "1hew_ImageBatchGroup": ImageBatchGroup,
    "1hew_ImageListAppend": ImageListAppend,
    "1hew_MaskBatchMathOps": MaskBatchMathOps,
    "1hew_MaskBatchSplit": MaskBatchSplit,
    "1hew_VideoCutGroup": VideoCutGroup,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "1hew_ImageBatchExtract": "Image Batch Extract",
    "1hew_ImageBatchSplit": "Image Batch Split",
    "1hew_ImageBatchGroup": "Image Batch Group",
    "1hew_ImageListAppend": "Image List Append",
    "1hew_MaskBatchMathOps": "Mask Batch Math Ops",
    "1hew_MaskBatchSplit": "Mask Batch Split",
    "1hew_VideoCutGroup": "Video Cut Group",
}
