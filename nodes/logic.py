import torch
import numpy as np
from PIL import Image, ImageColor


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
    CATEGORY = "1hewNodes/logic"
    
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
    CATEGORY = "1hewNodes/logic"
    
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
    CATEGORY = "1hewNodes/logic"
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
        
        for i, start_idx in enumerate(start_indices):
            if i == len(start_indices) - 1:
                # 最后一批
                if last_batch_mode == "fill_color":
                    # 补充彩色图模式：总是保持批次大小
                    batch_counts.append(batch_size)
                else:
                    # 其他模式：根据剩余图片数量确定
                    remaining = total_images - start_idx
                    if last_batch_mode in ["backtrack_last", "drop_incomplete"]:
                        # backtrack_last 和 drop_incomplete 模式：保持批次大小
                        batch_counts.append(batch_size)
                    else:
                        # keep_remaining 模式：使用实际剩余数量
                        batch_counts.append(remaining)
            else:
                # 非最后一批：总是使用批次大小
                batch_counts.append(batch_size)
        
        return batch_counts
    
    def _calculate_valid_counts(self, start_indices, batch_counts, overlap, last_batch_mode, total_images=None):
        """计算每批次的有效帧数"""
        valid_counts = []
        
        for i, (start_idx, batch_count) in enumerate(zip(start_indices, batch_counts)):
            if last_batch_mode == "fill_color":
                # fill_color模式：每批次的有效数量等于step_size，最后一批特殊处理
                if i == len(start_indices) - 1:
                    # 最后一批：计算实际有效的原始图像数量
                    if total_images is not None:
                        original_images_in_last_batch = total_images - start_idx
                        valid_counts.append(original_images_in_last_batch)
                    else:
                        # 如果没有传入total_images，使用batch_count
                        valid_counts.append(batch_count)
                else:
                    # 非最后一批：有效数量等于step_size
                    step_size = batch_counts[0] - overlap  # batch_size - overlap
                    valid_counts.append(step_size)
            else:
                # 其他模式的原有逻辑
                if i == len(start_indices) - 1:
                    # 最后一批：全部有效
                    valid_counts.append(batch_count)
                else:
                    # 非最后一批：有效数量 = 下一批的起始位置 - 当前批的起始位置
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
        batch_counts = self._calculate_batch_counts(start_indices, total_images, batch_size, last_batch_mode)
        
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
    CATEGORY = "1hewNodes/logic"
    
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


NODE_CLASS_MAPPINGS = {
    "ImageBatchSplit": ImageBatchSplit,
    "MaskBatchSplit": MaskBatchSplit,
    "ImageBatchGroup": ImageBatchGroup,
    "ImageListAppend": ImageListAppend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageBatchSplit": "Image Batch Split",
    "MaskBatchSplit": "Mask Batch Split",
    "ImageBatchGroup": "Image Batch Group",
    "ImageListAppend": "Image List Append",
}