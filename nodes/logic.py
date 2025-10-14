import torch
import numpy as np
from PIL import Image, ImageColor

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
                "last_batch_mode": (["keep_remaining", "backward_extend", "append_image"], {"default": "backward_extend"})
            },
            "optional": {
                "color": ("STRING", {"default": "1.0"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("image", "batch_total", "start_index", "batch_count", "effective_count")
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
            colored_image = torch.ones((1, height, width, channels), dtype=reference_image.dtype, device=reference_image.device)
        else:
            # 如果输入是3维 (height, width, channels)
            height, width, channels = reference_image.shape
            colored_image = torch.ones((1, height, width, channels), dtype=reference_image.dtype, device=reference_image.device)
        
        # 设置颜色
        colored_image[0, :, :, 0] = r
        colored_image[0, :, :, 1] = g
        colored_image[0, :, :, 2] = b
        
        return colored_image
    
    def _validate_parameters(self, total_images, batch_size, overlap):
        """验证参数有效性"""
        if overlap >= batch_size:
            raise ValueError(f"重叠帧数 ({overlap}) 必须小于批次大小 ({batch_size})")
        
        if total_images < 1:
            raise ValueError("输入图片数量必须大于0")
    
    def _calculate_batches_direct(self, total_images, batch_size, overlap):
        """计算直接输出模式的批次信息"""
        batches = []
        start_idx = 0
        
        while start_idx < total_images:
            end_idx = min(start_idx + batch_size, total_images)
            actual_count = end_idx - start_idx
            batches.append({
                'start': start_idx,
                'count': actual_count,
                'end': end_idx
            })
            
            # 如果当前批次已经包含了最后一张图片，停止计算
            if end_idx >= total_images:
                break
            
            # 计算下一批的起始位置
            start_idx = start_idx + batch_size - overlap
            
            # 如果下一批的起始位置已经超出范围，退出循环
            if start_idx >= total_images:
                break
        
        return batches
    
    def _calculate_batches_backward(self, total_images, batch_size, overlap):
        """计算向前推导模式的批次信息"""
        if total_images <= batch_size:
            # 如果总数不超过批次大小，直接返回一个批次
            return [{
                'start': 0,
                'count': total_images,
                'end': total_images
            }]
        
        # 计算理论批次数量
        step_size = batch_size - overlap
        if step_size <= 0:
            raise ValueError("批次大小必须大于重叠数")
        
        # 从前往后计算批次，但确保最后一批包含所有剩余帧
        batches = []
        start_idx = 0
        
        while start_idx < total_images:
            # 检查是否为最后一批
            remaining = total_images - start_idx
            if remaining <= batch_size:
                # 最后一批，包含所有剩余帧
                batches.append({
                    'start': start_idx,
                    'count': remaining,
                    'end': total_images
                })
                break
            else:
                # 普通批次
                batches.append({
                    'start': start_idx,
                    'count': batch_size,
                    'end': start_idx + batch_size
                })
                start_idx += step_size
        
        return batches
    
    def _calculate_batches_pad(self, total_images, batch_size, overlap):
        """计算补充白图模式的批次信息"""
        batches = []
        start_idx = 0
        
        while start_idx < total_images:
            end_idx = start_idx + batch_size
            actual_count = min(batch_size, total_images - start_idx)
            
            batches.append({
                'start': start_idx,
                'count': batch_size,  # 总是保持批次大小
                'actual_count': actual_count,  # 实际有效图片数量
                'end': end_idx
            })
            
            # 如果当前批次已经包含了最后一张图片，停止计算
            if start_idx + batch_size >= total_images:
                break
            
            # 计算下一批的起始位置
            start_idx = start_idx + batch_size - overlap
            
            # 如果下一批的起始位置已经超出范围，退出循环
            if start_idx >= total_images:
                break
        
        return batches
    
    def _calculate_effective_counts(self, batches, overlap, last_batch_mode):
        """计算每批次的有效帧数"""
        effective_counts = []
        
        for i, batch in enumerate(batches):
            if i == len(batches) - 1:
                # 最后一批
                if last_batch_mode == "append_image":
                    # 补充彩色图模式下，有效帧数为实际图片数量
                    effective_counts.append(batch.get('actual_count', batch['count']))
                else:
                    # 其他模式下，最后一批全部有效
                    effective_counts.append(batch['count'])
            else:
                # 非最后一批，有效帧数 = 批次大小 - 重叠数
                effective_count = batch['count'] - overlap
                effective_counts.append(max(0, effective_count))
        
        return effective_counts
    
    def split_batch_sequential(self, image, batch_size, overlap, last_batch_mode, color="1.0"):
        """
        顺序分割批量图片
        """
        # 验证参数
        total_images = len(image)
        self._validate_parameters(total_images, batch_size, overlap)
        
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
        
        # 根据模式计算批次信息
        if last_batch_mode == "keep_remaining":
            batches = self._calculate_batches_direct(total_images, batch_size, overlap)
        elif last_batch_mode == "backward_extend":
            batches = self._calculate_batches_backward(total_images, batch_size, overlap)
        else:  # append_image
            batches = self._calculate_batches_pad(total_images, batch_size, overlap)
            
            # 补充指定颜色图模式需要添加额外的彩色图
            max_needed = max(batch['end'] for batch in batches)
            if max_needed > total_images:
                colored_images = []
                for _ in range(max_needed - total_images):
                    colored_img = self._create_white_image(image, color)
                    colored_images.append(colored_img)
                # 将彩色图列表合并为一个tensor
                colored_batch = torch.cat(colored_images, dim=0)
                image = torch.cat([image, colored_batch], dim=0)
        
        # 计算有效帧数
        effective_counts = self._calculate_effective_counts(batches, overlap, last_batch_mode)
        
        # 生成输出数据
        start_indices = []
        batch_counts = []
        
        for batch in batches:
            start_indices.append(batch['start'])
            batch_counts.append(batch['count'])
        
        # 确定输出图像：只有append_image模式需要输出包含彩色图的图像，其他模式直接输出原始图像
        if last_batch_mode == "append_image":
            output_image = image  # 已经包含了彩色图填充
        else:
            # 恢复到原始输入图像（去除可能添加的彩色图填充）
            output_image = image[:original_total]
        
        return (output_image, len(batches), start_indices, batch_counts, effective_counts)


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


# 节点映射
NODE_CLASS_MAPPINGS = {
    "ImageBatchGroup": ImageBatchGroup,
    "ImageListAppend": ImageListAppend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageBatchGroup": "Image Batch Group",
    "ImageListAppend": "Image List Append",
}