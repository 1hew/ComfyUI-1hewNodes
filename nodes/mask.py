import torch
import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage

class MaskFillHole:
    """
    遮罩孔洞填充节点 - 支持填充封闭区域内的孔洞
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "invert_mask": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "fill_holes"
    CATEGORY = "1hewNodes/mask"

    def fill_holes(self, mask, invert_mask):
        # 确保mask是3D张量 [batch, height, width]
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
            
        batch_size = mask.shape[0]
        output_masks = []
        
        for b in range(batch_size):
            current_mask = mask[b]
            
            # 将mask转换为numpy数组
            if mask.is_cuda:
                mask_np = (current_mask.cpu().numpy() * 255).astype(np.uint8)
            else:
                mask_np = (current_mask.numpy() * 255).astype(np.uint8)
            
            # 转换为PIL图像进行处理
            mask_pil = Image.fromarray(mask_np, mode="L")
            
            # 执行填充操作
            filled_mask = self._fill_holes_internal(mask_pil, invert_mask)
            
            # 转换回tensor
            filled_np = np.array(filled_mask).astype(np.float32) / 255.0
            filled_tensor = torch.from_numpy(filled_np)
            output_masks.append(filled_tensor)
        
        # 合并批次
        output_tensor = torch.stack(output_masks)
        
        return (output_tensor,)
    
    def _fill_holes_internal(self, mask_pil, invert_mask):
        """
        填充封闭区域内的孔洞
        """
        # 转换为numpy数组
        mask_array = np.array(mask_pil)
        
        # 二值化处理
        binary_mask = mask_array > 127
        
        # 使用scipy的binary_fill_holes填充孔洞（使用8连通性）
        structure = ndimage.generate_binary_structure(2, 2)
        filled_mask = ndimage.binary_fill_holes(binary_mask, structure=structure)
        
        # 如果需要反转mask，在填充完成后再反转
        if invert_mask:
            filled_mask = ~filled_mask
        
        # 转换回PIL图像
        filled_array = (filled_mask * 255).astype(np.uint8)
        return Image.fromarray(filled_array, mode="L")


class MaskMathOps:
    """
    蒙版操作节点 - 支持蒙版之间的相交、相加、相减、异或等操作，并支持批处理
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_1": ("MASK",),
                "mask_2": ("MASK",),
                "operation": (["or", "and", "subtract (a-b)", "subtract (b-a)", "xor"], 
                             {"default": "or"})
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "mask_math_ops"
    CATEGORY = "1hewNodes/mask"

    def mask_math_ops(self, mask_1, mask_2, operation):
        # 获取蒙版尺寸
        batch_size_a = mask_1.shape[0]
        batch_size_b = mask_2.shape[0]
        
        # 创建输出蒙版列表
        output_masks = []
        
        # 确定批处理大小（使用最大的批次大小）
        max_batch_size = max(batch_size_a, batch_size_b)
        
        for b in range(max_batch_size):
            # 获取当前批次的蒙版（循环使用如果批次大小不匹配）
            current_mask_a = mask_1[b % batch_size_a]
            current_mask_b = mask_2[b % batch_size_b]
            
            # 将蒙版转换为PIL格式以便处理
            if mask_1.is_cuda:
                mask_a_np = (current_mask_a.cpu().numpy() * 255).astype(np.uint8)
                mask_b_np = (current_mask_b.cpu().numpy() * 255).astype(np.uint8)
            else:
                mask_a_np = (current_mask_a.numpy() * 255).astype(np.uint8)
                mask_b_np = (current_mask_b.numpy() * 255).astype(np.uint8)
            
            mask_a_pil = Image.fromarray(mask_a_np)
            mask_b_pil = Image.fromarray(mask_b_np)
            
            # 调整蒙版大小以匹配
            if mask_a_pil.size != mask_b_pil.size:
                mask_b_pil = mask_b_pil.resize(mask_a_pil.size, Image.Resampling.LANCZOS)
            
            # 将PIL图像转换为numpy数组进行操作
            mask_a_array = np.array(mask_a_pil).astype(np.float32) / 255.0
            mask_b_array = np.array(mask_b_pil).astype(np.float32) / 255.0
            
            # 应用选定的操作
            if operation == "and":
                result_array = np.minimum(mask_a_array, mask_b_array)
                    
            elif operation == "or":
                result_array = np.maximum(mask_a_array, mask_b_array)
                    
            elif operation == "subtract (a-b)":
                result_array = np.clip(mask_a_array - mask_b_array, 0, 1)
                    
            elif operation == "subtract (b-a)":
                result_array = np.clip(mask_b_array - mask_a_array, 0, 1)
                    
            elif operation == "xor":
                result_array = np.abs(mask_a_array - mask_b_array)
            
            # 转换回tensor
            result_tensor = torch.from_numpy(result_array)
            output_masks.append(result_tensor)
        
        # 合并批次
        output_tensor = torch.stack(output_masks)
        
        return (output_tensor,)


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
    CATEGORY = "1hewNodes/mask"

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


class MaskCropByBBoxMask:
    """
    遮罩检测框裁剪 - 根据边界框遮罩信息批量裁剪遮罩
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "bbox_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("cropped_mask",)
    FUNCTION = "mask_crop_by_bbox_mask"
    CATEGORY = "1hewNodes/mask"

    def mask_crop_by_bbox_mask(self, mask, bbox_mask):
        # 确保mask是3D张量 [batch, height, width]
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        if bbox_mask.dim() == 2:
            bbox_mask = bbox_mask.unsqueeze(0)
            
        # 获取遮罩尺寸
        batch_size = mask.shape[0]
        bbox_batch_size = bbox_mask.shape[0]

        # 创建输出遮罩列表
        output_masks = []
        
        for b in range(batch_size):
            # 获取当前批次对应的边界框遮罩
            bbox_idx = b % bbox_batch_size
            current_bbox_mask = bbox_mask[bbox_idx]
            current_mask = mask[b]
            
            # 将遮罩转换为PIL格式
            if mask.is_cuda:
                mask_np = (current_mask.cpu().numpy() * 255).astype(np.uint8)
                bbox_np = (current_bbox_mask.cpu().numpy() * 255).astype(np.uint8)
            else:
                mask_np = (current_mask.numpy() * 255).astype(np.uint8)
                bbox_np = (current_bbox_mask.numpy() * 255).astype(np.uint8)

            mask_pil = Image.fromarray(mask_np).convert("L")
            bbox_pil = Image.fromarray(bbox_np).convert("L")
            
            # 从bbox_mask中获取边界框坐标
            bbox = self.get_bbox_from_mask(bbox_pil)
            
            if bbox is None:
                # 如果没有找到有效区域，返回原始遮罩
                output_masks.append(current_mask)
                continue
            
            x_min, y_min, x_max, y_max = bbox
            
            # 确保边界框不超出图像范围
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(mask_pil.width, x_max)
            y_max = min(mask_pil.height, y_max)
            
            # 裁切遮罩
            cropped_mask = mask_pil.crop((x_min, y_min, x_max, y_max))

            # 转换回tensor
            cropped_mask_np = np.array(cropped_mask).astype(np.float32) / 255.0
            output_masks.append(torch.from_numpy(cropped_mask_np))

        # 合并批次
        if output_masks:
            # 检查所有遮罩是否具有相同尺寸
            sizes = [mask.shape for mask in output_masks]
            if len(set(sizes)) == 1:
                output_mask_tensor = torch.stack(output_masks)
            else:
                # 如果尺寸不同，填充到相同尺寸
                output_masks = self.pad_to_same_size(output_masks)
                output_mask_tensor = torch.stack(output_masks)
            return (output_mask_tensor,)
        else:
            # 如果没有有效的输出遮罩，返回原始遮罩
            return (mask,)
    
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
    
    def pad_to_same_size(self, masks):
        """将所有遮罩填充到相同尺寸"""
        max_height = max(mask.shape[0] for mask in masks)
        max_width = max(mask.shape[1] for mask in masks)
        
        padded_masks = []
        
        for mask in masks:
            h, w = mask.shape
            pad_h = max_height - h
            pad_w = max_width - w
            
            padded_mask = torch.nn.functional.pad(mask, (0, pad_w, 0, pad_h), value=0)
            padded_masks.append(padded_mask)
        
        return padded_masks


# 节点类映射
NODE_CLASS_MAPPINGS = {
    "MaskFillHole": MaskFillHole,
    "MaskMathOps": MaskMathOps,
    "MaskBatchMathOps": MaskBatchMathOps,
    "MaskCropByBBoxMask": MaskCropByBBoxMask,
}

# 节点显示名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskFillHole": "Mask Fill Hole",
    "MaskMathOps": "Mask Math Ops",
    "MaskBatchMathOps": "Mask Batch Math Ops",
    "MaskCropByBBoxMask": "Mask Crop by BBox Mask",
}