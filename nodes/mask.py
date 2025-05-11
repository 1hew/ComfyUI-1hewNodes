import torch
import numpy as np
from PIL import Image, ImageOps

class MaskMathOps:
    """
    蒙版操作节点 - 支持蒙版之间的相交、相加、相减、异或等操作，并支持批处理
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_a": ("MASK",),
                "mask_b": ("MASK",),
                "operation": (["or", "and", "subtract (a-b)", "subtract (b-a)", "xor"], 
                             {"default": "or"})
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "mask_math_ops"
    CATEGORY = "1hewNodes/mask"

    def mask_math_ops(self, mask_a, mask_b, operation):
        # 获取蒙版尺寸
        batch_size_a = mask_a.shape[0]
        batch_size_b = mask_b.shape[0]
        
        # 创建输出蒙版列表
        output_masks = []
        
        # 确定批处理大小（使用最大的批次大小）
        max_batch_size = max(batch_size_a, batch_size_b)
        
        for b in range(max_batch_size):
            # 获取当前批次的蒙版（循环使用如果批次大小不匹配）
            current_mask_a = mask_a[b % batch_size_a]
            current_mask_b = mask_b[b % batch_size_b]
            
            # 将蒙版转换为PIL格式以便处理
            if mask_a.is_cuda:
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


class BatchMaskMathOps:
    """
    批量蒙版数学运算节点 - 支持批量处理所有图层的OR和AND功能
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
                "operation": (["or", "and"], {"default": "or"})
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "batch_mask_math_ops"
    CATEGORY = "1hewNodes/mask"

    def batch_mask_math_ops(self, masks, operation):
        # 获取批次大小
        batch_size = masks.shape[0]
        
        # 如果批次大小为1，直接返回
        if batch_size <= 1:
            return (masks,)
        
        # 创建输出蒙版
        output_mask = None
        
        # 对每个批次进行处理
        for b in range(batch_size):
            current_mask = masks[b]
            
            # 将蒙版转换为numpy数组
            if masks.is_cuda:
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


class MaskBlend:
    """
    蒙版混合器 - 支持两个蒙版之间的混合，可调整混合比例
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_a": ("MASK",),
                "mask_b": ("MASK",),
                "blend_factor": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "label": "混合比例 (0=仅A, 1=仅B)"})
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "blend_masks"
    CATEGORY = "1hewNodes/mask"

    def blend_masks(self, mask_a, mask_b, blend_factor=0.5):
        # 获取蒙版尺寸
        batch_size_a = mask_a.shape[0]
        batch_size_b = mask_b.shape[0]
        
        # 创建输出蒙版列表
        output_masks = []
        
        # 确定批处理大小（使用最大的批次大小）
        max_batch_size = max(batch_size_a, batch_size_b)
        
        for b in range(max_batch_size):
            # 获取当前批次的蒙版（循环使用如果批次大小不匹配）
            current_mask_a = mask_a[b % batch_size_a]
            current_mask_b = mask_b[b % batch_size_b]
            
            # 将蒙版转换为PIL格式以便处理
            if mask_a.is_cuda:
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
            
            # 将PIL图像转换为numpy数组进行混合
            mask_a_array = np.array(mask_a_pil).astype(np.float32) / 255.0
            mask_b_array = np.array(mask_b_pil).astype(np.float32) / 255.0
            
            # 线性混合
            result_array = mask_a_array * (1.0 - blend_factor) + mask_b_array * blend_factor
            
            # 转换回tensor
            result_tensor = torch.from_numpy(result_array)
            output_masks.append(result_tensor)
        
        # 合并批次
        output_tensor = torch.stack(output_masks)
        
        return (output_tensor,)


# 注册节点
NODE_CLASS_MAPPINGS = {
    "MaskMathOps": MaskMathOps,
    "BatchMaskMathOps": BatchMaskMathOps,
    "MaskBlend": MaskBlend,
}

# 节点显示名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskMathOps": "Mask Math Ops",
    "BatchMaskMathOps": "Batch Mask Math Ops",
    "MaskBlend": "Mask Blend",
}
