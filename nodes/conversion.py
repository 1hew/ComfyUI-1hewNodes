import json
import ast
import re
import torch
from typing import List, Dict, Any, Union


class StringCoordinateToBBoxes:
    """
    将字符串格式的坐标列表转换为 BBOXES 格式
    支持多种输入格式："x1,y1,x2,y2" 或 "[x1,y1,x2,y2]" 或多行坐标
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "coordinates_string": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": 'Supports "[x1,y1,x2,y2]" or multi-line coordinates'
                })
            }
        }
    
    RETURN_TYPES = ("BBOXES",)
    RETURN_NAMES = ("bboxes",)
    FUNCTION = "string_coordinate_to_bboxes"
    CATEGORY = "1hewNodes/conversion"
    
    def string_coordinate_to_bboxes(self, coordinates_string: str):
        """将字符串坐标转换为 BBOXES 格式"""
        if not coordinates_string.strip():
            return ([[]])
        
        lines = coordinates_string.strip().split('\n')
        bboxes = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 清理格式
            line = line.replace('[', '').replace(']', '').replace('(', '').replace(')', '')
            
            # 分割坐标
            coords = []
            for part in line.replace(',', ' ').split():
                try:
                    coords.append(int(float(part)))
                except ValueError:
                    continue
            
            if len(coords) >= 4:
                bboxes.append(coords[:4])
        
        if not bboxes:
            return ([[]])
        
        # 转换为 SAM2 兼容格式
        sam2_bboxes = [bboxes]
        
        return (sam2_bboxes,)


class ImageBatchToList:
    """
    将批量Image转换为Image列表
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_batch": ("IMAGE",)
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_list",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "image_batch_to_list"
    CATEGORY = "1hewNodes/conversion"
    
    def image_batch_to_list(self, image_batch):
        """将批量Image转换为Image列表"""
        if image_batch is None or image_batch.shape[0] == 0:
            return ([])
        
        # 将批量维度拆分为列表
        image_list = [image_batch[i:i+1] for i in range(image_batch.shape[0])]
        
        return (image_list,)


class ImageListToBatch:
    """
    将Image列表转换为批量Image
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_list": ("IMAGE",)
            }
        }
    
    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_batch",)
    FUNCTION = "image_list_to_batch"
    CATEGORY = "1hewNodes/conversion"
    
    def image_list_to_batch(self, image_list):
        """将Image列表转换为批量Image"""
        if not image_list:
            # 返回空的图像张量
            return (torch.zeros((0, 64, 64, 3)),)
        
        # 确保所有图像具有相同的尺寸
        if len(image_list) == 1:
            return (image_list[0],)
        
        # 获取最大尺寸
        max_height = max(img.shape[-3] for img in image_list)
        max_width = max(img.shape[-2] for img in image_list)
        channels = image_list[0].shape[-1]
        
        # 填充所有图像到相同尺寸
        padded_images = []
        for img in image_list:
            if len(img.shape) == 3:
                img = img.unsqueeze(0)  # 添加批量维度
            
            # 计算填充
            pad_h = max_height - img.shape[-3]
            pad_w = max_width - img.shape[-2]
            
            if pad_h > 0 or pad_w > 0:
                # 使用torch.nn.functional.pad进行填充 (left, right, top, bottom)
                img = torch.nn.functional.pad(img, (0, 0, 0, pad_w, 0, pad_h), mode='constant', value=0)
            
            padded_images.append(img)
        
        # 连接所有图像
        batch_image = torch.cat(padded_images, dim=0)
        
        return (batch_image,)


class MaskBatchToList:
    """
    将批量Mask转换为Mask列表
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_batch": ("MASK",)
            }
        }
    
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask_list",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "mask_batch_to_list"
    CATEGORY = "1hewNodes/conversion"
    
    def mask_batch_to_list(self, mask_batch):
        """将批量Mask转换为Mask列表"""
        if mask_batch is None or mask_batch.shape[0] == 0:
            return ([])
        
        # 将批量维度拆分为列表
        mask_list = [mask_batch[i:i+1] for i in range(mask_batch.shape[0])]
        
        return (mask_list,)


class MaskListToBatch:
    """
    将Mask列表转换为批量Mask
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_list": ("MASK",)
            }
        }
    
    INPUT_IS_LIST = True
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask_batch",)
    FUNCTION = "mask_list_to_batch"
    CATEGORY = "1hewNodes/conversion"
    
    def mask_list_to_batch(self, mask_list):
        """将Mask列表转换为批量Mask"""
        if not mask_list:
            # 返回空的mask张量
            return (torch.zeros((0, 64, 64)),)
        
        # 确保所有mask具有相同的尺寸
        if len(mask_list) == 1:
            return (mask_list[0],)
        
        # 获取最大尺寸
        max_height = max(mask.shape[-2] for mask in mask_list)
        max_width = max(mask.shape[-1] for mask in mask_list)
        
        # 填充所有mask到相同尺寸
        padded_masks = []
        for mask in mask_list:
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)  # 添加批量维度
            
            # 计算填充
            pad_h = max_height - mask.shape[-2]
            pad_w = max_width - mask.shape[-1]
            
            if pad_h > 0 or pad_w > 0:
                # 使用torch.nn.functional.pad进行填充
                mask = torch.nn.functional.pad(mask, (0, pad_w, 0, pad_h), mode='constant', value=0)
            
            padded_masks.append(mask)
        
        # 连接所有mask
        batch_mask = torch.cat(padded_masks, dim=0)
        
        return (batch_mask,)


# 节点映射
NODE_CLASS_MAPPINGS = {
    "StringCoordinateToBBoxes": StringCoordinateToBBoxes,
    "ImageBatchToList": ImageBatchToList,
    "ImageListToBatch": ImageListToBatch,
    "MaskBatchToList": MaskBatchToList,
    "MaskListToBatch": MaskListToBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StringCoordinateToBBoxes": "String Coordinate to BBoxes",
    "ImageBatchToList": "Image Batch to List",
    "ImageListToBatch": "Image List to Batch",
    "MaskBatchToList": "Mask Batch to List",
    "MaskListToBatch": "Mask List to Batch",
}