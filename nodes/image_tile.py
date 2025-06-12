import torch
import numpy as np
from PIL import Image
import math

class ImageTileSplit:
    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def pil2tensor(image: Image) -> torch.Tensor:
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

    @staticmethod
    def tensor2pil(t_image: torch.Tensor) -> Image:
        return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "split_mode": (["auto", "custom", "2x2", "2x3", "2x4", "3x2", "3x3", "3x4", "4x2", "4x3", "4x4"], {"default": "auto"}),
                "overlap_amount": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 512.0, "step": 0.01}),
            },
            "optional": {
                "custom_rows": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1}),
                "custom_cols": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1}),
                "divisible_by": ("INT", {"default": 8, "min": 1, "max": 64, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "DICT")
    RETURN_NAMES = ("tiles", "tiles_meta")
    FUNCTION = "image_tile_split"
    CATEGORY = "1hewNodes/image/tile"

    def calculate_auto_grid(self, width, height, target_size=1024):
        """自动计算最佳网格划分，目标是每个tile尽可能接近1024x1024像素"""
        target_pixels = target_size * target_size
        total_pixels = width * height
        estimated_tiles = max(1, total_pixels // target_pixels)
        
        best_score = float('inf')
        best_grid = (1, 1)
        
        max_search = max(10, int(math.sqrt(estimated_tiles)) + 5)
        
        for rows in range(1, max_search):
            for cols in range(1, max_search):
                if rows * cols >= estimated_tiles * 0.8:
                    tile_w = width // cols
                    tile_h = height // rows
                    
                    size_diff = abs(tile_w - target_size) + abs(tile_h - target_size)
                    ratio_diff = abs(tile_w - tile_h)
                    score = size_diff + ratio_diff * 0.5
                    
                    if score < best_score:
                        best_score = score
                        best_grid = (rows, cols)
        
        return best_grid

    def calculate_overlap_pixels(self, overlap_amount, image_width, image_height):
        """根据overlap_amount计算实际的重叠像素数（优化版本）"""
        if overlap_amount <= 1.0:
            # 比例模式：基于图像尺寸计算重叠像素
            overlap_width = int(image_width * overlap_amount)
            overlap_height = int(image_height * overlap_amount)
            mode = "ratio"
        else:
            # 像素模式：直接使用像素值
            overlap_width = int(overlap_amount)
            overlap_height = int(overlap_amount)
            mode = "pixels"
        
        return overlap_width, overlap_height, mode

    def calculate_tile_size_with_overlap(self, image_width, image_height, cols, rows, overlap_amount, divisible_by):
        """优化版本：先计算重叠像素，再确定tile尺寸"""
        # 1. 直接基于图像尺寸计算重叠像素数
        overlap_width, overlap_height, overlap_mode = self.calculate_overlap_pixels(
            overlap_amount, image_width, image_height
        )
        
        # 2. 计算基础tile尺寸（不含重叠）
        base_tile_width = image_width // cols
        base_tile_height = image_height // rows
        
        # 3. 计算实际tile尺寸（含重叠）
        actual_tile_width = base_tile_width + overlap_width
        actual_tile_height = base_tile_height + overlap_height
        
        # 4. 确保尺寸是divisible_by的倍数
        actual_tile_width = ((actual_tile_width + divisible_by - 1) // divisible_by) * divisible_by
        actual_tile_height = ((actual_tile_height + divisible_by - 1) // divisible_by) * divisible_by
        
        # 5. 重新计算实际的重叠像素数（考虑divisible_by调整）
        final_overlap_width = actual_tile_width - base_tile_width
        final_overlap_height = actual_tile_height - base_tile_height
        
        return actual_tile_width, actual_tile_height, final_overlap_width, final_overlap_height, overlap_mode

    def calculate_step_with_overlap(self, size, tile_size, overlap):
        """计算步长，考虑重叠"""
        if size <= tile_size:
            return 1, 0
        else:
            # 计算需要的tile数量
            num_tiles = math.ceil((size - overlap) / (tile_size - overlap))
            if num_tiles <= 1:
                return 1, 0
            
            # 计算实际步长
            step = (size - tile_size) // (num_tiles - 1)
            return num_tiles, step

    def tile_image(self, image, tile_width, tile_height, overlap_width, overlap_height, rows, cols):
        """使用重叠方式分割图像"""
        image_pil = self.tensor2pil(image.squeeze(0))
        img_width, img_height = image_pil.size
        
        # 计算步长
        num_cols, step_x = self.calculate_step_with_overlap(img_width, tile_width, overlap_width)
        num_rows, step_y = self.calculate_step_with_overlap(img_height, tile_height, overlap_height)
        
        tiles = []
        tile_metas = []
        
        for row in range(num_rows):
            for col in range(num_cols):
                # 计算裁剪位置
                left = col * step_x
                top = row * step_y
                right = min(left + tile_width, img_width)
                bottom = min(top + tile_height, img_height)
                
                # 确保tile尺寸一致
                if right - left < tile_width:
                    left = max(0, img_width - tile_width)
                    right = img_width
                if bottom - top < tile_height:
                    top = max(0, img_height - tile_height)
                    bottom = img_height
                
                # 裁剪tile
                tile = image_pil.crop((left, top, right, bottom))
                tile_tensor = self.pil2tensor(tile)
                tiles.append(tile_tensor)
                
                # 记录tile信息
                tile_meta_item = {
                    'crop_region': (left, top, right, bottom),
                    'tile_size': (tile_width, tile_height),
                    'position': (col, row)
                }
                tile_metas.append(tile_meta_item)
        
        return tiles, tile_metas, (img_width, img_height), (num_cols, num_rows)
    
    def image_tile_split(self, image, split_mode, overlap_amount, divisible_by, custom_rows=2, custom_cols=2):
        _, img_height, img_width, _ = image.shape
        
        if split_mode == "auto":
            rows, cols = self.calculate_auto_grid(img_width, img_height)
        elif split_mode == "custom":
            rows, cols = custom_rows, custom_cols
        else:
            parts = split_mode.split('x')
            rows = int(parts[0])
            cols = int(parts[1])
        
        tile_width, tile_height, overlap_width, overlap_height, overlap_mode = self.calculate_tile_size_with_overlap(
            img_width, img_height, cols, rows, overlap_amount, divisible_by
        )
        
        tiles_list, tile_metas, original_size, grid_size = self.tile_image(
            image, tile_width, tile_height, overlap_width, overlap_height, rows, cols
        )
        
        if len(tiles_list) > 0:
            tiles_output = torch.cat(tiles_list, dim=0)
        else:
            tiles_output = torch.empty(0, tile_height, tile_width, 3)
        
        tile_meta = {
            "tile_metas": tile_metas,
            "original_size": original_size,
            "grid_size": grid_size,
            "tile_width": tile_width,
            "tile_height": tile_height,
            "rows": rows,
            "cols": cols,
            "overlap_amount": overlap_amount,
            "overlap_mode": overlap_mode,
            "overlap_width": overlap_width,
            "overlap_height": overlap_height,
            "split_mode": split_mode,
            "divisible_by": divisible_by
        }
        
        return (tiles_output, tile_meta)


class ImageTileMerge:
    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def pil2tensor(image: Image) -> torch.Tensor:
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

    @staticmethod
    def tensor2pil(t_image: torch.Tensor) -> Image:
        return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tiles": ("IMAGE",),
                "tile_meta": ("DICT",),
                "blend_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "blend_mode": (["none", "linear", "gaussian"], {"default": "linear"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("merged_image",)
    FUNCTION = "image_tile_merge"
    CATEGORY = "1hewNodes/image/tile"

    def apply_blend(self, base_image, overlay_image, mask, blend_mode):
        """应用混合模式"""
        if blend_mode == "none":
            return overlay_image
        
        base_array = np.array(base_image, dtype=np.float32)
        overlay_array = np.array(overlay_image, dtype=np.float32)
        mask_array = np.array(mask, dtype=np.float32) / 255.0
        
        if len(mask_array.shape) == 2:
            mask_array = np.stack([mask_array] * 3, axis=2)
        
        # 线性混合
        if blend_mode == "linear":
            blended = base_array * (1 - mask_array) + overlay_array * mask_array
        elif blend_mode == "gaussian":
            # 高斯混合（更平滑）
            sigma = 2.0
            mask_smooth = mask_array ** sigma
            blended = base_array * (1 - mask_smooth) + overlay_array * mask_smooth
        else:
            blended = overlay_image
        
        return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))

    def create_blend_mask(self, overlap_width, overlap_height, blend_strength):
        """创建混合蒙版（优化版本）"""
        if blend_strength <= 0:
            # 不混合，返回全白蒙版（完全使用新tile）
            return Image.new("L", (overlap_width, overlap_height), 255)
        
        # 计算实际混合区域像素数（关键优化）
        blend_width = int(overlap_width * blend_strength)
        blend_height = int(overlap_height * blend_strength)
        
        if blend_width <= 0 or blend_height <= 0:
            return Image.new("L", (overlap_width, overlap_height), 255)
        
        # 创建渐变蒙版
        mask = Image.new("L", (overlap_width, overlap_height), 0)
        mask_array = np.array(mask, dtype=np.float32)
        
        # 水平渐变（在混合区域内）
        for x in range(blend_width):
            alpha = x / blend_width if blend_width > 0 else 1.0
            mask_array[:, x] = alpha * 255
        
        # 非混合区域直接使用新tile（全白）
        if blend_width < overlap_width:
            mask_array[:, blend_width:] = 255
        
        # 垂直渐变处理（如果需要）
        if blend_height < overlap_height:
            for y in range(blend_height, overlap_height):
                mask_array[y, :] = 255
        
        return Image.fromarray(mask_array.astype(np.uint8))

    def image_tile_merge(self, tiles, tile_meta, blend_strength, blend_mode="linear"):
        # 从tile_meta中提取信息
        tile_metas = tile_meta["tile_metas"]
        original_size = tile_meta["original_size"]
        grid_size = tile_meta["grid_size"]
        overlap_width = tile_meta.get("overlap_width", 0)
        overlap_height = tile_meta.get("overlap_height", 0)
        overlap_mode = tile_meta.get("overlap_mode", "unknown")
        overlap_amount = tile_meta.get("overlap_amount", 0)
        
        # 打印重叠信息（用于调试）
        print(f"重叠模式: {overlap_mode}, 重叠值: {overlap_amount}")
        print(f"实际重叠像素: {overlap_width}x{overlap_height}")
        print(f"混合比例: {blend_strength}, 混合像素: {int(overlap_width * blend_strength)}x{int(overlap_height * blend_strength)}")
        
        num_cols, num_rows = grid_size
        orig_width, orig_height = original_size
        
        # 处理批量tensor格式输入
        tiles_tensor = []
        
        if hasattr(tiles, 'shape') and len(tiles.shape) == 4:
            # 标准的批量tensor格式 [batch, height, width, channels]
            for i in range(tiles.shape[0]):
                tiles_tensor.append(tiles[i])
        elif hasattr(tiles, 'shape') and len(tiles.shape) == 3:
            # 单个3D tensor
            tiles_tensor.append(tiles)
        else:
            # 其他格式，尝试转换
            tiles_tensor.append(tiles)
        
        # 验证tiles数量是否匹配
        expected_tiles = num_rows * num_cols
        if len(tiles_tensor) != expected_tiles:
            print(f"警告：期望 {expected_tiles} 个tiles，但收到 {len(tiles_tensor)} 个")
            # 如果数量不匹配，尝试调整
            if len(tiles_tensor) > expected_tiles:
                tiles_tensor = tiles_tensor[:expected_tiles]
            else:
                # 如果tiles不够，重复最后一个tile
                while len(tiles_tensor) < expected_tiles:
                    if tiles_tensor:
                        tiles_tensor.append(tiles_tensor[-1])
                    else:
                        # 如果没有任何tile，创建一个空的
                        empty_tile = torch.zeros((tile_meta.get('tile_height', 512), tile_meta.get('tile_width', 512), 3))
                        tiles_tensor.append(empty_tile)
        
        # 创建最终图像
        final_image = Image.new("RGB", original_size)
        
        # 使用优化的智能混合拼接
        for i, tile_tensor in enumerate(tiles_tensor):
            if i >= len(tile_metas):
                break
                
            # 确保tile_tensor是正确的格式
            if isinstance(tile_tensor, torch.Tensor):
                if len(tile_tensor.shape) == 3:
                    tile_image = self.tensor2pil(tile_tensor.unsqueeze(0))
                elif len(tile_tensor.shape) == 4:
                    tile_image = self.tensor2pil(tile_tensor)
                else:
                    tile_image = self.tensor2pil(tile_tensor.unsqueeze(0) if len(tile_tensor.shape) == 3 else tile_tensor)
            else:
                tile_image = self.tensor2pil(self.pil2tensor(tile_tensor))
            
            tile_meta_item = tile_metas[i]
            crop_region = tile_meta_item['crop_region']
            paste_x, paste_y = crop_region[0], crop_region[1]
            
            if i == 0 or blend_strength <= 0 or overlap_width <= 0 or overlap_height <= 0:
                # 第一个tile或不需要混合，直接粘贴
                final_image.paste(tile_image, (paste_x, paste_y))
            else:
                # 需要混合的tile
                # 获取重叠区域
                overlap_region = final_image.crop((paste_x, paste_y, paste_x + overlap_width, paste_y + overlap_height))
                tile_overlap = tile_image.crop((0, 0, overlap_width, overlap_height))
                
                # 创建优化的混合蒙版
                blend_mask = self.create_blend_mask(overlap_width, overlap_height, blend_strength)
                
                # 应用混合
                blended_overlap = self.apply_blend(overlap_region, tile_overlap, blend_mask, blend_mode)
                
                # 粘贴混合后的重叠区域
                final_image.paste(blended_overlap, (paste_x, paste_y))
                
                # 粘贴tile的非重叠部分
                if tile_image.width > overlap_width:
                    non_overlap_right = tile_image.crop((overlap_width, 0, tile_image.width, tile_image.height))
                    final_image.paste(non_overlap_right, (paste_x + overlap_width, paste_y))
                
                if tile_image.height > overlap_height:
                    non_overlap_bottom = tile_image.crop((0, overlap_height, tile_image.width, tile_image.height))
                    final_image.paste(non_overlap_bottom, (paste_x, paste_y + overlap_height))
        
        return (self.pil2tensor(final_image),)


NODE_CLASS_MAPPINGS = {
    "ImageTileSplit": ImageTileSplit,
    "ImageTileMerge": ImageTileMerge, 
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageTileSplit": "Image Tile Split",
    "ImageTileMerge": "Image Tile Merge",
}