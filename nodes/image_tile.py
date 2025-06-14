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
        """根据overlap_amount计算实际的重叠像素数"""
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
        """计算tile尺寸，考虑重叠和divisible_by约束"""
        # 1. 计算重叠像素数
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

    def calculate_precise_positions(self, image_size, tile_size, overlap, num_tiles):
        """精确计算tile位置，确保完美覆盖"""
        if num_tiles == 1:
            return [0]
        
        positions = []
        effective_step = (image_size - tile_size) / (num_tiles - 1)
        
        for i in range(num_tiles):
            if i == num_tiles - 1:
                # 最后一个tile，确保右/下边界对齐
                pos = image_size - tile_size
            else:
                pos = int(i * effective_step)
            positions.append(max(0, pos))
        
        return positions

    def calculate_grid_layout(self, image_width, image_height, tile_width, tile_height, overlap_width, overlap_height):
        """计算网格布局，确保完美覆盖"""
        # 计算需要的tile数量
        effective_tile_width = tile_width - overlap_width
        effective_tile_height = tile_height - overlap_height
        
        num_cols = max(1, math.ceil((image_width - overlap_width) / effective_tile_width))
        num_rows = max(1, math.ceil((image_height - overlap_height) / effective_tile_height))
        
        # 计算精确位置
        col_positions = self.calculate_precise_positions(image_width, tile_width, overlap_width, num_cols)
        row_positions = self.calculate_precise_positions(image_height, tile_height, overlap_height, num_rows)
        
        return num_cols, num_rows, col_positions, row_positions

    def tile_image(self, image, tile_width, tile_height, overlap_width, overlap_height, rows, cols):
        """改进的图像分割方法，确保完美覆盖"""
        image_pil = self.tensor2pil(image.squeeze(0))
        img_width, img_height = image_pil.size
        
        # 计算精确的网格布局
        num_cols, num_rows, col_positions, row_positions = self.calculate_grid_layout(
            img_width, img_height, tile_width, tile_height, overlap_width, overlap_height
        )
        
        tiles = []
        tile_metas = []
        
        for row_idx, top in enumerate(row_positions):
            for col_idx, left in enumerate(col_positions):
                # 计算tile边界
                right = min(left + tile_width, img_width)
                bottom = min(top + tile_height, img_height)
                
                # 裁剪tile
                tile = image_pil.crop((left, top, right, bottom))
                
                # 如果tile尺寸不足，进行填充（保持一致的tile尺寸）
                actual_width = right - left
                actual_height = bottom - top
                
                if actual_width < tile_width or actual_height < tile_height:
                    # 创建标准尺寸的tile，用黑色填充
                    padded_tile = Image.new('RGB', (tile_width, tile_height), (0, 0, 0))
                    padded_tile.paste(tile, (0, 0))
                    tile = padded_tile
                
                tile_tensor = self.pil2tensor(tile)
                tiles.append(tile_tensor)
                
                # 记录tile信息
                tile_meta_item = {
                    'crop_region': (left, top, right, bottom),
                    'tile_size': (tile_width, tile_height),
                    'position': (col_idx, row_idx),
                    'actual_crop_size': (actual_width, actual_height)
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
                "blend_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("merged_image",)
    FUNCTION = "image_tile_merge"
    CATEGORY = "1hewNodes/image/tile"

    def create_weight_mask(self, tile_width, tile_height, overlap_width, overlap_height, position, grid_size, blend_strength):
        """创建更平滑的权重蒙版，模仿ttp.py的渐变效果"""
        col, row = position
        num_cols, num_rows = grid_size
        
        # 创建基础权重蒙版
        weight = np.ones((tile_height, tile_width), dtype=np.float32)
        
        if blend_strength <= 0 or (overlap_width <= 0 and overlap_height <= 0):
            return weight
        
        # 计算混合区域大小
        blend_width = max(1, int(overlap_width * blend_strength)) if overlap_width > 0 else 0
        blend_height = max(1, int(overlap_height * blend_strength)) if overlap_height > 0 else 0
        
        # 创建更平滑的渐变 - 使用余弦函数或高斯函数
        def smooth_transition(x, length):
            # 使用余弦函数创建更平滑的过渡
            return 0.5 * (1 + np.cos(np.pi * (1 - x / length)))
        
        # 左边缘渐变
        if col > 0 and blend_width > 0:
            fade_width = min(blend_width, tile_width)
            for x in range(fade_width):
                alpha = smooth_transition(x + 1, fade_width)
                weight[:, x] *= alpha
        
        # 右边缘渐变
        if col < num_cols - 1 and blend_width > 0:
            fade_width = min(blend_width, tile_width)
            start_x = max(0, tile_width - fade_width)
            for x in range(start_x, tile_width):
                alpha = smooth_transition(tile_width - x, fade_width)
                weight[:, x] *= alpha
        
        # 上边缘渐变
        if row > 0 and blend_height > 0:
            fade_height = min(blend_height, tile_height)
            for y in range(fade_height):
                alpha = smooth_transition(y + 1, fade_height)
                weight[y, :] *= alpha
        
        # 下边缘渐变
        if row < num_rows - 1 and blend_height > 0:
            fade_height = min(blend_height, tile_height)
            start_y = max(0, tile_height - fade_height)
            for y in range(start_y, tile_height):
                alpha = smooth_transition(tile_height - y, fade_height)
                weight[y, :] *= alpha
        
        return weight

    def image_tile_merge(self, tiles, tile_meta, blend_strength):
        """改进的拼接方法，使用权重图确保完美匹配"""
        # 从tile_meta中提取信息
        tile_metas = tile_meta["tile_metas"]
        original_size = tile_meta["original_size"]
        grid_size = tile_meta["grid_size"]
        overlap_width = tile_meta.get("overlap_width", 0)
        overlap_height = tile_meta.get("overlap_height", 0)
        tile_width = tile_meta.get("tile_width", 512)
        tile_height = tile_meta.get("tile_height", 512)
        
        orig_width, orig_height = original_size
        num_cols, num_rows = grid_size
        
        # 处理tiles输入格式
        tiles_tensor = []
        if hasattr(tiles, 'shape') and len(tiles.shape) == 4:
            for i in range(tiles.shape[0]):
                tiles_tensor.append(tiles[i])
        elif hasattr(tiles, 'shape') and len(tiles.shape) == 3:
            tiles_tensor.append(tiles)
        else:
            tiles_tensor.append(tiles)
        
        # 验证tiles数量
        expected_tiles = num_rows * num_cols
        if len(tiles_tensor) != expected_tiles:
            print(f"警告：期望 {expected_tiles} 个tiles，但收到 {len(tiles_tensor)} 个")
            if len(tiles_tensor) > expected_tiles:
                tiles_tensor = tiles_tensor[:expected_tiles]
            else:
                while len(tiles_tensor) < expected_tiles:
                    if tiles_tensor:
                        tiles_tensor.append(tiles_tensor[-1])
                    else:
                        empty_tile = torch.zeros((tile_height, tile_width, 3))
                        tiles_tensor.append(empty_tile)
        
        # 创建累加图像和权重图（使用更高精度）
        final_image_array = np.zeros((orig_height, orig_width, 3), dtype=np.float64)
        weight_map = np.zeros((orig_height, orig_width), dtype=np.float64)
        
        # 逐个处理每个tile
        for i, tile_tensor in enumerate(tiles_tensor):
            if i >= len(tile_metas):
                break
            
            # 转换tile为PIL图像
            if isinstance(tile_tensor, torch.Tensor):
                if len(tile_tensor.shape) == 3:
                    tile_image = self.tensor2pil(tile_tensor.unsqueeze(0))
                else:
                    tile_image = self.tensor2pil(tile_tensor)
            else:
                tile_image = tile_tensor
            
            tile_meta_item = tile_metas[i]
            crop_region = tile_meta_item['crop_region']
            position = tile_meta_item['position']
            actual_crop_size = tile_meta_item.get('actual_crop_size', (tile_width, tile_height))
            
            left, top, right, bottom = crop_region
            actual_width, actual_height = actual_crop_size
            
            # 只使用tile的有效区域（去除填充部分）
            if tile_image.size != (actual_width, actual_height):
                tile_image = tile_image.crop((0, 0, actual_width, actual_height))
            
            # 转换为numpy数组（使用更高精度）
            tile_array = np.array(tile_image, dtype=np.float64)
            
            # 创建权重蒙版
            tile_weight = self.create_weight_mask(
                actual_width, actual_height, overlap_width, overlap_height, 
                position, grid_size, blend_strength
            )
            
            # 确保权重蒙版尺寸匹配
            if tile_weight.shape != (actual_height, actual_width):
                tile_weight = tile_weight[:actual_height, :actual_width]
            
            # 累加到最终图像
            final_image_array[top:bottom, left:right] += tile_array * tile_weight[:, :, np.newaxis]
            weight_map[top:bottom, left:right] += tile_weight
        
        # 归一化处理
        # 避免除零错误，使用更小的epsilon值
        epsilon = 1e-8
        weight_map = np.maximum(weight_map, epsilon)
        final_image_array = final_image_array / weight_map[:, :, np.newaxis]
        
        # 确保像素值在有效范围内
        final_image_array = np.clip(final_image_array, 0, 255)
        
        # 转换回PIL图像
        final_image = Image.fromarray(final_image_array.astype(np.uint8))
        
        return (self.pil2tensor(final_image),)


NODE_CLASS_MAPPINGS = {
    "ImageTileSplit": ImageTileSplit,
    "ImageTileMerge": ImageTileMerge, 
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageTileSplit": "Image Tile Split",
    "ImageTileMerge": "Image Tile Merge",
}