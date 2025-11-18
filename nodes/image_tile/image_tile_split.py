import asyncio
import math
from typing import Any

import numpy as np
from PIL import Image
import torch

from comfy_api.latest import io, ui

class ImageTileSplit(io.ComfyNode):
    def __init__(self, *args, **kwargs):
        pass


    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_ImageTileSplit",
            display_name="Image Tile Split",
            category="1hewNodes/image/tile",
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("split_mode", options=["auto", "custom", "2x2", "2x3", "2x4", "3x2", "3x3", "3x4", "4x2", "4x3", "4x4"], default="auto"),
                io.Float.Input("overlap_amount", default=0.05, min=0.0, max=512.0, step=0.01),
                io.Int.Input("custom_rows", default=2, min=1, max=10, optional=True, step=1),
                io.Int.Input("custom_cols", default=2, min=1, max=10, optional=True, step=1),
                io.Int.Input("divisible_by", default=8, min=1, max=64, optional=True, step=1),
            ],
            outputs=[
                io.Image.Output(display_name="tile"),
                io.Custom("DICT").Output(display_name="tile_meta"),
            ],
        )

    @classmethod
    async def execute(
        cls,
        image: torch.Tensor,
        split_mode: str,
        overlap_amount: float,
        custom_rows: int | None = 2,
        custom_cols: int | None = 2,
        divisible_by: int | None = 8,
    ) -> io.NodeOutput:
        if image.shape[0] > 1:
            image = image[:1]
        _, img_height, img_width, _ = image.shape
        
        if split_mode == "auto":
            rows, cols = await asyncio.to_thread(
                cls.calculate_auto_grid, img_width, img_height
            )
        elif split_mode == "custom":
            rows, cols = custom_rows, custom_cols
        else:
            parts = split_mode.split('x')
            rows = int(parts[0])
            cols = int(parts[1])
        
        div = divisible_by if divisible_by is not None else 8
        tile_width, tile_height, overlap_width, overlap_height, overlap_mode = await asyncio.to_thread(
            cls.calculate_tile_size_with_overlap,
            img_width,
            img_height,
            cols,
            rows,
            overlap_amount,
            div,
        )
        
        tiles_list, tile_metas, original_size, grid_size = await cls.tile_image(
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
        
        return io.NodeOutput(tiles_output, tile_meta)


    @staticmethod
    def calculate_auto_grid(width, height, target_size=1024):
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

    @staticmethod
    def calculate_overlap_pixels(overlap_amount, image_width, image_height):
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

    @classmethod
    def calculate_tile_size_with_overlap(cls, image_width, image_height, cols, rows, overlap_amount, divisible_by):
        """计算tile尺寸，考虑重叠和divisible_by约束"""
        # 1. 计算重叠像素数
        overlap_width, overlap_height, overlap_mode = cls.calculate_overlap_pixels(
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

    @staticmethod
    def calculate_precise_positions(image_size, tile_size, overlap, num_tiles):
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

    @classmethod
    def calculate_grid_layout(cls, image_width, image_height, tile_width, tile_height, overlap_width, overlap_height):
        """计算网格布局，确保完美覆盖"""
        # 计算需要的tile数量
        effective_tile_width = tile_width - overlap_width
        effective_tile_height = tile_height - overlap_height
        
        num_cols = max(1, math.ceil((image_width - overlap_width) / effective_tile_width))
        num_rows = max(1, math.ceil((image_height - overlap_height) / effective_tile_height))
        
        # 计算精确位置
        col_positions = cls.calculate_precise_positions(image_width, tile_width, overlap_width, num_cols)
        row_positions = cls.calculate_precise_positions(image_height, tile_height, overlap_height, num_rows)
        
        return num_cols, num_rows, col_positions, row_positions

    @classmethod
    async def tile_image(cls, image, tile_width, tile_height, overlap_width, overlap_height, rows, cols):
        image_pil = cls.tensor2pil(image[0])
        img_width, img_height = image_pil.size
        num_cols, num_rows, col_positions, row_positions = cls.calculate_grid_layout(
            img_width, img_height, tile_width, tile_height, overlap_width, overlap_height
        )
        tasks = []
        order = 0
        for row_idx, top in enumerate(row_positions):
            for col_idx, left in enumerate(col_positions):
                right = min(left + tile_width, img_width)
                bottom = min(top + tile_height, img_height)
                actual_width = right - left
                actual_height = bottom - top
                async def _one(idx, l, t, r, b, aw, ah, ci, ri):
                    def _do():
                        tile = image_pil.crop((l, t, r, b))
                        if aw < tile_width or ah < tile_height:
                            pad = Image.new('RGB', (tile_width, tile_height), (0, 0, 0))
                            pad.paste(tile, (0, 0))
                            tile = pad
                        tile_tensor = cls.pil2tensor(tile)
                        meta_item = {
                            'crop_region': (l, t, r, b),
                            'tile_size': (tile_width, tile_height),
                            'position': (ci, ri),
                            'actual_crop_size': (aw, ah)
                        }
                        return idx, tile_tensor, meta_item
                    return await asyncio.to_thread(_do)
                tasks.append(_one(order, left, top, right, bottom, actual_width, actual_height, col_idx, row_idx))
                order += 1
        results = await asyncio.gather(*tasks)
        results.sort(key=lambda x: x[0])
        tiles = [t for _, t, _ in results]
        tile_metas = [m for _, _, m in results]
        return tiles, tile_metas, (img_width, img_height), (num_cols, num_rows)
 
    @staticmethod
    def pil2tensor(image: Image) -> torch.Tensor:
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

    @staticmethod
    def tensor2pil(t_image: torch.Tensor) -> Image:
        arr = t_image.detach().cpu().numpy()
        if arr.ndim == 4:
            arr = arr[0]
        return Image.fromarray(np.clip(255.0 * arr, 0, 255).astype(np.uint8))

