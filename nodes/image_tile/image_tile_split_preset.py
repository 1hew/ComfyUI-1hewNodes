import asyncio
import math
from typing import Any

import numpy as np
from PIL import Image
import torch

from comfy_api.latest import io, ui




class ImageTileSplitPreset(io.ComfyNode):
    def __init__(self, *args, **kwargs):
        pass

    # 预定义的尺寸列表
    PRESET_RESOLUTIONS = [
        ("672x1568 [1:2.33] (3:7)", 672, 1568),
        ("688x1504 [1:2.19]", 688, 1504),
        ("720x1456 [1:2.00] (1:2)", 720, 1456),
        ("752x1392 [1:1.85]", 752, 1392),
        ("800x1328 [1:1.66]", 800, 1328),
        ("832x1248 [1:1.50] (2:3)", 832, 1248),
        ("880x1184 [1:1.35]", 880, 1184),
        ("944x1104 [1:1.17]", 944, 1104),
        ("1024x1024 [1:1.00] (1:1)", 1024, 1024),
        ("1104x944 [1.17:1]", 1104, 944),
        ("1184x880 [1.35:1]", 1184, 880),
        ("1248x832 [1.50:1] (3:2)", 1248, 832),
        ("1328x800 [1.66:1]", 1328, 800),
        ("1392x752 [1.85:1]", 1392, 752),
        ("1456x720 [2.00:1] (2:1)", 1456, 720),
        ("1504x688 [2.19:1]", 1504, 688),
        ("1568x672 [2.33:1] (7:3)", 1568, 672),
    ]

    @classmethod
    def define_schema(cls) -> io.Schema:
        preset_options = ["auto"] + [p[0] for p in cls.PRESET_RESOLUTIONS]
        return io.Schema(
            node_id="1hew_ImageTileSplitPreset",
            display_name="Image Tile Split Preset",
            category="1hewNodes/image/tile",
            inputs=[
                io.Image.Input("image"),
                io.Float.Input("overlap_amount", default=0.05, min=0.0, max=512.0, step=0.01),
                io.Combo.Input("tile_preset_size", options=preset_options, default="auto"),
            ],
            outputs=[
                io.Image.Output(display_name="tile"),
                io.Custom("DICT").Output(display_name="tile_meta"),
            ],
        )

    @classmethod
    async def execute(
        cls, image: torch.Tensor, overlap_amount: float, tile_preset_size: str
    ) -> io.NodeOutput:
        if image.shape[0] > 1:
            image = image[:1]
        _, img_height, img_width, _ = image.shape
        
        (tile_width, tile_height), (rows, cols), size_info = await asyncio.to_thread(
            cls.calculate_best_tile_size, img_width, img_height, tile_preset_size
        )
        
        # 计算重叠像素
        overlap_width, overlap_height, overlap_mode = cls.calculate_overlap_pixels(
            overlap_amount, img_width, img_height
        )
        
        tiles_list, tile_metas, original_size, grid_size = await cls.tile_image(
            image, tile_width, tile_height, overlap_width, overlap_height
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
            "tile_preset_size": tile_preset_size,
            "selected_size_info": size_info,
            "predefined_size": f"{tile_width}x{tile_height}"
        }
        
        return io.NodeOutput(tiles_output, tile_meta)

    @classmethod
    def calculate_best_tile_size(cls, image_width, image_height, tile_preset_size):
        """根据用户偏好选择tile尺寸"""
        # 如果用户指定了具体尺寸
        if tile_preset_size != "auto":
            # 从预设中查找匹配的尺寸
            for preset in cls.PRESET_RESOLUTIONS:
                if preset[0] == tile_preset_size:
                    tile_width, tile_height = preset[1], preset[2]
                    
                    # 计算网格信息
                    cols = max(1, math.ceil(image_width / tile_width))
                    rows = max(1, math.ceil(image_height / tile_height))
                    
                    # 计算效率
                    total_tile_area = cols * rows * tile_width * tile_height
                    image_area = image_width * image_height
                    efficiency = image_area / total_tile_area
                    
                    return (tile_width, tile_height), (rows, cols), {
                        'efficiency': efficiency,
                        'total_tiles': cols * rows,
                        'waste_pixels': total_tile_area - image_area,
                        'selected_size': preset[0],
                        'preset_name': preset[0]
                    }
        
        # 自动选择最佳尺寸
        image_area = image_width * image_height
        image_aspect = image_width / image_height
        best_size = None
        best_score = float('inf')
        best_grid = (1, 1)
        best_info = {}
        
        for preset in cls.PRESET_RESOLUTIONS:
            tile_width, tile_height = preset[1], preset[2]
            tile_aspect = tile_width / tile_height
            
            # 计算可以放置多少个这样的tile
            cols = max(1, math.ceil(image_width / tile_width))
            rows = max(1, math.ceil(image_height / tile_height))
            
            # 计算总的tile面积和浪费的像素
            total_tile_area = cols * rows * tile_width * tile_height
            waste_pixels = total_tile_area - image_area
            
            # 计算效率（图像面积 / 总tile面积）
            efficiency = image_area / total_tile_area
            
            # 计算宽高比匹配度
            aspect_diff = abs(image_aspect - tile_aspect)
            
            # 综合评分：效率 + 宽高比匹配 + tile数量
            score = (1 - efficiency) * 100 + aspect_diff * 50 + (cols * rows) * 0.1
            
            if score < best_score:
                best_score = score
                best_size = (tile_width, tile_height)
                best_grid = (rows, cols)
                best_info = {
                    'efficiency': efficiency,
                    'total_tiles': cols * rows,
                    'waste_pixels': waste_pixels,
                    'selected_size': f"{tile_width}x{tile_height}",
                    'coverage_ratio': efficiency,
                    'aspect_ratio': tile_width / tile_height,
                    'aspect_match_score': 1 / (1 + aspect_diff),
                    'preset_name': preset[0]
                }
        
        if best_size is None:
            # 如果没有找到合适的尺寸，使用最小的尺寸
            smallest_preset = min(cls.PRESET_RESOLUTIONS, key=lambda x: x[1] * x[2])
            tile_width, tile_height = smallest_preset[1], smallest_preset[2]
            best_size = (tile_width, tile_height)
            cols = max(1, math.ceil(image_width / tile_width))
            rows = max(1, math.ceil(image_height / tile_height))
            best_grid = (rows, cols)
            best_info = {
                'coverage_ratio': image_width * image_height / (cols * rows * tile_width * tile_height),
                'total_tiles': cols * rows,
                'waste_pixels': cols * rows * tile_width * tile_height - image_width * image_height,
                'aspect_ratio': tile_width / tile_height,
                'preset_name': smallest_preset[0]
            }
        
        return best_size, best_grid, best_info

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
    async def tile_image(cls, image, tile_width, tile_height, overlap_width, overlap_height):
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
