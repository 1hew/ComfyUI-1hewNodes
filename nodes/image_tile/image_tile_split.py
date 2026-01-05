import asyncio
import math
from typing import Any

import numpy as np
from PIL import Image
import torch

from comfy_api.latest import io


class ImageTileSplit(io.ComfyNode):
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

    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def define_schema(cls) -> io.Schema:
        preset_options = [p[0] for p in cls.PRESET_RESOLUTIONS]
        mode_options = ["auto", "grid"] + preset_options

        return io.Schema(
            node_id="1hew_ImageTileSplit",
            display_name="Image Tile Split",
            category="1hewNodes/image/tile",
            inputs=[
                io.Image.Input("image"),
                io.Image.Input("get_tile_size", optional=True),
                io.Combo.Input("mode", options=mode_options, default="auto"),
                io.Float.Input("overlap_amount", default=0.05, min=0.0, max=1024.0, step=0.01),
                io.Int.Input("grid_row", default=2, min=1, max=10, step=1),
                io.Int.Input("grid_col", default=2, min=1, max=10, step=1),
                io.Int.Input("divisible_by", default=8, min=1, max=1024, step=1),
            ],
            outputs=[
                io.Image.Output(display_name="tile"),
                io.Custom("DICT").Output(display_name="tile_meta"),
                io.Mask.Output(display_name="bbox_mask"),
            ],
        )

    @classmethod
    async def execute(
        cls,
        image: torch.Tensor,
        mode: str,
        overlap_amount: float,
        grid_row: int | None = 2,
        grid_col: int | None = 2,
        divisible_by: int | None = 8,
        get_tile_size: torch.Tensor | None = None,
    ) -> io.NodeOutput:
        if image.shape[0] > 1:
            image = image[:1]
        _, img_height, img_width, _ = image.shape

        # 初始化变量
        size_info = {}
        preset_name = None

        # 优先处理 get_tile_size
        if get_tile_size is not None:
            # 使用参考图片的尺寸作为 tile 尺寸
            _, ref_h, ref_w, _ = get_tile_size.shape
            tile_width, tile_height = ref_w, ref_h
            preset_name = "custom_image_reference"
            
            # 计算网格信息（复用 calculate_best_tile_size 的部分逻辑或直接计算）
            cols = max(1, math.ceil(img_width / tile_width))
            rows = max(1, math.ceil(img_height / tile_height))
            
            # 计算重叠像素
            # 参考图片模式下，是否应该受 divisible_by 影响？
            # 通常如果指定了具体尺寸，就像 preset 一样，应该尽量保持该尺寸。
            # 这里我们按照 preset 的逻辑处理：不受 divisible_by 影响，直接使用尺寸。
            (
                overlap_width,
                overlap_height,
                overlap_mode
            ) = cls.calculate_overlap_pixels(
                overlap_amount, img_width, img_height
            )
            
        elif mode in ["auto", "grid"]:
            # Auto 或 Grid 模式
            if mode == "auto":
                rows, cols = await asyncio.to_thread(
                    cls.calculate_auto_grid, img_width, img_height
                )
            else:  # grid
                rows, cols = grid_row, grid_col

            div = divisible_by if divisible_by is not None else 8
            (
                tile_width,
                tile_height,
                overlap_width,
                overlap_height,
                overlap_mode
            ) = await asyncio.to_thread(
                cls.calculate_tile_size_with_overlap,
                img_width,
                img_height,
                cols,
                rows,
                overlap_amount,
                div,
            )
        else:
            # Preset 模式
            # 查找对应的 preset 配置
            (
                (tile_width, tile_height),
                (rows, cols),
                size_info
            ) = await asyncio.to_thread(
                cls.calculate_best_tile_size,
                img_width,
                img_height,
                mode
            )

            # 计算重叠像素
            (
                overlap_width,
                overlap_height,
                overlap_mode
            ) = cls.calculate_overlap_pixels(
                overlap_amount, img_width, img_height
            )
            
            preset_name = mode

        # 执行切片
        (
            tiles_list,
            tile_metas,
            original_size,
            grid_size
        ) = await cls.tile_image(
            image,
            tile_width,
            tile_height,
            overlap_width,
            overlap_height
        )

        if len(tiles_list) > 0:
            tiles_output = torch.cat(tiles_list, dim=0)
        else:
            tiles_output = torch.empty(0, tile_height, tile_width, 3)

        bbox_masks = cls.build_bbox_masks(tile_metas, original_size)
        if bbox_masks.numel() > 0:
            bbox_masks = bbox_masks.to(device=image.device, dtype=image.dtype)

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
            "mode": mode,
            "divisible_by": divisible_by,
        }
        
        # 如果有额外信息（来自 preset），合并进去
        if size_info:
            tile_meta["selected_size_info"] = size_info
        if preset_name:
            tile_meta["tile_preset_size"] = preset_name
            tile_meta["predefined_size"] = f"{tile_width}x{tile_height}"

        return io.NodeOutput(tiles_output, tile_meta, bbox_masks)

    @staticmethod
    def build_bbox_masks(
        tile_metas: list[dict[str, Any]],
        original_size: tuple[int, int],
    ) -> torch.Tensor:
        img_width, img_height = original_size
        if not tile_metas:
            return torch.empty(0, img_height, img_width)

        masks = np.zeros((len(tile_metas), img_height, img_width), dtype=np.float32)
        for i, meta_item in enumerate(tile_metas):
            left, top, right, bottom = meta_item["crop_region"]
            left = max(0, int(left))
            top = max(0, int(top))
            right = min(img_width, int(right))
            bottom = min(img_height, int(bottom))
            if right > left and bottom > top:
                masks[i, top:bottom, left:right] = 1.0

        return torch.from_numpy(masks)

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

                    size_diff = (
                        abs(tile_w - target_size) + abs(tile_h - target_size)
                    )
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
    def calculate_tile_size_with_overlap(
        cls,
        image_width,
        image_height,
        cols,
        rows,
        overlap_amount,
        divisible_by
    ):
        """计算tile尺寸，考虑重叠和divisible_by约束"""
        # 1. 计算重叠像素数
        (
            overlap_width,
            overlap_height,
            overlap_mode
        ) = cls.calculate_overlap_pixels(
            overlap_amount, image_width, image_height
        )

        # 2. 计算基础tile尺寸（不含重叠）
        base_tile_width = image_width // cols
        base_tile_height = image_height // rows

        # 3. 计算实际tile尺寸（含重叠）
        actual_tile_width = base_tile_width + overlap_width
        actual_tile_height = base_tile_height + overlap_height

        # 4. 确保尺寸是divisible_by的倍数
        actual_tile_width = (
            (actual_tile_width + divisible_by - 1) // divisible_by
        ) * divisible_by
        actual_tile_height = (
            (actual_tile_height + divisible_by - 1) // divisible_by
        ) * divisible_by

        # 5. 重新计算实际的重叠像素数（考虑divisible_by调整）
        final_overlap_width = actual_tile_width - base_tile_width
        final_overlap_height = actual_tile_height - base_tile_height

        return (
            actual_tile_width,
            actual_tile_height,
            final_overlap_width,
            final_overlap_height,
            overlap_mode
        )

    @classmethod
    def calculate_best_tile_size(
        cls, image_width, image_height, tile_preset_size
    ):
        """根据用户偏好选择tile尺寸"""
        # 注意：此方法主要服务于 Preset 模式
        # 这里的 tile_preset_size 应该是 PRESET_RESOLUTIONS 中的某一项名称
        
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
        
        # 如果没找到匹配项（理论上不应发生，除非直接传了错误的字符串）
        # 返回默认值
        return (512, 512), (2, 2), {}

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
    def calculate_grid_layout(
        cls,
        image_width,
        image_height,
        tile_width,
        tile_height,
        overlap_width,
        overlap_height
    ):
        """计算网格布局，确保完美覆盖"""
        # 计算需要的tile数量
        effective_tile_width = tile_width - overlap_width
        effective_tile_height = tile_height - overlap_height

        num_cols = max(
            1,
            math.ceil((image_width - overlap_width) / effective_tile_width)
        )
        num_rows = max(
            1,
            math.ceil((image_height - overlap_height) / effective_tile_height)
        )

        # 计算精确位置
        col_positions = cls.calculate_precise_positions(
            image_width, tile_width, overlap_width, num_cols
        )
        row_positions = cls.calculate_precise_positions(
            image_height, tile_height, overlap_height, num_rows
        )

        return num_cols, num_rows, col_positions, row_positions

    @classmethod
    async def tile_image(
        cls,
        image,
        tile_width,
        tile_height,
        overlap_width,
        overlap_height
    ):
        image_pil = cls.tensor2pil(image[0])
        img_width, img_height = image_pil.size
        (
            num_cols,
            num_rows,
            col_positions,
            row_positions
        ) = cls.calculate_grid_layout(
            img_width,
            img_height,
            tile_width,
            tile_height,
            overlap_width,
            overlap_height
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
                            pad = Image.new(
                                'RGB', (tile_width, tile_height), (0, 0, 0)
                            )
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

                tasks.append(
                    _one(
                        order,
                        left,
                        top,
                        right,
                        bottom,
                        actual_width,
                        actual_height,
                        col_idx,
                        row_idx
                    )
                )
                order += 1
        results = await asyncio.gather(*tasks)
        results.sort(key=lambda x: x[0])
        tiles = [t for _, t, _ in results]
        tile_metas = [m for _, _, m in results]
        return (
            tiles,
            tile_metas,
            (img_width, img_height),
            (num_cols, num_rows)
        )

    @staticmethod
    def pil2tensor(image: Image) -> torch.Tensor:
        return torch.from_numpy(
            np.array(image).astype(np.float32) / 255.0
        ).unsqueeze(0)

    @staticmethod
    def tensor2pil(t_image: torch.Tensor) -> Image:
        arr = t_image.detach().cpu().numpy()
        if arr.ndim == 4:
            arr = arr[0]
        return Image.fromarray(np.clip(255.0 * arr, 0, 255).astype(np.uint8))
