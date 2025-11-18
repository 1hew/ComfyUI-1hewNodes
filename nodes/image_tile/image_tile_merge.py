import asyncio
from comfy_api.latest import io, ui
import numpy as np
from PIL import Image
import torch
from typing import Any


class ImageTileMerge(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_ImageTileMerge",
            display_name="Image Tile Merge",
            category="1hewNodes/image/tile",
            inputs=[
                io.Image.Input("tile"),
                io.Custom("DICT").Input("tile_meta"),
                io.Float.Input("blend_strength", default=1.0, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    async def execute(
        cls,
        tile: torch.Tensor,
        tile_meta: dict[str, Any],
        blend_strength: float,
    ) -> io.NodeOutput:
        metas: list[dict[str, Any]] = tile_meta["tile_metas"]
        orig_w, orig_h = tile_meta["original_size"]
        num_cols, num_rows = tile_meta["grid_size"]
        ov_w = int(tile_meta.get("overlap_width", 0))
        ov_h = int(tile_meta.get("overlap_height", 0))
        tile_w = int(tile_meta.get("tile_width", 512))
        tile_h = int(tile_meta.get("tile_height", 512))

        expected = int(num_rows) * int(num_cols)
        tiles_list: list[torch.Tensor] = []
        if hasattr(tile, "shape") and len(tile.shape) == 4:
            for i in range(tile.shape[0]):
                tiles_list.append(tile[i])
        elif hasattr(tile, "shape") and len(tile.shape) == 3:
            tiles_list.append(tile)
        else:
            tiles_list.append(tile)

        if len(tiles_list) > expected:
            tiles_list = tiles_list[:expected]
        while len(tiles_list) < expected:
            if tiles_list:
                tiles_list.append(tiles_list[-1])
            else:
                tiles_list.append(
                    torch.zeros((tile_h, tile_w, 3), dtype=torch.float32)
                )

        final_img = np.zeros((orig_h, orig_w, 3), dtype=np.float64)
        weight_map = np.zeros((orig_h, orig_w), dtype=np.float64)

        async def _prepare(idx: int, t: torch.Tensor):
            def _do():
                arr = t.detach().cpu().numpy().astype(np.float64)
                meta = metas[idx]
                left, top, right, bottom = meta["crop_region"]
                pos = meta["position"]
                act_w, act_h = meta.get("actual_crop_size", (tile_w, tile_h))
                arr = arr[:act_h, :act_w, :]
                wmask = cls._create_weight_mask(
                    act_w,
                    act_h,
                    ov_w,
                    ov_h,
                    pos,
                    (num_cols, num_rows),
                    blend_strength,
                )
                if wmask.shape != (act_h, act_w):
                    wmask = wmask[:act_h, :act_w]
                return idx, arr, wmask, left, top, right, bottom
            return await asyncio.to_thread(_do)

        tasks = []
        for i, t in enumerate(tiles_list):
            if i >= len(metas):
                break
            tasks.append(_prepare(i, t))
        results = await asyncio.gather(*tasks)

        for _, arr, wmask, left, top, right, bottom in results:
            final_img[top:bottom, left:right] += arr * wmask[:, :, None]
            weight_map[top:bottom, left:right] += wmask

        eps = 1e-8
        weight_map = np.maximum(weight_map, eps)
        final_img = final_img / weight_map[:, :, None]
        final_img = np.clip(final_img, 0.0, 1.0)

        out = torch.from_numpy(final_img.astype(np.float32)).unsqueeze(0)
        return io.NodeOutput(out)

    @staticmethod
    def _create_weight_mask(
        tile_w: int,
        tile_h: int,
        overlap_w: int,
        overlap_h: int,
        position: tuple[int, int],
        grid_size: tuple[int, int],
        blend_strength: float,
    ) -> np.ndarray:
        col, row = position
        num_cols, num_rows = grid_size

        weight = np.ones((tile_h, tile_w), dtype=np.float64)
        if blend_strength <= 0 or (overlap_w <= 0 and overlap_h <= 0):
            return weight

        b_w = max(1, int(overlap_w * blend_strength)) if overlap_w > 0 else 0
        b_h = max(1, int(overlap_h * blend_strength)) if overlap_h > 0 else 0

        def smooth(x: int, length: int) -> float:
            return 0.5 * (1.0 + np.cos(np.pi * (1.0 - x / float(length))))

        if col > 0 and b_w > 0:
            fw = min(b_w, tile_w)
            for x in range(fw):
                a = smooth(x + 1, fw)
                weight[:, x] *= a

        if col < num_cols - 1 and b_w > 0:
            fw = min(b_w, tile_w)
            start_x = max(0, tile_w - fw)
            for x in range(start_x, tile_w):
                a = smooth(tile_w - x, fw)
                weight[:, x] *= a

        if row > 0 and b_h > 0:
            fh = min(b_h, tile_h)
            for y in range(fh):
                a = smooth(y + 1, fh)
                weight[y, :] *= a

        if row < num_rows - 1 and b_h > 0:
            fh = min(b_h, tile_h)
            start_y = max(0, tile_h - fh)
            for y in range(start_y, tile_h):
                a = smooth(tile_h - y, fh)
                weight[y, :] *= a

        return weight

    @staticmethod
    def pil2tensor(image: Image) -> torch.Tensor:
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

    @staticmethod
    def tensor2pil(t_image: torch.Tensor) -> Image:
        arr = t_image.detach().cpu().numpy()
        if arr.ndim == 4:
            arr = arr[0]
        return Image.fromarray(np.clip(255.0 * arr, 0, 255).astype(np.uint8))
