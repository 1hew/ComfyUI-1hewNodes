import asyncio
import os
from comfy_api.latest import io
import numpy as np
from PIL import Image
import torch


class MaskPasteByBBoxMask(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_MaskPasteByBBoxMask",
            display_name="Mask Paste by BBox Mask",
            category="1hewNodes/mask",
            inputs=[
                io.Mask.Input("paste_mask"),
                io.Mask.Input("bbox_mask"),
                io.Mask.Input("base_mask", optional=True),
            ],
            outputs=[io.Mask.Output(display_name="mask")],
        )

    @classmethod
    async def execute(cls, paste_mask, bbox_mask, base_mask=None):
        if base_mask is None:
            base_mask = torch.zeros_like(bbox_mask)

        base_batch_size = base_mask.shape[0]
        paste_batch_size = paste_mask.shape[0]
        bbox_batch_size = bbox_mask.shape[0]
        max_batch_size = max(base_batch_size, paste_batch_size, bbox_batch_size)
        concurrency = max(1, min(max_batch_size, os.cpu_count() or 1))
        sem = asyncio.Semaphore(concurrency)
        tasks = []
        for b in range(max_batch_size):
            async def run_one(idx=b):
                async with sem:
                    base_idx = idx % base_batch_size
                    paste_idx = idx % paste_batch_size
                    bbox_idx = idx % bbox_batch_size
                    return await asyncio.to_thread(
                        cls._paste_one,
                        base_mask[base_idx],
                        paste_mask[paste_idx],
                        bbox_mask[bbox_idx],
                    )
            tasks.append(run_one())

        output_masks = await asyncio.gather(*tasks)
        output_mask_tensor = torch.stack(output_masks)
        return io.NodeOutput(output_mask_tensor)

    @staticmethod
    def get_bbox_from_mask(mask_pil):
        mask_np = np.array(mask_pil)
        rows = np.any(mask_np > 10, axis=1)
        cols = np.any(mask_np > 10, axis=0)
        if not np.any(rows) or not np.any(cols):
            return None
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        return (x_min, y_min, x_max + 1, y_max + 1)

    @staticmethod
    def paste_mask_simple(base_pil, paste_pil, bbox):
        x_min, y_min, x_max, y_max = bbox
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        if paste_pil.size != (bbox_width, bbox_height):
            paste_pil = paste_pil.resize(
                (bbox_width, bbox_height), Image.Resampling.LANCZOS
            )
        result_pil = base_pil.copy()
        result_pil.paste(paste_pil, (x_min, y_min))
        return result_pil

    @classmethod
    def _paste_one(cls, base_item, paste_item, bbox_item):
        base_np = (base_item.cpu().numpy() * 255).astype(np.uint8)
        paste_np = (paste_item.cpu().numpy() * 255).astype(np.uint8)
        bbox_np = (bbox_item.cpu().numpy() * 255).astype(np.uint8)
        base_pil = Image.fromarray(base_np).convert("L")
        paste_pil = Image.fromarray(paste_np).convert("L")
        bbox_pil = Image.fromarray(bbox_np).convert("L")
        bbox = cls.get_bbox_from_mask(bbox_pil)
        if bbox is None:
            return base_item
        result_pil = cls.paste_mask_simple(base_pil, paste_pil, bbox)
        result_np = np.array(result_pil).astype(np.float32) / 255.0
        return torch.from_numpy(result_np)

