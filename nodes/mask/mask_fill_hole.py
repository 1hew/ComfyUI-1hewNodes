import asyncio
import os
from comfy_api.latest import io
import numpy as np
from PIL import Image
from scipy import ndimage
import torch


class MaskFillHole(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_MaskFillHole",
            display_name="Mask Fill Hole",
            category="1hewNodes/mask",
            inputs=[
                io.Mask.Input("mask"),
                io.Boolean.Input("invert_mask", default=False),
                io.Float.Input(
                    "threshold",
                    default=0.2,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                ),
            ],
            outputs=[io.Mask.Output(display_name="mask")],
        )

    @classmethod
    async def execute(cls, mask, invert_mask, threshold=0.2):
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        threshold = min(1.0, max(0.0, float(threshold)))
        batch_size = mask.shape[0]
        concurrency = max(1, min(batch_size, os.cpu_count() or 1))
        sem = asyncio.Semaphore(concurrency)
        tasks = []
        for b in range(batch_size):
            async def run_one(idx=b):
                async with sem:
                    return await asyncio.to_thread(
                        cls._fill_one,
                        mask[idx],
                        invert_mask,
                        threshold,
                    )
            tasks.append(run_one())

        results = await asyncio.gather(*tasks)
        output_tensor = torch.stack(results)
        return io.NodeOutput(output_tensor)

    @staticmethod
    def _fill_holes_internal(mask_pil, invert_mask, threshold=0.2):
        mask_array = np.array(mask_pil)
        cutoff = min(1.0, max(0.0, float(threshold))) * 255.0
        binary_mask = mask_array > cutoff
        structure = ndimage.generate_binary_structure(2, 2)
        filled_mask = ndimage.binary_fill_holes(binary_mask, structure=structure)
        if invert_mask:
            filled_mask = ~filled_mask
        filled_array = (filled_mask * 255).astype(np.uint8)
        return Image.fromarray(filled_array, mode="L")

    @classmethod
    def _fill_one(cls, current_mask, invert_mask, threshold=0.2):
        mask_np = (current_mask.cpu().numpy() * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_np, mode="L")
        filled_mask = cls._fill_holes_internal(mask_pil, invert_mask, threshold)
        filled_np = np.array(filled_mask).astype(np.float32) / 255.0
        return torch.from_numpy(filled_np)

