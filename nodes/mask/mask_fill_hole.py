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
    def _mask_without_border(mask_array):
        """Treat the outermost 1px as background."""
        cleared = mask_array.copy()
        cleared[0, :] = 0
        cleared[-1, :] = 0
        cleared[:, 0] = 0
        cleared[:, -1] = 0
        return cleared

    @staticmethod
    def _is_border_frame(mask_array):
        """True when the outermost 1px is a closed frame around the canvas.

        A full 1px frame is an export artifact, unlike real content that only
        touches part of an edge; both nodes ignore the artifact but keep
        genuine border content.
        """
        height, width = mask_array.shape
        if height < 3 or width < 3:
            return False
        border = np.concatenate(
            [
                mask_array[0, :],
                mask_array[-1, :],
                mask_array[1:-1, 0],
                mask_array[1:-1, -1],
            ]
        )
        if np.count_nonzero(border) < 0.98 * border.size:
            return False
        inner = np.concatenate(
            [
                mask_array[1, :],
                mask_array[-2, :],
                mask_array[1:-1, 1],
                mask_array[1:-1, -2],
            ]
        )
        return np.count_nonzero(inner) < 0.5 * inner.size

    @classmethod
    def _mask_ignoring_frame(cls, mask_array):
        """Drop a detected 1px frame; leave every other border pixel alone."""
        if cls._is_border_frame(mask_array):
            return cls._mask_without_border(mask_array)
        return mask_array.copy()

    @staticmethod
    def _fill_holes_internal(mask_pil, invert_mask, threshold=0.2):
        mask_array = np.array(mask_pil)
        cutoff = min(1.0, max(0.0, float(threshold))) * 255.0
        binary_mask = mask_array > cutoff
        fill_input = MaskFillHole._mask_ignoring_frame(binary_mask)
        structure = ndimage.generate_binary_structure(2, 2)
        filled_mask = ndimage.binary_fill_holes(fill_input, structure=structure)
        filled_mask |= binary_mask
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

