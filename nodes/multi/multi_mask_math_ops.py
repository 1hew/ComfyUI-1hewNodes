import asyncio
import os
from comfy_api.latest import io
import numpy as np
from PIL import Image
import torch


class MultiMaskMathOps(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_MaskMathOps",
            display_name="Multi Mask Math Ops",
            category="1hewNodes/multi",
            inputs=[
                io.Mask.Input("mask_1"),
                io.Mask.Input("mask_2", optional=True),
                io.Combo.Input( "operation", options=["or", "and", "subtract (a-b)", "subtract (b-a)", "xor"], default="or"),
            ],
            outputs=[io.Mask.Output(display_name="mask")],
        )

    @classmethod
    async def execute(cls, mask_1=None, mask_2=None, operation=None, **kwargs):
        masks = []
        if mask_1 is not None:
            masks.append(mask_1)
        if mask_2 is not None:
            masks.append(mask_2)
        ordered = []
        for k in kwargs.keys():
            if k.startswith("mask_"):
                suf = k[len("mask_") :]
                if suf.isdigit():
                    ordered.append((int(suf), k))
        ordered.sort(key=lambda x: x[0])
        for _, key in ordered:
            val = kwargs.get(key)
            if val is not None:
                masks.append(val)
        if not masks:
            return io.NodeOutput(torch.zeros((0, 64, 64)))
        batch_sizes = [m.shape[0] for m in masks]
        max_batch_size = max(batch_sizes)
        concurrency = max(1, min(max_batch_size, os.cpu_count() or 1))
        sem = asyncio.Semaphore(concurrency)
        tasks = []
        for b in range(max_batch_size):
            async def run_one(idx=b):
                async with sem:
                    items = [m[idx % s] for m, s in zip(masks, batch_sizes)]
                    return await asyncio.to_thread(
                        cls._math_many,
                        items,
                        operation,
                    )
            tasks.append(run_one())

        output_masks = await asyncio.gather(*tasks)
        output_tensor = torch.stack(output_masks)
        return io.NodeOutput(output_tensor)

    @classmethod
    def _math_many(cls, current_masks, operation):
        ref_np = (current_masks[0].cpu().numpy() * 255).astype(np.uint8)
        ref_pil = Image.fromarray(ref_np)
        arrays = []
        for m in current_masks:
            m_np = (m.cpu().numpy() * 255).astype(np.uint8)
            m_pil = Image.fromarray(m_np)
            if m_pil.size != ref_pil.size:
                m_pil = m_pil.resize(ref_pil.size, Image.Resampling.LANCZOS)
            arrays.append(np.array(m_pil).astype(np.float32) / 255.0)
        if operation == "and":
            result = arrays[0]
            for a in arrays[1:]:
                result = np.minimum(result, a)
        elif operation == "or":
            result = arrays[0]
            for a in arrays[1:]:
                result = np.maximum(result, a)
        elif operation == "subtract (a-b)":
            result = arrays[0]
            for a in arrays[1:]:
                result = np.clip(result - a, 0, 1)
        elif operation == "subtract (b-a)":
            if len(arrays) >= 2:
                result = np.clip(arrays[1] - arrays[0], 0, 1)
            else:
                result = arrays[0]
            for a in arrays[2:]:
                result = np.clip(result - a, 0, 1)
        elif operation == "xor":
            result = arrays[0]
            for a in arrays[1:]:
                result = np.abs(result - a)
        else:
            result = arrays[0]
        return torch.from_numpy(result)

