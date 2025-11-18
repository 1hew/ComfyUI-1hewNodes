import asyncio
import os
from comfy_api.latest import io
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from server import PromptServer


class MaskCropByBBoxMask(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_MaskCropByBBoxMask",
            display_name="Mask Crop by BBox Mask",
            category="1hewNodes/mask",
            inputs=[
                io.Mask.Input("mask"),
                io.Mask.Input("bbox_mask"),
            ],
            outputs=[io.Mask.Output(display_name="cropped_mask")],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    async def execute(cls, mask, bbox_mask, **kwargs):
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        if bbox_mask.dim() == 2:
            bbox_mask = bbox_mask.unsqueeze(0)

        batch_size = mask.shape[0]
        bbox_batch_size = bbox_mask.shape[0]
        concurrency = max(1, min(batch_size, os.cpu_count() or 1))
        sem = asyncio.Semaphore(concurrency)
        tasks = []
        uid = kwargs.get("unique_id")
        try:
            PromptServer.instance.send_sync("1hew.mask.crop.progress", {"node": uid, "progress": 0.0})
        except Exception:
            pass
        for b in range(batch_size):
            async def run_one(idx=b):
                async with sem:
                    bbox_idx = idx % bbox_batch_size
                    return await asyncio.to_thread(
                        cls._crop_one,
                        mask[idx],
                        bbox_mask[bbox_idx],
                    )
            tasks.append(run_one())
        output_masks = []
        done = 0
        for fut in asyncio.as_completed(tasks):
            res = await fut
            output_masks.append(res)
            done += 1
            try:
                PromptServer.instance.send_sync("1hew.mask.crop.progress", {"node": uid, "progress": done / batch_size})
            except Exception:
                pass
        try:
            PromptServer.instance.send_sync("1hew.mask.crop.progress", {"node": uid, "progress": 1.0})
        except Exception:
            pass

        if output_masks:
            sizes = [m.shape for m in output_masks]
            if len(set(sizes)) == 1:
                output_mask_tensor = torch.stack(output_masks)
            else:
                output_masks = cls.pad_to_same_size(output_masks)
                output_mask_tensor = torch.stack(output_masks)
            return io.NodeOutput(output_mask_tensor)
        return io.NodeOutput(mask)

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
    def pad_to_same_size(masks):
        max_height = max(m.shape[0] for m in masks)
        max_width = max(m.shape[1] for m in masks)
        padded_masks = []
        for m in masks:
            h, w = m.shape
            pad_h = max_height - h
            pad_w = max_width - w
            padded_masks.append(F.pad(m, (0, pad_w, 0, pad_h), value=0))
        return padded_masks

    @classmethod
    def _crop_one(cls, current_mask, current_bbox_mask):
        mask_np = (current_mask.cpu().numpy() * 255).astype(np.uint8)
        bbox_np = (current_bbox_mask.cpu().numpy() * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_np).convert("L")
        bbox_pil = Image.fromarray(bbox_np).convert("L")
        bbox = cls.get_bbox_from_mask(bbox_pil)
        if bbox is None:
            return current_mask
        x_min, y_min, x_max, y_max = bbox
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(mask_pil.width, x_max)
        y_max = min(mask_pil.height, y_max)
        cropped_mask = mask_pil.crop((x_min, y_min, x_max, y_max))
        cropped_mask_np = np.array(cropped_mask).astype(np.float32) / 255.0
        return torch.from_numpy(cropped_mask_np)

