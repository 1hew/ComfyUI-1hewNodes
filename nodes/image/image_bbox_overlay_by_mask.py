from comfy_api.latest import io, ui
import asyncio
import numpy as np
import os
import torch
from PIL import Image, ImageDraw
from skimage.measure import label, regionprops


class ImageBBoxOverlayByMask(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_ImageBBoxOverlayByMask",
            display_name="Image BBox Overlay by Mask",
            category="1hewNodes/image",
            inputs=[
                io.Image.Input("image"),
                io.Mask.Input("mask"),
                io.Combo.Input("bbox_color", options=["red", "green", "blue", "yellow", "cyan", "magenta", "white", "black"], default="green"),
                io.Int.Input("stroke_width", default=4, min=1, max=100, step=1),
                io.Boolean.Input("fill", default=True),
                io.Int.Input("padding", default=0, min=0, max=1000, step=1),
                io.Combo.Input("output_mode", options=["separate", "merge"], default="separate"),
            ],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    async def execute(
        cls,
        image: torch.Tensor,
        mask: torch.Tensor,
        bbox_color: str,
        stroke_width: int,
        fill: bool,
        padding: int,
        output_mode: str,
    ) -> io.NodeOutput:
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        if mask.dim() == 4 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)

        bs = int(image.shape[0])
        ms = int(mask.shape[0])
        if ms == 1 and bs > 1:
            mask = mask.repeat(bs, 1, 1)
        elif ms != bs:
            if ms > bs:
                mask = mask[:bs]
            else:
                reps = bs // ms + (1 if bs % ms else 0)
                mask = mask.repeat(reps, 1, 1)[:bs]

        color_map = {
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "yellow": (255, 255, 0),
            "cyan": (0, 255, 255),
            "magenta": (255, 0, 255),
            "white": (255, 255, 255),
            "black": (0, 0, 0),
        }
        bbox_rgb = color_map.get(bbox_color, (0, 255, 0))

        def _process_one(b):
            img_np = (image[b].detach().cpu().numpy() * 255.0).astype(np.uint8)
            img_pil = Image.fromarray(img_np, "RGB")
            draw = ImageDraw.Draw(img_pil)

            m_np = (mask[b].detach().cpu().numpy() * 255.0).astype(np.uint8)
            m_pil = Image.fromarray(m_np, "L")
            if m_pil.size != img_pil.size:
                m_pil = m_pil.resize(img_pil.size, Image.LANCZOS)

            if output_mode == "merge":
                bbox = cls.get_single_bbox_from_mask(m_pil, padding)
                if bbox is not None:
                    x_min, y_min, x_max, y_max = bbox
                    cls.draw_bbox(draw, x_min, y_min, x_max, y_max, bbox_rgb,
                                  stroke_width, fill)
            else:
                bboxes = cls.get_multiple_bboxes_from_mask(m_pil, padding)
                for x_min, y_min, x_max, y_max in bboxes:
                    cls.draw_bbox(draw, x_min, y_min, x_max, y_max, bbox_rgb,
                                  stroke_width, fill)

            out_np = np.array(img_pil).astype(np.float32) / 255.0
            return torch.from_numpy(out_np)

        tasks = [asyncio.to_thread(_process_one, b) for b in range(bs)]
        out_imgs = await asyncio.gather(*tasks)

        out_t = torch.stack(out_imgs, dim=0).to(torch.float32).clamp(0.0, 1.0)
        out_t = out_t.to(image.device)
        return io.NodeOutput(out_t, ui=ui.PreviewImage(out_t, cls=cls))
    
    @staticmethod
    def draw_bbox(draw, x_min, y_min, x_max, y_max, color, stroke_width, fill):
        if fill:
            draw.rectangle([x_min, y_min, x_max, y_max], fill=color)
        else:
            draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=stroke_width)
    
    @staticmethod
    def get_single_bbox_from_mask(mask_pil, padding=0):
        mask_np = np.array(mask_pil)
        binary_mask = mask_np > 128
        y_coords, x_coords = np.where(binary_mask)
        if len(y_coords) == 0 or len(x_coords) == 0:
            return None
        x_min = int(np.min(x_coords))
        x_max = int(np.max(x_coords))
        y_min = int(np.min(y_coords))
        y_max = int(np.max(y_coords))
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(mask_pil.width - 1, x_max + padding)
        y_max = min(mask_pil.height - 1, y_max + padding)
        return (x_min, y_min, x_max, y_max)
    
    @staticmethod
    def get_multiple_bboxes_from_mask(mask_pil, padding=0):
        mask_np = np.array(mask_pil)
        binary_mask = mask_np > 128
        if not np.any(binary_mask):
            return []
        labeled_mask = label(binary_mask)
        regions = regionprops(labeled_mask)
        bboxes = []
        for region in regions:
            min_row, min_col, max_row, max_col = region.bbox
            x_min = max(0, min_col - padding)
            y_min = max(0, min_row - padding)
            x_max = min(mask_pil.width - 1, max_col - 1 + padding)
            y_max = min(mask_pil.height - 1, max_row - 1 + padding)
            bboxes.append((x_min, y_min, x_max, y_max))
        return bboxes

    @classmethod
    def validate_inputs(
        cls,
        image: torch.Tensor,
        mask: torch.Tensor,
        bbox_color: str,
        stroke_width: int,
        fill: bool,
        padding: int,
        output_mode: str,
    ):
        if bbox_color not in {"red", "green", "blue", "yellow", "cyan", "magenta", "white", "black"}:
            return "invalid bbox_color"
        if output_mode not in {"separate", "merge"}:
            return "invalid output_mode"
        if stroke_width < 1 or padding < 0:
            return "invalid stroke_width or padding"
        return True

    @classmethod
    def fingerprint_inputs(
        cls,
        image: torch.Tensor,
        mask: torch.Tensor,
        bbox_color: str,
        stroke_width: int,
        fill: bool,
        padding: int,
        output_mode: str,
    ):
        b = int(image.shape[0]) if isinstance(image, torch.Tensor) else 0
        h = int(image.shape[1]) if isinstance(image, torch.Tensor) else 0
        w = int(image.shape[2]) if isinstance(image, torch.Tensor) else 0
        mb = int(mask.shape[0]) if isinstance(mask, torch.Tensor) else 0
        return f"{b}x{h}x{w}|mb={mb}|{bbox_color}|{stroke_width}|{fill}|{padding}|{output_mode}"


