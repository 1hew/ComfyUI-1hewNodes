import asyncio
from comfy_api.latest import io
import numpy as np
from PIL import Image
import torch


class ImagePasteByBBoxMask(io.ComfyNode):
    """
    图像遮罩粘贴器 - 将处理后的裁剪图像根据边界框遮罩粘贴回原始图像的位置
    """
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_ImagePasteByBBoxMask",
            display_name="Image Paste By BBox Mask",
            category="1hewNodes/image/crop",
            inputs=[
                io.Image.Input("paste_image"),
                io.Image.Input("base_image"),
                io.Mask.Input("bbox_mask"),
                io.Mask.Input("paste_mask", optional=True),
                io.Int.Input("position_x", default=0, min=-4096, max=4096),
                io.Int.Input("position_y", default=0, min=-4096, max=4096),
                io.Float.Input("scale", default=1.0, min=0.1, max=10.0),
                io.Float.Input("rotation", default=0.0, min=-3600.0, max=3600.0),
                io.Float.Input("opacity", default=1.0, min=0.0, max=1.0),
                io.Boolean.Input("apply_paste_mask", default=False),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Mask.Output(display_name="mask"),
            ],
        )

    @classmethod
    async def execute(
        cls,
        paste_image: torch.Tensor,
        base_image: torch.Tensor,
        bbox_mask: torch.Tensor,
        paste_mask: torch.Tensor | None = None,
        position_x: int = 0,
        position_y: int = 0,
        scale: float = 1.0,
        rotation: float = 0.0,
        opacity: float = 1.0,
        apply_paste_mask: bool = False,
    ) -> io.NodeOutput:
        paste_image = paste_image.to(torch.float32).clamp(0.0, 1.0)
        base_image = base_image.to(torch.float32).clamp(0.0, 1.0)
        bbox_mask = bbox_mask.to(torch.float32).clamp(0.0, 1.0)
        if paste_mask is not None:
            paste_mask = paste_mask.to(torch.float32).clamp(0.0, 1.0)

        base_bs = base_image.shape[0]
        paste_bs = paste_image.shape[0]
        bbox_bs = bbox_mask.shape[0]
        mask_bs = paste_mask.shape[0] if paste_mask is not None else 1
        max_bs = max(base_bs, paste_bs, bbox_bs, mask_bs)

        async def _proc(b):
            def _do():
                base_idx = b % base_bs
                paste_idx = b % paste_bs
                bbox_idx = b % bbox_bs
                mask_idx = b % mask_bs if paste_mask is not None else 0
                base_np = (
                    base_image[base_idx].detach().cpu().numpy() * 255
                ).astype(np.uint8)
                paste_np = (
                    paste_image[paste_idx].detach().cpu().numpy() * 255
                ).astype(np.uint8)
                bbox_np = (
                    bbox_mask[bbox_idx].detach().cpu().numpy() * 255
                ).astype(np.uint8)
                base_pil = Image.fromarray(base_np)
                bbox_pil = Image.fromarray(bbox_np).convert("L")
                if paste_np.shape[2] == 4:
                    paste_pil = Image.fromarray(paste_np, "RGBA")
                else:
                    paste_pil = Image.fromarray(paste_np)
                bbox = cls.get_bbox_from_mask(bbox_pil)
                if bbox is None:
                    empty_mask = np.zeros((base_np.shape[0], base_np.shape[1]), dtype=np.float32)
                    return base_image[base_idx], torch.from_numpy(empty_mask)
                m_pil = None
                if paste_mask is not None:
                    pm_np = (
                        paste_mask[mask_idx].detach().cpu().numpy() * 255
                    ).astype(np.uint8)
                    m_pil = Image.fromarray(pm_np).convert("L")
                elif paste_pil.mode == "RGBA":
                    m_pil = paste_pil.split()[-1]
                result_pil, result_mask_pil = cls.paste_image_with_transform(
                    base_pil,
                    paste_pil,
                    bbox,
                    position_x,
                    position_y,
                    scale,
                    rotation,
                    m_pil,
                    opacity,
                    apply_paste_mask,
                )
                result_np = (
                    np.array(result_pil).astype(np.float32) / 255.0
                )
                mask_np = (
                    np.array(result_mask_pil).astype(np.float32) / 255.0
                )
                return torch.from_numpy(result_np), torch.from_numpy(mask_np)
            return await asyncio.to_thread(_do)

        tasks = [_proc(b) for b in range(max_bs)]
        results = await asyncio.gather(*tasks)
        out_images = [r[0] for r in results]
        out_masks = [r[1] for r in results]

        device = base_image.device
        images_t = torch.stack(out_images).to(device).to(torch.float32)
        masks_t = torch.stack(out_masks).to(device).to(torch.float32)
        images_t = images_t.clamp(0.0, 1.0)
        masks_t = masks_t.clamp(0.0, 1.0)
        return io.NodeOutput(images_t, masks_t)
    
    @staticmethod
    def get_bbox_from_mask(mask_pil: Image.Image):
        mask_np = np.array(mask_pil)
        rows = np.any(mask_np > 10, axis=1)
        cols = np.any(mask_np > 10, axis=0)
        if not np.any(rows) or not np.any(cols):
            return None
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        return (x_min, y_min, x_max + 1, y_max + 1)
    
    @staticmethod
    def paste_image_with_transform(
        base_pil: Image.Image,
        paste_pil: Image.Image,
        bbox: tuple[int, int, int, int],
        position_x: int,
        position_y: int,
        scale: float,
        rotation: float,
        mask_pil: Image.Image | None,
        opacity: float,
        apply_paste_mask: bool,
    ):
        x_min, y_min, x_max, y_max = bbox
        bbox_w = x_max - x_min
        bbox_h = y_max - y_min
        cx = x_min + bbox_w // 2
        cy = y_min + bbox_h // 2

        if apply_paste_mask and mask_pil is not None:
            if paste_pil.size != mask_pil.size:
                mask_pil = mask_pil.resize(
                    paste_pil.size, Image.Resampling.LANCZOS
                )
            mask_bbox = mask_pil.getbbox()
            if mask_bbox is None:
                ref_w, ref_h = paste_pil.size
                eff_paste = paste_pil
                eff_mask = mask_pil
            else:
                eff_paste = paste_pil.crop(mask_bbox)
                eff_mask = mask_pil.crop(mask_bbox)
                ref_w, ref_h = eff_paste.size
        else:
            ref_w, ref_h = paste_pil.size
            eff_paste = paste_pil
            if mask_pil is not None:
                if paste_pil.size != mask_pil.size:
                    mask_pil = mask_pil.resize(
                        paste_pil.size, Image.Resampling.LANCZOS
                    )
                eff_mask = mask_pil
            else:
                eff_mask = Image.new("L", paste_pil.size, 255)

        ref_ratio = ref_w / max(ref_h, 1)
        bbox_ratio = bbox_w / max(bbox_h, 1)
        if ref_ratio > bbox_ratio:
            fitted_w = bbox_w
            fitted_h = int(bbox_w / max(ref_ratio, 1e-6))
        else:
            fitted_h = bbox_h
            fitted_w = int(bbox_h * ref_ratio)

        new_w = max(int(fitted_w * scale), 1)
        new_h = max(int(fitted_h * scale), 1)

        if eff_paste.size != (new_w, new_h):
            paste_pil = eff_paste.resize((new_w, new_h), Image.Resampling.LANCZOS)
            eff_mask = eff_mask.resize((new_w, new_h), Image.Resampling.LANCZOS)
        else:
            paste_pil = eff_paste
            eff_mask = eff_mask

        if rotation != 0.0:
            paste_pil, eff_mask = ImagePasteByBBoxMask.apply_rotation(
                paste_pil, rotation, eff_mask
            )
            new_w, new_h = paste_pil.size

        if opacity < 1.0:
            if paste_pil.mode != "RGBA":
                paste_pil = paste_pil.convert("RGBA")
            alpha = paste_pil.split()[-1]
            alpha = alpha.point(lambda p: int(p * opacity))
            paste_pil.putalpha(alpha)
            if eff_mask is not None:
                eff_mask = eff_mask.point(lambda p: int(p * opacity))
            else:
                eff_mask = alpha

        new_x = cx - new_w // 2 + position_x
        new_y = cy - new_h // 2 + position_y

        base_w, base_h = base_pil.size
        paste_x = max(0, new_x)
        paste_y = max(0, new_y)
        paste_x_end = min(base_w, new_x + new_w)
        paste_y_end = min(base_h, new_y + new_h)
        if paste_x >= paste_x_end or paste_y >= paste_y_end:
            return base_pil, Image.new("L", (base_w, base_h), 0)

        crop_x = paste_x - new_x
        crop_y = paste_y - new_y
        crop_x_end = crop_x + (paste_x_end - paste_x)
        crop_y_end = crop_y + (paste_y_end - paste_y)
        if crop_x > 0 or crop_y > 0 or crop_x_end < new_w or crop_y_end < new_h:
            paste_pil = paste_pil.crop((crop_x, crop_y, crop_x_end, crop_y_end))
            if eff_mask is not None:
                eff_mask = eff_mask.crop((crop_x, crop_y, crop_x_end, crop_y_end))

        result_pil = base_pil.copy()
        out_mask_pil = Image.new("L", (base_w, base_h), 0)
        if opacity < 1.0 or eff_mask is not None:
            if paste_pil.mode != "RGBA":
                paste_pil = paste_pil.convert("RGBA")
            result_pil.paste(paste_pil, (paste_x, paste_y), eff_mask)
            if eff_mask is not None:
                out_mask_pil.paste(eff_mask, (paste_x, paste_y))
            else:
                area_mask = Image.new("L", (paste_x_end - paste_x, paste_y_end - paste_y), 255)
                out_mask_pil.paste(area_mask, (paste_x, paste_y))
        else:
            result_pil.paste(paste_pil, (paste_x, paste_y))
            area_mask = Image.new("L", (paste_x_end - paste_x, paste_y_end - paste_y), 255)
            out_mask_pil.paste(area_mask, (paste_x, paste_y))
        return result_pil, out_mask_pil
    
    @staticmethod
    def apply_rotation(paste_pil: Image.Image, rotation_angle: float, mask_pil: Image.Image | None = None):
        actual_angle = -rotation_angle
        if paste_pil.mode == "RGBA":
            rotated_paste = paste_pil.rotate(actual_angle, expand=True, fillcolor=(0, 0, 0, 0))
        else:
            paste_rgba = paste_pil.convert("RGBA")
            alpha_mask = Image.new("L", paste_pil.size, 255)
            paste_rgba.putalpha(alpha_mask)
            rotated_paste = paste_rgba.rotate(actual_angle, expand=True, fillcolor=(0, 0, 0, 0))
        rotated_mask = None
        if mask_pil is not None:
            rotated_mask = mask_pil.rotate(actual_angle, expand=True, fillcolor=0)
        elif rotated_paste.mode == "RGBA":
            rotated_mask = rotated_paste.split()[-1]
        return rotated_paste, rotated_mask


