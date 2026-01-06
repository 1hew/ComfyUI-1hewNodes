import asyncio
import numpy as np
from comfy_api.latest import io
from PIL import Image
import torch
import torch.nn.functional as F

class ImageMaskCrop(io.ComfyNode):
    """
    图像遮罩裁剪（ImageMaskCrop）- 基于遮罩边界框进行裁剪，或保持原尺寸

    参数语义：
    - output_alpha：是否输出带 alpha 通道的图像（由 mask 提供），
      True 输出 RGBA，False 时输入为 RGBA 则保持 RGBA。
    - output_crop：是否根据 mask 的边界框进行裁剪，
      True 裁剪到 bbox，False 保持原输入尺寸。

    批处理规则：图像与遮罩数量不一致时按最大批次循环使用。
    输出与公式：
    - 当 output_alpha=True：alpha = mask（0–255），图像输出为 RGBA；
      当 output_alpha=False：输入为 RGBA 则保持 RGBA，输入为 RGB 则输出 RGB。
    - 当 output_crop=True：对图像与遮罩进行 bbox 裁剪后输出；
      当 output_crop=False：保持原尺寸，整幅图应用 alpha 或保持 RGB。
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_ImageMaskCrop",
            display_name="Image Mask Crop",
            category="1hewNodes/image/crop",
            inputs=[
                io.Image.Input("image"),
                io.Mask.Input("mask"),
                io.Boolean.Input("output_crop", default=True),
                io.Boolean.Input("output_alpha", default=False),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Mask.Output(display_name="mask"),
            ],
        )

    @classmethod
    async def execute(
        cls,
        image: torch.Tensor,
        mask: torch.Tensor,
        output_crop: bool = True,
        output_alpha: bool = False,
    ) -> io.NodeOutput:
        image = image.to(torch.float32).clamp(0.0, 1.0)
        mask = mask.to(torch.float32).clamp(0.0, 1.0)

        batch_size, height, width, channels = image.shape
        mask_batch_size = mask.shape[0]
        max_batch = max(batch_size, mask_batch_size)
        async def _proc(b):
            def _do():
                img_idx = b % batch_size
                mask_idx = b % mask_batch_size
                img_np = (
                    image[img_idx].detach().cpu().numpy() * 255
                ).astype(np.uint8)
                mask_np = (
                    mask[mask_idx].detach().cpu().numpy() * 255
                ).astype(np.uint8)
                img_pil = Image.fromarray(img_np)
                mask_pil = Image.fromarray(mask_np).convert("L")
                bbox = cls._get_bbox_from_mask(mask_pil)
                if bbox is None:
                    if output_alpha:
                        base_rgba = img_pil.convert("RGBA")
                        rgba_data = np.array(base_rgba)
                        full_mask = mask_pil
                        if full_mask.size != base_rgba.size:
                            new_mask = Image.new("L", base_rgba.size, 0)
                            paste_x = max(
                                0, (base_rgba.width - full_mask.width) // 2
                            )
                            paste_y = max(
                                0, (base_rgba.height - full_mask.height) // 2
                            )
                            new_mask.paste(full_mask, (paste_x, paste_y))
                            full_mask = new_mask
                        mask_data = np.array(full_mask)
                        rgba_data[:, :, 3] = mask_data
                        result_np = rgba_data.astype(np.float32) / 255.0
                        img_t = torch.from_numpy(result_np)
                    else:
                        rgb_np = np.array(img_pil)
                        if rgb_np.ndim == 2:
                            rgb_np = np.stack([rgb_np] * 3, axis=-1)
                        img_t = torch.from_numpy(
                            rgb_np.astype(np.float32) / 255.0
                        )
                    mk_t = torch.from_numpy(mask_np.astype(np.float32) / 255.0)
                    return img_t, mk_t
                x_min, y_min, x_max, y_max = bbox
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(img_pil.width, x_max)
                y_max = min(img_pil.height, y_max)
                if output_crop:
                    cropped_img = img_pil.crop((x_min, y_min, x_max, y_max))
                    cropped_mask = mask_pil.crop((x_min, y_min, x_max, y_max))
                    cropped_mask_np = (
                        np.array(cropped_mask).astype(np.float32) / 255.0
                    )
                    mk_t = torch.from_numpy(cropped_mask_np)
                    if output_alpha:
                        cropped_rgba = cropped_img.convert("RGBA")
                        img_data = np.array(cropped_rgba)
                        mask_data = np.array(cropped_mask)
                        img_data[:, :, 3] = mask_data
                        result_np = img_data.astype(np.float32) / 255.0
                        img_t = torch.from_numpy(result_np)
                    else:
                        cropped_np = np.array(cropped_img)
                        if cropped_np.ndim == 2:
                            cropped_np = np.stack([cropped_np] * 3, axis=-1)
                        img_t = torch.from_numpy(
                            cropped_np.astype(np.float32) / 255.0
                        )
                else:
                    full_mask = mask_pil
                    if full_mask.size != img_pil.size:
                        new_mask = Image.new("L", img_pil.size, 0)
                        paste_x = max(0, (img_pil.width - full_mask.width) // 2)
                        paste_y = max(0, (img_pil.height - full_mask.height) // 2)
                        new_mask.paste(full_mask, (paste_x, paste_y))
                        full_mask = new_mask
                    full_mask_np = (
                        np.array(full_mask).astype(np.float32) / 255.0
                    )
                    mk_t = torch.from_numpy(full_mask_np)
                    if output_alpha:
                        base_rgba = img_pil.convert("RGBA")
                        rgba_data = np.array(base_rgba)
                        mask_data = np.array(full_mask)
                        rgba_data[:, :, 3] = mask_data
                        result_np = rgba_data.astype(np.float32) / 255.0
                        img_t = torch.from_numpy(result_np)
                    else:
                        rgb_np = np.array(img_pil)
                        if rgb_np.ndim == 2:
                            rgb_np = np.stack([rgb_np] * 3, axis=-1)
                        img_t = torch.from_numpy(
                            rgb_np.astype(np.float32) / 255.0
                        )
                return img_t, mk_t
            return await asyncio.to_thread(_do)

        results = await asyncio.gather(*[_proc(b) for b in range(max_batch)])
        output_images = [r[0] for r in results]
        output_masks = [r[1] for r in results]

        device = image.device
        if output_images and output_masks:
            image_sizes = [img.shape for img in output_images]
            if len(set(image_sizes)) == 1:
                output_image_tensor = torch.stack(output_images)
            else:
                output_images = cls._pad_to_same_size(output_images)
                output_image_tensor = torch.stack(output_images)

            mask_sizes = [m.shape for m in output_masks]
            if len(set(mask_sizes)) == 1:
                output_mask_tensor = torch.stack(output_masks)
            else:
                output_masks = cls._pad_masks_to_same_size(output_masks)
                output_mask_tensor = torch.stack(output_masks)

            output_image_tensor = (
                output_image_tensor.to(device).to(torch.float32).clamp(0.0, 1.0)
            )
            output_mask_tensor = (
                output_mask_tensor.to(device).to(torch.float32).clamp(0.0, 1.0)
            )
            return io.NodeOutput(output_image_tensor, output_mask_tensor)
        return io.NodeOutput(image, mask)
    
    @staticmethod
    def _get_bbox_from_mask(mask_pil):
        mask_np = np.array(mask_pil)
        rows = np.any(mask_np > 10, axis=1)
        cols = np.any(mask_np > 10, axis=0)
        if not np.any(rows) or not np.any(cols):
            return None
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        return (x_min, y_min, x_max + 1, y_max + 1)
    
    @staticmethod
    def _pad_to_same_size(images):
        max_height = max(img.shape[0] for img in images)
        max_width = max(img.shape[1] for img in images)
        max_channels = max(img.shape[2] for img in images)
        padded_images = []
        for img in images:
            h, w, c = img.shape
            pad_h = max_height - h
            pad_w = max_width - w
            pad_c = max_channels - c
            padded_spatial = F.pad(img, (0, 0, 0, pad_w, 0, pad_h), value=0)
            if pad_c <= 0:
                padded_images.append(padded_spatial)
                continue

            if max_channels == 4 and c == 3 and pad_c == 1:
                extra = torch.ones(
                    (max_height, max_width, 1),
                    dtype=padded_spatial.dtype,
                    device=padded_spatial.device,
                )
            else:
                extra = torch.zeros(
                    (max_height, max_width, pad_c),
                    dtype=padded_spatial.dtype,
                    device=padded_spatial.device,
                )

            padded_images.append(torch.cat([padded_spatial, extra], dim=2))
        return padded_images
    
    @staticmethod
    def _pad_masks_to_same_size(masks):
        max_height = max(m.shape[0] for m in masks)
        max_width = max(m.shape[1] for m in masks)
        padded_masks = []
        for m in masks:
            h, w = m.shape
            pad_h = max_height - h
            pad_w = max_width - w
            padded_mask = F.pad(m, (0, pad_w, 0, pad_h), value=0)
            padded_masks.append(padded_mask)
        return padded_masks
