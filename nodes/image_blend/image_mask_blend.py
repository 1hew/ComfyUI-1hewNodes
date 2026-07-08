import asyncio
from comfy_api.latest import io
import numpy as np
import torch
from PIL import Image, ImageColor, ImageFilter, ImageOps
from scipy import ndimage

class ImageMaskBlend(io.ComfyNode):
    """
    遮罩混合 - 支持批量处理图像与遮罩，增强填充与扩展控制
    
    批量处理逻辑：
    - 当图像和遮罩数量不同时，按最大数量输出，较少的批次会循环复制
    - 例如：5张图片 + 2个遮罩 = 输出5张处理结果（遮罩按[1,2,1,2,1]循环使用）
    - 例如：2张图片 + 5个遮罩 = 输出5张处理结果（图片按[1,2,1,2,1]循环使用）
    
    支持多种颜色格式：灰度值、HEX、RGB、颜色名称、edge（边缘色）、average（平均色）
    
    可选参数：fill_hole、invert、feather、expansion、opacity、background_color、background_opacity
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_ImageMaskBlend",
            display_name="Image Mask Blend",
            category="1hewNodes/image/blend",
            inputs=[
                io.Image.Input("image"),
                io.Mask.Input("mask", optional=True),
                io.Boolean.Input("fill_hole", default=True),
                io.Boolean.Input("invert", default=False),
                io.Int.Input("feather", default=0, min=0, max=50, step=1),
                io.Float.Input("opacity", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Int.Input("expansion", default=0, min=-100, max=100, step=1),
                io.String.Input("background_color", default="1.0"),
                io.Float.Input("background_opacity", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Boolean.Input("output_mask_invert", default=False),
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
        fill_hole: bool,
        invert: bool,
        feather: int,
        opacity: float,
        expansion: int,
        background_color: str,
        background_opacity: float,
        output_mask_invert: bool,
        mask: torch.Tensor | None = None,
    ) -> io.NodeOutput:
        """
        AE 对齐的遮罩融合（无 alpha 分支，输出 image 与 mask）：
        - fill_hole：填补遮罩孔洞，确保连续形状
        - expansion：扩展/收缩遮罩（像素）
        - feather：高斯羽化（像素半径）
        - invert：形态/羽化完成后反转选择区域
        - opacity：遮罩不透明度（0–1），缩放遮罩强度
        - background_color：非遮罩区域底色；opacity<1 时遮罩区会按 (1-mask*opacity) 透出底色
        - background_opacity：底色与原图在非遮罩区域的混合强度（0–1），1 为完全底色，0.5 为半透明底色叠加到原图
        - output_mask_invert：仅在输出端反转遮罩（不影响融合）

        顺序：尺寸对齐 → 填孔 → 扩展/收缩 → 羽化 → 反转 → 应用 opacity
        输出：
        - image：final = image*mask + mixed_bg*(1-mask)
          其中 mixed_bg = (1-background_opacity)*image + background_opacity*background
        - mask：处理后的遮罩张量（0–1），mask = mask_gray*opacity
        """
        # 规范化输入数据
        image = torch.clamp(image, min=0.0, max=1.0).to(torch.float32)
        if mask is None:
            mask = torch.zeros(image.shape[:-1], dtype=torch.float32, device=image.device)
        mask = torch.clamp(mask, min=0.0, max=1.0).to(torch.float32)
        device = image.device

        image_batch_size = int(image.shape[0])
        mask_batch_size = int(mask.shape[0])
        max_batch_size = max(image_batch_size, mask_batch_size)

        async def _proc(b):
            def _do():
                image_index = b % image_batch_size
                mask_index = b % mask_batch_size
                img_tensor = image[image_index]
                img_np = (img_tensor.detach().cpu().numpy() * 255).astype(np.uint8)
                if img_np.ndim == 3 and img_np.shape[2] == 4:
                    img_np = img_np[:, :, :3]
                img_pil = Image.fromarray(img_np)
                mask_tensor = mask[mask_index]
                mask_np = (mask_tensor.detach().cpu().numpy() * 255).astype(np.uint8)
                if mask_np.ndim == 3 and mask_np.shape[2] == 1:
                    mask_np = mask_np.squeeze(2)
                elif mask_np.ndim == 3 and mask_np.shape[2] >= 3:
                    mask_np = np.mean(mask_np[:, :, :3], axis=2).astype(np.uint8)
                mask_pil = Image.fromarray(mask_np).convert("L")
                if img_pil.size != mask_pil.size:
                    mask_pil = mask_pil.resize(img_pil.size, Image.Resampling.LANCZOS)
                if fill_hole:
                    mask_pil = cls._fill_hole_pil(mask_pil)
                if expansion != 0:
                    mask_pil = cls._morph_adjust(mask_pil, expansion)
                if feather > 0:
                    mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=feather))
                if invert:
                    mask_pil = ImageOps.invert(mask_pil)
                img_np_f = np.array(img_pil).astype(np.float32) / 255.0
                mask_f = np.array(mask_pil).astype(np.float32) / 255.0
                bg_spec = cls._parse_color_advanced(background_color, img_pil, mask_pil)
                if isinstance(bg_spec, str) and bg_spec == "extend":
                    bg_np_f = cls._make_mask_extend_background(img_np_f, mask_f)
                else:
                    if isinstance(bg_spec, str) and bg_spec == "edge":
                        bg_rgb01 = cls._get_mask_edge_color(img_np_f, mask_f)
                    else:
                        bg_rgb01 = bg_spec
                    bg_color = np.array(
                        [
                            max(0.0, min(1.0, float(bg_rgb01[0]))),
                            max(0.0, min(1.0, float(bg_rgb01[1]))),
                            max(0.0, min(1.0, float(bg_rgb01[2]))),
                        ],
                        dtype=np.float32,
                    )
                    bg_np_f = np.broadcast_to(bg_color, img_np_f.shape).copy()
                mask_f = np.clip(mask_f * float(opacity), 0.0, 1.0)
                non_mask = 1.0 - mask_f
                mask_3 = np.expand_dims(mask_f, axis=2)
                non_mask_3 = np.expand_dims(non_mask, axis=2)
                try:
                    bg_op = max(0.0, min(1.0, float(background_opacity)))
                except Exception:
                    bg_op = 1.0
                mixed_bg = (1.0 - bg_op) * img_np_f + bg_op * bg_np_f
                final_np = img_np_f * mask_3 + mixed_bg * non_mask_3
                result_np = final_np.astype(np.float32)
                img_t = torch.from_numpy(result_np)
                mask_out = 1.0 - mask_f if output_mask_invert else mask_f
                mask_t = torch.from_numpy(mask_out.astype(np.float32))
                return img_t, mask_t
            return await asyncio.to_thread(_do)

        tasks = [_proc(b) for b in range(max_batch_size)]
        results = await asyncio.gather(*tasks)
        output_images_tensor = torch.stack([r[0] for r in results]).to(device)
        output_masks_tensor = torch.stack([r[1] for r in results]).to(device)
        return io.NodeOutput(output_images_tensor, output_masks_tensor)

    @staticmethod
    def _fill_hole_pil(mask_pil):
        mask_array = np.array(mask_pil)
        binary_mask = mask_array > 127
        structure = ndimage.generate_binary_structure(2, 2)
        filled_mask = ndimage.binary_fill_holes(binary_mask, structure=structure)
        filled_array = (filled_mask * 255).astype(np.uint8)
        return Image.fromarray(filled_array, mode="L")

    @staticmethod
    def _morph_adjust(mask_pil, expansion):
        mask_array = np.array(mask_pil)
        binary_mask = mask_array > 127
        structure = ndimage.generate_binary_structure(2, 2)
        if expansion > 0:
            adjusted = ndimage.binary_dilation(binary_mask, structure=structure, iterations=expansion)
        else:
            adjusted = ndimage.binary_erosion(binary_mask, structure=structure, iterations=abs(expansion))
        adjusted_array = (adjusted * 255).astype(np.uint8)
        return Image.fromarray(adjusted_array, mode="L")
    
    @staticmethod
    def _parse_color_advanced(color_str, img_pil=None, mask_pil=None):
        if color_str is None:
            return (1.0, 1.0, 1.0)
        text = str(color_str).strip().lower()
        if text in ("edge", "e"):
            return "edge"
        if text in ("extend", "ex"):
            return "extend"
        if text in ("a", "average", "avg") and img_pil is not None:
            img_np = (
                np.array(img_pil.convert("RGB")).astype(np.float32) / 255.0
            )
            avg = img_np.mean(axis=(0, 1))
            return (float(avg[0]), float(avg[1]), float(avg[2]))
        if text in ("mk", "mask") and img_pil is not None and mask_pil is not None:
            img_np = (
                np.array(img_pil.convert("RGB")).astype(np.float32) / 255.0
            )
            m = np.array(mask_pil).astype(np.float32) / 255.0
            if m.shape[:2] != img_np.shape[:2]:
                m = (
                    np.array(
                        mask_pil.resize(
                            img_pil.size, Image.Resampling.LANCZOS
                        )
                    ).astype(np.float32)
                    / 255.0
                )
            c = float(m.sum())
            if c > 0.0:
                m3 = np.repeat(m[:, :, None], 3, axis=2)
                avg = (img_np * m3).sum(axis=(0, 1)) / c
            else:
                avg = img_np.mean(axis=(0, 1))
            return (float(avg[0]), float(avg[1]), float(avg[2]))
        if text.startswith("(") and text.endswith(")"):
            text = text[1:-1].strip()
        single = {
            "r": "red",
            "g": "lime",
            "b": "blue",
            "c": "cyan",
            "m": "magenta",
            "y": "yellow",
            "k": "black",
            "w": "white",
            "o": "orange",
            "p": "purple",
            "n": "brown",
            "s": "silver",
            "l": "lime",
            "i": "indigo",
            "v": "violet",
            "t": "turquoise",
            "f": "fuchsia",
            "h": "hotpink",
            "d": "darkblue",
        }
        if len(text) == 1 and text in single:
            text = single[text]
        try:
            v = float(text)
            if 0.0 <= v <= 1.0:
                return (v, v, v)
        except Exception:
            pass
        if "," in text:
            try:
                parts = [p.strip() for p in text.split(",")]
                if len(parts) >= 3:
                    r = float(parts[0])
                    g = float(parts[1])
                    b = float(parts[2])
                    if max(r, g, b) <= 1.0:
                        return (r, g, b)
                    return (r / 255.0, g / 255.0, b / 255.0)
            except Exception:
                pass
        if text.startswith("#") and len(text) in (4, 7):
            try:
                hex_str = text[1:]
                if len(hex_str) == 3:
                    hex_str = "".join(ch * 2 for ch in hex_str)
                r = int(hex_str[0:2], 16) / 255.0
                g = int(hex_str[2:4], 16) / 255.0
                b = int(hex_str[4:6], 16) / 255.0
                return (r, g, b)
            except Exception:
                pass
        try:
            rgb = ImageColor.getrgb(text)
            return (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)
        except Exception:
            return (1.0, 1.0, 1.0)

    @staticmethod
    def _get_mask_edge_color(img_np_f, mask_f):
        """获取遮罩内侧边缘的平均颜色，用于生成协调的纯色背景。"""
        binary_mask = mask_f > 0.5
        if not np.any(binary_mask):
            return tuple(float(x) for x in img_np_f.mean(axis=(0, 1)))
        structure = ndimage.generate_binary_structure(2, 2)
        eroded = ndimage.binary_erosion(binary_mask, structure=structure, border_value=0)
        edge = binary_mask & ~eroded
        if not np.any(edge):
            edge = binary_mask
        avg = img_np_f[edge].mean(axis=0)
        return (float(avg[0]), float(avg[1]), float(avg[2]))

    @staticmethod
    def _make_mask_extend_background(img_np_f, mask_f):
        """将遮罩内侧颜色按最近邻扩展到背景区域。"""
        binary_mask = mask_f > 0.5
        if not np.any(binary_mask) or np.all(binary_mask):
            return img_np_f.copy()
        _, indices = ndimage.distance_transform_edt(
            ~binary_mask,
            return_indices=True,
        )
        return img_np_f[indices[0], indices[1]].astype(np.float32)

