from comfy_api.latest import io, ui
from PIL import Image, ImageColor
import math
import numpy as np
import torch
import torch.nn.functional as F


class ImageEditStitch(io.ComfyNode):
    """
        图像编辑缝合 - 将参考图像和编辑图像拼接在一起，支持上下左右四种拼接方式
        优化版本：当match_edit_size为false时，保持reference_image的原始比例
    """
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_ImageEditStitch",
            display_name="Image Edit Stitch",
            category="1hewNodes/image",
            inputs=[
                io.Image.Input("reference_image"),
                io.Image.Input("edit_image"),
                io.Mask.Input("edit_mask", optional=True),
                io.Combo.Input("edit_image_position", options=["top", "bottom", "left", "right"], default="right"),
                io.Boolean.Input("match_edit_size", default=False),
                io.Int.Input("spacing", default=0, min=0, max=1000, step=1),
                io.String.Input("spacing_color", default="1.0"),
                io.String.Input("pad_color", default="1.0"),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Mask.Output(display_name="mask"),
                io.Mask.Output(display_name="split_mask"),
            ],
        )

    @classmethod
    async def execute(
        cls,
        reference_image: torch.Tensor | None,
        edit_image: torch.Tensor | None,
        edit_image_position: str,
        match_edit_size: bool,
        spacing: int,
        spacing_color: str,
        pad_color: str,
        edit_mask: torch.Tensor | None = None,
    ) -> io.NodeOutput:
        # 检查输入
        if reference_image is None and edit_image is None:
            default_image = torch.ones((1, 512, 512, 3), dtype=torch.float32)
            default_mask = torch.ones((1, 512, 512), dtype=torch.float32)
            return io.NodeOutput(default_image, default_mask, default_mask)

        # 如果只有一个图像存在，直接返回该图像
        if reference_image is None:
            bs = edit_image.shape[0]
            edit_mask = cls._ensure_mask_3d(edit_mask)
            if edit_mask is None:
                edit_mask = torch.ones(
                    (bs, edit_image.shape[1], edit_image.shape[2]),
                    dtype=torch.float32,
                )
            else:
                edit_mask = cls._broadcast_mask(edit_mask, bs)
            split_mask = torch.zeros_like(edit_mask)
            return io.NodeOutput(edit_image, edit_mask, split_mask)

        if edit_image is None:
            edit_image = torch.zeros_like(reference_image)
            bs = reference_image.shape[0]
            white_mask = torch.ones(
                (bs, reference_image.shape[1], reference_image.shape[2]),
                dtype=torch.float32,
            )
            split_mask = torch.ones_like(white_mask)
            return io.NodeOutput(reference_image, white_mask, split_mask)

        # 确保编辑遮罩存在，如果不存在则创建全白遮罩
        edit_mask = cls._ensure_mask_3d(edit_mask)
        if edit_mask is None:
            edit_mask = torch.ones(
                (edit_image.shape[0], edit_image.shape[1], edit_image.shape[2]),
                dtype=torch.float32,
            )

        # 统一批量尺寸（广播到最大批次）
        bs = max(reference_image.shape[0], edit_image.shape[0], edit_mask.shape[0])
        reference_image = cls._broadcast_image(reference_image, bs)
        edit_image = cls._broadcast_image(edit_image, bs)
        edit_mask = cls._broadcast_mask(edit_mask, bs)

        # 获取图像尺寸
        ref_batch, ref_height, ref_width, _ = reference_image.shape
        edit_batch, edit_height, edit_width, _ = edit_image.shape

        # 遮罩尺寸与编辑图像对齐（最近邻），避免拼接时高度/宽度不一致
        edit_mask = cls._resize_mask_to_image(edit_mask, edit_image)

        # 处理尺寸调整逻辑（批量按样本处理）
        if match_edit_size:
            if ref_height != edit_height or ref_width != edit_width:
                reference_image = cls._resize_with_padding(
                    reference_image, edit_width, edit_height, pad_color
                )
                ref_batch, ref_height, ref_width, _ = reference_image.shape
        else:
            reference_image = cls._resize_keeping_aspect_ratio(
                reference_image, edit_image, edit_image_position
            )
            ref_batch, ref_height, ref_width, _ = reference_image.shape

        # 间隔条颜色解析（不支持 edge，严格 0..1 RGB）
        color = cls._parse_spacing_color(spacing_color)
        space_rgb = (float(color[0]), float(color[1]), float(color[2]))
        spacing_color_tensor = torch.tensor(
            space_rgb, dtype=torch.float32
        ).view(1, 1, 1, 3)
        
        # 根据编辑图像位置拼接图像
        if edit_image_position == "right":
            # 参考图像在左，编辑图像在右
            if spacing > 0:
                # 创建垂直间距条
                spacing_strip = spacing_color_tensor.expand(bs, ref_height, spacing, 3)
                combined_image = torch.cat([
                    reference_image,
                    spacing_strip,
                    edit_image
                ], dim=2)  # 水平拼接

                # 拼接遮罩（参考区域为0，间距区域为0，编辑区域保持原样）
                zero_mask_ref = torch.zeros((bs, ref_height, ref_width), dtype=torch.float32)
                zero_mask_spacing = torch.zeros((bs, ref_height, spacing), dtype=torch.float32)
                combined_mask = torch.cat([zero_mask_ref, zero_mask_spacing, edit_mask], dim=2)
                
                # 创建分离遮罩（参考区域为黑色，间距区域为黑色，编辑区域为白色）
                split_mask_left = torch.zeros((bs, ref_height, ref_width), dtype=torch.float32)
                split_mask_spacing = torch.zeros((bs, ref_height, spacing), dtype=torch.float32)
                split_mask_right = torch.ones((bs, edit_height, edit_width), dtype=torch.float32)
                split_mask = torch.cat([split_mask_left, split_mask_spacing, split_mask_right], dim=2)
            else:
                combined_image = torch.cat([
                    reference_image,
                    edit_image
                ], dim=2)  # 水平拼接
                
                # 拼接遮罩（参考区域为0，编辑区域保持原样）
                zero_mask = torch.zeros((bs, ref_height, ref_width), dtype=torch.float32)
                combined_mask = torch.cat([zero_mask, edit_mask], dim=2)
                
                # 创建分离遮罩（参考区域为黑色，编辑区域为白色）
                split_mask_left = torch.zeros((bs, ref_height, ref_width), dtype=torch.float32)
                split_mask_right = torch.ones((bs, edit_height, edit_width), dtype=torch.float32)
                split_mask = torch.cat([split_mask_left, split_mask_right], dim=2)

        elif edit_image_position == "left":
            # 编辑图像在左，参考图像在右
            if spacing > 0:
                # 创建垂直间距条
                spacing_strip = spacing_color_tensor.expand(bs, edit_height, spacing, 3)
                combined_image = torch.cat([
                    edit_image,
                    spacing_strip,
                    reference_image
                ], dim=2)  # 水平拼接
                
                # 拼接遮罩（编辑区域保持原样，间距区域为0，参考区域为0）
                zero_mask_spacing = torch.zeros((bs, edit_height, spacing), dtype=torch.float32)
                zero_mask_ref = torch.zeros((bs, ref_height, ref_width), dtype=torch.float32)
                combined_mask = torch.cat([edit_mask, zero_mask_spacing, zero_mask_ref], dim=2)
                
                # 创建分离遮罩（编辑区域为白色，间距区域为黑色，参考区域为黑色）
                split_mask_left = torch.ones((bs, edit_height, edit_width), dtype=torch.float32)
                split_mask_spacing = torch.zeros((bs, edit_height, spacing), dtype=torch.float32)
                split_mask_right = torch.zeros((bs, ref_height, ref_width), dtype=torch.float32)
                split_mask = torch.cat([split_mask_left, split_mask_spacing, split_mask_right], dim=2)
            else:
                combined_image = torch.cat([
                    edit_image,
                    reference_image
                ], dim=2)  # 水平拼接
                
                # 拼接遮罩（编辑区域保持原样，参考区域为0）
                zero_mask = torch.zeros((bs, ref_height, ref_width), dtype=torch.float32)
                combined_mask = torch.cat([edit_mask, zero_mask], dim=2)
                
                # 创建分离遮罩（编辑区域为白色，参考区域为黑色）
                split_mask_left = torch.ones((bs, edit_height, edit_width), dtype=torch.float32)
                split_mask_right = torch.zeros((bs, ref_height, ref_width), dtype=torch.float32)
                split_mask = torch.cat([split_mask_left, split_mask_right], dim=2)

        elif edit_image_position == "bottom":
            # 参考图像在上，编辑图像在下
            if spacing > 0:
                # 创建水平间距条
                spacing_strip = spacing_color_tensor.expand(bs, spacing, ref_width, 3)
                combined_image = torch.cat([
                    reference_image,
                    spacing_strip,
                    edit_image
                ], dim=1)  # 垂直拼接
                
                # 拼接遮罩（参考区域为0，间距区域为0，编辑区域保持原样）
                zero_mask_ref = torch.zeros((bs, ref_height, ref_width), dtype=torch.float32)
                zero_mask_spacing = torch.zeros((bs, spacing, ref_width), dtype=torch.float32)
                combined_mask = torch.cat([zero_mask_ref, zero_mask_spacing, edit_mask], dim=1)
                
                # 创建分离遮罩（参考区域为黑色，间距区域为黑色，编辑区域为白色）
                split_mask_top = torch.zeros((bs, ref_height, ref_width), dtype=torch.float32)
                split_mask_spacing = torch.zeros((bs, spacing, ref_width), dtype=torch.float32)
                split_mask_bottom = torch.ones((bs, edit_height, edit_width), dtype=torch.float32)
                split_mask = torch.cat([split_mask_top, split_mask_spacing, split_mask_bottom], dim=1)
            else:
                combined_image = torch.cat([
                    reference_image,
                    edit_image
                ], dim=1)  # 垂直拼接
                
                # 拼接遮罩（参考区域为0，编辑区域保持原样）
                zero_mask = torch.zeros((bs, ref_height, ref_width), dtype=torch.float32)
                combined_mask = torch.cat([zero_mask, edit_mask], dim=1)
                
                # 创建分离遮罩（参考区域为黑色，编辑区域为白色）
                split_mask_top = torch.zeros((bs, ref_height, ref_width), dtype=torch.float32)
                split_mask_bottom = torch.ones((bs, edit_height, edit_width), dtype=torch.float32)
                split_mask = torch.cat([split_mask_top, split_mask_bottom], dim=1)

        elif edit_image_position == "top":
            # 编辑图像在上，参考图像在下
            if spacing > 0:
                # 创建水平间距条
                spacing_strip = spacing_color_tensor.expand(bs, spacing, edit_width, 3)
                combined_image = torch.cat([
                    edit_image,
                    spacing_strip,
                    reference_image
                ], dim=1)  # 垂直拼接
                
                # 拼接遮罩（编辑区域保持原样，间距区域为0，参考区域为0）
                zero_mask_spacing = torch.zeros((bs, spacing, edit_width), dtype=torch.float32)
                zero_mask_ref = torch.zeros((bs, ref_height, ref_width), dtype=torch.float32)
                combined_mask = torch.cat([edit_mask, zero_mask_spacing, zero_mask_ref], dim=1)
                
                # 创建分离遮罩（编辑区域为白色，间距区域为黑色，参考区域为黑色）
                split_mask_top = torch.ones((bs, edit_height, edit_width), dtype=torch.float32)
                split_mask_spacing = torch.zeros((bs, spacing, edit_width), dtype=torch.float32)
                split_mask_bottom = torch.zeros((bs, ref_height, ref_width), dtype=torch.float32)
                split_mask = torch.cat([split_mask_top, split_mask_spacing, split_mask_bottom], dim=1)
            else:
                combined_image = torch.cat([
                    edit_image,
                    reference_image
                ], dim=1)  # 垂直拼接
                
                # 拼接遮罩（编辑区域保持原样，参考区域为0）
                zero_mask = torch.zeros((bs, ref_height, ref_width), dtype=torch.float32)
                combined_mask = torch.cat([edit_mask, zero_mask], dim=1)
                
                # 创建分离遮罩（编辑区域为白色，参考区域为黑色）
                split_mask_top = torch.ones((bs, edit_height, edit_width), dtype=torch.float32)
                split_mask_bottom = torch.zeros((bs, ref_height, ref_width), dtype=torch.float32)
                split_mask = torch.cat([split_mask_top, split_mask_bottom], dim=1)

        return io.NodeOutput(combined_image, combined_mask, split_mask)

    @staticmethod
    def _resize_with_padding(image, target_width, target_height, pad_color):
        """
        原有的resize逻辑：使用padding填充到目标尺寸（pad_color 控制颜色）
        """
        bs = image.shape[0]
        out = []
        for i in range(bs):
            fill = ImageEditStitch._parse_pad_color(pad_color)
            img_np = (image[i].cpu().numpy() * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)

            img_width, img_height = img_pil.size
            img_aspect = img_width / max(img_height, 1)
            target_aspect = target_width / max(target_height, 1)

            if img_aspect > target_aspect:
                new_width = target_width
                new_height = int(target_width / img_aspect)
            else:
                new_height = target_height
                new_width = int(target_height * img_aspect)

            img_pil = img_pil.resize((new_width, new_height), Image.LANCZOS)
            arr = np.array(img_pil).astype(np.float32) / 255.0
            t = torch.from_numpy(arr).unsqueeze(0)
            padded = ImageEditStitch._pad_to_rgb_like(t, target_height, target_width, fill)
            out.append(padded.squeeze(0))

        return torch.stack(out, dim=0)

    @staticmethod
    def _parse_pad_color(color_str):
        if color_str is None:
            return (1.0, 1.0, 1.0)
        text = str(color_str).strip().lower()
        if text in ("edge", "e"):
            return "edge"
        if text in ("extend", "ex"):
            return "extend"
        if text in ("mirror", "mr"):
            return "mirror"
        if text in ("average", "a"):
            return "average"
        if text.startswith("(") and text.endswith(")"):
            text = text[1:-1].strip()
        single = {
            "r": "red",
            "g": "green",
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
            "q": "aqua",
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
                    r, g, b = [float(parts[i]) for i in range(3)]
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
    def _parse_spacing_color(color_str):
        if color_str is None:
            return (1.0, 1.0, 1.0)
        text = str(color_str).strip().lower()
        if text.startswith("(") and text.endswith(")"):
            text = text[1:-1].strip()
        single = {
            "r": "red",
            "g": "green",
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
                    r, g, b = [float(parts[i]) for i in range(3)]
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
    def _pad_to_rgb_like(img, target_h, target_w, fill_rgb):
        b, h, w, c = img.shape
        top = max((target_h - h) // 2, 0)
        left = max((target_w - w) // 2, 0)
        h_end = min(top + h, target_h)
        w_end = min(left + w, target_w)
        if isinstance(fill_rgb, str) and fill_rgb == "extend":
            pad_top = top
            pad_bottom = max(target_h - (top + h), 0)
            pad_left = left
            pad_right = max(target_w - (left + w), 0)
            nchw = img.permute(0, 3, 1, 2)
            padded = F.pad(nchw, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")
            return padded.permute(0, 2, 3, 1)
        if isinstance(fill_rgb, str) and fill_rgb == "mirror":
            pad_top = top
            pad_bottom = max(target_h - (top + h), 0)
            pad_left = left
            pad_right = max(target_w - (left + w), 0)
            nchw = img.permute(0, 3, 1, 2)
            while pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
                hh = int(nchw.shape[2])
                ww = int(nchw.shape[3])
                st = min(pad_top, max(hh - 1, 0))
                sb = min(pad_bottom, max(hh - 1, 0))
                sl = min(pad_left, max(ww - 1, 0))
                sr = min(pad_right, max(ww - 1, 0))
                nchw = F.pad(nchw, (sl, sr, st, sb), mode="reflect")
                pad_top -= st
                pad_bottom -= sb
                pad_left -= sl
                pad_right -= sr
            return nchw.permute(0, 2, 3, 1)
        out = torch.zeros((b, target_h, target_w, c), dtype=img.dtype, device=img.device)
        if isinstance(fill_rgb, str) and fill_rgb == "edge":
            if top > 0 or (target_h - h_end) > 0:
                top_row = img[:, 0:1, :, :].mean(dim=(1, 2))
                top_color = top_row.view(b, 1, 1, c)
                bottom_row = img[:, -1:, :, :].mean(dim=(1, 2))
                bottom_color = bottom_row.view(b, 1, 1, c)
                if top > 0:
                    out[:, :top, :, :] = top_color.expand(b, top, target_w, c)
                if (target_h - h_end) > 0:
                    bottom_pad = target_h - h_end
                    out[:, h_end:, :, :] = bottom_color.expand(b, bottom_pad, target_w, c)
            if left > 0 or (target_w - w_end) > 0:
                left_col = img[:, :, 0:1, :].mean(dim=(1, 2))
                left_color = left_col.view(b, 1, 1, c)
                right_col = img[:, :, -1:, :].mean(dim=(1, 2))
                right_color = right_col.view(b, 1, 1, c)
                if left > 0:
                    out[:, :, :left, :] = left_color.expand(b, target_h, left, c)
                if (target_w - w_end) > 0:
                    right_pad = target_w - w_end
                    out[:, :, w_end:, :] = right_color.expand(b, target_h, right_pad, c)
        elif isinstance(fill_rgb, str) and fill_rgb == "average":
            avg = img.mean(dim=(1, 2))
            avg_color = avg.view(b, 1, 1, c)
            out[:] = avg_color.expand(b, target_h, target_w, c)
        else:
            fill_t = torch.tensor(fill_rgb, dtype=img.dtype, device=img.device)
            out[:] = fill_t
        out[:, top:h_end, left:w_end, :] = img[:, : h_end - top, : w_end - left, :]
        return out


    @staticmethod
    def _resize_keeping_aspect_ratio(reference_image, edit_image, edit_image_position):
        """
        新的resize逻辑：保持reference_image的原始比例，根据拼接方向调整尺寸
        """
        bs = max(reference_image.shape[0], edit_image.shape[0])
        out = []
        for i in range(bs):
            # 单样本 (H, W, C)
            ref_arr = (reference_image[i].cpu().numpy() * 255).astype(np.uint8)
            ref_pil = Image.fromarray(ref_arr)

            edit_h = int(edit_image[i].shape[0])
            edit_w = int(edit_image[i].shape[1])
            ref_h = int(reference_image[i].shape[0])
            ref_w = int(reference_image[i].shape[1])

            if edit_image_position in ["left", "right"]:
                target_height = max(edit_h, 1)
                aspect_ratio = ref_w / max(ref_h, 1)
                target_width = max(int(round(target_height * aspect_ratio)), 1)
            else:
                target_width = max(edit_w, 1)
                aspect_ratio = ref_h / max(ref_w, 1)
                target_height = max(int(round(target_width * aspect_ratio)), 1)

            ref_pil = ref_pil.resize((target_width, target_height), Image.LANCZOS)
            arr = np.array(ref_pil).astype(np.float32) / 255.0
            out.append(torch.from_numpy(arr))

        return torch.stack(out, dim=0)

    @staticmethod
    def _broadcast_image(img, batch_size):
        b = img.shape[0]
        if b == batch_size:
            return img
        if b == 1:
            return img.repeat(batch_size, 1, 1, 1)
        reps = int(math.ceil(batch_size / b))
        tiled = img.repeat(reps, 1, 1, 1)[:batch_size]
        return tiled

    @staticmethod
    def _broadcast_mask(mask, batch_size):
        b = mask.shape[0]
        if b == batch_size:
            return mask
        if b == 1:
            return mask.repeat(batch_size, 1, 1)
        reps = int(math.ceil(batch_size / b))
        tiled = mask.repeat(reps, 1, 1)[:batch_size]
        return tiled

    @staticmethod
    def _ensure_mask_3d(mask):
        """确保遮罩为 (batch, height, width) 形状。
        - (batch, 1, height, width) -> (batch, height, width)
        - (height, width) -> (1, height, width)
        其他情况原样返回。
        """
        if mask is None:
            return None
        if mask.dim() == 4:
            if mask.shape[1] == 1:
                return mask.squeeze(1)
            # 取第一个通道作为遮罩
            return mask[:, 0, :, :]
        if mask.dim() == 2:
            return mask.unsqueeze(0)
        return mask

    @staticmethod
    def _resize_mask_to_image(mask, image):
        """将遮罩尺寸重采样到与图像一致 (H, W)。
        使用最近邻以保持二值或离散值。
        """
        if mask is None or image is None:
            return mask
        target_h = int(image.shape[1])
        target_w = int(image.shape[2])
        if int(mask.shape[1]) == target_h and int(mask.shape[2]) == target_w:
            return mask
        mask4 = mask.unsqueeze(1)
        resized4 = F.interpolate(
            mask4, size=(target_h, target_w), mode="nearest"
        )
        return resized4.squeeze(1)
    @classmethod
    def validate_inputs(
        cls,
        reference_image: torch.Tensor | None,
        edit_image: torch.Tensor | None,
        edit_image_position: str,
        match_edit_size: bool,
        spacing: int,
        spacing_color: str,
        pad_color: str,
        edit_mask: torch.Tensor | None = None,
    ):
        if edit_image_position not in {"top", "bottom", "left", "right"}:
            return "invalid edit_image_position"
        if spacing < 0:
            return "invalid spacing"
        return True

    @classmethod
    def fingerprint_inputs(
        cls,
        reference_image: torch.Tensor | None,
        edit_image: torch.Tensor | None,
        edit_image_position: str,
        match_edit_size: bool,
        spacing: int,
        spacing_color: str,
        pad_color: str,
        edit_mask: torch.Tensor | None = None,
    ):
        rb = int(reference_image.shape[0]) if isinstance(reference_image, torch.Tensor) else 0
        eb = int(edit_image.shape[0]) if isinstance(edit_image, torch.Tensor) else 0
        return f"rb={rb}|eb={eb}|pos={edit_image_position}|match={match_edit_size}|sp={spacing}|sc={spacing_color}|pc={pad_color}"
