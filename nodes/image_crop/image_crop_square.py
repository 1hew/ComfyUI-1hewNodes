import asyncio
from comfy_api.latest import io
import numpy as np
from PIL import Image, ImageDraw, ImageColor
import torch
import torch.nn.functional as F



class ImageCropSquare(io.ComfyNode):
    """
    图像方形裁剪器 - 根据遮罩裁切图像为方形，支持放大系数和填充颜色
    """
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_ImageCropSquare",
            display_name="Image Crop Square",
            category="1hewNodes/image/crop",
            inputs=[
                io.Image.Input("image"),
                io.Mask.Input("mask", optional=True),
                io.Float.Input("scale_factor", default=1.0, min=0.1, max=3.0, step=0.01),
                io.Int.Input("extra_padding", default=0, min=0, max=512, step=1),
                io.String.Input("fill_color", default="1.0"),
                io.Boolean.Input("apply_mask", default=False),
                io.Int.Input("divisible_by", default=8, min=1, max=1024, step=1),
            ],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    async def execute(
        cls,
        image: torch.Tensor,
        scale_factor: float,
        apply_mask: bool,
        extra_padding: int,
        fill_color: str,
        divisible_by: int,
        mask: torch.Tensor | None = None,
    ) -> io.NodeOutput:
        image = image.to(torch.float32).clamp(0.0, 1.0)
        if mask is not None:
            mask = mask.to(torch.float32).clamp(0.0, 1.0)

        batch_size, height, width, channels = image.shape

        async def _proc(b):
            def _do():
                img_np = (image[b].detach().cpu().numpy() * 255).astype(np.uint8)
                if mask is not None:
                    m = mask[b % mask.shape[0]]
                    mask_np = (m.detach().cpu().numpy() * 255).astype(np.uint8)
                else:
                    mask_np = np.zeros((height, width), dtype=np.uint8)

                img_pil = Image.fromarray(img_np)
                mask_pil = Image.fromarray(mask_np).convert("L")

                if img_pil.size != mask_pil.size:
                    new_mask = Image.new("L", img_pil.size, 0)
                    paste_x0 = max(0, (img_pil.width - mask_pil.width) // 2)
                    paste_y0 = max(0, (img_pil.height - mask_pil.height) // 2)
                    new_mask.paste(mask_pil, (paste_x0, paste_y0))
                    mask_pil = new_mask

                bbox = cls._get_bbox(mask_pil)

                if bbox is None:
                    square_size = min(img_pil.width, img_pil.height)
                    scaled_size = int(square_size * scale_factor)
                    center_x = img_pil.width // 2
                    center_y = img_pil.height // 2
                    crop_x1 = center_x - scaled_size // 2
                    crop_y1 = center_y - scaled_size // 2
                    crop_x2 = crop_x1 + scaled_size
                    crop_y2 = crop_y1 + scaled_size
                    crop_x1 = max(0, crop_x1)
                    crop_y1 = max(0, crop_y1)
                    crop_x2 = min(img_pil.width, crop_x2)
                    crop_y2 = min(img_pil.height, crop_y2)
                    final_size = scaled_size + extra_padding * 2
                    final_size = (final_size // divisible_by) * divisible_by
                    if final_size <= 0:
                        final_size = divisible_by
                    pad_mode = cls._parse_pad_mode(fill_color)
                    square_img = None
                    cropped_region = img_pil.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                    paste_x = (final_size - cropped_region.width) // 2
                    paste_y = (final_size - cropped_region.height) // 2
                    if square_img is None:
                        if pad_mode == "extend" or pad_mode == "mirror":
                            pad_left = paste_x
                            pad_top = paste_y
                            pad_right = final_size - cropped_region.width - paste_x
                            pad_bottom = final_size - cropped_region.height - paste_y
                            mode = "edge" if pad_mode == "extend" else "reflect"
                            square_np = cls._pad_cropped_numpy(
                                np.array(cropped_region),
                                pad_left,
                                pad_top,
                                pad_right,
                                pad_bottom,
                                mode,
                            )
                            square_img = Image.fromarray(square_np.astype(np.uint8))
                        elif pad_mode == "edge":
                            edge_colors = cls._get_four_edge_colors(
                                img_pil, crop_x1, crop_y1, crop_x2, crop_y2
                            )
                            square_img = Image.new(
                                "RGB", (final_size, final_size), (255, 255, 255)
                            )
                            cls._fill_pad_areas_with_edge_colors(
                                square_img,
                                edge_colors,
                                paste_x,
                                paste_y,
                                final_size - cropped_region.width - paste_x,
                                final_size - cropped_region.height - paste_y,
                            )
                        elif pad_mode == "average":
                            avg_rgb = cls._compute_average_rgb(img_pil)
                            square_img = Image.new("RGB", (final_size, final_size), avg_rgb)
                        else:
                            bg_color = cls._parse_color(fill_color)
                            square_img = Image.new("RGB", (final_size, final_size), bg_color)
                    square_img.paste(cropped_region, (paste_x, paste_y))
                    square_img_np = np.array(square_img).astype(np.float32) / 255.0
                    return torch.from_numpy(square_img_np)

                bbox_width = bbox[2] - bbox[0]
                bbox_height = bbox[3] - bbox[1]
                center_x = (bbox[0] + bbox[2]) // 2
                center_y = (bbox[1] + bbox[3]) // 2
                square_size = max(bbox_width, bbox_height)
                scaled_size = int(square_size * scale_factor)
                scaled_x1 = center_x - scaled_size // 2
                scaled_y1 = center_y - scaled_size // 2
                scaled_x2 = scaled_x1 + scaled_size
                scaled_y2 = scaled_y1 + scaled_size
                final_size = scaled_size + extra_padding * 2
                final_size = (final_size // divisible_by) * divisible_by
                if final_size <= 0:
                    final_size = divisible_by
                square_x1 = center_x - final_size // 2
                square_y1 = center_y - final_size // 2
                square_x2 = square_x1 + final_size
                square_y2 = square_y1 + final_size
                pad_mode = cls._parse_pad_mode(fill_color)
                square_img = None
                paste_x = max(0, -scaled_x1) + extra_padding
                paste_y = max(0, -scaled_y1) + extra_padding
                crop_x1 = max(0, scaled_x1)
                crop_y1 = max(0, scaled_y1)
                crop_x2 = min(img_pil.width, scaled_x2)
                crop_y2 = min(img_pil.height, scaled_y2)
                if crop_x1 < crop_x2 and crop_y1 < crop_y2:
                    cropped_region = img_pil.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                    if apply_mask:
                        cropped_mask = mask_pil.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                        fc = fill_color.lower()
                        if fc in ("a", "average", "avg", "mk", "mask"):
                            bg_rgb = cls._parse_fill_color_mask(fill_color, img_pil, mask_pil)
                        elif fc in ("e", "edge"):
                            bg_rgb = cls._get_mask_area_color(img_pil, mask_pil, bbox)
                        else:
                            bg_rgb = cls._parse_color(fill_color)
                        square_img = Image.new("RGB", (final_size, final_size), bg_rgb)
                    if square_img is None:
                        if pad_mode == "extend" or pad_mode == "mirror":
                            pad_left = paste_x
                            pad_top = paste_y
                            pad_right = final_size - cropped_region.width - paste_x
                            pad_bottom = final_size - cropped_region.height - paste_y
                            mode = "edge" if pad_mode == "extend" else "reflect"
                            square_np = cls._pad_cropped_numpy(
                                np.array(cropped_region),
                                pad_left,
                                pad_top,
                                pad_right,
                                pad_bottom,
                                mode,
                            )
                            square_img = Image.fromarray(square_np.astype(np.uint8))
                        elif pad_mode == "edge":
                            edge_colors = cls._get_four_edge_colors(
                                img_pil, scaled_x1, scaled_y1, scaled_x2, scaled_y2
                            )
                            square_img = Image.new(
                                "RGB", (final_size, final_size), (255, 255, 255)
                            )
                            cls._fill_pad_areas_with_edge_colors(
                                square_img,
                                edge_colors,
                                paste_x,
                                paste_y,
                                final_size - cropped_region.width - paste_x,
                                final_size - cropped_region.height - paste_y,
                            )
                        elif pad_mode == "average":
                            avg_rgb = cls._compute_average_rgb(img_pil)
                            square_img = Image.new("RGB", (final_size, final_size), avg_rgb)
                        else:
                            bg_color = cls._parse_color(fill_color)
                            square_img = Image.new("RGB", (final_size, final_size), bg_color)
                    if apply_mask:
                        square_img.paste(cropped_region, (paste_x, paste_y), mask=cropped_mask)
                    else:
                        square_img.paste(cropped_region, (paste_x, paste_y))
                square_img_np = np.array(square_img).astype(np.float32) / 255.0
                return torch.from_numpy(square_img_np)
            return await asyncio.to_thread(_do)

        results = await asyncio.gather(*[_proc(b) for b in range(batch_size)])

        output_images = list(results)
        device = image.device
        if output_images:
            shapes = [img.shape for img in output_images]
            if len(set(shapes)) > 1:
                output_images = cls._pad_images_to_same_size(output_images)
            output_image_tensor = torch.stack(output_images).to(device)
            output_image_tensor = output_image_tensor.to(torch.float32).clamp(0.0, 1.0)
            return io.NodeOutput(output_image_tensor)
        return io.NodeOutput(image)

    @staticmethod
    def _get_bbox(mask_pil):
        # 将遮罩转换为numpy数组
        mask_np = np.array(mask_pil)

        # 找到非零区域的坐标
        rows = np.any(mask_np > 10, axis=1)
        cols = np.any(mask_np > 10, axis=0)

        # 如果没有找到非零区域，返回None
        if not np.any(rows) or not np.any(cols):
            return None

        # 获取边界框坐标
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        # 返回边界框 (left, top, right, bottom)
        return (x_min, y_min, x_max + 1, y_max + 1)
    
    @staticmethod
    def _get_mask_area_color(img, mask, bbox):
        """获取遮罩区域的平均颜色"""
        # 裁剪到边界框区域
        img_crop = img.crop(bbox)
        mask_crop = mask.crop(bbox)
        
        # 将图像和遮罩转换为numpy数组
        img_np = np.array(img_crop)
        mask_np = np.array(mask_crop)
        
        # 找到遮罩中非零区域
        mask_indices = mask_np > 10
        
        # 如果没有有效区域，返回白色
        if not np.any(mask_indices):
            return (255, 255, 255)
        
        # 获取遮罩区域内的像素
        masked_pixels = img_np[mask_indices]
        
        # 计算平均颜色
        r_mean = int(np.mean(masked_pixels[:, 0]))
        g_mean = int(np.mean(masked_pixels[:, 1]))
        b_mean = int(np.mean(masked_pixels[:, 2]))
        
        return (r_mean, g_mean, b_mean)
    
    @staticmethod
    def _get_four_edge_colors(img, x1, y1, x2, y2):
        """获取图像四个边缘的平均颜色"""
        width, height = img.size
        
        # 确保坐标在图像范围内
        x1 = max(0, min(x1, width-1))
        y1 = max(0, min(y1, height-1))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        
        # 收集四个边缘的像素
        top_pixels = []
        bottom_pixels = []
        left_pixels = []
        right_pixels = []
        
        # 上边缘
        if y1 >= 0 and y1 < height:
            for x in range(max(0, x1), min(width, x2)):
                if 0 <= x < width:
                    top_pixels.append(img.getpixel((x, y1)))
        
        # 下边缘
        if y2-1 >= 0 and y2-1 < height:
            for x in range(max(0, x1), min(width, x2)):
                if 0 <= x < width:
                    bottom_pixels.append(img.getpixel((x, y2-1)))
        
        # 左边缘
        if x1 >= 0 and x1 < width:
            for y in range(max(0, y1), min(height, y2)):
                if 0 <= y < height:
                    left_pixels.append(img.getpixel((x1, y)))
        
        # 右边缘
        if x2-1 >= 0 and x2-1 < width:
            for y in range(max(0, y1), min(height, y2)):
                if 0 <= y < height:
                    right_pixels.append(img.getpixel((x2-1, y)))
        
        # 计算每个边缘的平均颜色
        top_color = ImageCropSquare._calculate_average_color(top_pixels)
        bottom_color = ImageCropSquare._calculate_average_color(bottom_pixels)
        left_color = ImageCropSquare._calculate_average_color(left_pixels)
        right_color = ImageCropSquare._calculate_average_color(right_pixels)
        
        return {
            'top': top_color,
            'bottom': bottom_color,
            'left': left_color,
            'right': right_color
        }
    
    @staticmethod
    def _calculate_average_color(pixels):
        """计算像素列表的平均颜色"""
        if not pixels:
            return (255, 255, 255)
        
        r_sum = sum(p[0] for p in pixels)
        g_sum = sum(p[1] for p in pixels)
        b_sum = sum(p[2] for p in pixels)
        
        pixel_count = len(pixels)
        return (r_sum // pixel_count, g_sum // pixel_count, b_sum // pixel_count)
    
    @staticmethod
    def _fill_edges_with_colors(img, edge_colors, padding):
        """使用四个边缘颜色填充图像的边缘区域"""
        if padding <= 0:
            return
        
        width, height = img.size
        draw = ImageDraw.Draw(img)
        
        # 填充上边缘
        draw.rectangle([0, 0, width, padding], fill=edge_colors['top'])
        
        # 填充下边缘
        draw.rectangle([0, height-padding, width, height], fill=edge_colors['bottom'])
        
        # 填充左边缘 (不包括已经填充的角落)
        draw.rectangle([0, padding, padding, height-padding], fill=edge_colors['left'])
        
        # 填充右边缘 (不包括已经填充的角落)
        draw.rectangle([width-padding, padding, width, height-padding], fill=edge_colors['right'])
    
    @staticmethod
    def _get_edge_colors(img, x1, y1, x2, y2):
        """获取图像边缘的平均颜色 (兼容旧代码)"""
        width, height = img.size
        
        # 确保坐标在图像范围内
        x1 = max(0, min(x1, width-1))
        y1 = max(0, min(y1, height-1))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        
        # 收集边缘像素
        edge_pixels = []
        
        # 上边缘
        if y1 >= 0 and y1 < height:
            for x in range(max(0, x1), min(width, x2)):
                if 0 <= x < width:
                    edge_pixels.append(img.getpixel((x, y1)))
        
        # 下边缘
        if y2-1 >= 0 and y2-1 < height:
            for x in range(max(0, x1), min(width, x2)):
                if 0 <= x < width:
                    edge_pixels.append(img.getpixel((x, y2-1)))
        
        # 左边缘
        if x1 >= 0 and x1 < width:
            for y in range(max(0, y1), min(height, y2)):
                if 0 <= y < height:
                    edge_pixels.append(img.getpixel((x1, y)))
        
        # 右边缘
        if x2-1 >= 0 and x2-1 < width:
            for y in range(max(0, y1), min(height, y2)):
                if 0 <= y < height:
                    edge_pixels.append(img.getpixel((x2-1, y)))
        
        # 如果没有有效的边缘像素，返回白色
        if not edge_pixels:
            return (255, 255, 255)
        
        # 计算平均颜色
        r_sum = sum(p[0] for p in edge_pixels)
        g_sum = sum(p[1] for p in edge_pixels)
        b_sum = sum(p[2] for p in edge_pixels)
        
        pixel_count = len(edge_pixels)
        return (r_sum // pixel_count, g_sum // pixel_count, b_sum // pixel_count)
    
    @staticmethod
    def _parse_color(color_str):
        """解析颜色字符串，增强支持与 multi_image_stitch 一致的映射"""
        if color_str is None:
            return (255, 255, 255)
        s = str(color_str).strip()
        lower = s.lower()
        # 去掉括号包裹
        if lower.startswith("(") and lower.endswith(")"):
            lower = lower[1:-1].strip()
        # 单字母映射到颜色名称
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
        if len(lower) == 1 and lower in single:
            lower = single[lower]
        # 灰度数值
        try:
            v = float(lower)
            if 0.0 <= v <= 1.0:
                g = int(v * 255)
                return (g, g, g)
            if 1.0 < v <= 255.0:
                g = int(v)
                return (g, g, g)
        except Exception:
            pass
        # 逗号分隔 RGB，支持 0–1 或 0–255
        if "," in lower:
            try:
                parts = [p.strip() for p in lower.split(",")]
                if len(parts) >= 3:
                    r = float(parts[0])
                    g = float(parts[1])
                    b = float(parts[2])
                    if max(r, g, b) <= 1.0:
                        return (int(r * 255), int(g * 255), int(b * 255))
                    return (int(r), int(g), int(b))
            except Exception:
                pass
        # 十六进制 #RGB/#RRGGBB
        if lower.startswith("#") and len(lower) in (4, 7):
            try:
                hex_str = lower[1:]
                if len(hex_str) == 3:
                    hex_str = "".join(ch * 2 for ch in hex_str)
                r = int(hex_str[0:2], 16)
                g = int(hex_str[2:4], 16)
                b = int(hex_str[4:6], 16)
                return (r, g, b)
            except Exception:
                pass
        # 颜色名称
        try:
            rgb = ImageColor.getrgb(lower)
            return (int(rgb[0]), int(rgb[1]), int(rgb[2]))
        except Exception:
            return (255, 255, 255)

    @staticmethod
    def _parse_pad_mode(color_str):
        s = str(color_str).strip().lower()
        if s in ("edge", "e"):
            return "edge"
        if s in ("extend", "ex"):
            return "extend"
        if s in ("mirror", "mr"):
            return "mirror"
        if s in ("average", "a"):
            return "average"
        return "color"

    @staticmethod
    def _compute_average_rgb(img):
        rgb = np.array(img.convert("RGB")).astype(np.float32)
        avg = rgb.mean(axis=(0, 1))
        return (int(avg[0]), int(avg[1]), int(avg[2]))

    @staticmethod
    def _compute_mask_average_rgb(img, mask):
        rgb = np.array(img.convert("RGB")).astype(np.float32)
        m = np.array(mask).astype(np.float32) / 255.0
        if m.shape[:2] != rgb.shape[:2]:
            m = (
                np.array(mask.resize(img.size, Image.Resampling.LANCZOS)).astype(np.float32)
                / 255.0
            )
        c = float(m.sum())
        if c > 0.0:
            m3 = np.repeat(m[:, :, None], 3, axis=2)
            avg = (rgb * m3).sum(axis=(0, 1)) / c
        else:
            avg = rgb.mean(axis=(0, 1))
        return (int(avg[0]), int(avg[1]), int(avg[2]))

    @staticmethod
    def _parse_fill_color_mask(color_str, img, mask):
        s = str(color_str).strip().lower()
        if s in ("a", "average", "avg"):
            return ImageCropSquare._compute_average_rgb(img)
        if s in ("mk", "mask"):
            return ImageCropSquare._compute_mask_average_rgb(img, mask)
        return ImageCropSquare._parse_color(color_str)

    @staticmethod
    def _pad_reflect_np(arr, pad_top, pad_bottom, pad_left, pad_right):
        out = arr
        pt, pb, pl, pr = pad_top, pad_bottom, pad_left, pad_right
        while pt > 0 or pb > 0 or pl > 0 or pr > 0:
            h = int(out.shape[0])
            w = int(out.shape[1])
            st = min(pt, max(h - 1, 0))
            sb = min(pb, max(h - 1, 0))
            sl = min(pl, max(w - 1, 0))
            sr = min(pr, max(w - 1, 0))
            out = np.pad(out, ((st, sb), (sl, sr), (0, 0)), mode="reflect")
            pt -= st
            pb -= sb
            pl -= sl
            pr -= sr
        return out

    @staticmethod
    def _pad_cropped_numpy(arr, pad_left, pad_top, pad_right, pad_bottom, mode):
        if mode == "edge":
            return np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="edge")
        return ImageCropSquare._pad_reflect_np(arr, pad_top, pad_bottom, pad_left, pad_right)

    @staticmethod
    def _fill_pad_areas_with_edge_colors(img, edge_colors, pad_left, pad_top, pad_right, pad_bottom):
        width, height = img.size
        draw = ImageDraw.Draw(img)
        if pad_top > 0:
            draw.rectangle([0, 0, width, pad_top], fill=edge_colors["top"])
        if pad_bottom > 0:
            draw.rectangle([0, height - pad_bottom, width, height], fill=edge_colors["bottom"])
        if pad_left > 0:
            draw.rectangle([0, pad_top, pad_left, height - pad_bottom], fill=edge_colors["left"])
        if pad_right > 0:
            draw.rectangle([width - pad_right, pad_top, width, height - pad_bottom], fill=edge_colors["right"])

    @staticmethod
    def _pad_images_to_same_size(images):
        max_h = max(img.shape[0] for img in images)
        max_w = max(img.shape[1] for img in images)
        max_c = max(img.shape[2] for img in images)
        padded = []
        for img in images:
            h, w, c = img.shape
            pad_h = max_h - h
            pad_w = max_w - w
            pad_c = max_c - c
            padded_img = F.pad(img, (0, pad_c, 0, pad_w, 0, pad_h), value=0)
            padded.append(padded_img)
        return padded
