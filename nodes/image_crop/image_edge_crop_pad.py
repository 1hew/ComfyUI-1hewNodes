import asyncio
from comfy_api.latest import io
import torch
import torch.nn.functional as F
from PIL import ImageColor

class ImageEdgeCropPad(io.ComfyNode):
    """
    图像边缘裁剪填充 - 支持向内裁剪和向外填充
    负数值：向内裁剪
    正数值：向外填充（pad）
    支持多种颜色格式的填充颜色和边缘填充模式
    输出 mask：裁剪或填充的区域为白色，原图区域为黑色
    """
    MAX_RESOLUTION = 8192

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_ImageEdgeCropPad",
            display_name="Image Edge Crop/Pad",
            category="1hewNodes/image/crop",
            inputs=[
                io.Image.Input("image"),
                io.Float.Input("uniform_amount", default=0.0, min=-cls.MAX_RESOLUTION, max=cls.MAX_RESOLUTION, step=0.01),
                io.Float.Input("top_amount", default=0.0, min=-cls.MAX_RESOLUTION, max=cls.MAX_RESOLUTION, step=0.01),
                io.Float.Input("bottom_amount", default=0.0, min=-cls.MAX_RESOLUTION, max=cls.MAX_RESOLUTION, step=0.01),
                io.Float.Input("left_amount", default=0.0, min=-cls.MAX_RESOLUTION, max=cls.MAX_RESOLUTION, step=0.01),
                io.Float.Input("right_amount", default=0.0, min=-cls.MAX_RESOLUTION, max=cls.MAX_RESOLUTION, step=0.01),
                io.String.Input("pad_color", default="0.0"),
                io.Int.Input("divisible_by", default=8, min=1, max=1024, step=1),
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
        uniform_amount: float,
        top_amount: float,
        bottom_amount: float,
        left_amount: float,
        right_amount: float,
        pad_color: str,
        divisible_by: int,
    ) -> io.NodeOutput:
        image = image.to(torch.float32).clamp(0.0, 1.0)
        batch_size, height, width, channels = image.shape
        
        def process_value(value, dimension):
            if value == 0:
                return 0
            if abs(value) < 1:
                return int(dimension * value)
            return int(value)

        if abs(uniform_amount) < 1 and uniform_amount != 0:
            half_w = int(width * (uniform_amount / 2.0))
            half_h = int(height * (uniform_amount / 2.0))
            uniform_left = half_w
            uniform_right = half_w
            uniform_top = half_h
            uniform_bottom = half_h
        elif uniform_amount != 0:
            u = int(uniform_amount)
            uniform_left = u
            uniform_right = u
            uniform_top = u
            uniform_bottom = u
        else:
            uniform_left = 0
            uniform_right = 0
            uniform_top = 0
            uniform_bottom = 0

        left = uniform_left + process_value(left_amount, width)
        right = uniform_right + process_value(right_amount, width)
        top = uniform_top + process_value(top_amount, height)
        bottom = uniform_bottom + process_value(bottom_amount, height)

        # 确保值为divisible_by的倍数
        left = (abs(left) // divisible_by * divisible_by) * (1 if left >= 0 else -1)
        right = (abs(right) // divisible_by * divisible_by) * (1 if right >= 0 else -1)
        top = (abs(top) // divisible_by * divisible_by) * (1 if top >= 0 else -1)
        bottom = (abs(bottom) // divisible_by * divisible_by) * (1 if bottom >= 0 else -1)

        # 如果所有值为0，直接返回原图和全黑mask
        if left == 0 and right == 0 and bottom == 0 and top == 0:
            mask = torch.zeros((batch_size, height, width), dtype=torch.float32, device=image.device)
            return io.NodeOutput(image, mask)

        # 处理批量图像
        async def _proc(b):
            def _do():
                img_tensor = image[b:b+1]
                res_t, res_m = cls._crop_or_pad_tensor(
                    img_tensor, left, right, top, bottom, pad_color
                )
                return res_t.squeeze(0), res_m.squeeze(0)
            return await asyncio.to_thread(_do)

        results = await asyncio.gather(*[_proc(b) for b in range(batch_size)])
        output_images = [r[0] for r in results]
        output_masks = [r[1] for r in results]

        # 合并批次
        output_tensor = torch.stack(output_images).to(image.device).to(torch.float32).clamp(0.0, 1.0)
        output_mask = torch.stack(output_masks).to(image.device).to(torch.float32).clamp(0.0, 1.0)
        return io.NodeOutput(output_tensor, output_mask)

    @staticmethod
    def _crop_or_pad_tensor(img_tensor, left, right, top, bottom, pad_color):
        """
        对图像tensor执行裁剪或填充操作，同时生成对应的mask
        """
        B, H, W, C = img_tensor.shape
        original_H, original_W = H, W  # 保存原始尺寸
        
        # 计算裁剪和填充的数量
        crop_left = max(0, -left)
        crop_right = max(0, -right)
        crop_top = max(0, -top)
        crop_bottom = max(0, -bottom)
        
        pad_left = max(0, left)
        pad_right = max(0, right)
        pad_top = max(0, top)
        pad_bottom = max(0, bottom)
        
        # 创建原始尺寸的mask，初始为全黑（0）
        original_mask = torch.zeros((B, original_H, original_W), dtype=torch.float32, device=img_tensor.device)
        
        # 先执行裁剪
        if crop_left > 0 or crop_right > 0 or crop_top > 0 or crop_bottom > 0:
            # 计算裁剪边界
            crop_x1 = crop_left
            crop_y1 = crop_top
            crop_x2 = W - crop_right
            crop_y2 = H - crop_bottom
            
            # 验证裁剪尺寸
            if crop_x1 >= crop_x2 or crop_y1 >= crop_y2:
                raise ValueError(f"裁剪尺寸无效：裁剪后图像尺寸为 {crop_x2-crop_x1}x{crop_y2-crop_y1}")
            
            # 在原始尺寸的mask中标记被裁剪的区域为白色（1）
            for b in range(B):
                # 标记被裁剪的区域
                if crop_top > 0:
                    original_mask[b, :crop_top, :] = 1.0  # 顶部裁剪区域
                if crop_bottom > 0:
                    original_mask[b, original_H-crop_bottom:, :] = 1.0  # 底部裁剪区域
                if crop_left > 0:
                    original_mask[b, :, :crop_left] = 1.0  # 左侧裁剪区域
                if crop_right > 0:
                    original_mask[b, :, original_W-crop_right:] = 1.0  # 右侧裁剪区域
            
            # 执行图像裁剪
            img_tensor = img_tensor[:, crop_y1:crop_y2, crop_x1:crop_x2, :]
            B, H, W, C = img_tensor.shape
            
        
        # 再执行填充
        if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
            # 计算新的尺寸
            new_height = H + pad_top + pad_bottom
            new_width = W + pad_left + pad_right
            
            # 创建输出tensor和mask
            out_tensor = torch.zeros((B, new_height, new_width, C), dtype=img_tensor.dtype, device=img_tensor.device)
            out_mask = torch.ones((B, new_height, new_width), dtype=torch.float32, device=img_tensor.device)
            
            fill_spec = ImageEdgeCropPad._parse_pad_color(pad_color)
            if isinstance(fill_spec, str) and fill_spec in ('extend', 'mirror'):
                nchw = img_tensor.permute(0, 3, 1, 2)
                if fill_spec == 'extend':
                    padded = F.pad(
                        nchw,
                        (pad_left, pad_right, pad_top, pad_bottom),
                        mode='replicate',
                    )
                else:
                    pl = pad_left
                    pr = pad_right
                    pt = pad_top
                    pb = pad_bottom
                    padded = nchw
                    while pt > 0 or pb > 0 or pl > 0 or pr > 0:
                        hh = int(padded.shape[2])
                        ww = int(padded.shape[3])
                        st = min(pt, max(hh - 1, 0))
                        sb = min(pb, max(hh - 1, 0))
                        sl = min(pl, max(ww - 1, 0))
                        sr = min(pr, max(ww - 1, 0))
                        padded = F.pad(padded, (sl, sr, st, sb), mode='reflect')
                        pt -= st
                        pb -= sb
                        pl -= sl
                        pr -= sr
                out_tensor = padded.permute(0, 2, 3, 1)
                out_mask[:, pad_top:pad_top+H, pad_left:pad_left+W] = 0.0
            elif isinstance(fill_spec, str) and fill_spec == 'edge':
                for b in range(B):
                    out_tensor[b, pad_top:pad_top+H, pad_left:pad_left+W, :] = img_tensor[b]
                    out_mask[b, pad_top:pad_top+H, pad_left:pad_left+W] = 0.0
                    top_edge = img_tensor[b, 0, :, :]
                    bottom_edge = img_tensor[b, H-1, :, :]
                    left_edge = img_tensor[b, :, 0, :]
                    right_edge = img_tensor[b, :, W-1, :]
                    top_color = top_edge.mean(dim=0)
                    bottom_color = bottom_edge.mean(dim=0)
                    left_color = left_edge.mean(dim=0)
                    right_color = right_edge.mean(dim=0)
                    if pad_top > 0:
                        out_tensor[b, :pad_top, :, :] = top_color.unsqueeze(0).unsqueeze(0)
                    if pad_bottom > 0:
                        out_tensor[b, pad_top+H:, :, :] = bottom_color.unsqueeze(0).unsqueeze(0)
                    if pad_left > 0:
                        out_tensor[b, :, :pad_left, :] = left_color.unsqueeze(0).unsqueeze(0)
                    if pad_right > 0:
                        out_tensor[b, :, pad_left+W:, :] = right_color.unsqueeze(0).unsqueeze(0)
            elif isinstance(fill_spec, str) and fill_spec == 'average':
                avg = img_tensor.mean(dim=(1, 2)).view(B, 1, 1, C)
                out_tensor[:] = avg
                out_tensor[:, pad_top:pad_top+H, pad_left:pad_left+W, :] = img_tensor
                out_mask[:, pad_top:pad_top+H, pad_left:pad_left+W] = 0.0
            else:
                bg = torch.tensor(fill_spec, dtype=img_tensor.dtype, device=img_tensor.device)
                out_tensor[:] = bg.view(1, 1, 1, C)
                out_tensor[:, pad_top:pad_top+H, pad_left:pad_left+W, :] = img_tensor
                out_mask[:, pad_top:pad_top+H, pad_left:pad_left+W] = 0.0
            
            img_tensor = out_tensor
            mask_tensor = out_mask
            return img_tensor, mask_tensor
        else:
            inv_mask = 1.0 - original_mask
            return img_tensor, inv_mask

    @staticmethod
    def _parse_color_advanced(color_str, img_tensor=None):
        """
        高级颜色解析，支持多种格式：
        - 灰度值: "0.5", "1.0"
        - HEX: "#FF0000", "FF0000"
        - RGB: "255,0,0", "1.0,0.0,0.0", "(255,128,64)"
        - 颜色名称: "red", "blue", "white"
        - 特殊值: "edge", "average"
        """
        if not color_str:
            return (0, 0, 0)
        
        # 检查特殊值
        color_lower = color_str.lower().strip()
        
        # 检查是否为 average 或其缩写
        if color_lower in ['average', 'avg', 'a', 'av', 'aver']:
            if img_tensor is not None:
                return ImageEdgeCropPad._get_average_color_tensor(img_tensor)
            return (128, 128, 128)  # 默认灰色
        
        # 检查是否为 edge 或其缩写
        if color_lower in ['edge', 'e', 'ed']:
            if img_tensor is not None:
                return ImageEdgeCropPad._get_edge_color_tensor(img_tensor)
            return (128, 128, 128)  # 默认灰色
        
        # 移除括号（如果存在）
        color_str = color_str.strip()
        if color_str.startswith('(') and color_str.endswith(')'):
            color_str = color_str[1:-1].strip()
        
        # 尝试解析为灰度值 (0.0-1.0)
        try:
            gray = float(color_str)
            if 0.0 <= gray <= 1.0:
                gray_int = int(gray * 255)
                return (gray_int, gray_int, gray_int)
            elif gray > 1.0 and gray <= 255:
                # 可能是0-255范围的灰度值
                gray_int = int(gray)
                return (gray_int, gray_int, gray_int)
        except ValueError:
            pass
        
        # 尝试解析为 RGB 格式
        if ',' in color_str:
            try:
                parts = [part.strip() for part in color_str.split(',')]
                if len(parts) >= 3:
                    r, g, b = [float(parts[i]) for i in range(3)]
                    # 判断是否为 0-1 范围
                    if max(r, g, b) <= 1.0:
                        return (int(r * 255), int(g * 255), int(b * 255))
                    else:
                        return (int(r), int(g), int(b))
            except (ValueError, IndexError):
                pass
        
        # 尝试解析为十六进制颜色
        hex_color = color_str
        if hex_color.startswith('#'):
            hex_color = hex_color[1:]
        
        if len(hex_color) == 6:
            try:
                r = int(hex_color[0:2], 16)
                g = int(hex_color[2:4], 16)
                b = int(hex_color[4:6], 16)
                return (r, g, b)
            except ValueError:
                pass
        elif len(hex_color) == 3:
            try:
                r = int(hex_color[0], 16) * 17  # 扩展单个十六进制数字
                g = int(hex_color[1], 16) * 17
                b = int(hex_color[2], 16) * 17
                return (r, g, b)
            except ValueError:
                pass
        
        # 尝试解析为颜色名称
        try:
            return ImageColor.getrgb(color_str)
        except ValueError:
            pass
        
        # 默认返回黑色
        return (0, 0, 0)
    
    @staticmethod
    def _parse_pad_color(color_str):
        if color_str is None:
            return (1.0, 1.0, 1.0)
        text = str(color_str).strip().lower()
        if text in ('edge', 'e'):
            return 'edge'
        if text in ('extend', 'ex'):
            return 'extend'
        if text in ('mirror', 'mr'):
            return 'mirror'
        if text in ('average', 'a'):
            return 'average'
        if text.startswith('(') and text.endswith(')'):
            text = text[1:-1].strip()
        single = {
            'r': 'red',
            'g': 'green',
            'b': 'blue',
            'c': 'cyan',
            'm': 'magenta',
            'y': 'yellow',
            'k': 'black',
            'w': 'white',
            'o': 'orange',
            'p': 'purple',
            'n': 'brown',
            's': 'silver',
            'l': 'lime',
            'i': 'indigo',
            'v': 'violet',
            't': 'turquoise',
            'q': 'aqua',
            'f': 'fuchsia',
            'h': 'hotpink',
            'd': 'darkblue',
        }
        if len(text) == 1 and text in single:
            text = single[text]
        try:
            v = float(text)
            if 0.0 <= v <= 1.0:
                return (v, v, v)
        except Exception:
            pass
        if ',' in text:
            try:
                parts = [p.strip() for p in text.split(',')]
                if len(parts) >= 3:
                    r, g, b = [float(parts[i]) for i in range(3)]
                    if max(r, g, b) <= 1.0:
                        return (r, g, b)
                    return (r / 255.0, g / 255.0, b / 255.0)
            except Exception:
                pass
        if text.startswith('#') and len(text) in (4, 7):
            try:
                hex_str = text[1:]
                if len(hex_str) == 3:
                    hex_str = ''.join(ch * 2 for ch in hex_str)
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
    def _get_average_color_tensor(img_tensor):
        """计算tensor图像的平均颜色"""
        # img_tensor shape: [H, W, C]
        avg_color = torch.mean(img_tensor, dim=(0, 1))  # 在H和W维度上求平均
        avg_color_255 = (avg_color * 255).int().tolist()
        return tuple(avg_color_255)
    
    @staticmethod
    def _get_edge_color_tensor(img_tensor):
        """获取tensor图像边缘的平均颜色"""
        H, W, C = img_tensor.shape
        
        # 获取所有边缘像素
        top_edge = img_tensor[0, :, :]  # 顶部边缘
        bottom_edge = img_tensor[H-1, :, :]  # 底部边缘
        left_edge = img_tensor[:, 0, :]  # 左侧边缘
        right_edge = img_tensor[:, W-1, :]  # 右侧边缘
        
        # 合并所有边缘像素
        all_edges = torch.cat([
            top_edge.reshape(-1, C),
            bottom_edge.reshape(-1, C),
            left_edge.reshape(-1, C),
            right_edge.reshape(-1, C)
        ], dim=0)
        
        # 计算平均颜色
        avg_color = torch.mean(all_edges, dim=0)
        avg_color_255 = (avg_color * 255).int().tolist()
        return tuple(avg_color_255)

