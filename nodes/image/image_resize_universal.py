



from comfy_api.latest import io
import math
import numpy as np
from PIL import Image, ImageColor
import torch
import torch.nn.functional as F

class ImageResizeUniversal(io.ComfyNode):
    """
    图像通用缩放器 - 支持多种纵横比和缩放模式，完整的 mask 处理逻辑
    
    主要功能：
    1. mask 输出端在没有输入时也能正常输出
    2. pad 模式下原图区域为白色，填充区域为黑色
    3. 其他模式下 mask 尺寸与图片一致
    """
    NODE_NAME = "ImageResizeUniversal"
    @classmethod
    def define_schema(cls) -> io.Schema:
        ratio_list = ['origin', 'custom', '1:1', '3:2', '4:3', '16:9', '21:9', '2:3', '3:4', '9:16', '9:21']
        fit_mode = ['crop', 'pad', 'stretch']
        method_mode = ['nearest', 'bilinear', 'lanczos', 'bicubic', 'hamming', 'box']
        scale_to_list = ['None', 'longest', 'shortest', 'width', 'height', 'length_to_sq_area']
        return io.Schema(
            node_id="1hew_ImageResizeUniversal",
            display_name="Image Resize Universal",
            category="1hewNodes/image",
            inputs=[
                io.Image.Input("image", optional=True),
                io.Mask.Input("mask", optional=True),
                io.Image.Input("get_image_size", optional=True),
                io.Combo.Input("preset_ratio", options=ratio_list, default="origin"),
                io.Int.Input("proportional_width", default=1, min=1, max=8192, step=1),
                io.Int.Input("proportional_height", default=1, min=1, max=8192, step=1),
                io.Combo.Input("method", options=method_mode, default="lanczos"),
                io.Combo.Input("scale_to_side", options=scale_to_list, default="None"),
                io.Int.Input("scale_to_length", default=1024, min=1, max=8192, step=1),
                io.Combo.Input("fit", options=fit_mode, default="crop"),
                io.String.Input("pad_color", default="1.0"),
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
        preset_ratio: str,
        proportional_width: int,
        proportional_height: int,
        method: str,
        scale_to_side: str,
        scale_to_length: int,
        fit: str,
        pad_color: str,
        divisible_by: int,
        image: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        get_image_size: torch.Tensor | None = None,
        ) -> io.NodeOutput:
        orig_images = []
        orig_masks = []
        orig_width = 0
        orig_height = 0
        target_width = 0
        target_height = 0
        ratio = 1.0
        ret_images = []
        ret_masks = []
        
        # 处理输入图像
        if image is not None:
            for i in image:
                i = torch.unsqueeze(i, 0)
                orig_images.append(i)
            orig_width, orig_height = cls.tensor2pil(orig_images[0]).size
            
        # 处理输入遮罩
        if mask is not None:
            if mask.dim() == 2:
                mask = torch.unsqueeze(mask, 0)
            for m in mask:
                m = torch.unsqueeze(m, 0)
                if not cls.is_valid_mask(m) and m.shape == torch.Size([1, 64, 64]):
                    cls.log(f"警告: {cls.NODE_NAME} 输入遮罩为空，已忽略。", message_type='warning')
                else:
                    orig_masks.append(m)

            if len(orig_masks) > 0:
                _width, _height = cls.tensor2pil(orig_masks[0]).size
                if (orig_width > 0 and orig_width != _width) or (orig_height > 0 and orig_height != _height):
                    cls.log(f"错误: {cls.NODE_NAME} 执行失败，因为遮罩与图像尺寸不匹配。", message_type='error')
                    return io.NodeOutput(None, None)
                elif orig_width + orig_height == 0:
                    orig_width = _width
                    orig_height = _height

        # 允许无输入场景，通过 preset_ratio/scale_to_side 或 get_image_size 推导尺寸

        # 确定目标尺寸
        if get_image_size is not None:
            # 从输入图像获取尺寸
            size_img = cls.tensor2pil(get_image_size[0])
            target_width, target_height = size_img.size
            cls.log(f"使用输入图像尺寸: {target_width}x{target_height}", message_type='info')
        else:
            # 根据纵横比计算目标尺寸
            if preset_ratio == 'origin':
                ratio = orig_width / orig_height
            elif preset_ratio == 'custom':
                ratio = proportional_width / proportional_height
            else:
                s = preset_ratio.split(":")
                ratio = int(s[0]) / int(s[1])

            # 根据不同缩放模式计算目标宽度和高度
            if ratio > 1:  # 宽大于高
                if scale_to_side == 'longest':
                    target_width = scale_to_length
                    target_height = int(target_width / ratio)
                elif scale_to_side == 'shortest':
                    target_height = scale_to_length
                    target_width = int(target_height * ratio)
                elif scale_to_side == 'width':
                    target_width = scale_to_length
                    target_height = int(target_width / ratio)
                elif scale_to_side == 'height':
                    target_height = scale_to_length
                    target_width = int(target_height * ratio)
                elif scale_to_side == 'length_to_sq_area':
                    target_width = math.sqrt(ratio) * scale_to_length
                    target_height = target_width / ratio
                    target_width = int(target_width)
                    target_height = int(target_height)
                else:  # 'None'
                    target_width = orig_width
                    target_height = int(target_width / ratio)
            else:  # 高大于或等于宽
                if scale_to_side == 'longest':
                    target_height = scale_to_length
                    target_width = int(target_height * ratio)
                elif scale_to_side == 'shortest':
                    target_width = scale_to_length
                    target_height = int(target_width / ratio)
                elif scale_to_side == 'width':
                    target_width = scale_to_length
                    target_height = int(target_width / ratio)
                elif scale_to_side == 'height':
                    target_height = scale_to_length
                    target_width = int(target_height * ratio)
                elif scale_to_side == 'length_to_sq_area':
                    target_width = math.sqrt(ratio) * scale_to_length
                    target_height = target_width / ratio
                    target_width = int(target_width)
                    target_height = int(target_height)
                else:  # 'None'
                    target_height = orig_height
                    target_width = int(target_height * ratio)

        # 确保尺寸能被 divisible_by 整除
        if divisible_by > 1:
            target_width = cls.num_round_up_to_multiple(target_width, divisible_by)
            target_height = cls.num_round_up_to_multiple(target_height, divisible_by)

        # 设置缩放采样方法
        resize_sampler = Image.LANCZOS
        if method == "bicubic":
            resize_sampler = Image.BICUBIC
        elif method == "hamming":
            resize_sampler = Image.HAMMING
        elif method == "bilinear":
            resize_sampler = Image.BILINEAR
        elif method == "box":
            resize_sampler = Image.BOX
        elif method == "nearest":
            resize_sampler = Image.NEAREST

        # 处理图像缩放
        if len(orig_images) > 0:
            for i in orig_images:
                _image = cls.tensor2pil(i).convert('RGB')
                _image = cls.fit_resize_image(_image, target_width, target_height, fit, resize_sampler, pad_color)
                ret_images.append(cls.pil2tensor(_image))
        else:
            # 无图像输入时也输出图像：按目标尺寸生成纯色图
            bg = ImageResizeUniversal.parse_color(pad_color) if pad_color is not None else (255, 255, 255)
            if isinstance(bg, str) and bg in ['edge', 'average', 'extend', 'mirror']:
                bg = (255, 255, 255)
            if target_width > 0 and target_height > 0:
                if len(orig_masks) > 0:
                    for _ in orig_masks:
                        blank = Image.new('RGB', (target_width, target_height), bg)
                        ret_images.append(cls.pil2tensor(blank))
                else:
                    blank = Image.new('RGB', (target_width, target_height), bg)
                    ret_images.append(cls.pil2tensor(blank))
                
        # 处理遮罩缩放逻辑
        if len(orig_masks) > 0:
            # 有输入 mask 的情况
            for m in orig_masks:
                _mask = cls.tensor2pil(m).convert('L')
                _mask = cls.fit_resize_mask(_mask, target_width, target_height, fit, resize_sampler, orig_width, orig_height)
                ret_masks.append(cls.image2mask(_mask))
        else:
            # 没有输入 mask 时，根据图像数量生成对应的 mask
            if len(orig_images) > 0:
                for _ in orig_images:
                    _mask = cls.generate_default_mask(target_width, target_height, fit, orig_width, orig_height)
                    ret_masks.append(cls.image2mask(_mask))
            else:
                # 只有尺寸信息或无任何输入，生成默认 mask
                _mask = cls.generate_default_mask(target_width, target_height, fit, orig_width, orig_height)
                ret_masks.append(cls.image2mask(_mask))
                
        # 返回结果 - 确保总是返回 mask
        if len(ret_images) > 0 and len(ret_masks) > 0:
            cls.log(f"{cls.NODE_NAME} 已处理 {len(ret_images)} 张图像和 {len(ret_masks)} 张遮罩。", message_type='finish')
            return io.NodeOutput(torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0))
        elif len(ret_images) > 0 and len(ret_masks) == 0:
            cls.log(f"{cls.NODE_NAME} 已处理 {len(ret_images)} 张图像。", message_type='finish')
            return io.NodeOutput(torch.cat(ret_images, dim=0), None)
        elif len(ret_images) == 0 and len(ret_masks) > 0:
            cls.log(f"{cls.NODE_NAME} 已处理 {len(ret_masks)} 张遮罩。", message_type='finish')
            return io.NodeOutput(None, torch.cat(ret_masks, dim=0))
        else:
            # 若仍无法确定输出，返回空
            cls.log(f"错误: {cls.NODE_NAME} 跳过，因为没有找到可用的图像或遮罩。", message_type='error')
            return io.NodeOutput(None, None)

    @staticmethod
    def generate_default_mask(target_width, target_height, fit_mode, orig_width, orig_height):
        """
        生成默认 mask
        - pad 模式：原图区域为白色，填充区域为黑色
        - 其他模式：按模式生成
        """
        if fit_mode == 'pad':
            if orig_width <= 0 or orig_height <= 0:
                return Image.new('L', (target_width, target_height), 255)
            # 计算原图在目标尺寸中的位置和大小
            orig_ratio = orig_width / orig_height
            target_ratio = target_width / target_height
            
            # 创建黑色背景
            mask = Image.new('L', (target_width, target_height), 0)
            
            if orig_ratio > target_ratio:
                # 原图更宽，按宽度缩放
                new_width = target_width
                new_height = int(new_width / orig_ratio)
                pad_top = (target_height - new_height) // 2
                
                # 在原图区域填充白色
                white_region = Image.new('L', (new_width, new_height), 255)
                mask.paste(white_region, (0, pad_top))
            else:
                # 原图更高，按高度缩放
                new_height = target_height
                new_width = int(new_height * orig_ratio)
                pad_left = (target_width - new_width) // 2
                
                # 在原图区域填充白色
                white_region = Image.new('L', (new_width, new_height), 255)
                mask.paste(white_region, (pad_left, 0))
                
            return mask
        elif fit_mode == 'crop':
            if orig_width <= 0 or orig_height <= 0:
                return Image.new('L', (target_width, target_height), 255)
            orig_ratio = orig_width / orig_height
            target_ratio = target_width / target_height
            if orig_ratio > target_ratio:
                scale = target_height / orig_height
                new_width = int(orig_width * scale)
                left_resized = max(0, (new_width - target_width) // 2)
                left_original = int(left_resized / scale)
                crop_w_original = int(target_width / scale)
                crop_w_original = max(0, min(crop_w_original, orig_width))
                if left_original + crop_w_original > orig_width:
                    crop_w_original = orig_width - left_original
                mask = Image.new('L', (orig_width, orig_height), 0)
                white_region = Image.new('L', (crop_w_original, orig_height), 255)
                mask.paste(white_region, (left_original, 0))
                return mask
            elif orig_ratio < target_ratio:
                scale = target_width / orig_width
                new_height = int(orig_height * scale)
                top_resized = max(0, (new_height - target_height) // 2)
                top_original = int(top_resized / scale)
                crop_h_original = int(target_height / scale)
                crop_h_original = max(0, min(crop_h_original, orig_height))
                if top_original + crop_h_original > orig_height:
                    crop_h_original = orig_height - top_original
                mask = Image.new('L', (orig_width, orig_height), 0)
                white_region = Image.new('L', (orig_width, crop_h_original), 255)
                mask.paste(white_region, (0, top_original))
                return mask
            else:
                return Image.new('L', (orig_width, orig_height), 255)
        else:
            return Image.new('L', (target_width, target_height), 255)

    @staticmethod
    def fit_resize_mask(mask, target_width, target_height, fit_mode, resize_sampler, orig_width, orig_height):
        """
        根据不同适应模式调整 mask 大小
        """
        if fit_mode == 'pad':
            # pad 模式下的特殊处理
            orig_ratio = orig_width / orig_height
            target_ratio = target_width / target_height
            
            # 创建黑色背景
            result_mask = Image.new('L', (target_width, target_height), 0)
            
            if orig_ratio > target_ratio:
                # 原图更宽，按宽度缩放
                new_width = target_width
                new_height = int(new_width / orig_ratio)
                resized_mask = mask.resize((new_width, new_height), resize_sampler)
                pad_top = (target_height - new_height) // 2
                result_mask.paste(resized_mask, (0, pad_top))
            else:
                # 原图更高，按高度缩放
                new_height = target_height
                new_width = int(new_height * orig_ratio)
                resized_mask = mask.resize((new_width, new_height), resize_sampler)
                pad_left = (target_width - new_width) // 2
                result_mask.paste(resized_mask, (pad_left, 0))
                
            return result_mask
        else:
            # stretch 和 crop 模式：直接使用原有逻辑
            return ImageResizeUniversal.fit_resize_image(mask, target_width, target_height, fit_mode, resize_sampler).convert('L')

    @staticmethod
    def parse_color(color_str):
        """解析不同格式的颜色输入"""
        if not color_str:
            return (255, 255, 255)
        
        if color_str.lower() in ['edge', 'e']:
            return 'edge'
        if color_str.lower() in ['average', 'a']:
            return 'average'
        if color_str.lower() in ['extend', 'ex']:
            return 'extend'
        if color_str.lower() in ['mirror', 'mr']:
            return 'mirror'
        
        # 移除括号（如果存在）
        color_str = color_str.strip()
        if color_str.startswith('(') and color_str.endswith(')'):
            color_str = color_str[1:-1].strip()
        
        # 单字母颜色缩写映射
        single_letter_colors = {
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
            'd': 'darkblue'
        }
        
        # 检查是否为单字母颜色缩写
        if len(color_str) == 1 and color_str.lower() in single_letter_colors:
            color_str = single_letter_colors[color_str.lower()]
        
        # 尝试解析为灰度值
        try:
            gray = float(color_str)
            return (int(gray * 255), int(gray * 255), int(gray * 255))
        except ValueError:
            pass
        
        # 尝试解析为 RGB 格式 (如 "0.5,0.7,0.9" 或 "128,192,255")
        if ',' in color_str:
            try:
                # 分割并清理每个部分
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
        
        # 尝试解析为十六进制或颜色名称
        try:
            return ImageColor.getrgb(color_str)
        except ValueError:
            return (255, 255, 255)

    @staticmethod
    def get_average_color(img):
        arr = np.array(img.convert('RGB'))
        v = arr.reshape(-1, 3).mean(axis=0)
        return tuple(int(x) for x in v)

    @staticmethod
    def get_edge_color(img, side):
        """获取图像边缘的平均颜色"""
        width, height = img.size
        img = img.convert('RGB')
        
        if side == 'left':
            edge = img.crop((0, 0, 1, height))
        elif side == 'right':
            edge = img.crop((width-1, 0, width, height))
        elif side == 'top':
            edge = img.crop((0, 0, width, 1))
        elif side == 'bottom':
            edge = img.crop((0, height-1, width, height))
        else:
            # 所有边缘
            top = np.array(img.crop((0, 0, width, 1)))
            bottom = np.array(img.crop((0, height-1, width, height)))
            left = np.array(img.crop((0, 0, 1, height)))
            right = np.array(img.crop((width-1, 0, width, height)))
            
            # 合并所有边缘像素并计算平均值
            all_edges = np.vstack([top.reshape(-1, 3), bottom.reshape(-1, 3), 
                                left.reshape(-1, 3), right.reshape(-1, 3)])
            return tuple(np.mean(all_edges, axis=0).astype(int))
        
        # 计算平均颜色
        edge_array = np.array(edge)
        return tuple(np.mean(edge_array.reshape(-1, 3), axis=0).astype(int))

    @staticmethod
    def fit_resize_image(img, target_width, target_height, fit_mode, resize_sampler, background_color=None):
        """根据不同适应模式调整图像大小"""
        # 解析背景颜色
        bg_color = ImageResizeUniversal.parse_color(background_color) if background_color else (0, 0, 0)
        
        # 获取原始尺寸
        orig_width, orig_height = img.size
        
        # 计算宽高比
        orig_ratio = orig_width / orig_height
        target_ratio = target_width / target_height
        
        # 根据适应模式处理图像
        if fit_mode == 'stretch':  # 直接拉伸
            # 直接拉伸到目标尺寸
            return img.resize((target_width, target_height), resize_sampler)
        
        elif fit_mode == 'crop':  # 裁剪
            # 调整大小保持比例，然后裁剪
            if orig_ratio > target_ratio:
                # 原图更宽，按高度缩放后裁剪宽度
                new_height = target_height
                new_width = int(new_height * orig_ratio)
                img = img.resize((new_width, new_height), resize_sampler)
                # 居中裁剪
                left = (new_width - target_width) // 2
                img = img.crop((left, 0, left + target_width, new_height))
            else:
                # 原图更高，按宽度缩放后裁剪高度
                new_width = target_width
                new_height = int(new_width / orig_ratio)
                img = img.resize((new_width, new_height), resize_sampler)
                # 居中裁剪
                top = (new_height - target_height) // 2
                img = img.crop((0, top, new_width, top + target_height))
            return img
        
        elif fit_mode == 'pad':  # 填充
            # 调整大小保持比例，然后填充
            if orig_ratio > target_ratio:
                # 原图更宽，按宽度缩放后填充高度
                new_width = target_width
                new_height = int(new_width / orig_ratio)
                img = img.resize((new_width, new_height), resize_sampler)
                
                # 计算需要填充的像素数
                pad_top = (target_height - new_height) // 2
                pad_bottom = target_height - new_height - pad_top
                
                if bg_color == 'extend':
                    arr = np.array(img).astype(np.float32) / 255.0
                    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
                    padded = F.pad(t, (0, 0, pad_top, pad_bottom), mode='replicate')
                    out = (padded.squeeze(0).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
                    return Image.fromarray(out)
                elif bg_color == 'mirror':
                    arr = np.array(img).astype(np.float32) / 255.0
                    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
                    pt = pad_top
                    pb = pad_bottom
                    while pt > 0 or pb > 0:
                        hh = int(t.shape[2])
                        st = min(pt, max(hh - 1, 0))
                        sb = min(pb, max(hh - 1, 0))
                        t = F.pad(t, (0, 0, st, sb), mode='reflect')
                        pt -= st
                        pb -= sb
                    out = (t.squeeze(0).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
                    return Image.fromarray(out)
                if bg_color == 'edge':
                    top_color = ImageResizeUniversal.get_edge_color(img, 'top')
                    bottom_color = ImageResizeUniversal.get_edge_color(img, 'bottom')
                    
                    # 创建填充图像
                    padded = Image.new('RGB', (target_width, target_height), (0, 0, 0))
                    # 填充顶部
                    if pad_top > 0:
                        top_pad = Image.new('RGB', (target_width, pad_top), top_color)
                        padded.paste(top_pad, (0, 0))
                    # 粘贴原图
                    padded.paste(img, (0, pad_top))
                    # 填充底部
                    if pad_bottom > 0:
                        bottom_pad = Image.new('RGB', (target_width, pad_bottom), bottom_color)
                        padded.paste(bottom_pad, (0, pad_top + new_height))
                elif bg_color == 'average':
                    avg_color = ImageResizeUniversal.get_average_color(img)
                    padded = Image.new('RGB', (target_width, target_height), avg_color)
                    padded.paste(img, (0, pad_top))
                else:
                    padded = Image.new('RGB', (target_width, target_height), bg_color)
                    padded.paste(img, (0, pad_top))
            else:
                # 原图更高，按高度缩放后填充宽度
                new_height = target_height
                new_width = int(new_height * orig_ratio)
                img = img.resize((new_width, new_height), resize_sampler)
                
                # 计算需要填充的像素数
                pad_left = (target_width - new_width) // 2
                pad_right = target_width - new_width - pad_left
                
                if bg_color == 'extend':
                    arr = np.array(img).astype(np.float32) / 255.0
                    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
                    padded = F.pad(t, (pad_left, pad_right, 0, 0), mode='replicate')
                    out = (padded.squeeze(0).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
                    return Image.fromarray(out)
                elif bg_color == 'mirror':
                    arr = np.array(img).astype(np.float32) / 255.0
                    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
                    pl = pad_left
                    pr = pad_right
                    while pl > 0 or pr > 0:
                        ww = int(t.shape[3])
                        sl = min(pl, max(ww - 1, 0))
                        sr = min(pr, max(ww - 1, 0))
                        t = F.pad(t, (sl, sr, 0, 0), mode='reflect')
                        pl -= sl
                        pr -= sr
                    out = (t.squeeze(0).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
                    return Image.fromarray(out)
                if bg_color == 'edge':
                    left_color = ImageResizeUniversal.get_edge_color(img, 'left')
                    right_color = ImageResizeUniversal.get_edge_color(img, 'right')
                    
                    # 创建填充图像
                    padded = Image.new('RGB', (target_width, target_height), (0, 0, 0))
                    # 填充左侧
                    if pad_left > 0:
                        left_pad = Image.new('RGB', (pad_left, target_height), left_color)
                        padded.paste(left_pad, (0, 0))
                    # 粘贴原图
                    padded.paste(img, (pad_left, 0))
                    # 填充右侧
                    if pad_right > 0:
                        right_pad = Image.new('RGB', (pad_right, target_height), right_color)
                        padded.paste(right_pad, (pad_left + new_width, 0))
                elif bg_color == 'average':
                    avg_color = ImageResizeUniversal.get_average_color(img)
                    padded = Image.new('RGB', (target_width, target_height), avg_color)
                    padded.paste(img, (pad_left, 0))
                else:
                    padded = Image.new('RGB', (target_width, target_height), bg_color)
                    padded.paste(img, (pad_left, 0))
            
            return padded
        
        # 默认情况下直接调整大小
        return img.resize((target_width, target_height), resize_sampler)

    @staticmethod
    def tensor2pil(image):
        """将张量转换为PIL图像"""
        return Image.fromarray(np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

    @staticmethod
    def pil2tensor(image):
        """将PIL图像转换为张量"""
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

    @staticmethod
    def image2mask(image):
        """将图像转换为遮罩"""
        return torch.from_numpy(np.array(image.convert("L")).astype(np.float32) / 255.0).unsqueeze(0)

    @staticmethod
    def is_valid_mask(mask):
        """检查遮罩是否有效"""
        if mask is None:
            return False
        return torch.min(mask) < 0.9999

    @staticmethod
    def num_round_up_to_multiple(num, multiple):
        """将数字向上取整到指定倍数"""
        return math.ceil(num / multiple) * multiple

    @staticmethod
    def log(message, message_type='info'):
        """输出日志信息"""
        print(f"[{message_type.upper()}] {message}")