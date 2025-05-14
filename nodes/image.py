import torch
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import os
import math


class ImageResizeUniversal:
    """
    图像通用缩放器 - 支持多种纵横比和缩放模式，可以按照不同方式调整图像大小
    """
    
    NODE_NAME = "ImageResizeUniversal"
    
    @classmethod
    def INPUT_TYPES(cls):
        ratio_list = ['original', 'custom', '1:1', '3:2', '4:3', '16:9', '21:9', '2:3', '3:4', '9:16', '9:21',]
        fit_mode = ['stretch', 'crop', 'pad']
        method_mode = ['nearest', 'bilinear', 'lanczos', 'bicubic', 'hamming',  'box']
        scale_to_list = ['None', 'longest', 'shortest', 'width', 'height', 'total_pixel']
        return {
            "required": {
                "aspect_ratio": (ratio_list,),
                "proportional_width": ("INT", {"default": 1, "min": 1, "max": 1e8, "step": 1}),
                "proportional_height": ("INT", {"default": 1, "min": 1, "max": 1e8, "step": 1}),
                "method": (method_mode, {"default": 'lanczos'}),
                "scale_to_side": (scale_to_list, {"default": 'None', "label": "按边缩放"}),
                "scale_to_length": ("INT", {"default": 1024, "min": 4, "max": 1e8, "step": 1}),
                "fit": (fit_mode, {"default": "crop", "label": "适应方式"}),
                "pad_color": ("STRING", {"default": "1.0", "label": "背景颜色 (灰度/HEX/RGB/edge)"}),
                "divisible_by": ("INT", {"default": 8, "min": 1, "max": 1024, "step": 1, "label": "尺寸整除数"}),
            },
            "optional": {
                "image": ("IMAGE",),  
                "mask": ("MASK",),  
                "get_image_size": ("IMAGE",), 
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = 'image_resize'
    CATEGORY = '1hewNodes/image'

    def image_resize(self, aspect_ratio, proportional_width, proportional_height,
                     fit, method, divisible_by, scale_to_side, scale_to_length,
                     pad_color,
                     image=None, mask=None, get_image_size=None
                     ):
        # 初始化变量
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
            orig_width, orig_height = self.tensor2pil(orig_images[0]).size
            
        # 处理输入遮罩
        if mask is not None:
            if mask.dim() == 2:
                mask = torch.unsqueeze(mask, 0)
            for m in mask:
                m = torch.unsqueeze(m, 0)
                if not self.is_valid_mask(m) and m.shape==torch.Size([1,64,64]):
                    self.log(f"警告: {self.NODE_NAME} 输入遮罩为空，已忽略。", message_type='warning')
                else:
                    orig_masks.append(m)

            if len(orig_masks) > 0:
                _width, _height = self.tensor2pil(orig_masks[0]).size
                if (orig_width > 0 and orig_width != _width) or (orig_height > 0 and orig_height != _height):
                    self.log(f"错误: {self.NODE_NAME} 执行失败，因为遮罩与图像尺寸不匹配。", message_type='error')
                    return (None, None)
                elif orig_width + orig_height == 0:
                    orig_width = _width
                    orig_height = _height

        # 检查是否有有效输入
        if orig_width + orig_height == 0:
            self.log(f"错误: {self.NODE_NAME} 执行失败，至少需要输入图像或遮罩。", message_type='error')
            return (None, None)

        # 确定目标尺寸
        if get_image_size is not None:
            # 从输入图像获取尺寸
            size_img = self.tensor2pil(get_image_size[0])
            target_width, target_height = size_img.size
            self.log(f"使用输入图像尺寸: {target_width}x{target_height}", message_type='info')
        else:
            # 根据纵横比计算目标尺寸
            if aspect_ratio == 'original':
                ratio = orig_width / orig_height
            elif aspect_ratio == 'custom':
                ratio = proportional_width / proportional_height
            else:
                s = aspect_ratio.split(":")
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
                elif scale_to_side == 'total_pixel':
                    target_width = math.sqrt(ratio * scale_to_length * 1000)
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
                elif scale_to_side == 'total_pixel':
                    target_width = math.sqrt(ratio * scale_to_length * 1000)
                    target_height = target_width / ratio
                    target_width = int(target_width)
                    target_height = int(target_height)
                else:  # 'None'
                    target_height = orig_height
                    target_width = int(target_height * ratio)

        # 确保尺寸能被 divisible_by 整除
        if divisible_by > 1:
            target_width = self.num_round_up_to_multiple(target_width, divisible_by)
            target_height = self.num_round_up_to_multiple(target_height, divisible_by)

        # 创建默认图像和遮罩
        _mask = Image.new('L', size=(target_width, target_height), color='black')
        _image = Image.new('RGB', size=(target_width, target_height), color='black')

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
                _image = self.tensor2pil(i).convert('RGB')
                _image = self.fit_resize_image(_image, target_width, target_height, fit, resize_sampler, pad_color)
                ret_images.append(self.pil2tensor(_image))
                
        # 处理遮罩缩放
        if len(orig_masks) > 0:
            for m in orig_masks:
                _mask = self.tensor2pil(m).convert('L')
                _mask = self.fit_resize_image(_mask, target_width, target_height, fit, resize_sampler).convert('L')
                ret_masks.append(self.image2mask(_mask))
                
        # 返回结果
        if len(ret_images) > 0 and len(ret_masks) > 0:
            self.log(f"{self.NODE_NAME} 已处理 {len(ret_images)} 张图像。", message_type='finish')
            return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0))
        elif len(ret_images) > 0 and len(ret_masks) == 0:
            self.log(f"{self.NODE_NAME} 已处理 {len(ret_images)} 张图像。", message_type='finish')
            return (torch.cat(ret_images, dim=0), None)
        elif len(ret_images) == 0 and len(ret_masks) > 0:
            self.log(f"{self.NODE_NAME} 已处理 {len(ret_masks)} 张遮罩。", message_type='finish')
            return (None, torch.cat(ret_masks, dim=0))
        else:
            self.log(f"错误: {self.NODE_NAME} 跳过，因为没有找到可用的图像或遮罩。", message_type='error')
            return (None, None)

    def parse_color(self, color_str):
        """解析不同格式的颜色输入"""
        if not color_str:
            return (0, 0, 0)
        
        # 检查是否为 edge 或 e (不区分大小写)
        if color_str.lower() in ['edge', 'e']:
            return 'edge'
        
        # 移除括号（如果存在）
        color_str = color_str.strip()
        if color_str.startswith('(') and color_str.endswith(')'):
            color_str = color_str[1:-1].strip()
        
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
            # 默认返回黑色
            return (0, 0, 0)

    def get_edge_color(self, img, side):
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

    def fit_resize_image(self, img, target_width, target_height, fit_mode, resize_sampler, background_color=None):
        """根据不同适应模式调整图像大小"""
        # 解析背景颜色
        bg_color = self.parse_color(background_color) if background_color else (0, 0, 0)
        
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
                
                # 处理边缘颜色填充
                if bg_color == 'edge':
                    top_color = self.get_edge_color(img, 'top')
                    bottom_color = self.get_edge_color(img, 'bottom')
                    
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
                else:
                    # 使用指定颜色填充
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
                
                # 处理边缘颜色填充
                if bg_color == 'edge':
                    left_color = self.get_edge_color(img, 'left')
                    right_color = self.get_edge_color(img, 'right')
                    
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
                else:
                    # 使用指定颜色填充
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


class ImageEditStitch:
    """
    图像编辑缝合 - 将参考图像和编辑图像拼接在一起，支持上下左右四种拼接方式
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_image": ("IMAGE",),
                "edit_image": ("IMAGE",),
                "position": (["top", "bottom", "left", "right"], {"default": "right", "label": "拼接位置"}),
                "match_size": ("BOOLEAN", {"default": True, "label": "匹配尺寸"}),
                "fill_color": (
                "FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "label": "填充颜色(0黑-1白)"})
            },
            "optional": {
                "edit_mask": ("MASK",)
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("image", "mask", "split_mask")
    FUNCTION = "image_edit_stitch"
    CATEGORY = "1hewNodes/image"

    def image_edit_stitch(self, reference_image, edit_image, edit_mask=None, position='right', match_size=True,
                          fill_color=1.0):
        # 检查输入
        if reference_image is None and edit_image is None:
            # 如果两个图像都为空，创建默认图像
            default_image = torch.ones((1, 512, 512, 3), dtype=torch.float32)
            default_mask = torch.ones((1, 512, 512), dtype=torch.float32)
            return default_image, default_mask, default_mask

        # 如果只有一个图像存在，直接返回该图像
        if reference_image is None:
            # 如果没有编辑遮罩，创建全白遮罩
            if edit_mask is None:
                edit_mask = torch.ones((1, edit_image.shape[1], edit_image.shape[2]), dtype=torch.float32)
            # 创建分离遮罩（全黑，表示全部是编辑区域）
            split_mask = torch.zeros_like(edit_mask)
            return edit_image, edit_mask, split_mask

        if edit_image is None:
            # 创建与参考图像相同尺寸的空白图像
            edit_image = torch.zeros_like(reference_image)
            # 创建全白遮罩
            white_mask = torch.ones((1, reference_image.shape[1], reference_image.shape[2]), dtype=torch.float32)
            # 创建分离遮罩（全白，表示全部是参考区域）
            split_mask = torch.ones_like(white_mask)
            return reference_image, white_mask, split_mask

        # 确保编辑遮罩存在，如果不存在则创建全白遮罩
        if edit_mask is None:
            edit_mask = torch.ones((1, edit_image.shape[1], edit_image.shape[2]), dtype=torch.float32)

        # 获取图像尺寸
        ref_batch, ref_height, ref_width, ref_channels = reference_image.shape
        edit_batch, edit_height, edit_width, edit_channels = edit_image.shape

        # 处理尺寸不匹配的情况
        if match_size and (ref_height != edit_height or ref_width != edit_width):
            # 将图像转换为PIL格式以便于处理
            if reference_image.is_cuda:
                ref_np = (reference_image[0].cpu().numpy() * 255).astype(np.uint8)
            else:
                ref_np = (reference_image[0].numpy() * 255).astype(np.uint8)

            ref_pil = Image.fromarray(ref_np)

            # 计算等比例缩放的尺寸
            ref_aspect = ref_width / ref_height
            edit_aspect = edit_width / edit_height

            # 等比例缩放参考图像以匹配编辑图像
            if ref_aspect > edit_aspect:
                # 宽度优先
                new_width = edit_width
                new_height = int(edit_width / ref_aspect)
            else:
                # 高度优先
                new_height = edit_height
                new_width = int(edit_height * ref_aspect)

            # 调整参考图像大小，保持纵横比
            ref_pil = ref_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # 创建一个与编辑图像相同大小的填充颜色图像
            fill_color_rgb = int(fill_color * 255)
            new_ref_pil = Image.new("RGB", (edit_width, edit_height), (fill_color_rgb, fill_color_rgb, fill_color_rgb))

            # 将调整大小后的参考图像粘贴到中心位置
            paste_x = (edit_width - new_width) // 2
            paste_y = (edit_height - new_height) // 2
            new_ref_pil.paste(ref_pil, (paste_x, paste_y))

            # 转换回tensor
            ref_np = np.array(new_ref_pil).astype(np.float32) / 255.0
            reference_image = torch.from_numpy(ref_np).unsqueeze(0)

            # 更新尺寸
            ref_height, ref_width = edit_height, edit_width

        # 根据位置拼接图像
        if position == "right":
            # 参考图像在左，编辑图像在右
            combined_image = torch.cat([
                reference_image,
                edit_image
            ], dim=2)  # 水平拼接

            # 拼接遮罩（参考区域为0，编辑区域保持原样）
            zero_mask = torch.zeros((1, ref_height, ref_width), dtype=torch.float32)
            combined_mask = torch.cat([zero_mask, edit_mask], dim=2)

            # 创建分离遮罩（参考区域为黑色，编辑区域为白色）
            split_mask_left = torch.zeros((1, ref_height, ref_width), dtype=torch.float32)
            split_mask_right = torch.ones((1, edit_height, edit_width), dtype=torch.float32)
            split_mask = torch.cat([split_mask_left, split_mask_right], dim=2)

        elif position == "left":
            # 编辑图像在左，参考图像在右
            combined_image = torch.cat([
                edit_image,
                reference_image
            ], dim=2)  # 水平拼接

            # 拼接遮罩（编辑区域保持原样，参考区域为0）
            zero_mask = torch.zeros((1, ref_height, ref_width), dtype=torch.float32)
            combined_mask = torch.cat([edit_mask, zero_mask], dim=2)

            # 创建分离遮罩（编辑区域为白色，参考区域为黑色）
            split_mask_left = torch.ones((1, edit_height, edit_width), dtype=torch.float32)
            split_mask_right = torch.zeros((1, ref_height, ref_width), dtype=torch.float32)
            split_mask = torch.cat([split_mask_left, split_mask_right], dim=2)

        elif position == "bottom":
            # 参考图像在上，编辑图像在下
            combined_image = torch.cat([
                reference_image,
                edit_image
            ], dim=1)  # 垂直拼接

            # 拼接遮罩（参考区域为0，编辑区域保持原样）
            zero_mask = torch.zeros((1, ref_height, ref_width), dtype=torch.float32)
            combined_mask = torch.cat([zero_mask, edit_mask], dim=1)

            # 创建分离遮罩（参考区域为黑色，编辑区域为白色）
            split_mask_top = torch.zeros((1, ref_height, ref_width), dtype=torch.float32)
            split_mask_bottom = torch.ones((1, edit_height, edit_width), dtype=torch.float32)
            split_mask = torch.cat([split_mask_top, split_mask_bottom], dim=1)

        elif position == "top":
            # 编辑图像在上，参考图像在下
            combined_image = torch.cat([
                edit_image,
                reference_image
            ], dim=1)  # 垂直拼接

            # 拼接遮罩（编辑区域保持原样，参考区域为0）
            zero_mask = torch.zeros((1, ref_height, ref_width), dtype=torch.float32)
            combined_mask = torch.cat([edit_mask, zero_mask], dim=1)

            # 创建分离遮罩（编辑区域为白色，参考区域为黑色）
            split_mask_top = torch.ones((1, edit_height, edit_width), dtype=torch.float32)
            split_mask_bottom = torch.zeros((1, ref_height, ref_width), dtype=torch.float32)
            split_mask = torch.cat([split_mask_top, split_mask_bottom], dim=1)

        return combined_image, combined_mask, split_mask


class ImageCropSquare:
    """
    图像方形裁剪器 - 根据遮罩裁切图像为方形，支持放大系数和填充颜色
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "scale_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.01, "label": "放大系数"}),
                "apply_mask": ("BOOLEAN", {"default": False, "label": "应用遮罩抠图"}),
                "extra_padding": ("INT", {"default": 0, "min": 0, "max": 512, "step": 1, "label": "额外边距(像素)"}),
                "fill_color": ("STRING", {"default": "1.0", "label": "背景颜色 (灰度/HEX/RGB/edge)"})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "image_crop_square"
    CATEGORY = "1hewNodes/image"

    def image_crop_square(self, image, mask, scale_factor=1.0, fill_color="1.0", apply_mask=False, extra_padding=0):
        # 获取图像尺寸
        batch_size, height, width, channels = image.shape

        # 创建输出图像列表
        output_images = []

        for b in range(batch_size):
            # 将图像转换为PIL格式
            if image.is_cuda:
                img_np = (image[b].cpu().numpy() * 255).astype(np.uint8)
                mask_np = (mask[b % mask.shape[0]].cpu().numpy() * 255).astype(np.uint8)
            else:
                img_np = (image[b].numpy() * 255).astype(np.uint8)
                mask_np = (mask[b % mask.shape[0]].numpy() * 255).astype(np.uint8)

            img_pil = Image.fromarray(img_np)
            mask_pil = Image.fromarray(mask_np).convert("L")

            # 调整遮罩大小以匹配图像 - 使用填充而非缩放
            if img_pil.size != mask_pil.size:
                # 创建一个与图像相同大小的空白遮罩
                new_mask = Image.new("L", img_pil.size, 0)

                # 计算居中位置
                paste_x = max(0, (img_pil.width - mask_pil.width) // 2)
                paste_y = max(0, (img_pil.height - mask_pil.height) // 2)

                # 将原始遮罩粘贴到中心位置
                new_mask.paste(mask_pil, (paste_x, paste_y))
                mask_pil = new_mask

            # 找到遮罩中非零区域的边界框
            bbox = self.get_bbox(mask_pil)

            # 如果没有找到有效区域，返回原始图像
            if bbox is None:
                output_images.append(image[b])
                continue

            # 计算边界框的宽度和高度
            bbox_width = bbox[2] - bbox[0]
            bbox_height = bbox[3] - bbox[1]
            
            # 计算中心点
            center_x = (bbox[0] + bbox[2]) // 2
            center_y = (bbox[1] + bbox[3]) // 2
            
            # 计算方形边长 (取最大值)
            square_size = max(bbox_width, bbox_height)
            
            # 应用放大系数
            scaled_size = int(square_size * scale_factor)
            
            # 计算方形边界框的左上角和右下角坐标 (仅应用scale_factor)
            scaled_x1 = center_x - scaled_size // 2
            scaled_y1 = center_y - scaled_size // 2
            scaled_x2 = scaled_x1 + scaled_size
            scaled_y2 = scaled_y1 + scaled_size
            
            # 最终尺寸 (包含额外边距)
            final_size = scaled_size + extra_padding * 2
            
            # 最终边界框坐标
            square_x1 = center_x - final_size // 2
            square_y1 = center_y - final_size // 2
            square_x2 = square_x1 + final_size
            square_y2 = square_y1 + final_size
            
            # 处理填充颜色
            if fill_color.lower() in ["e", "edge"]:
                if apply_mask:
                    # 当应用遮罩时，获取遮罩区域的平均颜色
                    mask_area_color = self.get_mask_area_color(img_pil, mask_pil, bbox)
                    # 创建方形画布 - 使用遮罩区域的平均颜色
                    square_img = Image.new("RGB", (final_size, final_size), mask_area_color)
                else:
                    # 当不应用遮罩时，获取四个边的平均颜色
                    edge_colors = self.get_four_edge_colors(img_pil, scaled_x1, scaled_y1, scaled_x2, scaled_y2)
                    # 创建方形画布 - 使用四边颜色填充
                    square_img = Image.new("RGB", (final_size, final_size), (255, 255, 255))
                    # 填充四个边缘区域
                    self.fill_edges_with_colors(square_img, edge_colors, extra_padding)
            else:
                # 解析填充颜色
                bg_color = self.parse_color(fill_color)
                # 创建方形画布
                square_img = Image.new("RGB", (final_size, final_size), bg_color)
            
            # 计算粘贴位置 - 考虑额外边距
            paste_x = max(0, -scaled_x1) + extra_padding
            paste_y = max(0, -scaled_y1) + extra_padding
            
            # 计算从原图裁剪的区域 - 只考虑scale_factor
            crop_x1 = max(0, scaled_x1)
            crop_y1 = max(0, scaled_y1)
            crop_x2 = min(img_pil.width, scaled_x2)
            crop_y2 = min(img_pil.height, scaled_y2)
            
            # 裁剪原图并粘贴到方形画布上
            if crop_x1 < crop_x2 and crop_y1 < crop_y2:
                cropped_region = img_pil.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                
                # 如果需要应用遮罩抠图
                if apply_mask:
                    # 裁剪遮罩
                    cropped_mask = mask_pil.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                    # 将遮罩应用到裁剪区域
                    if fill_color.lower() in ["e", "edge"]:
                        # 使用遮罩区域的平均颜色作为背景
                        bg_img = Image.new("RGB", cropped_region.size, mask_area_color)
                    else:
                        # 使用指定的填充颜色作为背景
                        bg_img = Image.new("RGB", cropped_region.size, self.parse_color(fill_color))
                    
                    # 合成图像
                    cropped_region = Image.composite(cropped_region, bg_img, cropped_mask)
                
                square_img.paste(cropped_region, (paste_x, paste_y))
            
            # 转换回tensor
            square_img_np = np.array(square_img).astype(np.float32) / 255.0
            output_images.append(torch.from_numpy(square_img_np))

        # 合并批次
        if output_images:
            output_image_tensor = torch.stack(output_images)
            return (output_image_tensor,)
        else:
            # 如果没有有效的输出图像，返回原始图像
            return (image,)

    def get_bbox(self, mask_pil):
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
    
    def get_mask_area_color(self, img, mask, bbox):
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
    
    def get_four_edge_colors(self, img, x1, y1, x2, y2):
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
        top_color = self.calculate_average_color(top_pixels)
        bottom_color = self.calculate_average_color(bottom_pixels)
        left_color = self.calculate_average_color(left_pixels)
        right_color = self.calculate_average_color(right_pixels)
        
        return {
            'top': top_color,
            'bottom': bottom_color,
            'left': left_color,
            'right': right_color
        }
    
    def calculate_average_color(self, pixels):
        """计算像素列表的平均颜色"""
        if not pixels:
            return (255, 255, 255)
        
        r_sum = sum(p[0] for p in pixels)
        g_sum = sum(p[1] for p in pixels)
        b_sum = sum(p[2] for p in pixels)
        
        pixel_count = len(pixels)
        return (r_sum // pixel_count, g_sum // pixel_count, b_sum // pixel_count)
    
    def fill_edges_with_colors(self, img, edge_colors, padding):
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
    
    def get_edge_colors(self, img, x1, y1, x2, y2):
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
    
    def parse_color(self, color_str):
        """解析颜色字符串，支持灰度值、HEX和RGB格式"""
        # 尝试作为灰度值解析
        try:
            gray_value = float(color_str)
            gray_int = int(gray_value * 255)
            return (gray_int, gray_int, gray_int)
        except ValueError:
            pass
        
        # 尝试作为HEX解析
        if color_str.startswith('#'):
            # 移除 # 符号
            color_str = color_str[1:]
            
            # 处理不同长度的HEX
            if len(color_str) == 3:  # 短格式 #RGB
                r = int(color_str[0] + color_str[0], 16)
                g = int(color_str[1] + color_str[1], 16)
                b = int(color_str[2] + color_str[2], 16)
                return (r, g, b)
            elif len(color_str) == 6:  # 标准格式 #RRGGBB
                r = int(color_str[0:2], 16)
                g = int(color_str[2:4], 16)
                b = int(color_str[4:6], 16)
                return (r, g, b)
        
        # 尝试作为RGB格式解析 "255,0,0"
        try:
            rgb_values = color_str.split(',')
            if len(rgb_values) == 3:
                r = int(rgb_values[0].strip())
                g = int(rgb_values[1].strip())
                b = int(rgb_values[2].strip())
                return (r, g, b)
        except ValueError:
            pass
        
        # 默认返回白色
        return (255, 255, 255)


class ImageCropWithBBox:
    """
    图像裁切器 - 根据遮罩裁切图像，并返回边界框信息以便后续粘贴回原位置
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "aspect_ratio": (["mask_ratio", "1:1", "3:2", "4:3", "16:9", "21:9", "2:3", "3:4", "9:16", "9:21"], {"default": "mask_ratio", "label": "输出比例"}),
                "scale_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1, "label": "缩放系数"}),
                "extra_padding": ("INT", {"default": 0, "min": 0, "max": 512, "step": 1, "label": "边距(像素)"}),
                "exceed_image": ("BOOLEAN", {"default": False, "label": "允许超出原图"}),
                "invert_mask": ("BOOLEAN", {"default": False, "label": "反转遮罩"}),
                "fill_color": ("STRING", {"default": "1.0", "label": "背景颜色 (灰度/HEX/RGB/edge)"}),
                "divisible_by": ("INT", {"default": 8, "min": 1, "max": 1024, "step": 1, "label": "尺寸整除数"})
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "CROP_BBOX")  # 将 "STRING" 改为 "CROP_BBOX"
    RETURN_NAMES = ("cropped_image", "cropped_mask", "crop_bbox")
    FUNCTION = "image_crop_with_bbox"
    CATEGORY = "1hewNodes/image"

    def image_crop_with_bbox(self, image, mask, invert_mask=False, extra_padding=0, aspect_ratio="mask_ratio", scale_factor=1.0, 
                            exceed_image=False, fill_color="1.0", divisible_by=8):
        # 获取图像尺寸
        batch_size, height, width, channels = image.shape

        # 创建输出图像和遮罩列表
        output_images = []
        output_masks = []
        output_bboxes_str = []

        for b in range(batch_size):
            # 将图像转换为PIL格式
            if image.is_cuda:
                img_np = (image[b].cpu().numpy() * 255).astype(np.uint8)
                mask_np = (mask[b % mask.shape[0]].cpu().numpy() * 255).astype(np.uint8)
            else:
                img_np = (image[b].numpy() * 255).astype(np.uint8)
                mask_np = (mask[b % mask.shape[0]].numpy() * 255).astype(np.uint8)

            img_pil = Image.fromarray(img_np)
            mask_pil = Image.fromarray(mask_np).convert("L")

            # 调整遮罩大小以匹配图像 - 使用填充而非缩放
            if img_pil.size != mask_pil.size:
                # 创建一个与图像相同大小的空白遮罩
                new_mask = Image.new("L", img_pil.size, 0)

                # 计算居中位置
                paste_x = max(0, (img_pil.width - mask_pil.width) // 2)
                paste_y = max(0, (img_pil.height - mask_pil.height) // 2)

                # 将原始遮罩粘贴到中心位置
                new_mask.paste(mask_pil, (paste_x, paste_y))
                mask_pil = new_mask

            # 如果需要反转遮罩
            if invert_mask:
                mask_pil = ImageOps.invert(mask_pil)

            # 找到遮罩中非零区域的边界框
            bbox = self.get_bbox(mask_pil, extra_padding)

            # 如果没有找到有效区域，返回原始图像
            if bbox is None:
                output_images.append(image[b])
                output_masks.append(mask[b % mask.shape[0]])
                # 使用整个图像作为边界框，并转换为字符串
                output_bboxes_str.append(f"{0},{0},{width},{height}")
                continue

            # 根据选择的比例调整边界框
            if aspect_ratio != "mask_ratio":
                bbox = self.adjust_bbox_aspect_ratio(bbox, aspect_ratio, img_pil.size, exceed_image)
            
            # 应用缩放系数
            if scale_factor != 1.0:
                bbox = self.apply_scale_factor(bbox, scale_factor, img_pil.size, exceed_image)
            
            # 调整尺寸以满足整除要求
            if divisible_by > 1:
                bbox = self.adjust_for_divisibility(bbox, divisible_by, img_pil.size, exceed_image)

            # 获取最终的边界框坐标
            x_min, y_min, x_max, y_max = bbox
            crop_width = x_max - x_min
            crop_height = y_max - y_min
            
            # 如果允许超出图像范围，创建带填充的画布
            if exceed_image and (x_min < 0 or y_min < 0 or x_max > img_pil.width or y_max > img_pil.height):
                # 解析填充颜色
                bg_color = self.parse_color(fill_color, img_pil, bbox)
                
                # 创建新的画布
                canvas_img = Image.new("RGB", (crop_width, crop_height), bg_color)
                canvas_mask = Image.new("L", (crop_width, crop_height), 0)
                
                # 计算原图在新画布上的位置
                paste_x = max(0, -x_min)
                paste_y = max(0, -y_min)
                
                # 计算从原图裁剪的区域
                crop_x1 = max(0, x_min)
                crop_y1 = max(0, y_min)
                crop_x2 = min(img_pil.width, x_max)
                crop_y2 = min(img_pil.height, y_max)
                
                # 裁剪原图并粘贴到新画布上
                if crop_x1 < crop_x2 and crop_y1 < crop_y2:
                    cropped_region = img_pil.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                    cropped_mask_region = mask_pil.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                    
                    canvas_img.paste(cropped_region, (paste_x, paste_y))
                    canvas_mask.paste(cropped_mask_region, (paste_x, paste_y))
                
                cropped_img = canvas_img
                cropped_mask = canvas_mask
            else:
                # 确保边界框不超出图像范围
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(img_pil.width, x_max)
                y_max = min(img_pil.height, y_max)
                
                # 裁切图像和遮罩
                cropped_img = img_pil.crop((x_min, y_min, x_max, y_max))
                cropped_mask = mask_pil.crop((x_min, y_min, x_max, y_max))

            # 转换回tensor
            cropped_img_np = np.array(cropped_img).astype(np.float32) / 255.0
            cropped_mask_np = np.array(cropped_mask).astype(np.float32) / 255.0

            output_images.append(torch.from_numpy(cropped_img_np))
            output_masks.append(torch.from_numpy(cropped_mask_np))
            # 将边界框转换为字符串
            output_bboxes_str.append(f"{x_min},{y_min},{x_max},{y_max}")

        # 合并批次
        output_image_tensor = torch.stack(output_images)
        output_mask_tensor = torch.stack(output_masks)

        return (output_image_tensor, output_mask_tensor, output_bboxes_str)

    def get_bbox(self, mask_pil, extra_padding=0):
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

        # 添加边距
        x_min = max(0, x_min - extra_padding)
        y_min = max(0, y_min - extra_padding)
        x_max = min(mask_pil.width - 1, x_max + extra_padding)
        y_max = min(mask_pil.height - 1, y_max + extra_padding)

        # 返回边界框 (left, top, right, bottom)
        return (x_min, y_min, x_max + 1, y_max + 1)
    
    def adjust_bbox_aspect_ratio(self, bbox, aspect_ratio, img_size, exceed_image=False):
        """根据指定的宽高比调整边界框"""
        x_min, y_min, x_max, y_max = bbox
        
        # 计算当前宽高
        width = x_max - x_min
        height = y_max - y_min
        
        # 计算中心点
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        
        # 解析目标宽高比
        if aspect_ratio == "1:1":
            target_ratio = 1/1
        elif aspect_ratio == "3:2":
            target_ratio = 3/2
        elif aspect_ratio == "4:3":
            target_ratio = 4/3
        elif aspect_ratio == "16:9":
            target_ratio = 16/9
        elif aspect_ratio == "21:9":
            target_ratio = 21/9
        elif aspect_ratio == "2:3":
            target_ratio = 2/3
        elif aspect_ratio == "3:4":
            target_ratio = 3/4
        elif aspect_ratio == "9:16":
            target_ratio = 9/16
        elif aspect_ratio == "9:21":
            target_ratio = 9/21
        else:
            # 保持原始比例
            return bbox
        
        # 计算当前比例
        current_ratio = width / height if height > 0 else 1
        
        # 调整边界框以匹配目标比例
        if current_ratio > target_ratio:
            # 当前比例更宽，需要增加高度
            new_height = width / target_ratio
            y_min = center_y - new_height / 2
            y_max = center_y + new_height / 2
        else:
            # 当前比例更高，需要增加宽度
            new_width = height * target_ratio
            x_min = center_x - new_width / 2
            x_max = center_x + new_width / 2
        
        # 如果不允许超出图像范围，则进行限制
        if not exceed_image:
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(img_size[0], x_max)
            y_max = min(img_size[1], y_max)
        
        return (int(x_min), int(y_min), int(x_max), int(y_max))
    
    def apply_scale_factor(self, bbox, scale_factor, img_size, exceed_image=False):
        """应用缩放系数到边界框，保持中心点和比例不变"""
        x_min, y_min, x_max, y_max = bbox
        
        # 计算中心点
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        
        # 计算当前宽高
        width = x_max - x_min
        height = y_max - y_min
        
        # 应用缩放系数
        new_width = width * scale_factor
        new_height = height * scale_factor
        
        # 计算新的边界框
        new_x_min = center_x - new_width / 2
        new_y_min = center_y - new_height / 2
        new_x_max = center_x + new_width / 2
        new_y_max = center_y + new_height / 2
        
        # 如果不允许超出图像范围，则进行限制
        if not exceed_image:
            new_x_min = max(0, new_x_min)
            new_y_min = max(0, new_y_min)
            new_x_max = min(img_size[0], new_x_max)
            new_y_max = min(img_size[1], new_y_max)
        
        return (int(new_x_min), int(new_y_min), int(new_x_max), int(new_y_max))
    
    def adjust_for_divisibility(self, bbox, divisible_by, img_size, exceed_image=False):
        """调整边界框使宽高可被指定整数整除"""
        x_min, y_min, x_max, y_max = bbox
        
        # 计算当前宽高
        width = x_max - x_min
        height = y_max - y_min
        
        # 计算需要调整的量，使宽高可被整除
        width_remainder = width % divisible_by
        height_remainder = height % divisible_by
        
        # 如果已经可以整除，不需要调整
        if width_remainder == 0 and height_remainder == 0:
            return bbox
        
        # 计算需要增加的宽高
        width_add = 0 if width_remainder == 0 else divisible_by - width_remainder
        height_add = 0 if height_remainder == 0 else divisible_by - height_remainder
        
        # 计算中心点
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        
        # 计算新的边界框，保持中心点不变
        new_width = width + width_add
        new_height = height + height_add
        
        new_x_min = center_x - new_width / 2
        new_y_min = center_y - new_height / 2
        new_x_max = center_x + new_width / 2
        new_y_max = center_y + new_height / 2
        
        # 如果不允许超出图像范围，则进行限制
        if not exceed_image:
            new_x_min = max(0, new_x_min)
            new_y_min = max(0, new_y_min)
            new_x_max = min(img_size[0], new_x_max)
            new_y_max = min(img_size[1], new_y_max)
            
            # 重新检查调整后的尺寸是否满足整除要求
            adjusted_width = new_x_max - new_x_min
            adjusted_height = new_y_max - new_y_min
            
            # 如果调整后不满足整除要求，则缩小尺寸
            if adjusted_width % divisible_by != 0:
                new_width = (adjusted_width // divisible_by) * divisible_by
                new_x_min = center_x - new_width / 2
                new_x_max = center_x + new_width / 2
                
                # 确保不超出图像范围
                if new_x_min < 0:
                    new_x_min = 0
                    new_x_max = new_width
                if new_x_max > img_size[0]:
                    new_x_max = img_size[0]
                    new_x_min = new_x_max - new_width
            
            if adjusted_height % divisible_by != 0:
                new_height = (adjusted_height // divisible_by) * divisible_by
                new_y_min = center_y - new_height / 2
                new_y_max = center_y + new_height / 2
                
                # 确保不超出图像范围
                if new_y_min < 0:
                    new_y_min = 0
                    new_y_max = new_height
                if new_y_max > img_size[1]:
                    new_y_max = img_size[1]
                    new_y_min = new_y_max - new_height
        
        return (int(new_x_min), int(new_y_min), int(new_x_max), int(new_y_max))
    
    def parse_color(self, color_str, img, bbox=None):
        """解析颜色字符串为RGB元组"""
        # 处理边缘颜色
        if color_str.lower() in ["e", "edge"]:
            if bbox is not None:
                # 获取边缘颜色
                return self.get_edge_color(img, bbox)
            else:
                return (255, 255, 255)  # 默认白色
        
        # 处理灰度值 (0.0-1.0)
        try:
            gray_value = float(color_str)
            if 0 <= gray_value <= 1:
                gray_int = int(gray_value * 255)
                return (gray_int, gray_int, gray_int)
        except ValueError:
            pass
        
        # 处理十六进制颜色
        if color_str.startswith("#"):
            color_str = color_str.lstrip("#")
            try:
                if len(color_str) == 6:
                    r = int(color_str[0:2], 16)
                    g = int(color_str[2:4], 16)
                    b = int(color_str[4:6], 16)
                    return (r, g, b)
                elif len(color_str) == 3:
                    r = int(color_str[0] + color_str[0], 16)
                    g = int(color_str[1] + color_str[1], 16)
                    b = int(color_str[2] + color_str[2], 16)
                    return (r, g, b)
            except ValueError:
                pass
        
        # 处理RGB格式 (r,g,b) 或 r,g,b
        if color_str.startswith("(") and color_str.endswith(")"):
            color_str = color_str.strip("()")
        
        # 处理不带括号的RGB格式
        try:
            rgb = color_str.split(",")
            if len(rgb) == 3:
                r = int(rgb[0].strip())
                g = int(rgb[1].strip())
                b = int(rgb[2].strip())
                return (r, g, b)
        except ValueError:
            pass
        
        # 默认返回白色
        return (255, 255, 255)
    
    def get_edge_color(self, img, bbox):
        """获取边界框边缘的平均颜色"""
        x_min, y_min, x_max, y_max = bbox
        width, height = img.size
        
        # 确保坐标在图像范围内
        x_min = max(0, min(x_min, width-1))
        y_min = max(0, min(y_min, height-1))
        x_max = max(0, min(x_max, width))
        y_max = max(0, min(y_max, height))
        
        # 收集边缘像素
        edge_pixels = []
        
        # 上边缘
        if 0 <= y_min < height:
            for x in range(max(0, x_min), min(width, x_max)):
                edge_pixels.append(img.getpixel((x, y_min)))
        
        # 下边缘
        if 0 <= y_max-1 < height:
            for x in range(max(0, x_min), min(width, x_max)):
                edge_pixels.append(img.getpixel((x, y_max-1)))
        
        # 左边缘
        if 0 <= x_min < width:
            for y in range(max(0, y_min+1), min(height, y_max-1)):
                edge_pixels.append(img.getpixel((x_min, y)))
        
        # 右边缘
        if 0 <= x_max-1 < width:
            for y in range(max(0, y_min+1), min(height, y_max-1)):
                edge_pixels.append(img.getpixel((x_max-1, y)))
        
        # 如果没有有效的边缘像素，返回白色
        if not edge_pixels:
            return (255, 255, 255)
        
        # 计算平均颜色
        r_sum = sum(p[0] for p in edge_pixels)
        g_sum = sum(p[1] for p in edge_pixels)
        b_sum = sum(p[2] for p in edge_pixels)
        
        pixel_count = len(edge_pixels)
        return (r_sum // pixel_count, g_sum // pixel_count, b_sum // pixel_count)


class ImageBBoxCrop:
    """
    图像检测框裁剪 - 根据边界框信息批量裁剪图像
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "crop_bbox": ("CROP_BBOX",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("cropped_image",)
    FUNCTION = "image_bbox_crop"
    CATEGORY = "1hewNodes/image"

    def image_bbox_crop(self, image, crop_bbox):
        # 获取图像尺寸
        batch_size, height, width, channels = image.shape

        # 创建输出图像列表
        output_images = []

        for b in range(batch_size):
            # 将图像转换为PIL格式
            if image.is_cuda:
                img_np = (image[b].cpu().numpy() * 255).astype(np.uint8)
            else:
                img_np = (image[b].numpy() * 255).astype(np.uint8)

            img_pil = Image.fromarray(img_np)

            # 获取当前批次对应的边界框
            bbox_str = crop_bbox[b % len(crop_bbox)]
            x_min, y_min, x_max, y_max = map(int, bbox_str.split(","))
            
            # 确保边界框不超出图像范围
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(img_pil.width, x_max)
            y_max = min(img_pil.height, y_max)
            
            # 裁切图像
            cropped_img = img_pil.crop((x_min, y_min, x_max, y_max))

            # 转换回tensor
            cropped_img_np = np.array(cropped_img).astype(np.float32) / 255.0
            output_images.append(torch.from_numpy(cropped_img_np))

        # 合并批次
        if output_images:
            output_image_tensor = torch.stack(output_images)
            return (output_image_tensor,)
        else:
            # 如果没有有效的输出图像，返回原始图像
            return (image,)


class ImageCroppedPaste:
    """
    图像裁切后粘贴器 - 将处理后的裁剪图像粘贴回原始图像的位置
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "detail_image": ("IMAGE",),
                "processed_image": ("IMAGE",),
                "crop_bbox": ("CROP_BBOX",),
                "blend_mode": (
                ["normal", "multiply", "screen", "overlay", "soft_light", "difference"], {"default": "normal"}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "label": "不透明度"})
            },
            "optional": {
                "mask": ("MASK",)
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("pasted_image",)
    FUNCTION = "image_cropped_paste"
    CATEGORY = "1hewNodes/image"

    def iamge_cropped_paste(self, detail_image, processed_image, crop_bbox, blend_mode="normal", opacity=1.0,
                            mask=None):
        try:
            # 获取图像尺寸
            batch_size, height, width, channels = detail_image.shape
            proc_batch_size = processed_image.shape[0]

            # 创建输出图像列表
            output_images = []

            for b in range(batch_size):
                # 获取当前批次的图像
                orig_img = detail_image[b]
                proc_img = processed_image[b % proc_batch_size]

                # 将字符串转换为边界框坐标
                bbox_str = crop_bbox[b % len(crop_bbox)]
                bbox = list(map(int, bbox_str.split(",")))

                # 将图像转换为PIL格式
                if detail_image.is_cuda:
                    orig_np = (orig_img.cpu().numpy() * 255).astype(np.uint8)
                    proc_np = (proc_img.cpu().numpy() * 255).astype(np.uint8)
                else:
                    orig_np = (orig_img.numpy() * 255).astype(np.uint8)
                    proc_np = (proc_img.numpy() * 255).astype(np.uint8)

                orig_pil = Image.fromarray(orig_np)
                proc_pil = Image.fromarray(proc_np)

                # 如果处理后的图像尺寸与裁剪区域不匹配，调整大小
                crop_width = bbox[2] - bbox[0]
                crop_height = bbox[3] - bbox[1]

                if proc_pil.size != (crop_width, crop_height):
                    proc_pil = proc_pil.resize((crop_width, crop_height), Image.Resampling.LANCZOS)

                # 创建结果图像的副本
                result_pil = orig_pil.copy()

                # 准备遮罩
                paste_mask = None
                if mask is not None and b < mask.shape[0]:
                    if mask.is_cuda:
                        mask_np = (mask[b].cpu().numpy() * 255).astype(np.uint8)
                    else:
                        mask_np = (mask[b].numpy() * 255).astype(np.uint8)

                    mask_pil = Image.fromarray(mask_np).convert("L")

                    # 调整遮罩大小以匹配处理后的图像
                    if mask_pil.size != proc_pil.size:
                        mask_pil = mask_pil.resize(proc_pil.size, Image.Resampling.LANCZOS)

                    paste_mask = mask_pil

                # 应用混合模式
                if blend_mode != "normal":
                    # 创建裁剪区域的原始图像
                    orig_crop = orig_pil.crop(bbox)

                    # 根据混合模式混合图像
                    blended_img = self.blend_images(orig_crop, proc_pil, blend_mode)

                    # 应用不透明度
                    if opacity < 1.0:
                        proc_pil = Image.blend(orig_crop, blended_img, opacity)
                    else:
                        proc_pil = blended_img
                elif opacity < 1.0:
                    # 仅应用不透明度
                    orig_crop = orig_pil.crop(bbox)
                    proc_pil = Image.blend(orig_crop, proc_pil, opacity)

                # 粘贴处理后的图像
                result_pil.paste(proc_pil, (bbox[0], bbox[1]), paste_mask)

                # 转换回tensor
                result_np = np.array(result_pil).astype(np.float32) / 255.0
                output_images.append(torch.from_numpy(result_np))

            # 合并批次
            output_tensor = torch.stack(output_images)

            return (output_tensor,)
        except Exception as e:
            print(f"图像粘贴器错误: {str(e)}")
            # 出错时返回原始图像
            return (detail_image,)

    def blend_images(self, img1, img2, mode):
        """应用不同的混合模式"""
        img1 = img1.convert("RGB")
        img2 = img2.convert("RGB")

        if mode == "normal":
            return img2

        # 将图像转换为numpy数组以便进行混合计算
        img1_np = np.array(img1).astype(np.float32) / 255.0
        img2_np = np.array(img2).astype(np.float32) / 255.0

        if mode == "multiply":
            result_np = img1_np * img2_np
        elif mode == "screen":
            result_np = 1 - (1 - img1_np) * (1 - img2_np)
        elif mode == "overlay":
            mask = img1_np <= 0.5
            result_np = np.zeros_like(img1_np)
            result_np[mask] = 2 * img1_np[mask] * img2_np[mask]
            result_np[~mask] = 1 - 2 * (1 - img1_np[~mask]) * (1 - img2_np[~mask])
        elif mode == "soft_light":
            result_np = (1 - 2 * img2_np) * img1_np ** 2 + 2 * img2_np * img1_np
        elif mode == "difference":
            result_np = np.abs(img1_np - img2_np)
        else:
            return img2

        # 将结果转换回PIL图像
        result_np = np.clip(result_np * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(result_np)


class ImageBlendModesByCSS:
    """
    CSS 图层叠加模式 - 基于 Pilgram 库实现的 CSS 混合模式
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "overlay_image": ("IMAGE",),
                "base_image": ("IMAGE",),
                "blend_mode": (["normal", "multiply", "screen", "overlay", "darken", "lighten", 
                                "color_dodge", "color_burn", "hard_light", "soft_light", 
                                "difference", "exclusion", "hue", "saturation", "color", "luminosity"], 
                               {"default": "normal"}),
                "blend_percentage": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})
            },
            "optional": {
                "overlay_mask": ("MASK",),
                "invert_mask": ("BOOLEAN", {"default": False, "label": "反转遮罩"})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "image_blend_modes_by_css"
    CATEGORY = "1hewNodes/image"

    def image_blend_modes_by_css(self, base_image, overlay_image, blend_mode, blend_percentage, overlay_mask=None, invert_mask=False):
        # 检查并安装 pilgram 库
        if not self._check_pilgram():
            raise ImportError("无法导入 pilgram 库，请确保已安装。可以使用 pip install pilgram 安装。")
        
        import pilgram.css.blending as blending
        
        # 初始化结果为基础图层
        result = base_image.clone()
        
        # 检查并转换 RGBA 图像为 RGB
        base_image = self._convert_rgba_to_rgb(base_image)
        overlay_image = self._convert_rgba_to_rgb(overlay_image)
        
        # 获取批次大小
        base_batch_size = base_image.shape[0]
        overlay_batch_size = overlay_image.shape[0]
        
        # 创建输出图像列表
        output_images = []
        
        # 处理每个批次的图像
        for b in range(base_batch_size):
            # 获取当前批次的基础图像
            current_base = base_image[b]
            
            # 确定使用哪个叠加图像（如果叠加图像数量少于基础图像数量，则循环使用）
            overlay_index = b % overlay_batch_size
            current_overlay = overlay_image[overlay_index]
            
            # 将张量转换为PIL图像
            base_pil = self._tensor_to_pil(current_base)
            overlay_pil = self._tensor_to_pil(current_overlay)
            
            # 确保两个图像具有相同的尺寸
            if base_pil.size != overlay_pil.size:
                overlay_pil = overlay_pil.resize(base_pil.size, Image.Resampling.LANCZOS)
            
            # 应用混合模式
            blended_pil = self._apply_css_blend(base_pil, overlay_pil, blend_mode, blending)
            
            # 应用混合百分比
            if blend_percentage < 1.0:
                # 创建不透明度蒙版
                opacity_mask = Image.new("L", base_pil.size, int(blend_percentage * 255))
                # 反转蒙版
                opacity_mask = ImageOps.invert(opacity_mask)
                # 合成图像
                blended_pil = Image.composite(base_pil, blended_pil, opacity_mask)
            
            # 如果提供了遮罩，则应用遮罩
            if overlay_mask is not None:
                # 确定使用哪个遮罩（如果遮罩数量少于图像数量，则循环使用）
                mask_batch_size = overlay_mask.shape[0]
                mask_index = b % mask_batch_size
                current_mask = overlay_mask[mask_index]
                
                # 如果需要反转遮罩
                if invert_mask:
                    current_mask = 1.0 - current_mask
                
                # 将遮罩转换为PIL格式
                if overlay_mask.is_cuda:
                    mask_np = (current_mask.cpu().numpy() * 255).astype(np.uint8)
                else:
                    mask_np = (current_mask.numpy() * 255).astype(np.uint8)
                mask_pil = Image.fromarray(mask_np)
                
                # 调整遮罩大小以匹配图像
                if mask_pil.size != base_pil.size:
                    mask_pil = mask_pil.resize(base_pil.size, Image.Resampling.LANCZOS)
                
                # 合成图像
                final_pil = Image.composite(base_pil, blended_pil, mask_pil)
            else:
                final_pil = blended_pil
            
            # 转换回张量
            final_tensor = self._pil_to_tensor(final_pil)
            output_images.append(final_tensor)
        
        # 合并批次
        result = torch.stack(output_images)
        
        return (result,)
    
    def _check_pilgram(self):
        """检查是否已安装 pilgram 库"""
        try:
            import pilgram
            return True
        except ImportError:
            try:
                import pip
                pip.main(['install', 'pilgram'])
                import pilgram
                return True
            except:
                return False
    
    def _convert_rgba_to_rgb(self, image):
        """将RGBA图像转换为RGB图像"""
        # 检查图像是否为RGBA格式（4通道）
        if image.shape[3] == 4:
            # 提取RGB通道
            rgb_image = image[:, :, :, :3]
            
            # 获取Alpha通道
            alpha_channel = image[:, :, :, 3:4]
            
            # 使用Alpha通道混合RGB与白色背景
            white_bg = torch.ones_like(rgb_image)
            rgb_image = rgb_image * alpha_channel + white_bg * (1 - alpha_channel)
            
            return rgb_image
        else:
            # 如果已经是RGB格式，直接返回
            return image
    
    def _tensor_to_pil(self, tensor):
        """将张量转换为PIL图像"""
        # 确保张量在CPU上
        if tensor.is_cuda:
            tensor = tensor.cpu()
        
        # 转换为numpy数组
        np_array = (tensor.numpy() * 255).astype(np.uint8)
        
        # 创建PIL图像
        if np_array.shape[2] == 3:
            return Image.fromarray(np_array, 'RGB')
        elif np_array.shape[2] == 4:
            return Image.fromarray(np_array, 'RGBA')
        else:
            raise ValueError(f"不支持的通道数: {np_array.shape[2]}")
    
    def _pil_to_tensor(self, pil_image):
        """将PIL图像转换为张量"""
        # 确保图像是RGB模式
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # 转换为numpy数组
        np_array = np.array(pil_image).astype(np.float32) / 255.0
        
        # 转换为张量
        return torch.from_numpy(np_array)
    
    def _apply_css_blend(self, base_pil, overlay_pil, blend_mode, blending):
        """应用CSS混合模式"""
        # 将CSS混合模式名称转换为pilgram函数名
        mode_mapping = {
            "normal": "normal",
            "multiply": "multiply",
            "screen": "screen",
            "overlay": "overlay",
            "darken": "darken",
            "lighten": "lighten",
            "color_dodge": "color_dodge",
            "color_burn": "color_burn",
            "hard_light": "hard_light",
            "soft_light": "soft_light",
            "difference": "difference",
            "exclusion": "exclusion",
            "hue": "hue",
            "saturation": "saturation",
            "color": "color",
            "luminosity": "luminosity"
        }
        
        # 获取对应的混合函数
        blend_func_name = mode_mapping.get(blend_mode, "normal")
        blend_func = getattr(blending, blend_func_name)
        
        # 应用混合
        try:
            result = blend_func(base_pil, overlay_pil)
            return result
        except Exception as e:
            print(f"混合模式 {blend_mode} 应用失败: {str(e)}")
            # 如果混合失败，返回原始图像
            return base_pil


class ImageDetailHLFreqSeparation:
    """
    图像细节保留-高低频分离技术
    执行流程：
    1. 图像A和B分别进行反转和高斯模糊处理
    2. 将反转和模糊后的图像混合（使用CSS混合模式）
    3. 将混合结果再次反转
    4. 使用遮罩C混合两组混合结果
    5. 进行最终混合和色阶调整
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generate_image": ("IMAGE",),
                "detail_image": ("IMAGE",),
                "detail_mask": ("MASK",),
                "gaussian_blur": ("FLOAT", {"default": 10.00, "min": 0.00, "max": 1000.00, "step": 0.01})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process_images"
    CATEGORY = "1hewNodes/image"

    def process_images(self, detail_image, generate_image, detail_mask, gaussian_blur=10.00):
        # 获取批次大小
        batch_size_a = detail_image.shape[0]
        batch_size_b = generate_image.shape[0]
        batch_size_c = detail_mask.shape[0]
        
        # 使用最大批次大小
        max_batch_size = max(batch_size_a, batch_size_b, batch_size_c)
        
        # 创建输出图像列表
        output_images = []
        
        for b in range(max_batch_size):
            # 获取当前批次的图像和遮罩
            current_image_a = detail_image[b % batch_size_a]
            current_image_b = generate_image[b % batch_size_b]
            current_mask_c = detail_mask[b % batch_size_c]
            
            # 将图像转换为PIL格式
            pil_image_a = self._tensor_to_pil(current_image_a)
            pil_image_b = self._tensor_to_pil(current_image_b)
            
            # 步骤1: 图像A处理 - 反转得到a1
            a1 = ImageOps.invert(pil_image_a)
            
            # 步骤2: 图像A处理 - 高斯模糊得到a2
            a2 = self._gaussian_blur(pil_image_a, gaussian_blur)
            
            # 步骤3: 混合a1和a2得到c1 - 使用CSS混合模式normal
            c1 = self._blend_images_css(a1, a2, "normal", 0.5)
            
            # 新增步骤: 反转c1得到c1-1
            c1_1 = ImageOps.invert(c1)
            
            # 步骤4: 图像B处理 - 反转得到b1
            b1 = ImageOps.invert(pil_image_b)
            
            # 步骤5: 图像B处理 - 高斯模糊得到b2
            b2 = self._gaussian_blur(pil_image_b, gaussian_blur)
            
            # 步骤6: 混合b1和b2得到c2 - 使用CSS混合模式normal
            c2 = self._blend_images_css(b1, b2, "normal", 0.5)
            
            # 新增步骤: 反转c2得到c2-1
            c2_1 = ImageOps.invert(c2)
            
            # 步骤7: 使用遮罩C混合c1-1和c2-1得到d
            # 将遮罩转换为PIL格式
            if detail_mask.is_cuda:
                mask_np = (current_mask_c.cpu().numpy() * 255).astype(np.uint8)
            else:
                mask_np = (current_mask_c.numpy() * 255).astype(np.uint8)
            mask_pil = Image.fromarray(mask_np)
            
            # 调整遮罩大小以匹配图像
            if mask_pil.size != pil_image_a.size:
                mask_pil = mask_pil.resize(pil_image_a.size, Image.Resampling.LANCZOS)
            
            # 使用遮罩混合c1-1和c2-1
            # 注意：这里c2-1是base_img，c1-1是overlay_img
            d = Image.composite(c1_1, c2_1, mask_pil)
            
            # 步骤8: 混合d和b2得到e - 使用CSS混合模式normal
            # 注意：这里d是overlay_img，b2是base_img
            e = self._blend_images_css(d, b2, "normal", 0.65)
            
            # 步骤9: 应用色阶调整得到f
            f = self._adjust_levels(e, 83, 172, 1.0, 0, 255)
            
            # 转换回tensor
            result_tensor = self._pil_to_tensor(f)
            output_images.append(result_tensor)
        
        # 合并批次
        output_tensor = torch.stack(output_images)
        
        return (output_tensor,)
    
    def _tensor_to_pil(self, tensor):
        """将张量转换为PIL图像"""
        # 确保张量在CPU上
        if tensor.is_cuda:
            tensor = tensor.cpu()
        
        # 转换为numpy数组
        np_array = (tensor.numpy() * 255).astype(np.uint8)
        
        # 创建PIL图像
        if np_array.shape[2] == 3:
            return Image.fromarray(np_array, 'RGB')
        elif np_array.shape[2] == 4:
            return Image.fromarray(np_array, 'RGBA')
        else:
            raise ValueError(f"不支持的通道数: {np_array.shape[2]}")
    
    def _pil_to_tensor(self, pil_image):
        """将PIL图像转换为张量"""
        # 确保图像是RGB模式
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # 转换为numpy数组
        np_array = np.array(pil_image).astype(np.float32) / 255.0
        
        # 转换为张量
        return torch.from_numpy(np_array)
    
    def _gaussian_blur(self, image, blur):
        """应用高斯模糊"""
        if blur <= 0:
            return image
        
        # 参考 LS_GaussianBlurV2 节点的实现
        return image.filter(ImageFilter.GaussianBlur(radius=blur))
    
    def _blend_images_css(self, overlay_img, base_img, blend_mode, blend_percentage):
        """使用CSS混合模式混合两个图像"""
        # 确保两个图像具有相同的尺寸和模式
        if overlay_img.size != base_img.size:
            overlay_img = overlay_img.resize(base_img.size, Image.Resampling.LANCZOS)
        
        if overlay_img.mode != base_img.mode:
            if 'A' in overlay_img.mode:
                base_img = base_img.convert(overlay_img.mode)
            else:
                overlay_img = overlay_img.convert(base_img.mode)
        
        # 转换为numpy数组
        overlay_array = np.array(overlay_img).astype(float)
        base_array = np.array(base_img).astype(float)
        
        # 应用CSS混合模式
        if blend_mode == "normal":
            # 普通混合模式
            blended_array = overlay_array * blend_percentage + base_array * (1 - blend_percentage)
        elif blend_mode == "multiply":
            # 正片叠底
            blended_array = (overlay_array * base_array) / 255.0
            blended_array = blended_array * blend_percentage + base_array * (1 - blend_percentage)
        elif blend_mode == "screen":
            # 滤色
            blended_array = 255.0 - ((255.0 - overlay_array) * (255.0 - base_array) / 255.0)
            blended_array = blended_array * blend_percentage + base_array * (1 - blend_percentage)
        elif blend_mode == "overlay":
            # 叠加
            mask = base_array <= 127.5
            blended_array = np.zeros_like(base_array)
            blended_array[mask] = (2 * overlay_array[mask] * base_array[mask]) / 255.0
            blended_array[~mask] = 255.0 - (2 * (255.0 - overlay_array[~mask]) * (255.0 - base_array[~mask]) / 255.0)
            blended_array = blended_array * blend_percentage + base_array * (1 - blend_percentage)
        elif blend_mode == "soft_light":
            # 柔光
            blended_array = np.zeros_like(base_array)
            mask = overlay_array <= 127.5
            blended_array[mask] = ((2 * overlay_array[mask] - 255.0) * (base_array[mask] - base_array[mask] * base_array[mask] / 255.0) / 255.0) + base_array[mask]
            blended_array[~mask] = ((2 * overlay_array[~mask] - 255.0) * (np.sqrt(base_array[~mask] / 255.0) * 255.0 - base_array[~mask]) / 255.0) + base_array[~mask]
            blended_array = blended_array * blend_percentage + base_array * (1 - blend_percentage)
        else:
            # 默认为普通混合
            blended_array = overlay_array * blend_percentage + base_array * (1 - blend_percentage)
        
        # 确保值在有效范围内
        blended_array = np.clip(blended_array, 0, 255).astype(np.uint8)
        
        # 转换回PIL图像
        if overlay_img.mode == 'RGB':
            return Image.fromarray(blended_array, 'RGB')
        elif overlay_img.mode == 'RGBA':
            return Image.fromarray(blended_array, 'RGBA')
        else:
            return Image.fromarray(blended_array)
    
    def _adjust_levels(self, image, black_point, white_point, gray_point=1.0, output_black_point=0, output_white_point=255):
        """应用色阶调整，参考 ColorCorrectLevels 节点"""
        # 确保值在0-255范围内
        black_point = max(0, min(255, black_point))
        white_point = max(0, min(255, white_point))
        output_black_point = max(0, min(255, output_black_point))
        output_white_point = max(0, min(255, output_white_point))
        
        # 转换为numpy数组
        img_array = np.array(image)
        
        # 分别处理每个通道
        result_array = np.zeros_like(img_array)
        
        for i in range(img_array.shape[2]):
            channel = img_array[:, :, i].astype(float)
            
            # 应用黑白点调整
            channel = np.clip(channel, black_point, white_point)
            channel = (channel - black_point) / (white_point - black_point) * 255.0
            
            # 应用灰度点调整（gamma校正）
            if gray_point != 1.0:
                channel = 255.0 * (channel / 255.0) ** (1.0 / gray_point)
            
            # 应用输出黑白点调整
            channel = (channel / 255.0) * (output_white_point - output_black_point) + output_black_point
            
            # 确保值在有效范围内
            result_array[:, :, i] = np.clip(channel, 0, 255).astype(np.uint8)
        
        # 转换回PIL图像
        return Image.fromarray(result_array)


class ImageAddLabel:
    """
    为图像添加标签文本 - 支持批量图像和批量标签
    """

    @classmethod
    def INPUT_TYPES(s):
        # 获取字体目录中的所有字体文件
        font_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fonts")
        font_files = []
        if os.path.exists(font_dir):
            for file in os.listdir(font_dir):
                if file.lower().endswith(('.ttf', '.otf')):
                    font_files.append(file)

        if not font_files:
            font_files = ["FreeMono.ttf"]  # 默认字体

        return {
            "required": {
                "image": ("IMAGE",),
                "height": ("INT", {"default": 60, "min": 1, "max": 1024}),
                "font_size": ("INT", {"default": 36, "min": 1, "max": 256}),
                "invert_colors": ("BOOLEAN", {"default": True}),
                "font": (font_files, {"default": "arial.ttf", "label": "字体文件"}),
                "text": ("STRING", {"default": "", "multiline": True, "label": "标签文本(多行时每行对应一张图像)"}),
                "direction": (["top", "bottom", "left", "right"], {"default": "top", "label": "标签位置"})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "image_add_label"
    CATEGORY = "1hewNodes/image"

    def image_add_label(self, image, height, font_size, invert_colors, font, text, direction):
        # 设置颜色，根据invert_colors决定黑白配色
        if invert_colors:
            font_color = "black"
            label_color = "white"
        else:
            font_color = "white"
            label_color = "black"

        # 获取图像尺寸
        result = []
        
        # 处理多行文本，分割成标签列表
        text_lines = text.strip().split('\n')
        
        for i, img in enumerate(image):
            # 选择对应的标签文本，如果标签数量少于图像数量，则循环使用
            current_text = text_lines[i % len(text_lines)] if text_lines else ""
            
            # 将图像转换为PIL格式
            i = 255. * img.cpu().numpy()
            img_pil = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            width, orig_height = img_pil.size

            # 创建标签区域
            if direction in ["top", "bottom"]:
                label_img = Image.new("RGB", (width, height), label_color)
                # 创建绘图对象
                draw = ImageDraw.Draw(label_img)

                # 尝试加载字体，如果失败则使用默认字体
                try:
                    # 检查字体文件是否存在
                    font_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fonts", font)
                    if not os.path.exists(font_path):
                        # 尝试在系统字体目录查找
                        system_font_dirs = [
                            "C:/Windows/Fonts",  # Windows
                            "/usr/share/fonts",  # Linux
                            "/System/Library/Fonts"  # macOS
                        ]
                        for font_dir in system_font_dirs:
                            if os.path.exists(os.path.join(font_dir, font)):
                                font_path = os.path.join(font_dir, font)
                                break

                    font_obj = ImageFont.truetype(font_path, font_size)
                except Exception as e:
                    print(f"无法加载字体 {font}: {e}，使用默认字体")
                    font_obj = ImageFont.load_default()

                # 计算文本尺寸
                try:
                    # 对于较新版本的PIL
                    text_bbox = draw.textbbox((0, 0), current_text, font=font_obj)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                except AttributeError:
                    # 对于较旧版本的PIL
                    text_width, text_height = draw.textsize(current_text, font=font_obj)

                # 计算文本位置 - 左对齐，空出10像素，垂直居中
                text_x = 10  # 左边距10像素
                text_y = (height - text_height) // 2  # 垂直居中

                # 绘制文本
                draw.text((text_x, text_y), current_text, fill=font_color, font=font_obj)

                # 合并图像和标签
                if direction == "top":
                    new_img = Image.new("RGB", (width, orig_height + height))
                    new_img.paste(label_img, (0, 0))
                    new_img.paste(img_pil, (0, height))
                else:  # bottom
                    new_img = Image.new("RGB", (width, orig_height + height))
                    new_img.paste(img_pil, (0, 0))
                    new_img.paste(label_img, (0, orig_height))
            else:  # left or right
                # 对于左右方向，我们需要创建一个临时的水平标签，然后旋转它
                if direction == "left":
                    # 创建一个水平标签（类似于top标签）
                    temp_label_img = Image.new("RGB", (orig_height, height), label_color)
                    draw = ImageDraw.Draw(temp_label_img)

                    try:
                        font_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fonts", font)
                        if not os.path.exists(font_path):
                            system_font_dirs = [
                                "C:/Windows/Fonts",
                                "/usr/share/fonts",
                                "/System/Library/Fonts"
                            ]
                            for font_dir in system_font_dirs:
                                if os.path.exists(os.path.join(font_dir, font)):
                                    font_path = os.path.join(font_dir, font)
                                    break

                        font_obj = ImageFont.truetype(font_path, font_size)
                    except Exception as e:
                        print(f"无法加载字体 {font}: {e}，使用默认字体")
                        font_obj = ImageFont.load_default()

                    # 计算文本尺寸
                    try:
                        text_bbox = draw.textbbox((0, 0), current_text, font=font_obj)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                    except AttributeError:
                        text_width, text_height = draw.textsize(current_text, font=font_obj)

                    # 计算文本位置 - 左对齐，空出10像素，垂直居中
                    text_x = 10  # 左边距10像素
                    text_y = (height - text_height) // 2  # 垂直居中

                    # 绘制文本
                    draw.text((text_x, text_y), current_text, fill=font_color, font=font_obj)

                    # 旋转标签图像逆时针90度
                    label_img = temp_label_img.rotate(90, expand=True)

                    # 合并图像和标签
                    new_img = Image.new("RGB", (width + height, orig_height))
                    new_img.paste(label_img, (0, 0))
                    new_img.paste(img_pil, (height, 0))

                else:  # right
                    # 创建一个水平标签（类似于top标签）
                    temp_label_img = Image.new("RGB", (orig_height, height), label_color)
                    draw = ImageDraw.Draw(temp_label_img)

                    try:
                        font_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fonts", font)
                        if not os.path.exists(font_path):
                            system_font_dirs = [
                                "C:/Windows/Fonts",
                                "/usr/share/fonts",
                                "/System/Library/Fonts"
                            ]
                            for font_dir in system_font_dirs:
                                if os.path.exists(os.path.join(font_dir, font)):
                                    font_path = os.path.join(font_dir, font)
                                    break

                        font_obj = ImageFont.truetype(font_path, font_size)
                    except Exception as e:
                        print(f"无法加载字体 {font}: {e}，使用默认字体")
                        font_obj = ImageFont.load_default()

                    # 计算文本尺寸
                    try:
                        text_bbox = draw.textbbox((0, 0), current_text, font=font_obj)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                    except AttributeError:
                        text_width, text_height = draw.textsize(current_text, font=font_obj)

                    # 计算文本位置 - 左对齐，空出10像素，垂直居中
                    text_x = 10  # 左边距10像素
                    text_y = (height - text_height) // 2  # 垂直居中

                    # 绘制文本
                    draw.text((text_x, text_y), current_text, fill=font_color, font=font_obj)

                    # 旋转标签图像顺时针90度（即逆时针270度）
                    label_img = temp_label_img.rotate(270, expand=True)

                    # 合并图像和标签
                    new_img = Image.new("RGB", (width + height, orig_height))
                    new_img.paste(img_pil, (0, 0))
                    new_img.paste(label_img, (width, 0))

            # 转换回tensor
            img_np = np.array(new_img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np)[None,]
            result.append(img_tensor)

        return (torch.cat(result, dim=0),)


class ImagePlot:
    """
    将多张图像拼合成一张大图
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "layout": (["horizontal", "vertical", "grid"], {"default": "horizontal", "label": "排列方式"}),
                "gap": ("INT", {"default": 10, "min": 0, "max": 100, "label": "图像间隙"}),
                "columns": ("INT", {"default": 2, "min": 1, "max": 10, "label": "每行图像数量(网格模式)"}),
                "background_color": ("STRING", {"default": "1.0", "label": "背景颜色 (灰度/HEX/RGB)"})
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "plot_image"
    CATEGORY = "1hewNodes/image"
    
    def plot_image(self, images, layout, gap, columns, background_color):
        # 解析背景颜色
        bg_color = self._parse_color(background_color)
        
        # 获取图像数量
        num_images = images.shape[0]
        
        # 将所有图像转换为PIL格式
        pil_images = []
        for i in range(num_images):
            img = 255. * images[i].cpu().numpy()
            pil_img = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
            pil_images.append(pil_img)
        
        # 获取所有图像的尺寸
        widths = [img.width for img in pil_images]
        heights = [img.height for img in pil_images]
        
        # 根据布局计算最终图像的尺寸
        if layout == "horizontal":
            # 水平排列
            total_width = sum(widths) + gap * (num_images - 1)
            max_height = max(heights)
            result_img = Image.new("RGB", (total_width, max_height), bg_color)
            
            x_offset = 0
            for img in pil_images:
                # 垂直居中
                y_offset = (max_height - img.height) // 2
                result_img.paste(img, (x_offset, y_offset))
                x_offset += img.width + gap
                
        elif layout == "vertical":
            # 垂直排列
            max_width = max(widths)
            total_height = sum(heights) + gap * (num_images - 1)
            result_img = Image.new("RGB", (max_width, total_height), bg_color)
            
            y_offset = 0
            for img in pil_images:
                # 水平居中
                x_offset = (max_width - img.width) // 2
                result_img.paste(img, (x_offset, y_offset))
                y_offset += img.height + gap
                
        else:  # grid
            # 网格排列
            rows = math.ceil(num_images / columns)
            
            # 计算每行每列的最大尺寸
            max_width_per_col = []
            for col in range(columns):
                col_images = [pil_images[i] for i in range(num_images) if i % columns == col]
                max_width_per_col.append(max([img.width for img in col_images]) if col_images else 0)
                
            max_height_per_row = []
            for row in range(rows):
                row_images = [pil_images[i] for i in range(num_images) if i // columns == row]
                max_height_per_row.append(max([img.height for img in row_images]) if row_images else 0)
            
            # 计算总宽度和总高度
            total_width = sum(max_width_per_col) + gap * (columns - 1)
            total_height = sum(max_height_per_row) + gap * (rows - 1)
            
            result_img = Image.new("RGB", (total_width, total_height), bg_color)
            
            # 放置图像
            for i, img in enumerate(pil_images):
                row = i // columns
                col = i % columns
                
                # 计算当前位置的x和y偏移
                x_offset = sum(max_width_per_col[:col]) + gap * col
                y_offset = sum(max_height_per_row[:row]) + gap * row
                
                # 在当前单元格内居中
                x_center = (max_width_per_col[col] - img.width) // 2
                y_center = (max_height_per_row[row] - img.height) // 2
                
                result_img.paste(img, (x_offset + x_center, y_offset + y_center))
        
        # 转换回tensor
        result_np = np.array(result_img).astype(np.float32) / 255.0
        result_tensor = torch.from_numpy(result_np)[None,]
        
        return (result_tensor,)
    
    def _parse_color(self, color_str):
        """解析不同格式的颜色输入"""
        color_str = color_str.strip()
        
        # 尝试解析为灰度值 (0.0-1.0)
        try:
            gray_value = float(color_str)
            if 0.0 <= gray_value <= 1.0:
                # 灰度值转换为RGB
                gray_int = int(gray_value * 255)
                return (gray_int, gray_int, gray_int)
        except ValueError:
            pass
        
        # 尝试解析为十六进制颜色 (#RRGGBB 或 RRGGBB)
        if color_str.startswith('#'):
            hex_color = color_str[1:]
        else:
            hex_color = color_str
            
        if len(hex_color) == 6:
            try:
                r = int(hex_color[0:2], 16)
                g = int(hex_color[2:4], 16)
                b = int(hex_color[4:6], 16)
                return (r, g, b)
            except ValueError:
                pass
        
        # 尝试解析为RGB格式 (R,G,B)
        try:
            rgb = color_str.split(',')
            if len(rgb) == 3:
                r = int(rgb[0].strip())
                g = int(rgb[1].strip())
                b = int(rgb[2].strip())
                return (r, g, b)
        except ValueError:
            pass
        
        # 默认返回白色
        return (255, 255, 255)



NODE_CLASS_MAPPINGS = {
    "ImageResizeUniversal": ImageResizeUniversal,
    "ImageEditStitch": ImageEditStitch,
    "ImageCropSquare": ImageCropSquare,
    "ImageCropWithBBox": ImageCropWithBBox,
    "ImageBBoxCrop": ImageBBoxCrop,
    "ImageCroppedPaste": ImageCroppedPaste,
    "ImageBlendModesByCSS": ImageBlendModesByCSS,
    "ImageDetailHLFreqSeparation": ImageDetailHLFreqSeparation,
    "ImageAddLabel": ImageAddLabel,
    "ImagePlot": ImagePlot
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageResizeUniversal": "Image Resize Universal",
    "ImageEditStitch": "Image Edit Stitch",
    "ImageCropSquare": "Image Crop Square",
    "ImageCropWithBBox": "Image Crop With BBox",
    "ImageBBoxCrop": "Image BBox Crop",
    "ImageCroppedPaste": "Image Cropped Paste",
    "ImageBlendModesByCSS": "Image Blend Modes By CSS",
    "ImageDetailHLFreqSeparation": "Image Detail HL Freq Separation",
    "ImageAddLabel": "Image Add Label",
    "ImagePlot": "Image Plot"
}
