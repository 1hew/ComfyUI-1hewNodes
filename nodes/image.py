import torch
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont, ImageFilter, ImageEnhance, ImageColor
import torch.nn.functional as F
import os
import math
from skimage.measure import label, regionprops


class ImageSolid:
    """
    根据输入的颜色和尺寸生成纯色图像
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset_size": (["custom", "512×512 (1:1)", "768×768 (1:1)", "1024×1024 (1:1)", "1408×1408 (1:1)",
                                "768×512 (3:2)", "1728×1152 (3:2)",
                                "1024×768 (4:3)", "1664×1216 (4:3)",
                                "832×480 (16:9)", "1280×720 (16:9)", "1920×1080 (16:9)",
                                "2176×960 (21:9)",
                                "512×768 (2:3)", "1152×1728 (2:3)",
                                "768×1024 (3:4)", "1216×1664 (3:4)",
                                "480×832 (9:16)", "720×1280 (9:16)", "1080×1920 (9:16)",
                                "960×2176 (9:21)"],
                              {"default": "custom"}),
                "width": ("INT", {"default": 1024, "min": 1, "max": 8192, "step": 1}),
                "height": ("INT", {"default": 1024, "min": 1, "max": 8192, "step": 1}),
            },
            "optional": {
                "reference_image": ("IMAGE", ),
                "color": ("COLOR", {"default": "#FFFFFF"}),
                "alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "invert": ("BOOLEAN", {"default": False}),
                "mask_opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "divisible_by": ("INT", {"default": 8, "min": 1, "max": 1024, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "image_solid"
    CATEGORY = "1hewNodes/image"

    def image_solid(self, preset_size, width, height, divisible_by, color="#FFFFFF", alpha=1.0, invert=False, mask_opacity=1.0, reference_image=None):
        images = []
        masks = []

        if reference_image is not None:
            # 处理批量参考图像
            for ref_img in reference_image:
                # 从参考图像获取尺寸
                h, w, _ = ref_img.shape
                img_width = w
                img_height = h
        else:
            # 处理预设尺寸或自定义尺寸
            if preset_size != "custom":
                # 从预设尺寸中提取宽度和高度（去掉比例部分）
                dimensions = preset_size.split(" ")[0].split("×")
                img_width = int(dimensions[0])
                img_height = int(dimensions[1])
            else:
                img_width = width
                img_height = height

            # 确保尺寸能被 divisible_by 整除
            if divisible_by > 1:
                img_width = math.ceil(img_width / divisible_by) * divisible_by
                img_height = math.ceil(img_height / divisible_by) * divisible_by

            # 为了兼容批量处理，这里将单个尺寸的情况也当作一个批次处理
            num_images = 1
            reference_image = [None] * num_images

        # 解析颜色值
        if color.startswith("#"):
            color = color[1:]
        r = int(color[0:2], 16) / 255.0
        g = int(color[2:4], 16) / 255.0
        b = int(color[4:6], 16) / 255.0

        # 如果需要反转颜色
        if invert:
            r = 1.0 - r
            g = 1.0 - g
            b = 1.0 - b

        for ref_img in reference_image:
            if ref_img is not None:
                # 从参考图像获取尺寸
                h, w, _ = ref_img.shape
                img_width = w
                img_height = h

            # 创建纯色图像
            image = np.zeros((img_height, img_width, 3), dtype=np.float32)
            image[:, :, 0] = r
            image[:, :, 1] = g
            image[:, :, 2] = b
            # 应用 alpha 调整亮度
            image = image * alpha

            # 创建透明度蒙版
            mask = np.ones((img_height, img_width), dtype=np.float32) * mask_opacity

            # 转换为ComfyUI需要的格式 (批次, 高度, 宽度, 通道)
            image = torch.from_numpy(image).unsqueeze(0)
            mask = torch.from_numpy(mask).unsqueeze(0)

            images.append(image)
            masks.append(mask)

        # 合并所有图像和蒙版
        final_images = torch.cat(images, dim=0)
        final_masks = torch.cat(masks, dim=0)

        return (final_images, final_masks)


class ImageResizeUniversal:
    """
    图像通用缩放器 - 支持多种纵横比和缩放模式，可以按照不同方式调整图像大小
    """
    
    NODE_NAME = "ImageResizeUniversal"
    
    @classmethod
    def INPUT_TYPES(cls):
        ratio_list = ['origin', 'custom', '1:1', '3:2', '4:3', '16:9', '21:9', '2:3', '3:4', '9:16', '9:21',]
        fit_mode = ['stretch', 'crop', 'pad']
        method_mode = ['nearest', 'bilinear', 'lanczos', 'bicubic', 'hamming',  'box']
        scale_to_list = ['None', 'longest', 'shortest', 'width', 'height', 'mega_pixels_k']
        return {
            "required": {
                "preset_ratio": (ratio_list,),
                "proportional_width": ("INT", {"default": 1, "min": 1, "max": 1e8, "step": 1}),
                "proportional_height": ("INT", {"default": 1, "min": 1, "max": 1e8, "step": 1}),
                "method": (method_mode, {"default": 'lanczos'}),
                "scale_to_side": (scale_to_list, {"default": 'None'}),
                "scale_to_length": ("INT", {"default": 1024, "min": 4, "max": 1e8, "step": 1}),
                "fit": (fit_mode, {"default": "crop"}),
                "pad_color": ("STRING", {"default": "1.0"}),
                "divisible_by": ("INT", {"default": 8, "min": 1, "max": 1024, "step": 1}),
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

    def image_resize(self, preset_ratio, proportional_width, proportional_height,
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
                elif scale_to_side == 'mega_pixels_k':
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
                elif scale_to_side == 'mega_pixels_k':
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
                "edit_image_position": (["top", "bottom", "left", "right"], {"default": "right"}),
                "match_edit_size": ("BOOLEAN", {"default": True}),
                "spacing": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "fill_color": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})
            },
            "optional": {
                "edit_mask": ("MASK",)
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("image", "mask", "split_mask")
    FUNCTION = "image_edit_stitch"
    CATEGORY = "1hewNodes/image"

    def image_edit_stitch(self, reference_image, edit_image, edit_mask=None, edit_image_position='right', match_edit_size=True,
                          fill_color=1.0, spacing=0):
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
        if match_edit_size and (ref_height != edit_height or ref_width != edit_width):
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

        # 创建间距填充（如果spacing > 0）
        spacing_color = torch.full((1, 1, 1, 3), fill_color, dtype=torch.float32)
        
        # 根据编辑图像位置拼接图像
        if edit_image_position == "right":
            # 参考图像在左，编辑图像在右
            if spacing > 0:
                # 创建垂直间距条
                spacing_strip = spacing_color.expand(1, ref_height, spacing, 3)
                combined_image = torch.cat([
                    reference_image,
                    spacing_strip,
                    edit_image
                ], dim=2)  # 水平拼接
                
                # 拼接遮罩（参考区域为0，间距区域为0，编辑区域保持原样）
                zero_mask_ref = torch.zeros((1, ref_height, ref_width), dtype=torch.float32)
                zero_mask_spacing = torch.zeros((1, ref_height, spacing), dtype=torch.float32)
                combined_mask = torch.cat([zero_mask_ref, zero_mask_spacing, edit_mask], dim=2)
                
                # 创建分离遮罩（参考区域为黑色，间距区域为黑色，编辑区域为白色）
                split_mask_left = torch.zeros((1, ref_height, ref_width), dtype=torch.float32)
                split_mask_spacing = torch.zeros((1, ref_height, spacing), dtype=torch.float32)
                split_mask_right = torch.ones((1, edit_height, edit_width), dtype=torch.float32)
                split_mask = torch.cat([split_mask_left, split_mask_spacing, split_mask_right], dim=2)
            else:
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

        elif edit_image_position == "left":
            # 编辑图像在左，参考图像在右
            if spacing > 0:
                # 创建垂直间距条
                spacing_strip = spacing_color.expand(1, edit_height, spacing, 3)
                combined_image = torch.cat([
                    edit_image,
                    spacing_strip,
                    reference_image
                ], dim=2)  # 水平拼接
                
                # 拼接遮罩（编辑区域保持原样，间距区域为0，参考区域为0）
                zero_mask_spacing = torch.zeros((1, edit_height, spacing), dtype=torch.float32)
                zero_mask_ref = torch.zeros((1, ref_height, ref_width), dtype=torch.float32)
                combined_mask = torch.cat([edit_mask, zero_mask_spacing, zero_mask_ref], dim=2)
                
                # 创建分离遮罩（编辑区域为白色，间距区域为黑色，参考区域为黑色）
                split_mask_left = torch.ones((1, edit_height, edit_width), dtype=torch.float32)
                split_mask_spacing = torch.zeros((1, edit_height, spacing), dtype=torch.float32)
                split_mask_right = torch.zeros((1, ref_height, ref_width), dtype=torch.float32)
                split_mask = torch.cat([split_mask_left, split_mask_spacing, split_mask_right], dim=2)
            else:
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

        elif edit_image_position == "bottom":
            # 参考图像在上，编辑图像在下
            if spacing > 0:
                # 创建水平间距条
                spacing_strip = spacing_color.expand(1, spacing, ref_width, 3)
                combined_image = torch.cat([
                    reference_image,
                    spacing_strip,
                    edit_image
                ], dim=1)  # 垂直拼接
                
                # 拼接遮罩（参考区域为0，间距区域为0，编辑区域保持原样）
                zero_mask_ref = torch.zeros((1, ref_height, ref_width), dtype=torch.float32)
                zero_mask_spacing = torch.zeros((1, spacing, ref_width), dtype=torch.float32)
                combined_mask = torch.cat([zero_mask_ref, zero_mask_spacing, edit_mask], dim=1)
                
                # 创建分离遮罩（参考区域为黑色，间距区域为黑色，编辑区域为白色）
                split_mask_top = torch.zeros((1, ref_height, ref_width), dtype=torch.float32)
                split_mask_spacing = torch.zeros((1, spacing, ref_width), dtype=torch.float32)
                split_mask_bottom = torch.ones((1, edit_height, edit_width), dtype=torch.float32)
                split_mask = torch.cat([split_mask_top, split_mask_spacing, split_mask_bottom], dim=1)
            else:
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

        elif edit_image_position == "top":
            # 编辑图像在上，参考图像在下
            if spacing > 0:
                # 创建水平间距条
                spacing_strip = spacing_color.expand(1, spacing, edit_width, 3)
                combined_image = torch.cat([
                    edit_image,
                    spacing_strip,
                    reference_image
                ], dim=1)  # 垂直拼接
                
                # 拼接遮罩（编辑区域保持原样，间距区域为0，参考区域为0）
                zero_mask_spacing = torch.zeros((1, spacing, edit_width), dtype=torch.float32)
                zero_mask_ref = torch.zeros((1, ref_height, ref_width), dtype=torch.float32)
                combined_mask = torch.cat([edit_mask, zero_mask_spacing, zero_mask_ref], dim=1)
                
                # 创建分离遮罩（编辑区域为白色，间距区域为黑色，参考区域为黑色）
                split_mask_top = torch.ones((1, edit_height, edit_width), dtype=torch.float32)
                split_mask_spacing = torch.zeros((1, spacing, edit_width), dtype=torch.float32)
                split_mask_bottom = torch.zeros((1, ref_height, ref_width), dtype=torch.float32)
                split_mask = torch.cat([split_mask_top, split_mask_spacing, split_mask_bottom], dim=1)
            else:
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
    为图像添加标签文本 - 支持批量图像和批量标签，支持动态引用输入值
    支持 -- 分隔符功能，当存在只包含连字符的行时，-- 之间的内容作为完整标签
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
                "font": (font_files, {"default": "arial.ttf"}),
                "text": ("STRING", {
                    "default": "", 
                    "multiline": True,
                    "placeholder": "-- splits override separator\nelse use newline."
                }),
                "direction": (["top", "bottom", "left", "right"], {"default": "top"})
            },
            "optional": {
                "input1": ("STRING", {"default": ""}),
                "input2": ("STRING", {"default": ""})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "image_add_label"
    CATEGORY = "1hewNodes/image"

    def parse_text_with_inputs(self, text, input1=None, input2=None):
        """
        解析文本中的输入引用
        """
        parsed_text = text
        
        # 替换 {input1} 引用
        if input1 is not None and input1 != "":
            parsed_text = parsed_text.replace("{input1}", str(input1))
        
        # 替换 {input2} 引用
        if input2 is not None and input2 != "":
            parsed_text = parsed_text.replace("{input2}", str(input2))
            
        return parsed_text

    def parse_text_list(self, text):
        """
        解析文本列表，支持连字符分割和换行分割
        当有只包含连字符的行时，只按 -- 进行分割，其他分割方式失效
        否则按照换行符(\n) 分割
        """
        import re
        
        if not text.strip():
            return [""]
        
        # 检查是否有只包含连字符的行
        lines = text.split('\n')
        has_dash_separator = any(line.strip() and all(c == '-' for c in line.strip()) for line in lines)
        
        if has_dash_separator:
            # 按连字符分割，其他分割方式失效（包括换行符）
            sections = re.split(r'^\s*-+\s*$', text, flags=re.MULTILINE)
            all_lists = []
            
            for section in sections:
                section = section.strip()
                if not section:
                    continue
                    
                # 当有连字符分割时，每个段落作为一个完整项目，保留内部换行
                # 移除引号
                if (section.startswith('"') and section.endswith('"')) or (section.startswith("'") and section.endswith("'")):
                    section = section[1:-1]
                if section:
                    all_lists.append(str(section))
            
            return all_lists if all_lists else [""]
        else:
            # 按传统方式分割（换行符）
            text_lines = text.strip().split('\n')
            # 过滤空行
            text_lines = [line.strip() for line in text_lines if line.strip()]
            return text_lines if text_lines else [""]

    def image_add_label(self, image, height, font_size, invert_colors, font, text, direction, input1=None, input2=None):
        # 解析文本中的输入引用
        parsed_text = self.parse_text_with_inputs(text, input1, input2)
        
        # 设置颜色，根据invert_colors决定黑白配色
        if invert_colors:
            font_color = "black"
            label_color = "white"
        else:
            font_color = "white"
            label_color = "black"

        # 获取图像尺寸
        result = []
        
        # 处理文本，支持 -- 分隔符功能
        text_lines = self.parse_text_list(parsed_text)
        
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
    支持单张图像和批量图片收集，将输入按指定布局排列显示
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "layout": (["horizontal", "vertical", "grid"], {"default": "horizontal"}),
                "spacing": ("INT", {"default": 10, "min": 0, "max": 100}),
                "grid_columns": ("INT", {"default": 2, "min": 1, "max": 100}),
                "background_color": ("STRING", {"default": "1.0"})
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "image_plot"
    CATEGORY = "1hewNodes/image"
    
    def image_plot(self, image, layout, spacing, grid_columns, background_color):
        """
        主处理函数，自动检测输入类型并选择相应的处理方式
        """
        # 自动检测输入类型并选择处理方式
        if self._is_video_collection(image):
            return self._process_video_collection(image, layout, spacing, grid_columns, background_color)
        else:
            return self._process_standard_plot(image, layout, spacing, grid_columns, background_color)
    
    def _is_video_collection(self, image):
        """自动检测是否为视频收集数据"""
        # 只有当输入为列表格式时才认为是视频收集数据
        if isinstance(image, list):
            return True
        # 移除基于帧数的检测，避免误判
        return False
    
    def _process_standard_plot(self, image, layout, spacing, grid_columns, background_color):
        """标准图像拼接处理"""
        # 解析背景颜色
        bg_color = self._parse_color(background_color)
        
        # 获取图像数量
        num_images = image.shape[0]
        
        # 将所有图像转换为PIL格式
        pil_images = []
        for i in range(num_images):
            img = 255. * image[i].cpu().numpy()
            pil_img = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
            pil_images.append(pil_img)
        
        # 根据布局处理图像
        result_img = self._combine_images(pil_images, layout, spacing, grid_columns, bg_color)
        
        # 转换回tensor
        result_np = np.array(result_img).astype(np.float32) / 255.0
        result_tensor = torch.from_numpy(result_np)[None,]
        
        return (result_tensor,)
    
    def _process_video_collection(self, image, layout, spacing, grid_columns, background_color):
        """视频收集处理，支持多批次图像的时间序列显示"""
        try:
            # 处理输入数据
            batch_list = self._extract_batch_list(image)
            if not batch_list:
                print("警告: 没有找到有效的批量图片")
                return (torch.zeros(1, 256, 256, 3),)
            
            print(f"接收到{len(batch_list)}个批量图片组")
            
            # 获取所有批次的最大帧数
            max_frames = max(batch.shape[0] for batch in batch_list)
            print(f"最大帧数: {max_frames}")
            
            # 为每一帧创建并列显示
            combined_frames = []
            for frame_idx in range(max_frames):
                frame_images = []
                
                # 从每个批次中提取对应帧（循环使用如果帧数不足）
                for batch in batch_list:
                    actual_idx = frame_idx % batch.shape[0]
                    frame_images.append(batch[actual_idx])
                
                # 合并当前帧的所有图片
                combined_frame = self._combine_frame_images(
                    frame_images, layout, spacing, grid_columns, background_color
                )
                combined_frames.append(combined_frame)
            
            # 合并所有帧
            result_tensor = torch.stack(combined_frames, dim=0)
            print(f"输出形状: {result_tensor.shape}")
            
            return (result_tensor,)
            
        except Exception as e:
            print(f"视频收集显示错误: {str(e)}")
            return (torch.zeros(1, 256, 256, 3),)
    
    def _extract_batch_list(self, video_collection):
        """从video_collection中提取批量图片列表"""
        if isinstance(video_collection, list):
            return [item for item in video_collection if isinstance(item, torch.Tensor)]
        elif isinstance(video_collection, torch.Tensor):
            return [video_collection]
        else:
            print(f"未知的输入类型: {type(video_collection)}")
            return []
    
    def _combine_frame_images(self, frame_images, layout, spacing, grid_columns, background_color):
        """合并单帧的多个图片"""
        if not frame_images:
            return torch.zeros(256, 256, 3)
        
        # 统一图片尺寸
        normalized_images = self._normalize_image_sizes(frame_images)
        
        # 转换为PIL图像
        pil_images = []
        for img_tensor in normalized_images:
            img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            pil_images.append(pil_img)
        
        # 解析背景颜色
        bg_color = self._parse_color(background_color)
        
        # 合并图像
        result_img = self._combine_images(pil_images, layout, spacing, grid_columns, bg_color)
        
        # 转换回tensor
        result_np = np.array(result_img).astype(np.float32) / 255.0
        return torch.from_numpy(result_np)
    
    def _combine_images(self, pil_images, layout, spacing, grid_columns, bg_color):
        """统一的图像合并逻辑"""
        if not pil_images:
            return Image.new('RGB', (256, 256), bg_color)
        
        # 获取所有图像的尺寸
        widths = [img.width for img in pil_images]
        heights = [img.height for img in pil_images]
        num_images = len(pil_images)
        
        if layout == "horizontal":
            # 水平排列
            total_width = sum(widths) + spacing * (num_images - 1)
            max_height = max(heights)
            result_img = Image.new("RGB", (total_width, max_height), bg_color)
            
            x_offset = 0
            for img in pil_images:
                y_offset = (max_height - img.height) // 2
                result_img.paste(img, (x_offset, y_offset))
                x_offset += img.width + spacing
                
        elif layout == "vertical":
            # 垂直排列
            max_width = max(widths)
            total_height = sum(heights) + spacing * (num_images - 1)
            result_img = Image.new("RGB", (max_width, total_height), bg_color)
            
            y_offset = 0
            for img in pil_images:
                x_offset = (max_width - img.width) // 2
                result_img.paste(img, (x_offset, y_offset))
                y_offset += img.height + spacing
                
        else:  # "grid" - 网格模式
            # 使用grid_columns参数确定网格尺寸
            cols = grid_columns
            rows = math.ceil(num_images / cols)
            
            # 计算每行每列的最大尺寸
            max_width_per_col = []
            for col in range(cols):
                col_images = [pil_images[i] for i in range(num_images) if i % cols == col]
                max_width_per_col.append(max([img.width for img in col_images]) if col_images else 0)
                
            max_height_per_row = []
            for row in range(rows):
                row_images = [pil_images[i] for i in range(num_images) if i // cols == row]
                max_height_per_row.append(max([img.height for img in row_images]) if row_images else 0)
            
            # 计算总宽度和总高度
            total_width = sum(max_width_per_col) + spacing * (cols - 1)
            total_height = sum(max_height_per_row) + spacing * (rows - 1)
            
            result_img = Image.new("RGB", (total_width, total_height), bg_color)
            
            # 放置图像
            for i, img in enumerate(pil_images[:rows * cols]):
                row = i // cols
                col = i % cols
                
                # 计算当前位置的x和y偏移
                x_offset = sum(max_width_per_col[:col]) + spacing * col
                y_offset = sum(max_height_per_row[:row]) + spacing * row
                
                # 在当前单元格内居中
                x_center = (max_width_per_col[col] - img.width) // 2
                y_center = (max_height_per_row[row] - img.height) // 2
                
                result_img.paste(img, (x_offset + x_center, y_offset + y_center))
        
        return result_img
    
    def _normalize_image_sizes(self, images):
        """统一图片尺寸到最小公共尺寸"""
        if not images:
            return []
        
        # 找到最小尺寸
        min_height = min(img.shape[0] for img in images)
        min_width = min(img.shape[1] for img in images)
        
        normalized = []
        for img in images:
            if img.shape[0] != min_height or img.shape[1] != min_width:
                # 调整尺寸
                img = img.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
                img = F.interpolate(img, size=(min_height, min_width), mode='bilinear', align_corners=False)
                img = img.squeeze(0).permute(1, 2, 0)  # [H, W, C]
            normalized.append(img)
        
        return normalized
    
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


class ImageBBoxOverlayByMask:
    """
    基于遮罩的图像边界框叠加节点 - 根据遮罩生成检测框并以描边或填充形式叠加到图像上
    支持独立模式和合并模式
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "bbox_color": (["red", "green", "blue", "yellow", "cyan", "magenta", "white", "black"], 
                              {"default": "red"}),
                "stroke_width": ("INT", {"default": 4, "min": 1, "max": 100, "step": 1}),
                "fill": ("BOOLEAN", {"default": True}),
                "padding": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "output_mode": (["separate", "merge"], {"default": "separate"})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "overlay_bbox"
    CATEGORY = "1hewNodes/image"

    def overlay_bbox(self, image, mask, bbox_color="red", stroke_width=3, fill=False, padding=0, output_mode="separate"):
        # 确保输入是正确的维度
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
            
        batch_size = image.shape[0]
        mask_batch_size = mask.shape[0]
        
        # 颜色映射
        color_map = {
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "yellow": (255, 255, 0),
            "cyan": (0, 255, 255),
            "magenta": (255, 0, 255),
            "white": (255, 255, 255),
            "black": (0, 0, 0)
        }
        
        bbox_rgb = color_map.get(bbox_color, (255, 0, 0))
        
        output_images = []
        
        for b in range(batch_size):
            # 获取当前批次的图像
            current_image = image[b]
            
            # 转换图像为PIL格式
            if current_image.is_cuda:
                img_np = (current_image.cpu().numpy() * 255).astype(np.uint8)
            else:
                img_np = (current_image.numpy() * 255).astype(np.uint8)
            
            img_pil = Image.fromarray(img_np, 'RGB')
            draw = ImageDraw.Draw(img_pil)
            
            if output_mode == "merge":
                # 合并模式：将所有mask合并为一个边界框
                combined_mask = None
                for m in range(mask_batch_size):
                    mask_idx = m % mask_batch_size
                    current_mask = mask[mask_idx]
                    
                    # 转换遮罩为numpy格式
                    if current_mask.is_cuda:
                        mask_np = (current_mask.cpu().numpy() * 255).astype(np.uint8)
                    else:
                        mask_np = (current_mask.numpy() * 255).astype(np.uint8)
                    
                    if combined_mask is None:
                        combined_mask = mask_np
                    else:
                        # 合并mask（取最大值）
                        combined_mask = np.maximum(combined_mask, mask_np)
                
                # 调整合并后的遮罩尺寸
                mask_pil = Image.fromarray(combined_mask, 'L')
                if mask_pil.size != img_pil.size:
                    mask_pil = mask_pil.resize(img_pil.size, Image.Resampling.LANCZOS)
                
                # 获取合并后的边界框
                bbox = self.get_single_bbox_from_mask(mask_pil, padding)
                if bbox is not None:
                    x_min, y_min, x_max, y_max = bbox
                    self.draw_bbox(draw, x_min, y_min, x_max, y_max, bbox_rgb, stroke_width, fill)
                    print(f"合并边界框: ({x_min}, {y_min}, {x_max}, {y_max})")
            else:
                # 独立模式：为每个独立的mask区域生成单独的边界框
                all_bboxes = []
                for m in range(mask_batch_size):
                    mask_idx = m % mask_batch_size
                    current_mask = mask[mask_idx]
                    
                    # 转换遮罩为numpy格式
                    if current_mask.is_cuda:
                        mask_np = (current_mask.cpu().numpy() * 255).astype(np.uint8)
                    else:
                        mask_np = (current_mask.numpy() * 255).astype(np.uint8)
                    
                    mask_pil = Image.fromarray(mask_np, 'L')
                    
                    # 调整遮罩尺寸以匹配图像
                    if mask_pil.size != img_pil.size:
                        mask_pil = mask_pil.resize(img_pil.size, Image.Resampling.LANCZOS)
                    
                    # 获取当前mask的所有独立区域的边界框
                    bboxes = self.get_multiple_bboxes_from_mask(mask_pil, padding)
                    all_bboxes.extend(bboxes)
                
                # 绘制所有边界框
                for i, bbox in enumerate(all_bboxes):
                    x_min, y_min, x_max, y_max = bbox
                    self.draw_bbox(draw, x_min, y_min, x_max, y_max, bbox_rgb, stroke_width, fill)
                    print(f"独立边界框 {i+1}: ({x_min}, {y_min}, {x_max}, {y_max})")
            
            # 转换回tensor
            result_np = np.array(img_pil).astype(np.float32) / 255.0
            result_tensor = torch.from_numpy(result_np)
            output_images.append(result_tensor)
        
        # 合并批次
        output_tensor = torch.stack(output_images)
        
        return (output_tensor,)
    
    def draw_bbox(self, draw, x_min, y_min, x_max, y_max, color, stroke_width, fill):
        """绘制边界框，支持填充和描边模式"""
        if fill:
            # 填充模式：绘制填充的矩形
            draw.rectangle(
                [x_min, y_min, x_max, y_max],
                fill=color
            )
        else:
            # 描边模式：仅绘制边框
            draw.rectangle(
                [x_min, y_min, x_max, y_max],
                outline=color,
                width=stroke_width
            )
    
    def get_single_bbox_from_mask(self, mask_pil, padding=0):
        """从遮罩中提取单个边界框坐标（合并所有白色区域）"""
        mask_np = np.array(mask_pil)
        
        # 创建二值遮罩（mask值大于128认为是有效区域）
        binary_mask = mask_np > 128
        
        # 找到有效像素的坐标
        y_coords, x_coords = np.where(binary_mask)
        
        if len(y_coords) == 0 or len(x_coords) == 0:
            return None
        
        # 获取边界框坐标
        x_min = int(np.min(x_coords))
        x_max = int(np.max(x_coords))
        y_min = int(np.min(y_coords))
        y_max = int(np.max(y_coords))
        
        # 添加填充
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(mask_pil.width - 1, x_max + padding)
        y_max = min(mask_pil.height - 1, y_max + padding)
        
        return (x_min, y_min, x_max, y_max)
    
    def get_multiple_bboxes_from_mask(self, mask_pil, padding=0):
        """从遮罩中提取多个独立区域的边界框坐标"""
        mask_np = np.array(mask_pil)
        
        # 创建二值遮罩
        binary_mask = mask_np > 128
        
        if not np.any(binary_mask):
            return []
        
        # 使用连通组件分析找到独立的区域
        labeled_mask = label(binary_mask)
        regions = regionprops(labeled_mask)
        
        bboxes = []
        for region in regions:
            # 获取区域的边界框
            min_row, min_col, max_row, max_col = region.bbox
            
            # 添加填充
            x_min = max(0, min_col - padding)
            y_min = max(0, min_row - padding)
            x_max = min(mask_pil.width - 1, max_col - 1 + padding)
            y_max = min(mask_pil.height - 1, max_row - 1 + padding)
            
            bboxes.append((x_min, y_min, x_max, y_max))
        
        return bboxes



NODE_CLASS_MAPPINGS = {
    "ImageSolid": ImageSolid,
    "ImageResizeUniversal": ImageResizeUniversal,
    "ImageEditStitch": ImageEditStitch,
    "ImageDetailHLFreqSeparation": ImageDetailHLFreqSeparation,
    "ImageAddLabel": ImageAddLabel,
    "ImagePlot": ImagePlot,
    "ImageBBoxOverlayByMask": ImageBBoxOverlayByMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageSolid": "Image Solid",
    "ImageResizeUniversal": "Image Resize Universal",
    "ImageEditStitch": "Image Edit Stitch",
    "ImageDetailHLFreqSeparation": "Image Detail HL Freq Separation",
    "ImageAddLabel": "Image Add Label",
    "ImagePlot": "Image Plot",
    "ImageBBoxOverlayByMask": "Image BBox Overlay by Mask",
}