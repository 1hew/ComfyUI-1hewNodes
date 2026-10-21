



from comfy_api.latest import io

from ...utils._image_ops import (
    generate_default_mask,
    fit_resize_mask,
    parse_color,
    fit_resize_image,
    tensor2pil,
    pil2tensor,
    image2mask,
    is_valid_mask,
    num_round_up_to_multiple,
    log,
)
import math
from PIL import Image
import torch

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
            category="1hewNodes/image/resize",
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
            orig_width, orig_height = tensor2pil(orig_images[0]).size
            
        # 处理输入遮罩
        if mask is not None:
            if mask.dim() == 2:
                mask = torch.unsqueeze(mask, 0)
            for m in mask:
                m = torch.unsqueeze(m, 0)
                if not is_valid_mask(m) and m.shape == torch.Size([1, 64, 64]):
                    log(f"警告: {cls.NODE_NAME} 输入遮罩为空，已忽略。", message_type='warning')
                else:
                    orig_masks.append(m)

            if len(orig_masks) > 0:
                _width, _height = tensor2pil(orig_masks[0]).size
                if (orig_width > 0 and orig_width != _width) or (orig_height > 0 and orig_height != _height):
                    log(f"错误: {cls.NODE_NAME} 执行失败，因为遮罩与图像尺寸不匹配。", message_type='error')
                    return io.NodeOutput(None, None)
                elif orig_width + orig_height == 0:
                    orig_width = _width
                    orig_height = _height

        # 允许无输入场景，通过 preset_ratio/scale_to_side 或 get_image_size 推导尺寸

        # 确定目标尺寸
        if get_image_size is not None:
            # 从输入图像获取尺寸
            size_img = tensor2pil(get_image_size[0])
            target_width, target_height = size_img.size
            log(f"使用输入图像尺寸: {target_width}x{target_height}", message_type='info')
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
            target_width = num_round_up_to_multiple(target_width, divisible_by)
            target_height = num_round_up_to_multiple(target_height, divisible_by)

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
                src_image = tensor2pil(i)

                # 保持现有 RGB 流程；如果输入含 alpha，则在缩放后合并回 alpha 通道
                if 'A' in src_image.getbands() or ('transparency' in src_image.info):
                    rgba_image = src_image.convert('RGBA')
                    rgb_image = rgba_image.convert('RGB')
                    alpha_channel = rgba_image.getchannel('A')

                    resized_rgb = fit_resize_image(
                        rgb_image, target_width, target_height, fit, resize_sampler, pad_color
                    ).convert('RGB')
                    resized_alpha = fit_resize_mask(
                        alpha_channel, target_width, target_height, fit, resize_sampler, orig_width, orig_height
                    ).convert('L')

                    resized_rgb.putalpha(resized_alpha)
                    ret_images.append(pil2tensor(resized_rgb))
                else:
                    _image = src_image.convert('RGB')
                    _image = fit_resize_image(_image, target_width, target_height, fit, resize_sampler, pad_color)
                    ret_images.append(pil2tensor(_image))
        else:
            # 无图像输入时也输出图像：按目标尺寸生成纯色图
            bg = parse_color(pad_color) if pad_color is not None else (255, 255, 255)
            if isinstance(bg, str) and bg in ['edge', 'average', 'extend', 'mirror']:
                bg = (255, 255, 255)
            if target_width > 0 and target_height > 0:
                if len(orig_masks) > 0:
                    for _ in orig_masks:
                        blank = Image.new('RGB', (target_width, target_height), bg)
                        ret_images.append(pil2tensor(blank))
                else:
                    blank = Image.new('RGB', (target_width, target_height), bg)
                    ret_images.append(pil2tensor(blank))
                
        # 处理遮罩缩放逻辑
        if len(orig_masks) > 0:
            # 有输入 mask 的情况
            for m in orig_masks:
                _mask = tensor2pil(m).convert('L')
                _mask = fit_resize_mask(_mask, target_width, target_height, fit, resize_sampler, orig_width, orig_height)
                ret_masks.append(image2mask(_mask))
        else:
            # 没有输入 mask 时，根据图像数量生成对应的 mask
            if len(orig_images) > 0:
                for _ in orig_images:
                    _mask = generate_default_mask(target_width, target_height, fit, orig_width, orig_height)
                    ret_masks.append(image2mask(_mask))
            else:
                # 只有尺寸信息或无任何输入，生成默认 mask
                _mask = generate_default_mask(target_width, target_height, fit, orig_width, orig_height)
                ret_masks.append(image2mask(_mask))
                
        # 返回结果 - 确保总是返回 mask
        if len(ret_images) > 0 and len(ret_masks) > 0:
            log(f"{cls.NODE_NAME} 已处理 {len(ret_images)} 张图像和 {len(ret_masks)} 张遮罩。", message_type='finish')
            return io.NodeOutput(torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0))
        elif len(ret_images) > 0 and len(ret_masks) == 0:
            log(f"{cls.NODE_NAME} 已处理 {len(ret_images)} 张图像。", message_type='finish')
            return io.NodeOutput(torch.cat(ret_images, dim=0), None)
        elif len(ret_images) == 0 and len(ret_masks) > 0:
            log(f"{cls.NODE_NAME} 已处理 {len(ret_masks)} 张遮罩。", message_type='finish')
            return io.NodeOutput(None, torch.cat(ret_masks, dim=0))
        else:
            # 若仍无法确定输出，返回空
            log(f"错误: {cls.NODE_NAME} 跳过，因为没有找到可用的图像或遮罩。", message_type='error')
            return io.NodeOutput(None, None)
