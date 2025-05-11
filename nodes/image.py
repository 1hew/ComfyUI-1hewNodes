import torch
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont
import os

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


class ImageCropWithBBox:
    """
    图像裁切器增强版 - 根据遮罩裁切图像，并返回边界框信息以便后续粘贴回原位置
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "invert_mask": ("BOOLEAN", {"default": False, "label": "反转遮罩"}),
                "padding": ("INT", {"default": 0, "min": 0, "max": 512, "step": 1, "label": "边距(像素)"})
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "CROP_BBOX")  # 将 "STRING" 改为 "CROP_BBOX"
    RETURN_NAMES = ("cropped_image", "cropped_mask", "crop_bbox")
    FUNCTION = "image_crop_with_bbox"
    CATEGORY = "1hewNodes/image"

    def image_crop_with_bbox(self, image, mask, invert_mask=False, padding=0):
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
            bbox = self.get_bbox(mask_pil, padding)

            # 如果没有找到有效区域，返回原始图像
            if bbox is None:
                output_images.append(image[b])
                output_masks.append(mask[b % mask.shape[0]])
                # 使用整个图像作为边界框，并转换为字符串
                output_bboxes_str.append(f"{0},{0},{width},{height}")
                continue

            # 裁切图像和遮罩
            cropped_img = img_pil.crop(bbox)
            cropped_mask = mask_pil.crop(bbox)

            # 转换回tensor
            cropped_img_np = np.array(cropped_img).astype(np.float32) / 255.0
            cropped_mask_np = np.array(cropped_mask).astype(np.float32) / 255.0

            output_images.append(torch.from_numpy(cropped_img_np))
            output_masks.append(torch.from_numpy(cropped_mask_np))
            # 将边界框转换为字符串
            output_bboxes_str.append(f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}")

        # 合并批次
        output_image_tensor = torch.stack(output_images)
        output_mask_tensor = torch.stack(output_masks)

        return (output_image_tensor, output_mask_tensor, output_bboxes_str)

    def get_bbox(self, mask_pil, padding=0):
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
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(mask_pil.width - 1, x_max + padding)
        y_max = min(mask_pil.height - 1, y_max + padding)

        # 返回边界框 (left, top, right, bottom)
        return (x_min, y_min, x_max + 1, y_max + 1)


class CroppedImagePaste:
    """
    图像粘贴器 - 将处理后的裁剪图像粘贴回原始图像的位置
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
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
    FUNCTION = "cropped_image_paste"
    CATEGORY = "1hewNodes/image"

    def cropped_image_paste(self, original_image, processed_image, crop_bbox, blend_mode="normal", opacity=1.0,
                            mask=None):
        try:
            # 获取图像尺寸
            batch_size, height, width, channels = original_image.shape
            proc_batch_size = processed_image.shape[0]

            # 创建输出图像列表
            output_images = []

            for b in range(batch_size):
                # 获取当前批次的图像
                orig_img = original_image[b]
                proc_img = processed_image[b % proc_batch_size]

                # 将字符串转换为边界框坐标
                bbox_str = crop_bbox[b % len(crop_bbox)]
                bbox = list(map(int, bbox_str.split(",")))

                # 将图像转换为PIL格式
                if original_image.is_cuda:
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
            return (original_image,)

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


class ImageAddLabel:
    """
    为图像添加标签文本
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
                "invert_colors": ("BOOLEAN", {"default": False}),
                "font": (font_files, {"default": "arial.ttf", "label": "字体文件"}),
                "text": ("STRING", {"default": ""}),
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
        for img in image:
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
                    text_bbox = draw.textbbox((0, 0), text, font=font_obj)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                except AttributeError:
                    # 对于较旧版本的PIL
                    text_width, text_height = draw.textsize(text, font=font_obj)

                # 计算文本位置 - 左对齐，空出10像素，垂直居中
                text_x = 10  # 左边距10像素
                text_y = (height - text_height) // 2  # 垂直居中

                # 绘制文本
                draw.text((text_x, text_y), text, fill=font_color, font=font_obj)

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
                # 对于左右方向，创建垂直标签
                label_img = Image.new("RGB", (height, orig_height), label_color)
                draw = ImageDraw.Draw(label_img)

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
                    text_bbox = draw.textbbox((0, 0), text, font=font_obj)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                except AttributeError:
                    text_width, text_height = draw.textsize(text, font=font_obj)

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
                        text_bbox = draw.textbbox((0, 0), text, font=font_obj)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                    except AttributeError:
                        text_width, text_height = draw.textsize(text, font=font_obj)

                    # 计算文本位置 - 左对齐，空出10像素，垂直居中
                    text_x = 10  # 左边距10像素
                    text_y = (height - text_height) // 2  # 垂直居中

                    # 绘制文本
                    draw.text((text_x, text_y), text, fill=font_color, font=font_obj)

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
                        text_bbox = draw.textbbox((0, 0), text, font=font_obj)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                    except AttributeError:
                        text_width, text_height = draw.textsize(text, font=font_obj)

                    # 计算文本位置 - 左对齐，空出10像素，垂直居中
                    text_x = 10  # 左边距10像素
                    text_y = (height - text_height) // 2  # 垂直居中

                    # 绘制文本
                    draw.text((text_x, text_y), text, fill=font_color, font=font_obj)

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


NODE_CLASS_MAPPINGS = {
    "ImageEditStitch": ImageEditStitch,
    "ImageCropWithBBox": ImageCropWithBBox,
    "CroppedImagePaste": CroppedImagePaste,
    "ImageBlendModesByCSS": ImageBlendModesByCSS,
    "ImageAddLabel": ImageAddLabel
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageEditStitch": "Image Edit Stitch",
    "ImageCropWithBBox": "Image Crop With BBox",
    "CroppedImagePaste": "Cropped Image Paste",
    "ImageBlendModesByCSS": "Image Blend Modes By CSS",
    "ImageAddLabel": "Image Add Label"
}
