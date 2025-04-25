import torch
import numpy as np
from PIL import Image, ImageOps

class Solid:
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
                                "832×480 (16:9)", "1280×720 (16:9)", "1920×1088 (16:9)",
                                "2176×960 (21:9)"],
                              {"default": "custom"}),
                "flip_size": ("BOOLEAN", {"default": False}),
                "width": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 8}),
                "color": ("COLOR", {"default": "#FFFFFF"})
            },
            "optional": {
                "alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "invert": ("BOOLEAN", {"default": False}),
                "mask_opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "reference_images": ("IMAGE", )
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "solid"
    CATEGORY = "1hewNodes/image"

    def solid(self, preset_size, flip_size, width, height, color, alpha=1.0, invert=False, mask_opacity=1.0, reference_images=None):
        images = []
        masks = []

        if reference_images is not None:
            # 处理批量参考图像
            for reference_image in reference_images:
                # 从参考图像获取尺寸
                h, w, _ = reference_image.shape
                img_width = w
                img_height = h
        else:
            # 处理预设尺寸或自定义尺寸
            if preset_size != "custom":
                # 从预设尺寸中提取宽度和高度（去掉比例部分）
                dimensions = preset_size.split(" ")[0].split("×")
                img_width = int(dimensions[0])
                img_height = int(dimensions[1])

                # 如果选择了反转尺寸，交换宽高
                if flip_size:
                    img_width, img_height = img_height, img_width
            else:
                img_width = width
                img_height = height

            # 为了兼容批量处理，这里将单个尺寸的情况也当作一个批次处理
            num_images = 1
            reference_images = [None] * num_images

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

        for reference_image in reference_images:
            if reference_image is not None:
                # 从参考图像获取尺寸
                h, w, _ = reference_image.shape
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


class LumaMatte:
    """
    亮度蒙版 - 支持批量处理图像
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "mask": ("MASK",)
            },
            "optional": {
                "invert_mask": ("BOOLEAN", {"default": False, "label": "反转遮罩"}),
                "add_background": ("BOOLEAN", {"default": False, "label": "添加背景"}),
                "background_color": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "label": "底色灰度值"})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "luma_matte"

    CATEGORY = "1hewNodes/image"


    def luma_matte(self, images, mask, invert_mask=False, add_background=True, background_color=0.0):
        # 获取图像尺寸
        batch_size, height, width, channels = images.shape
        mask_batch_size = mask.shape[0]
        
        # 创建输出图像
        output_images = []
        
        for b in range(batch_size):
            # 将图像转换为PIL格式
            if images.is_cuda:
                img_np = (images[b].cpu().numpy() * 255).astype(np.uint8)
            else:
                img_np = (images[b].numpy() * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)

            # 确定使用哪个遮罩（如果遮罩数量少于图像数量，则循环使用）
            mask_index = b % mask_batch_size
            
            # 将遮罩转换为PIL格式
            if mask.is_cuda:
                mask_np = (mask[mask_index].cpu().numpy() * 255).astype(np.uint8)
            else:
                mask_np = (mask[mask_index].numpy() * 255).astype(np.uint8)
            mask_pil = Image.fromarray(mask_np)            

            # 调整遮罩大小以匹配图像
            if img_pil.size != mask_pil.size:
                mask_pil = mask_pil.resize(img_pil.size, Image.Resampling.LANCZOS)

            # 如果需要反转遮罩
            if invert_mask:
                mask_pil = ImageOps.invert(mask_pil)

            if add_background:
                # 使用 background_color 作为 3 通道的 RGB 背景色
                bg_color = tuple(int(background_color * 255) for _ in range(3))
                background = Image.new('RGB', img_pil.size, bg_color)
                background.paste(img_pil, (0, 0), mask_pil)

                # 转换回numpy格式
                background_np = np.array(background).astype(np.float32) / 255.0
                output_images.append(torch.from_numpy(background_np))
            else:
                # 创建透明图像
                transparent = Image.new('RGBA', img_pil.size)
                transparent.paste(img_pil, (0, 0), mask_pil)

                # 转换回numpy格式，保留所有4个通道（包括alpha）
                transparent_np = np.array(transparent).astype(np.float32) / 255.0
                output_images.append(torch.from_numpy(transparent_np))

        # 合并批次
        output_tensor = torch.stack(output_images)
        
        return (output_tensor,)


class ImageConcatenate:
    """
    图像拼接器 - 将参考图像和编辑图像拼接在一起，支持上下左右四种拼接方式
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_image": ("IMAGE",),
                "edit_image": ("IMAGE",),
                "position": (["right", "left", "bottom", "top"], {"default": "right", "label": "拼接位置"}),
                "match_size": ("BOOLEAN", {"default": True, "label": "匹配尺寸"}),
                "fill_color": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "label": "填充颜色(0黑-1白)"})
            },
            "optional": {
                "edit_mask": ("MASK",)
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("image", "mask", "split_mask")
    FUNCTION = "image_concatenate"
    CATEGORY = "1hewNodes/image"

    def image_concatenate(self, reference_image, edit_image, edit_mask=None, position='right', match_size=True, fill_color=1.0):
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
    FUNCTION = "image_crop"
    CATEGORY = "1hewNodes/image"

    def image_crop(self, image, mask, invert_mask=False, padding=0):
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


class ImagePaste:
    """
    图像粘贴器 - 将处理后的裁剪图像粘贴回原始图像的位置
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "processed_image": ("IMAGE",),
                "crop_bbox": ("CROP_BBOX",),  # 将 "STRING" 改为 "CROP_BBOX"
                "blend_mode": (["normal", "multiply", "screen", "overlay", "soft_light", "difference"], {"default": "normal"}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "label": "不透明度"})
            },
            "optional": {
                "mask": ("MASK",)
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("pasted_image",)
    FUNCTION = "image_paste"
    CATEGORY = "1hewNodes/image"

    def image_paste(self, original_image, processed_image, crop_bbox, blend_mode="normal", opacity=1.0, mask=None):
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
            result_np = (1 - 2 * img2_np) * img1_np**2 + 2 * img2_np * img1_np
        elif mode == "difference":
            result_np = np.abs(img1_np - img2_np)
        else:
            return img2
        
        # 将结果转换回PIL图像
        result_np = np.clip(result_np * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(result_np)

# 更新节点注册
NODE_CLASS_MAPPINGS = {
    "Solid": Solid,
    "LumaMatte": LumaMatte,
    "ImageConcatenate": ImageConcatenate,
    "ImageCropWithBBox": ImageCropWithBBox,
    "ImagePaste": ImagePaste
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Solid": "Solid",
    "LumaMatte": "Luma Matte",
    "ImageConcatenate": "Image Concatenate",
    "ImageCropWithBBox": "Image Crop With BBox",
    "ImagePaste": "Image Paste"
}
