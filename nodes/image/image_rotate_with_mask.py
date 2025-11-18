from comfy_api.latest import io
import cv2
import math
import numpy as np
import torch
from PIL import Image, ImageColor


class ImageRotateWithMask(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_ImageRotateWithMask",
            display_name="Image Rotate with Mask",
            category="1hewNodes/image",
            inputs=[
                io.Image.Input("image"),
                io.Mask.Input("mask", optional=True),
                io.Float.Input("angle", default=0.0, min=-3600.0, max=3600.0, step=0.01),
                io.String.Input("pad_color", default="0.0"),
                io.Boolean.Input("expand", default=True),
                io.Boolean.Input("mask_center", default=False),
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
        angle: float,
        pad_color: str,
        expand: bool,
        mask_center: bool,
        mask: torch.Tensor | None = None,
    ) -> io.NodeOutput:
        """
        旋转图像
        
        Args:
            image: 输入图像张量
            angle: 旋转角度（度）
            expand: 是否扩展画布以包含完整旋转后的图像
            fill_mode: 填充模式（color, edge_extend, mirror）
            use_mask_center: 是否以mask的白色区域为中心进行旋转
            mask: 可选的遮罩，与image进行相同的变换操作
            fill_color: 填充颜色（仅在color模式下使用）
        """
        # 修正角度方向
        angle = -angle
        
        device = image.device
        batch_size = int(image.shape[0])
        output_images = []
        output_masks = []
        
        for i in range(batch_size):
            # 转换为PIL图像
            img_tensor = image[i]
            img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            
            # 处理mask（如果提供）
            if isinstance(mask, torch.Tensor):
                mask_tensor = mask[i % int(mask.shape[0])]
                mask_np = (mask_tensor.cpu().numpy() * 255).astype(np.uint8)
                mask_pil = Image.fromarray(mask_np).convert("L")
                if pil_img.size != mask_pil.size:
                    mask_pil = mask_pil.resize(pil_img.size, Image.NEAREST)
            else:
                mask_pil = None
            
            # 获取图像尺寸和旋转中心
            width, height = pil_img.size
            
            # 根据use_mask_center参数确定旋转中心
            if mask_center and mask_pil is not None:
                center = cls._calculate_mask_center(mask_pil)
            else:
                center = (width // 2, height // 2)
            
            rad = math.radians(angle)
            cos_v = abs(math.cos(rad))
            sin_v = abs(math.sin(rad))
            new_w = int((height * sin_v) + (width * cos_v))
            new_h = int((height * cos_v) + (width * sin_v))
            pad_left = max((new_w - width) // 2, 0)
            pad_right = max(new_w - width - pad_left, 0)
            pad_top = max((new_h - height) // 2, 0)
            pad_bottom = max(new_h - height - pad_top, 0)
            rot_cx = pad_left + center[0]
            rot_cy = pad_top + center[1]
            text = str(pad_color).strip().lower()
            if text in ("ex", "extend") and cv2 is not None:
                img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                pre_cv = cv2.copyMakeBorder(
                    img_cv,
                    pad_top,
                    pad_bottom,
                    pad_left,
                    pad_right,
                    cv2.BORDER_REPLICATE,
                )
                if mask_pil is not None:
                    mask_cv = np.array(mask_pil)
                    pre_mask_cv = cv2.copyMakeBorder(
                        mask_cv,
                        pad_top,
                        pad_bottom,
                        pad_left,
                        pad_right,
                        cv2.BORDER_CONSTANT,
                        value=0,
                    )
                else:
                    pre_mask_cv = np.zeros((new_h, new_w), dtype=np.uint8)
                    pre_mask_cv[pad_top : pad_top + height, pad_left : pad_left + width] = 255
                rot_mat = cv2.getRotationMatrix2D((rot_cx, rot_cy), angle, 1.0)
                rotated_cv = cv2.warpAffine(
                    pre_cv,
                    rot_mat,
                    (new_w, new_h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                )
                rotated_mask_cv = cv2.warpAffine(
                    pre_mask_cv,
                    rot_mat,
                    (new_w, new_h),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                rotated_img = Image.fromarray(cv2.cvtColor(rotated_cv, cv2.COLOR_BGR2RGB))
                rotated_mask = Image.fromarray(rotated_mask_cv, mode="L")
            elif text in ("mr", "mirror") and cv2 is not None:
                img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                pre_cv = cv2.copyMakeBorder(
                    img_cv,
                    pad_top,
                    pad_bottom,
                    pad_left,
                    pad_right,
                    cv2.BORDER_REFLECT,
                )
                if mask_pil is not None:
                    mask_cv = np.array(mask_pil)
                    pre_mask_cv = cv2.copyMakeBorder(
                        mask_cv,
                        pad_top,
                        pad_bottom,
                        pad_left,
                        pad_right,
                        cv2.BORDER_CONSTANT,
                        value=0,
                    )
                else:
                    pre_mask_cv = np.zeros((new_h, new_w), dtype=np.uint8)
                    pre_mask_cv[pad_top : pad_top + height, pad_left : pad_left + width] = 255
                rot_mat = cv2.getRotationMatrix2D((rot_cx, rot_cy), angle, 1.0)
                rotated_cv = cv2.warpAffine(
                    pre_cv,
                    rot_mat,
                    (new_w, new_h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT,
                )
                rotated_mask_cv = cv2.warpAffine(
                    pre_mask_cv,
                    rot_mat,
                    (new_w, new_h),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                rotated_img = Image.fromarray(cv2.cvtColor(rotated_cv, cv2.COLOR_BGR2RGB))
                rotated_mask = Image.fromarray(rotated_mask_cv, mode="L")
            elif text in ("edge", "e") and cv2 is not None:
                src_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                top_col = src_cv[0:1, :, :].mean(axis=(0, 1)).astype(np.uint8)
                bot_col = src_cv[-1:, :, :].mean(axis=(0, 1)).astype(np.uint8)
                left_col = src_cv[:, 0:1, :].mean(axis=(0, 1)).astype(np.uint8)
                right_col = src_cv[:, -1:, :].mean(axis=(0, 1)).astype(np.uint8)
                pre_cv = np.zeros((new_h, new_w, 3), dtype=np.uint8)
                if pad_top > 0:
                    pre_cv[:pad_top, :, :] = top_col
                if pad_bottom > 0:
                    pre_cv[new_h - pad_bottom :, :, :] = bot_col
                if pad_left > 0:
                    pre_cv[:, :pad_left, :] = left_col
                if pad_right > 0:
                    pre_cv[:, new_w - pad_right :, :] = right_col
                pre_cv[pad_top : pad_top + height, pad_left : pad_left + width] = src_cv
                pre_mask_cv = np.zeros((new_h, new_w), dtype=np.uint8)
                if mask_pil is not None:
                    src_m = np.array(mask_pil)
                    pre_mask_cv[pad_top : pad_top + height, pad_left : pad_left + width] = src_m
                else:
                    pre_mask_cv[pad_top : pad_top + height, pad_left : pad_left + width] = 255
                rot_mat = cv2.getRotationMatrix2D((rot_cx, rot_cy), angle, 1.0)
                rotated_cv = cv2.warpAffine(
                    pre_cv,
                    rot_mat,
                    (new_w, new_h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                )
                rotated_mask_cv = cv2.warpAffine(
                    pre_mask_cv,
                    rot_mat,
                    (new_w, new_h),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                rotated_img = Image.fromarray(cv2.cvtColor(rotated_cv, cv2.COLOR_BGR2RGB))
                rotated_mask = Image.fromarray(rotated_mask_cv, mode="L")
            else:
                fill_rgb = cls._parse_color_advanced(pad_color, img_tensor)
                if cv2 is not None:
                    fill_bgr = (int(fill_rgb[2]), int(fill_rgb[1]), int(fill_rgb[0]))
                    pre_cv = np.full((new_h, new_w, 3), fill_bgr, dtype=np.uint8)
                    src_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    pre_cv[pad_top : pad_top + height, pad_left : pad_left + width] = src_cv
                    pre_mask_cv = np.zeros((new_h, new_w), dtype=np.uint8)
                    if mask_pil is not None:
                        src_m = np.array(mask_pil)
                        pre_mask_cv[pad_top : pad_top + height, pad_left : pad_left + width] = src_m
                    else:
                        pre_mask_cv[pad_top : pad_top + height, pad_left : pad_left + width] = 255
                    rot_mat = cv2.getRotationMatrix2D((rot_cx, rot_cy), angle, 1.0)
                    rotated_cv = cv2.warpAffine(
                        pre_cv,
                        rot_mat,
                        (new_w, new_h),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=fill_bgr,
                    )
                    rotated_mask_cv = cv2.warpAffine(
                        pre_mask_cv,
                        rot_mat,
                        (new_w, new_h),
                        flags=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0,
                    )
                    rotated_img = Image.fromarray(cv2.cvtColor(rotated_cv, cv2.COLOR_BGR2RGB))
                    rotated_mask = Image.fromarray(rotated_mask_cv, mode="L")
                else:
                    bg = Image.new("RGB", (new_w, new_h), tuple(fill_rgb))
                    bg.paste(pil_img, (pad_left, pad_top))
                    bg_mask = Image.new("L", (new_w, new_h), 0)
                    if mask_pil is not None:
                        bg_mask.paste(mask_pil, (pad_left, pad_top))
                    else:
                        temp_mask = Image.new("L", (width, height), 255)
                        bg_mask.paste(temp_mask, (pad_left, pad_top))
                    rotated_img = bg.rotate(
                        angle,
                        resample=Image.BILINEAR,
                        expand=False,
                        center=(rot_cx, rot_cy),
                        fillcolor=tuple(fill_rgb),
                    )
                    rotated_mask = bg_mask.rotate(
                        angle,
                        resample=Image.NEAREST,
                        expand=False,
                        center=(rot_cx, rot_cy),
                        fillcolor=0,
                    )
            if not expand:
                crop_left = max((new_w - width) // 2, 0)
                crop_top = max((new_h - height) // 2, 0)
                crop_box = (crop_left, crop_top, crop_left + width, crop_top + height)
                rotated_img = rotated_img.crop(crop_box)
                rotated_mask = rotated_mask.crop(crop_box) if rotated_mask is not None else rotated_mask
            
            if rotated_img.mode == "RGBA":
                background = Image.new("RGB", rotated_img.size, (255, 255, 255))
                background.paste(rotated_img, mask=rotated_img.split()[-1])
                rotated_img = background
            elif rotated_img.mode != "RGB":
                rotated_img = rotated_img.convert("RGB")
            
            img_np = np.array(rotated_img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np)
            output_images.append(img_tensor)
            
            if rotated_mask is not None:
                mask_np = np.array(rotated_mask).astype(np.float32) / 255.0
                mask_tensor = torch.from_numpy(mask_np)
            else:
                mask_tensor = torch.ones((img_tensor.shape[0], img_tensor.shape[1]), dtype=torch.float32)
            
            output_masks.append(mask_tensor)
        
        result_images = torch.stack(output_images, dim=0)
        result_masks = torch.stack(output_masks, dim=0)
        result_images = result_images.to(device)
        result_masks = result_masks.to(device)
        return io.NodeOutput(result_images, result_masks)
    
    @staticmethod
    def _calculate_mask_center(mask_pil):
        """
        计算mask白色区域的加权重心
        
        Args:
            mask_pil: PIL格式的mask图像
            
        Returns:
            tuple: (center_x, center_y) 重心坐标
        """
        # 转换为numpy数组
        mask_array = np.array(mask_pil, dtype=np.float32)
        
        # 获取图像尺寸
        height, width = mask_array.shape
        
        # 计算总权重（所有白色像素的强度总和）
        total_weight = np.sum(mask_array)
        
        # 如果没有白色区域，返回图像中心
        if total_weight == 0:
            return (width // 2, height // 2)
        
        # 创建坐标网格
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        
        # 计算加权重心
        center_x = np.sum(x_coords * mask_array) / total_weight
        center_y = np.sum(y_coords * mask_array) / total_weight
        
        # 返回整数坐标
        return (int(center_x), int(center_y))
    
    @staticmethod
    def _create_rotation_mask(original_size, angle, center, expand):
        """
        创建旋转区域遮罩，标识哪些区域是原始图像，哪些是填充区域
        """
        width, height = original_size
        
        # 创建原始图像的遮罩（全白）
        mask = Image.new("L", (width, height), 255)
        
        # 旋转遮罩
        rotated_mask = mask.rotate(
            angle,
            resample=Image.NEAREST,
            expand=expand,
            center=center,
            fillcolor=0  # 填充区域为黑色
        )
        
        return rotated_mask
    
    @staticmethod
    def _create_rotation_mask_cv(original_size, angle, center, expand, target_size):
        """
        使用OpenCV创建与旋转图像尺寸一致的遮罩
        """
        width, height = original_size
        target_width, target_height = target_size
        
        # 创建原始图像的遮罩（全白）
        mask_cv = np.ones((height, width), dtype=np.uint8) * 255
        
        # 计算旋转矩阵
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # 如果需要扩展，调整旋转矩阵
        if expand:
            rotation_matrix[0, 2] += (target_width / 2) - center[0]
            rotation_matrix[1, 2] += (target_height / 2) - center[1]
        
        # 执行旋转
        rotated_mask_cv = cv2.warpAffine(
            mask_cv,
            rotation_matrix,
            (target_width, target_height),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
            flags=cv2.INTER_NEAREST,
        )
        
        # 转换为PIL格式
        rotated_mask = Image.fromarray(rotated_mask_cv, mode='L')
        
        return rotated_mask
    
    @classmethod
    def _rotate_with_advanced_fill(cls, pil_img, mask_pil, angle, center, expand, fill_mode):
        """
        使用高级填充模式进行旋转，同时处理mask
        """
        if cv2 is None:
            rotated_pil = pil_img.rotate(
                angle,
                resample=Image.BILINEAR,
                expand=expand,
                center=center,
                fillcolor=(0, 0, 0),
            )
            if mask_pil is not None:
                rotated_mask = mask_pil.rotate(
                    angle,
                    resample=Image.NEAREST,
                    expand=expand,
                    center=center,
                    fillcolor=0,
                )
            else:
                rotated_mask = cls._create_rotation_mask(pil_img.size, angle, center, expand)
            return rotated_pil, rotated_mask

        img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        height, width = img_cv.shape[:2]
        
        # 计算旋转矩阵
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # 计算新的边界框尺寸
        if expand:
            cos_val = abs(rotation_matrix[0, 0])
            sin_val = abs(rotation_matrix[0, 1])
            new_width = int((height * sin_val) + (width * cos_val))
            new_height = int((height * cos_val) + (width * sin_val))
            
            # 调整旋转矩阵以居中图像
            rotation_matrix[0, 2] += (new_width / 2) - center[0]
            rotation_matrix[1, 2] += (new_height / 2) - center[1]
        else:
            new_width, new_height = width, height
        
        # 设置边界填充模式
        if fill_mode == "edge_extend":
            border_mode = cv2.BORDER_REPLICATE
        elif fill_mode == "mirror":
            border_mode = cv2.BORDER_REFLECT
        else:
            border_mode = cv2.BORDER_CONSTANT
        
        # 执行图像旋转
        rotated_cv = cv2.warpAffine(
            img_cv, 
            rotation_matrix, 
            (new_width, new_height),
            borderMode=border_mode,
        )
        
        # 转换回PIL格式
        rotated_pil = Image.fromarray(cv2.cvtColor(rotated_cv, cv2.COLOR_BGR2RGB))
        
        # 处理mask旋转
        if mask_pil is not None:
            # 将mask转换为OpenCV格式
            mask_cv = np.array(mask_pil)
            
            # 对mask执行相同的旋转操作
            rotated_mask_cv = cv2.warpAffine(
                mask_cv,
                rotation_matrix,
                (new_width, new_height),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
                flags=cv2.INTER_NEAREST,
            )
            
            # 转换回PIL格式
            rotated_mask = Image.fromarray(rotated_mask_cv, mode='L')
        else:
            # 创建与旋转图像尺寸一致的遮罩
            rotated_mask = cls._create_rotation_mask_cv(pil_img.size, angle, center, expand, (new_width, new_height))
        
        return rotated_pil, rotated_mask
    
    @staticmethod
    def _parse_color_advanced(color_str, img_tensor=None):
        if not color_str:
            return (255, 255, 255)
        text = str(color_str).strip().lower()
        if text in ("edge", "e"):
            if img_tensor is not None:
                return ImageRotateWithMask._get_edge_color_tensor(img_tensor)
            return (255, 255, 255)
        if text in ("average", "a"):
            if img_tensor is not None:
                return ImageRotateWithMask._get_average_color_tensor(img_tensor)
            return (255, 255, 255)
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
                vi = int(v * 255)
                return (vi, vi, vi)
        except Exception:
            pass
        if "," in text:
            try:
                parts = [p.strip() for p in text.split(",")]
                if len(parts) >= 3:
                    r, g, b = [float(parts[i]) for i in range(3)]
                    if max(r, g, b) <= 1.0:
                        return (int(r * 255), int(g * 255), int(b * 255))
                    return (int(r), int(g), int(b))
            except Exception:
                pass
        if text.startswith("#") and len(text) in (4, 7):
            try:
                hex_str = text[1:]
                if len(hex_str) == 3:
                    hex_str = "".join(ch * 2 for ch in hex_str)
                r = int(hex_str[0:2], 16)
                g = int(hex_str[2:4], 16)
                b = int(hex_str[4:6], 16)
                return (r, g, b)
            except Exception:
                pass
        try:
            rgb = ImageColor.getrgb(text)
            return (rgb[0], rgb[1], rgb[2])
        except Exception:
            return (255, 255, 255)
    
    @staticmethod
    def _get_average_color_tensor(img_tensor):
        """
        获取图像张量的平均颜色
        """
        # img_tensor shape: [H, W, C]
        mean_color = img_tensor.mean(dim=(0, 1))  # [C]
        mean_color_255 = (mean_color * 255).int().tolist()
        return tuple(mean_color_255)
    
    @staticmethod
    def _get_edge_color_tensor(img_tensor):
        """
        获取图像张量边缘的平均颜色
        """
        # img_tensor shape: [H, W, C]
        h, w, c = img_tensor.shape
        
        # 收集边缘像素
        edge_pixels = []
        
        # 上边缘和下边缘
        edge_pixels.append(img_tensor[0, :, :])  # 上边缘
        edge_pixels.append(img_tensor[-1, :, :])  # 下边缘
        
        # 左边缘和右边缘（排除角落避免重复）
        if h > 2:
            edge_pixels.append(img_tensor[1:-1, 0, :])  # 左边缘
            edge_pixels.append(img_tensor[1:-1, -1, :])  # 右边缘
        
        # 合并所有边缘像素
        all_edge_pixels = torch.cat([p.reshape(-1, c) for p in edge_pixels], dim=0)
        
        # 计算平均颜色
        mean_color = all_edge_pixels.mean(dim=0)
        mean_color_255 = (mean_color * 255).int().tolist()
        return tuple(mean_color_255)
