from comfy_api.latest import io
import cv2
import numpy as np
import torch
from PIL import Image, ImageColor

class ImageStrokeByMask(io.ComfyNode):
    """
    图像遮罩描边节点
    对输入图像的指定遮罩区域进行描边处理
    """
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_ImageStrokeByMask",
            display_name="Image Stroke by Mask",
            category="1hewNodes/image",
            inputs=[
                io.Image.Input("image"),
                io.Mask.Input("mask"),
                io.Int.Input("stroke_width", default=20, min=0, max=1000, step=1),
                io.String.Input("stroke_color", default="1.0"),
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
        mask: torch.Tensor,
        stroke_width: int,
        stroke_color: str,
    ) -> io.NodeOutput:
        """应用描边遮罩效果 - 支持批处理，简化逻辑"""
        # 获取批次大小
        batch_size = image.shape[0]
        mask_batch_size = mask.shape[0]
        
        # 确定最大批次大小
        max_batch_size = max(batch_size, mask_batch_size)
        
        output_images = []
        output_masks = []
        
        for i in range(max_batch_size):
            # 使用循环索引获取对应的输入
            img_idx = i % batch_size
            mask_idx = i % mask_batch_size
            
            # 转换当前批次的输入
            current_image = image[img_idx:img_idx+1]
            current_mask = mask[mask_idx:mask_idx+1]
            
            image_pil = cls.tensor_to_pil(current_image)
            mask_pil = cls.tensor_to_pil(current_mask)
            
            # 确保图像和遮罩尺寸一致
            if image_pil.size != mask_pil.size:
                mask_pil = mask_pil.resize(image_pil.size, Image.Resampling.LANCZOS)
            
            # 解析描边颜色
            stroke_rgb = cls.parse_color(stroke_color, image_pil, mask_pil)
            
            # 创建描边遮罩
            stroke_mask_pil = cls.create_stroke_mask(mask_pil, stroke_width)
            
            # 创建输出图像（黑色背景）
            output_image = Image.new('RGB', image_pil.size, (0, 0, 0))
            
            # 先填充描边区域为指定颜色
            stroke_color_image = Image.new('RGB', image_pil.size, stroke_rgb)
            output_image.paste(stroke_color_image, mask=stroke_mask_pil)
            
            # 再粘贴原始图像内容到原始遮罩区域
            output_image.paste(image_pil, mask=mask_pil)
            
            # 创建输出遮罩（先填充描边，然后在描边mask基础上直接填充，确保满fill）
            stroke_mask_np = np.array(stroke_mask_pil)
            mask_np = np.array(mask_pil)
            
            # 先创建描边遮罩作为基础
            output_mask_np = stroke_mask_np.copy()
            
            # 在描边mask基础上直接填充原始mask区域为满值（255）
            output_mask_np[mask_np > 0] = 255
            
            output_mask_pil = Image.fromarray(output_mask_np.astype(np.uint8), 'L')
            
            # 转换回tensor并添加到列表
            output_image_tensor = cls.pil_to_tensor(output_image)
            output_mask_tensor = cls.pil_to_tensor(output_mask_pil)
            
            output_images.append(output_image_tensor)
            output_masks.append(output_mask_tensor)
        
        # 合并批次
        device = image.device
        final_images = torch.cat(output_images, dim=0).to(device)
        final_masks = torch.cat(output_masks, dim=0).to(device)
        
        return io.NodeOutput(final_images, final_masks)
    @staticmethod
    def tensor_to_pil(tensor):
        """将tensor转换为PIL图像，自动处理alpha通道"""
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        if tensor.dim() == 3:
            if tensor.shape[2] == 3:
                # RGB图像
                np_img = (tensor.detach().cpu().numpy() * 255).astype(np.uint8)
                return Image.fromarray(np_img, 'RGB')
            elif tensor.shape[2] == 4:
                # RGBA图像，去除alpha通道
                tensor = tensor[:, :, :3]  # 只保留RGB通道
                np_img = (tensor.detach().cpu().numpy() * 255).astype(np.uint8)
                return Image.fromarray(np_img, 'RGB')
            elif tensor.shape[0] == 1:
                # 批次维度的遮罩 [1, H, W]
                tensor = tensor.squeeze(0)
                np_img = (tensor.detach().cpu().numpy() * 255).astype(np.uint8)
                return Image.fromarray(np_img, 'L')
            else:
                # 可能是 [H, W, 1] 格式的遮罩
                if tensor.shape[2] == 1:
                    tensor = tensor.squeeze(2)
                    np_img = (tensor.detach().cpu().numpy() * 255).astype(np.uint8)
                    return Image.fromarray(np_img, 'L')
                else:
                    raise ValueError(f"不支持的3维tensor形状: {tensor.shape}")
        elif tensor.dim() == 2:
            # 灰度图像/遮罩
            np_img = (tensor.detach().cpu().numpy() * 255).astype(np.uint8)
            return Image.fromarray(np_img, 'L')
        else:
            raise ValueError(f"不支持的tensor维度: {tensor.shape}")
    
    @staticmethod
    def pil_to_tensor(pil_img):
        """将PIL图像转换为tensor"""
        if pil_img.mode == 'RGB':
            np_img = np.array(pil_img).astype(np.float32) / 255.0
            return torch.from_numpy(np_img).unsqueeze(0)
        elif pil_img.mode == 'L':
            np_img = np.array(pil_img).astype(np.float32) / 255.0
            return torch.from_numpy(np_img).unsqueeze(0)  # 添加批次维度 [1, H, W]
        else:
            raise ValueError(f"不支持的PIL图像模式: {pil_img.mode}")
    
    @staticmethod
    def parse_color(color_str, image_pil=None, mask_pil=None):
        if color_str is None:
            return (255, 255, 255)
        text = str(color_str).strip().lower()
        if text in ("a", "average", "avg") and image_pil is not None:
            img = np.array(image_pil.convert("RGB")).astype(np.float32)
            avg = img.mean(axis=(0, 1))
            return (int(avg[0]), int(avg[1]), int(avg[2]))
        if text in ("mk", "mask") and image_pil is not None and mask_pil is not None:
            img = np.array(image_pil.convert("RGB")).astype(np.float32)
            mk = np.array(mask_pil).astype(np.float32) / 255.0
            if mk.shape[:2] != img.shape[:2]:
                mk = (
                    np.array(
                        mask_pil.resize(image_pil.size, Image.Resampling.LANCZOS)
                    ).astype(np.float32)
                    / 255.0
                )
            m3 = np.repeat(mk[:, :, None], 3, axis=2)
            c = float(mk.sum())
            if c > 0.0:
                avg = (img * m3).sum(axis=(0, 1)) / c
            else:
                avg = img.mean(axis=(0, 1))
            return (int(avg[0]), int(avg[1]), int(avg[2]))
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
                g = int(v * 255)
                return (g, g, g)
        except Exception:
            pass
        if "," in text:
            try:
                parts = [p.strip() for p in text.split(",")]
                if len(parts) >= 3:
                    r = float(parts[0])
                    g = float(parts[1])
                    b = float(parts[2])
                    if max(r, g, b) <= 1.0:
                        return (
                            int(r * 255),
                            int(g * 255),
                            int(b * 255),
                        )
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
    def create_stroke_mask(mask_pil, stroke_width):
        # 将PIL遮罩转换为numpy数组
        mask_np = np.array(mask_pil)
        
        # 使用形态学操作创建描边
        stroke_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (stroke_width*2+1, stroke_width*2+1))
        
        # 膨胀原始遮罩创建描边外边界
        dilated_mask = cv2.dilate(mask_np, stroke_kernel, iterations=1)
        
        # 描边区域 = 膨胀后的遮罩 - 原始遮罩
        stroke_mask = dilated_mask - mask_np
        
        # 确保值在0-255范围内
        stroke_mask = np.clip(stroke_mask, 0, 255)
        
        # 返回描边遮罩
        stroke_mask_pil = Image.fromarray(stroke_mask.astype(np.uint8), 'L')
        
        return stroke_mask_pil

