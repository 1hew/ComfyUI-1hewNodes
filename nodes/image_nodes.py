import torch
import numpy as np
from PIL import Image
import colorsys

class ColorImageGenerator:
    """
    通过拾色器面板生成纯色图像的节点
    支持多种颜色格式和透明度设置
    支持从输入图像获取尺寸
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset_size": (["custom", "512×512 (1:1)", "768×768 (1:1)", "1024×1024 (1:1)", "1408×1408 (1:1)",
                                "1728×1152 (3:2)",
                                "1280×720 (4:3)", "1664×1216 (4:3)",   
                                "832×480 (16:9)", "1280×720 (16:9)", "1920×1088 (16:9)",
                                "2176×960 (21:9)"],                             
                              {"default": "custom"}),
                "flip_dimensions": ("BOOLEAN", {"default": False, "label": "反转尺寸"}),
                "width": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 8}),
                "color": ("COLOR", {"default": "#FFFFFF"}),
                "alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "invert": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "mask_opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "reference_image": ("IMAGE", ),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "generate_color_image"
    CATEGORY = "1hewNodes/image"  # 更改类别以包含子类别

    def generate_color_image(self, preset_size, flip_dimensions, width, height, color, alpha=1.0, invert=False, mask_opacity=1.0, reference_image=None):
        # 处理预设尺寸
        if reference_image is not None:
            # 从参考图像获取尺寸
            _, h, w, _ = reference_image.shape
            width = w
            height = h
        elif preset_size != "custom":
            # 从预设尺寸中提取宽度和高度（去掉比例部分）
            dimensions = preset_size.split(" ")[0].split("×")
            width = int(dimensions[0])
            height = int(dimensions[1])
            
            # 如果选择了反转尺寸，交换宽高
            if flip_dimensions:
                width, height = height, width
        
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
        
        # 创建纯色图像
        image = np.zeros((height, width, 3), dtype=np.float32)
        image[:, :, 0] = r
        image[:, :, 1] = g
        image[:, :, 2] = b
        # 应用 alpha 调整亮度
        image = image * alpha
        
        # 创建透明度蒙版
        mask = np.ones((height, width), dtype=np.float32) * mask_opacity
        
        # 转换为ComfyUI需要的格式 (批次, 高度, 宽度, 通道)
        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)
        
        return (image, mask)


# 注册节点
NODE_CLASS_MAPPINGS = {
    "ColorImageGenerator": ColorImageGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorImageGenerator": "纯色图像生成器"
}