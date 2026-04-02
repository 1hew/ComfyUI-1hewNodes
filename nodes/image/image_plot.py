import asyncio
import math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from comfy_api.latest import io

class ImagePlot(io.ComfyNode):
    """
    支持单张图像和批量图片收集，将输入按指定布局排列显示
    """
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_ImagePlot",
            display_name="Image Plot",
            category="1hewNodes/image",
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("layout", options=["horizontal", "vertical", "grid"], default="horizontal"),
                io.Int.Input("spacing", default=10, min=0, max=1000, step=1),
                io.Int.Input("grid_columns", default=2, min=1, max=100, step=1),
                io.String.Input("background_color", default="1.0"),
            ],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    async def execute(
        cls,
        image: torch.Tensor,
        layout: str,
        spacing: int,
        grid_columns: int,
        background_color: str,
    ) -> io.NodeOutput:
        """
        主处理函数，自动检测输入类型并选择相应的处理方式
        """
        # 自动检测输入类型并选择处理方式
        if cls._is_video_collection(image):
            return await cls._process_video_collection(image, layout, spacing, grid_columns, background_color)
        return await cls._process_standard_plot(image, layout, spacing, grid_columns, background_color)
    
    @classmethod
    def _is_video_collection(cls, image):
        """自动检测是否为视频收集数据"""
        # 只有当输入为列表格式时才认为是视频收集数据
        if isinstance(image, list):
            return True
        # 移除基于帧数的检测，避免误判
        return False
    
    @classmethod
    async def _process_standard_plot(cls, image, layout, spacing, grid_columns, background_color):
        """标准图像拼接处理"""
        has_alpha = int(image.shape[-1]) == 4
        # 解析背景颜色
        bg_color = cls._parse_color(background_color, has_alpha=has_alpha)
        
        # 获取图像数量
        num_images = image.shape[0]
        
        async def _to_pil(frame):
            def _do(frame_):
                pil = cls._tensor_to_pil_image(frame_)
                return pil.convert("RGBA" if has_alpha else "RGB")
            return await asyncio.to_thread(_do, frame)

        tasks = [_to_pil(image[i]) for i in range(num_images)]
        pil_images = await asyncio.gather(*tasks)
        result_img = await asyncio.to_thread(
            cls._combine_images, pil_images, layout, spacing, grid_columns, bg_color
        )
        result_np = np.array(result_img).astype(np.float32) / 255.0
        result_tensor = torch.from_numpy(result_np)[None,]
        device = image.device if isinstance(image, torch.Tensor) else torch.device("cpu")
        result_tensor = result_tensor.to(device).to(torch.float32).clamp(0.0, 1.0)
        return io.NodeOutput(result_tensor)
    
    @classmethod
    async def _process_video_collection(cls, image, layout, spacing, grid_columns, background_color):
        """视频收集处理，支持多批次图像的时间序列显示"""
        try:
            # 处理输入数据
            batch_list = cls._extract_batch_list(image)
            if not batch_list:
                print("警告: 没有找到有效的批量图片")
                return io.NodeOutput(torch.zeros(1, 256, 256, 3))
            
            print(f"接收到{len(batch_list)}个批量图片组")
            
            # 获取所有批次的最大帧数
            max_frames = max(batch.shape[0] for batch in batch_list)
            print(f"最大帧数: {max_frames}")
            
            # 为每一帧创建并列显示
            async def _combine(idx):
                frames = []
                for batch in batch_list:
                    actual_idx = idx % batch.shape[0]
                    frames.append(batch[actual_idx])
                return await asyncio.to_thread(
                    cls._combine_frame_images,
                    frames,
                    layout,
                    spacing,
                    grid_columns,
                    background_color,
                )

            tasks = [_combine(i) for i in range(max_frames)]
            combined_frames = await asyncio.gather(*tasks)
            result_tensor = torch.stack(combined_frames, dim=0)
            device = batch_list[0].device if batch_list else torch.device("cpu")
            result_tensor = result_tensor.to(device).to(torch.float32).clamp(0.0, 1.0)
            print(f"输出形状: {result_tensor.shape}")
            return io.NodeOutput(result_tensor)
            
        except Exception as e:
            print(f"视频收集显示错误: {str(e)}")
            return io.NodeOutput(torch.zeros(1, 256, 256, 3))
    
    @classmethod
    def _extract_batch_list(cls, video_collection):
        """从video_collection中提取批量图片列表"""
        if isinstance(video_collection, list):
            return [item for item in video_collection if isinstance(item, torch.Tensor)]
        elif isinstance(video_collection, torch.Tensor):
            return [video_collection]
        else:
            print(f"未知的输入类型: {type(video_collection)}")
            return []
    
    @classmethod
    def _combine_frame_images(
        cls, frame_images, layout, spacing, grid_columns, background_color
    ):
        """合并单帧的多个图片"""
        if not frame_images:
            return torch.zeros(256, 256, 3)
        
        # 统一图片尺寸
        normalized_images = cls._normalize_image_sizes(frame_images)
        has_alpha = any(int(img.shape[-1]) == 4 for img in normalized_images)
        
        # 转换为PIL图像
        pil_images = []
        for img_tensor in normalized_images:
            pil_img = cls._tensor_to_pil_image(img_tensor).convert(
                "RGBA" if has_alpha else "RGB"
            )
            pil_images.append(pil_img)
        
        # 解析背景颜色
        bg_color = cls._parse_color(background_color, has_alpha=has_alpha)
        
        # 合并图像
        result_img = cls._combine_images(
            pil_images, layout, spacing, grid_columns, bg_color
        )
        
        # 转换回tensor
        result_np = np.array(result_img).astype(np.float32) / 255.0
        return torch.from_numpy(result_np)
    
    @classmethod
    def _combine_images(cls, pil_images, layout, spacing, grid_columns, bg_color):
        """统一的图像合并逻辑"""
        if not pil_images:
            mode = "RGBA" if isinstance(bg_color, tuple) and len(bg_color) == 4 else "RGB"
            return Image.new(mode, (256, 256), bg_color)
        
        has_alpha = any(img.mode == "RGBA" for img in pil_images)
        mode = "RGBA" if has_alpha else "RGB"
        if has_alpha:
            pil_images = [img.convert("RGBA") for img in pil_images]
            bg_color = cls._parse_color(bg_color, has_alpha=True)
        else:
            pil_images = [img.convert("RGB") for img in pil_images]
            bg_color = cls._parse_color(bg_color, has_alpha=False)
        
        # 获取所有图像的尺寸
        widths = [img.width for img in pil_images]
        heights = [img.height for img in pil_images]
        num_images = len(pil_images)

        placements: list[tuple[Image.Image, int, int]] = []
        
        if layout == "horizontal":
            # 水平排列
            total_width = sum(widths) + spacing * (num_images - 1)
            total_height = max(heights)
            
            x_offset = 0
            for img in pil_images:
                y_offset = (total_height - img.height) // 2
                placements.append((img, x_offset, y_offset))
                x_offset += img.width + spacing
                
        elif layout == "vertical":
            # 垂直排列
            total_width = max(widths)
            total_height = sum(heights) + spacing * (num_images - 1)
            
            y_offset = 0
            for img in pil_images:
                x_offset = (total_width - img.width) // 2
                placements.append((img, x_offset, y_offset))
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

                placements.append((img, x_offset + x_center, y_offset + y_center))

        canvas_size = (total_width, total_height)
        if mode == "RGBA":
            result_img = Image.new(mode, canvas_size, (0, 0, 0, 0))
            for img, x_offset, y_offset in placements:
                result_img.paste(img, (x_offset, y_offset), img)

            fill_mask = Image.new("L", canvas_size, 255)
            for img, x_offset, y_offset in placements:
                fill_mask.paste(
                    0,
                    (x_offset, y_offset, x_offset + img.width, y_offset + img.height),
                )
            background_img = Image.new(mode, canvas_size, bg_color)
            result_img.paste(background_img, (0, 0), fill_mask)
            return result_img

        result_img = Image.new(mode, canvas_size, bg_color)
        for img, x_offset, y_offset in placements:
            result_img.paste(img, (x_offset, y_offset), None)
        return result_img
    
    @classmethod
    def _normalize_image_sizes(cls, images):
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

    @staticmethod
    def _tensor_to_pil_image(image_tensor: torch.Tensor) -> Image.Image:
        image_np = np.clip(
            image_tensor.detach().cpu().numpy() * 255.0, 0, 255
        ).astype(np.uint8)
        channels = int(image_np.shape[2]) if image_np.ndim == 3 else 1
        if channels == 4:
            return Image.fromarray(image_np, mode="RGBA")
        if channels == 3:
            return Image.fromarray(image_np, mode="RGB")
        if channels == 1:
            return Image.fromarray(image_np[:, :, 0], mode="L")
        return Image.fromarray(image_np[:, :, :3], mode="RGB")
    
    @classmethod
    def _parse_color(cls, color_str, has_alpha=False):
        """解析不同格式的颜色输入"""
        if isinstance(color_str, tuple):
            if has_alpha:
                if len(color_str) == 4:
                    return color_str
                if len(color_str) == 3:
                    return (color_str[0], color_str[1], color_str[2], 255)
            return color_str[:3] if len(color_str) >= 3 else (255, 255, 255)
        color_str = str(color_str).strip()
        lower_text = color_str.lower()

        if has_alpha and lower_text in {"", "transparent", "none", "null", "off", "disable", "disabled"}:
            return (0, 0, 0, 0)
        
        # 尝试解析为灰度值 (0.0-1.0)
        try:
            gray_value = float(color_str)
            if 0.0 <= gray_value <= 1.0:
                # 灰度值转换为RGB
                gray_int = int(gray_value * 255)
                if has_alpha:
                    return (gray_int, gray_int, gray_int, 255)
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
                if has_alpha:
                    return (r, g, b, 255)
                return (r, g, b)
            except ValueError:
                pass
        if has_alpha and len(hex_color) == 8:
            try:
                r = int(hex_color[0:2], 16)
                g = int(hex_color[2:4], 16)
                b = int(hex_color[4:6], 16)
                a = int(hex_color[6:8], 16)
                return (r, g, b, a)
            except ValueError:
                pass
        
        # 尝试解析为RGB/RGBA格式
        try:
            rgb = color_str.split(',')
            if len(rgb) == 3:
                r = int(rgb[0].strip())
                g = int(rgb[1].strip())
                b = int(rgb[2].strip())
                if has_alpha:
                    return (r, g, b, 255)
                return (r, g, b)
            if has_alpha and len(rgb) == 4:
                r = int(rgb[0].strip())
                g = int(rgb[1].strip())
                b = int(rgb[2].strip())
                a = int(rgb[3].strip())
                return (r, g, b, a)
        except ValueError:
            pass
        
        # 默认返回白色
        return (255, 255, 255, 255) if has_alpha else (255, 255, 255)
