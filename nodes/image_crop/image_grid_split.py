import asyncio
from comfy_api.latest import io
import numpy as np
from PIL import Image
import torch

class ImageGridSplit(io.ComfyNode):
    """
    图片宫格分割器 - 将图片按指定行列分割成多个子图片
    """
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_ImageGridSplit",
            display_name="Image Grid Split",
            category="1hewNodes/image/crop",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("rows", default=2, min=1, max=10, step=1),
                io.Int.Input("columns", default=2, min=1, max=10, step=1),
                io.Int.Input("output_index", default=0, min=0, max=100, step=1),
            ],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    async def execute(
        cls,
        image: torch.Tensor,
        rows: int,
        columns: int,
        output_index: int,
    ) -> io.NodeOutput:
        """
        将图片按宫格分割
        
        Args:
            image: 输入图片张量 (batch, height, width, channels)
            rows: 分割行数
            columns: 分割列数
            output_index: 输出索引，0表示所有分割后的图片按批次输出，
                         1表示第1张，2表示第2张，以此类推（横向优先）
        
        Returns:
            分割后的图片张量
        """
        image = image.to(torch.float32).clamp(0.0, 1.0)
        batch_size, height, width, channels = image.shape
        
        # 计算每个网格的尺寸
        grid_height = height // rows
        grid_width = width // columns
        
        # 总的网格数量
        total_grids = rows * columns
        
        # 验证输出索引
        if output_index > total_grids:
            raise ValueError(f"输出索引 {output_index} 超出范围，最大值为 {total_grids}")
        
        async def _split_one(batch_idx):
            def _do():
                cur = image[batch_idx]
                out = []
                for row in range(rows):
                    for col in range(columns):
                        sy = row * grid_height
                        ey = sy + grid_height
                        sx = col * grid_width
                        ex = sx + grid_width
                        out.append(cur[sy:ey, sx:ex, :])
                return out
            return await asyncio.to_thread(_do)

        parts = await asyncio.gather(*[_split_one(b) for b in range(batch_size)])
        all_split_images = [img for sub in parts for img in sub]
        
        # 根据输出索引返回结果
        if output_index == 0:
            result = torch.stack(all_split_images, dim=0)
        else:
            selected_images = []
            for batch_idx in range(batch_size):
                grid_idx = (output_index - 1) % total_grids
                actual_idx = batch_idx * total_grids + grid_idx
                if actual_idx < len(all_split_images):
                    selected_images.append(all_split_images[actual_idx])
                else:
                    selected_images.append(all_split_images[0])
            result = torch.stack(selected_images, dim=0)
        return io.NodeOutput(result)

    @staticmethod
    def tensor_to_pil(tensor):
        """将张量转换为PIL图像"""
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        # 确保值在0-1范围内
        tensor = torch.clamp(tensor, 0, 1)
        
        # 转换为numpy数组
        np_image = (tensor.cpu().numpy() * 255).astype(np.uint8)
        
        return Image.fromarray(np_image)

    @staticmethod
    def pil_to_tensor(pil_image):
        """将PIL图像转换为张量"""
        np_image = np.array(pil_image).astype(np.float32) / 255.0
        
        # 确保是RGB格式
        if len(np_image.shape) == 2:
            np_image = np.stack([np_image] * 3, axis=-1)
        elif np_image.shape[-1] == 4:
            np_image = np_image[:, :, :3]
        
        tensor = torch.from_numpy(np_image)
        
        # 添加批次维度
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        
        return tensor
