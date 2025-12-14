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
                io.Int.Input("index", default=0, min=-100, max=100, step=1),
                io.Boolean.Input("all", default=False),
            ],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    async def execute(
        cls,
        image: torch.Tensor,
        rows: int,
        columns: int,
        index: int,
        all: bool,
    ) -> io.NodeOutput:
        """
        将图片按宫格分割
        
        Args:
            image: 输入图片张量 (batch, height, width, channels)
            rows: 分割行数
            columns: 分割列数
            index: 输出索引，支持 Python 风格索引（如 0 为第一个，-1 为最后一个）
            all: 是否输出所有分割后的图片。若为 True，则 index 参数无效。
        
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
        
        if all:
            # 返回所有分割后的图片
            result = torch.stack(all_split_images, dim=0)
        else:
            # 处理索引
            if index < 0:
                index += total_grids
            
            if index < 0 or index >= total_grids:
                raise ValueError(f"索引 {index} 超出范围，有效范围为 0 到 {total_grids - 1} (或 -{total_grids} 到 -1)")

            selected_images = []
            for batch_idx in range(batch_size):
                # 计算当前批次中对应的网格索引
                actual_idx = batch_idx * total_grids + index
                if actual_idx < len(all_split_images):
                    selected_images.append(all_split_images[actual_idx])
                else:
                    # 理论上不应该走到这里，因为上面已经做了范围检查
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
