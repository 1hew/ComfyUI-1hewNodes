import numpy as np
import torch

from comfy_api.latest import io
from PIL import Image, ImageFilter


class ImageBlur(io.ComfyNode):
    """对整张图像做高斯模糊，行为对标 LayerStyle 的 LayerFilter: Gaussian Blur V2。

    - blur 支持小数(FLOAT)，语义上更接近 V2 版本；
    - blur 为 0 时直接原图透传；
    - 按 batch 逐帧处理，保持通道数与 dtype 不变。
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_ImageBlur",
            display_name="Image Blur",
            category="1hewNodes/image",
            inputs=[
                io.Image.Input("image"),
                io.Float.Input("blur", default=20.0, min=0.0, max=1000.0, step=0.01),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
            ],
        )

    @classmethod
    async def execute(cls, image: torch.Tensor, blur: float) -> io.NodeOutput:
        if not blur or blur <= 0.0:
            # blur 为 0：不做处理，直接透传，与 LayerStyle V2 行为一致
            return io.NodeOutput(image)

        bsz = int(image.shape[0])
        img_np = (
            image.detach().cpu().numpy()
            if image.is_cuda
            else image.detach().numpy()
        )

        blurred_frames: list[torch.Tensor] = []
        for i in range(bsz):
            frame = img_np[i]  # (H, W, C)
            channel_count = int(frame.shape[2])

            channels: list[np.ndarray] = []
            for ch_idx in range(channel_count):
                ch = (frame[:, :, ch_idx] * 255.0).clip(0, 255).astype(np.uint8)
                pil_ch = Image.fromarray(ch, mode="L")
                blurred_pil = pil_ch.filter(ImageFilter.GaussianBlur(radius=float(blur)))
                channels.append(np.asarray(blurred_pil).astype(np.float32) / 255.0)

            blurred = np.stack(channels, axis=-1)
            blurred_frames.append(torch.from_numpy(blurred))

        return io.NodeOutput(torch.stack(blurred_frames))
