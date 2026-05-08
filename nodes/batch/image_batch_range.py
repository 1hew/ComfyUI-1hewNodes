import asyncio
from comfy_api.latest import io
import torch


class ImageBatchRange(io.ComfyNode):
    @staticmethod
    def _empty_image(image: torch.Tensor) -> torch.Tensor:
        return torch.empty(
            (0,) + tuple(image.shape[1:]),
            dtype=image.dtype,
            device=image.device,
        )

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_ImageBatchRange",
            display_name="Image Batch Range",
            category="1hewNodes/batch",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("start_index", default=0, min=0, max=8192, step=1),
                io.Int.Input("step", default=1, min=1, max=8192, step=1),
                io.Int.Input("num_frame", default=1, min=0, max=8192, step=1),
            ],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    async def execute(
        cls, image: torch.Tensor, start_index: int, step: int, num_frame: int
    ) -> io.NodeOutput:
        try:
            total = int(image.shape[0])
            start = max(0, int(start_index))
            stride = max(1, int(step))
            count = max(0, int(num_frame))

            if total <= 0 or start >= total:
                return io.NodeOutput(cls._empty_image(image))

            async def _slice():
                def _do():
                    if count == 0:
                        return image[start::stride]
                    stop = start + (stride * count)
                    return image[start:stop:stride]

                return await asyncio.to_thread(_do)

            selected_image = await _slice()
            if int(selected_image.shape[0]) == 0:
                return io.NodeOutput(cls._empty_image(image))
            return io.NodeOutput(selected_image)
        except Exception:
            return io.NodeOutput(cls._empty_image(image))