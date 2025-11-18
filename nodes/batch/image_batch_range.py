import asyncio
from comfy_api.latest import io
import torch


class ImageBatchRange(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_ImageBatchRange",
            display_name="Image Batch Range",
            category="1hewNodes/batch",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("start_index", default=0, min=0, max=8192, step=1),
                io.Int.Input("num_frame", default=1, min=1, max=8192, step=1),
            ],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    async def execute(
        cls, image: torch.Tensor, start_index: int, num_frame: int
    ) -> io.NodeOutput:
        try:
            total = int(image.shape[0])
            start = max(0, int(start_index))

            if total <= 0 or start >= total:
                empty_img = torch.empty(
                    (0,) + tuple(image.shape[1:]),
                    dtype=image.dtype,
                    device=image.device,
                )
                return io.NodeOutput(empty_img)

            take = max(0, min(int(num_frame), total - start))

            if take == 0:
                empty_img = torch.empty(
                    (0,) + tuple(image.shape[1:]),
                    dtype=image.dtype,
                    device=image.device,
                )
                return io.NodeOutput(empty_img)

            async def _slice():
                def _do():
                    return image[start : start + take]

                return await asyncio.to_thread(_do)

            selected_image = await _slice()
            return io.NodeOutput(selected_image)
        except Exception:
            empty_img = torch.empty(
                (0,) + tuple(image.shape[1:]), dtype=image.dtype, device=image.device
            )
            return io.NodeOutput(empty_img)