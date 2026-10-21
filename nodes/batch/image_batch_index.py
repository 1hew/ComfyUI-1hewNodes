import asyncio

from comfy_api.latest import io
import torch

from ...utils._index_range import parse_index_expressions


class ImageBatchIndex(io.ComfyNode):
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
            node_id="1hew_ImageBatchIndex",
            display_name="Image Batch Index",
            category="1hewNodes/batch",
            inputs=[
                io.Image.Input("image"),
                io.String.Input("index", default="0", multiline=True),
            ],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    async def execute(cls, image: torch.Tensor, index: str) -> io.NodeOutput:
        try:
            batch_size = int(image.shape[0])
            if batch_size <= 0:
                return io.NodeOutput(cls._empty_image(image))

            indices = await asyncio.to_thread(cls._parse_indices, index or "", batch_size)
            valid = [i for i in indices if 0 <= i < batch_size]

            if not valid:
                return io.NodeOutput(cls._empty_image(image))

            chunk_size = 512
            chunks = [valid[i : i + chunk_size] for i in range(0, len(valid), chunk_size)]

            async def _gather_chunk(chunk: list[int]) -> torch.Tensor:
                return await asyncio.to_thread(lambda: image[chunk])

            parts = await asyncio.gather(*[_gather_chunk(ch) for ch in chunks])
            extracted = torch.cat(parts, dim=0)
            return io.NodeOutput(extracted)
        except Exception:
            return io.NodeOutput(cls._empty_image(image))

    @staticmethod
    def _resolve_index(value: int, batch_size: int) -> int:
        if value < 0:
            return batch_size + value
        return value

    @classmethod
    def _parse_indices(cls, text: str, batch_size: int) -> list[int]:
        # Shared with Multi Switch Select. Negative endpoints are resolved
        # before a range is expanded, exactly like the previous local parser.
        return parse_index_expressions(
            text,
            resolve=lambda value: cls._resolve_index(value, batch_size),
        )
