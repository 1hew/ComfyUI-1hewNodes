import asyncio

from comfy_api.latest import io
import torch


class ImageBatchUniform(io.ComfyNode):
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
            node_id="1hew_ImageBatchUniform",
            display_name="Image Batch Uniform",
            category="1hewNodes/batch",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("num_frame", default=4, min=0, max=8192, step=1),
            ],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    async def execute(cls, image: torch.Tensor, num_frame: int) -> io.NodeOutput:
        try:
            batch_size = int(image.shape[0])
            if batch_size <= 0:
                return io.NodeOutput(cls._empty_image(image))

            indices = cls._calculate_uniform_indices(batch_size, max(0, int(num_frame)))
            if not indices:
                return io.NodeOutput(cls._empty_image(image))

            chunk_size = 512
            chunks = [indices[i : i + chunk_size] for i in range(0, len(indices), chunk_size)]

            async def _gather_chunk(chunk: list[int]) -> torch.Tensor:
                return await asyncio.to_thread(lambda: image[chunk])

            parts = await asyncio.gather(*[_gather_chunk(ch) for ch in chunks])
            extracted = torch.cat(parts, dim=0)
            return io.NodeOutput(extracted)
        except Exception:
            return io.NodeOutput(cls._empty_image(image))

    @staticmethod
    def _calculate_uniform_indices(batch_size: int, count: int) -> list[int]:
        if batch_size <= 0:
            return []
        if count == 0 or count >= batch_size:
            return list(range(batch_size))
        if count == 1:
            return [0]
        if count == 2:
            return [0, batch_size - 1] if batch_size > 1 else [0]

        raw_positions = [i * (batch_size - 1) / float(count - 1) for i in range(count)]
        indices: list[int] = []
        used: set[int] = set()

        for pos in raw_positions:
            base = int(round(pos))
            if base not in used:
                used.add(base)
                indices.append(base)
                continue

            for delta in range(1, batch_size):
                left = base - delta
                right = base + delta
                candidates = []
                if 0 <= left < batch_size:
                    candidates.append(left)
                if 0 <= right < batch_size:
                    candidates.append(right)

                chosen = next((idx for idx in candidates if idx not in used), None)
                if chosen is not None:
                    used.add(chosen)
                    indices.append(chosen)
                    break

        if len(indices) >= batch_size:
            return list(range(batch_size))
        return sorted(indices)
