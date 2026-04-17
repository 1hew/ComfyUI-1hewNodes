import torch
from comfy_api.latest import io


class ImageBatchInterleave(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_ImageBatchInterleave",
            display_name="Image Batch Interleave",
            category="1hewNodes/image",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("segment_count", default=2, min=1, max=100000, step=1),
            ],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    async def execute(
        cls,
        image: torch.Tensor,
        segment_count: int,
    ) -> io.NodeOutput:
        batch_size = int(image.shape[0])
        if batch_size <= 0:
            if image.ndim == 4:
                _, h, w, c = image.shape
                empty = torch.zeros(
                    (0, h, w, c), dtype=image.dtype, device=image.device
                )
            else:
                empty = torch.zeros((0, 64, 64, 3), dtype=torch.float32)
            return io.NodeOutput(empty)

        actual_segments = max(1, min(int(segment_count), batch_size))
        indices = cls._build_interleave_indices(batch_size, actual_segments)
        index_t = torch.tensor(indices, dtype=torch.long, device=image.device)
        return io.NodeOutput(image[index_t])

    @staticmethod
    def _build_interleave_indices(batch_size: int, segment_count: int) -> list[int]:
        base_size = batch_size // segment_count
        remainder = batch_size % segment_count

        segments: list[list[int]] = []
        start = 0
        for segment_idx in range(segment_count):
            current_size = base_size + (1 if segment_idx < remainder else 0)
            end = start + current_size
            segments.append(list(range(start, end)))
            start = end

        indices: list[int] = []
        max_segment_length = max((len(seg) for seg in segments), default=0)
        for offset in range(max_segment_length):
            for seg in segments:
                if offset < len(seg):
                    indices.append(seg[offset])
        return indices
