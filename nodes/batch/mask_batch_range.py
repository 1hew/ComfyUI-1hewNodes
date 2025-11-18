import asyncio
from comfy_api.latest import io
import torch


class MaskBatchRange(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_MaskBatchRange",
            display_name="Mask Batch Range",
            category="1hewNodes/batch",
            inputs=[
                io.Mask.Input("mask"),
                io.Int.Input("start_index", default=0, min=0, max=8192, step=1),
                io.Int.Input("num_frame", default=1, min=1, max=8192, step=1),
            ],
            outputs=[io.Mask.Output(display_name="mask")],
        )

    @classmethod
    async def execute(
        cls, mask: torch.Tensor, start_index: int, num_frame: int
    ) -> io.NodeOutput:
        try:
            total = int(mask.shape[0])
            start = max(0, int(start_index))

            if total <= 0 or start >= total:
                empty_msk = torch.empty(
                    (0,) + tuple(mask.shape[1:]),
                    dtype=mask.dtype,
                    device=mask.device,
                )
                return io.NodeOutput(empty_msk)

            take = max(0, min(int(num_frame), total - start))

            if take == 0:
                empty_msk = torch.empty(
                    (0,) + tuple(mask.shape[1:]),
                    dtype=mask.dtype,
                    device=mask.device,
                )
                return io.NodeOutput(empty_msk)

            async def _slice():
                def _do():
                    return mask[start : start + take]

                return await asyncio.to_thread(_do)

            selected_mask = await _slice()
            return io.NodeOutput(selected_mask)
        except Exception:
            empty_msk = torch.empty(
                (0,) + tuple(mask.shape[1:]), dtype=mask.dtype, device=mask.device
            )
            return io.NodeOutput(empty_msk)