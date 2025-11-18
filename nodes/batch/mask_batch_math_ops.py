
import asyncio
from comfy_api.latest import io
import torch


class MaskBatchMathOps(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_MaskBatchMathOps",
            display_name="Mask Batch Math Ops",
            category="1hewNodes/batch",
            inputs=[
                io.Mask.Input("mask"),
                io.Combo.Input("operation", options=["or", "and"], default="or"),
            ],
            outputs=[io.Mask.Output(display_name="mask")],
        )

    @classmethod
    async def execute(cls, mask: torch.Tensor, operation: str) -> io.NodeOutput:
        if mask.ndim == 4 and mask.shape[-1] == 1:
            mask = mask[..., 0]

        bs = int(mask.shape[0]) if mask.ndim == 3 else 0
        if bs <= 1:
            return io.NodeOutput(mask)

        chunk_size = 512
        chunks = [mask[i : i + chunk_size] for i in range(0, bs, chunk_size)]

        if operation == "or":
            async def _reduce(ch):
                def _do():
                    return torch.max(ch, dim=0).values
                return await asyncio.to_thread(_do)
            parts = await asyncio.gather(*[_reduce(ch) for ch in chunks])
            stacked = torch.stack(parts, dim=0)
            agg = torch.max(stacked, dim=0).values
        else:
            async def _reduce(ch):
                def _do():
                    return torch.min(ch, dim=0).values
                return await asyncio.to_thread(_do)
            parts = await asyncio.gather(*[_reduce(ch) for ch in chunks])
            stacked = torch.stack(parts, dim=0)
            agg = torch.min(stacked, dim=0).values

        out = agg.unsqueeze(0).to(torch.float32).clamp(0.0, 1.0)
        out = out.to(mask.device)
        return io.NodeOutput(out)