import torch
from comfy_api.latest import io


class MaskRepeat(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_MaskRepeat",
            display_name="Mask Repeat",
            category="1hewNodes/mask",
            inputs=[
                io.Mask.Input("mask"),
                io.Boolean.Input("invert", default=False),
                io.Int.Input("count", default=1, min=1, max=4096),
            ],
            outputs=[io.Mask.Output(display_name="mask")],
        )

    @classmethod
    async def execute(cls, mask: torch.Tensor, count: int, invert: bool):
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        if invert:
            mask = 1.0 - mask

        s = mask.shape
        # [B, H, W] -> [1, B, H, W] -> [count, B, H, W]
        repeated = mask.unsqueeze(0).repeat(count, 1, 1, 1)
        # -> [count*B, H, W]
        output = repeated.reshape((-1, s[1], s[2]))
        
        return io.NodeOutput(output)
