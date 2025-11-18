from comfy_api.latest import io
import torch


class IntMaskSideLength(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_IntMaskSideLength",
            display_name="Int Mask Side Length",
            category="1hewNodes/int",
            inputs=[
                io.Mask.Input("mask"),
                io.Combo.Input("mode", options=["longest", "shortest", "width", "height"], default="longest"),
            ],
            outputs=[io.Int.Output(display_name="int")],
        )

    @classmethod
    async def execute(cls, mask: torch.Tensor, mode: str) -> io.NodeOutput:
        h = int(mask.shape[-2])
        w = int(mask.shape[-1])
        if mode == "shortest":
            value = min(w, h)
        elif mode == "width":
            value = w
        elif mode == "height":
            value = h
        else:
            value = max(w, h)
        return io.NodeOutput(int(value))