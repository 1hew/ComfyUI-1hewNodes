from comfy_api.latest import io
import torch

from ...utils import first_torch_tensor, make_ui_text


class IntMaskSideLength(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_IntMaskSideLength",
            display_name="Int Mask Side Length",
            category="1hewNodes/int",
            inputs=[
                io.Mask.Input("mask"),
                io.Combo.Input("mode", options=["longest", "shortest", "width", "height"], default="shortest"),
            ],
            outputs=[io.Int.Output(display_name="int")],
        )

    @classmethod
    async def execute(cls, mask: torch.Tensor, mode: str) -> io.NodeOutput:
        tensor = first_torch_tensor(mask)
        if tensor is None or getattr(tensor, "ndim", 0) < 2:
            return io.NodeOutput(
                0,
                ui=make_ui_text("0"),
            )

        h = int(tensor.shape[-2])
        w = int(tensor.shape[-1])
        if mode == "shortest":
            value = min(w, h)
        elif mode == "width":
            value = w
        elif mode == "height":
            value = h
        else:
            value = max(w, h)
        value = int(value)
        return io.NodeOutput(
            value,
            ui=make_ui_text(str(value)),
        )
