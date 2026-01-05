from comfy_api.latest import io
import torch

from ...utils import first_torch_tensor, make_ui_text


class IntImageSize(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_IntImageSize",
            display_name="Int Image Size",
            category="1hewNodes/int",
            inputs=[
                io.Image.Input("image"),
            ],
            outputs=[
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
            ],
        )

    @classmethod
    async def execute(cls, image: torch.Tensor) -> io.NodeOutput:
        tensor = first_torch_tensor(image)
        if tensor is None or getattr(tensor, "ndim", 0) < 3:
            return io.NodeOutput(
                0,
                0,
                ui=make_ui_text("0x0"),
            )

        height = int(tensor.shape[-3])
        width = int(tensor.shape[-2])
        return io.NodeOutput(
            width,
            height,
            ui=make_ui_text(f"{width}x{height}"),
        )
