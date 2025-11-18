from comfy_api.latest import io
import torch


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
        if image.dim() == 3:
            image = image.unsqueeze(0)
        height = int(image.shape[1])
        width = int(image.shape[2])
        return io.NodeOutput(width, height)
