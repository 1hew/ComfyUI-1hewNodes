from comfy_api.latest import io
import torch


class IntImageSideLength(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_IntImageSideLength",
            display_name="Int Image Side Length",
            category="1hewNodes/int",
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("mode", options=["longest", "shortest", "width", "height"], default="longest"),
            ],
            outputs=[io.Int.Output(display_name="int")],
        )

    @classmethod
    async def execute(cls, image: torch.Tensor, mode: str) -> io.NodeOutput:
        h = int(image.shape[1])
        w = int(image.shape[2])
        if mode == "shortest":
            value = min(w, h)
        elif mode == "width":
            value = w
        elif mode == "height":
            value = h
        else:
            value = max(w, h)
        return io.NodeOutput(int(value))