from comfy_api.latest import io


class IntWan(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_IntWan",
            display_name="Int Wan",
            category="1hewNodes/int",
            inputs=[
                io.Int.Input("value", default=1, min=1, max=10000, step=4, display_mode=io.NumberDisplay.number),
            ],
            outputs=[io.Int.Output(display_name="int")],
        )

    @classmethod
    async def execute(cls, value: int) -> io.NodeOutput:
        return io.NodeOutput(int(value))

