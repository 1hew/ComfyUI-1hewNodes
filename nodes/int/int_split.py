from comfy_api.latest import io


class IntSplit(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_IntSplit",
            display_name="Int Split",
            category="1hewNodes/int",
            inputs=[
                io.Int.Input("total", default=20, min=1, max=10000, step=1),
                io.Float.Input("split_point", default=0.5, min=0.0, max=10000.0, step=0.01, display_mode=io.NumberDisplay.number),
            ],
            outputs=[
                io.Int.Output(display_name="int_total"),
                io.Int.Output(display_name="int_split"),
            ],
        )

    @classmethod
    async def execute(cls, total: int, split_point: float) -> io.NodeOutput:
        total_value = int(total)

        if 0.0 <= split_point <= 1.0:
            if split_point == 1.0:
                split_value = 1
            else:
                split_value = int(total_value * split_point)
        else:
            split_value = int(split_point)
            split_value = min(split_value, total_value)

        split_value = max(0, split_value)
        return io.NodeOutput(total_value, split_value)

