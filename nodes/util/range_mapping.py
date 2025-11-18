from comfy_api.latest import io


class RangeMapping(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_RangeMapping",
            display_name="Range Mapping",
            category="1hewNodes/util",
            description="将0-1范围值映射到[min,max]，支持小数精度控制。",
            inputs=[
                io.Float.Input(
                    "value",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.001,
                    display_mode=io.NumberDisplay.slider,
                ),
                io.Float.Input(
                    "min",
                    default=0.0,
                    min=-0xFFFFFFFFFFFFFFFF,
                    max=0xFFFFFFFFFFFFFFFF,
                    step=0.001,
                    display_mode=io.NumberDisplay.number,
                ),
                io.Float.Input(
                    "max",
                    default=1.0,
                    min=-0xFFFFFFFFFFFFFFFF,
                    max=0xFFFFFFFFFFFFFFFF,
                    step=0.001,
                    display_mode=io.NumberDisplay.number,
                ),
                io.Int.Input(
                    "rounding",
                    default=3,
                    min=0,
                    max=10,
                    step=1,
                    display_mode=io.NumberDisplay.number,
                ),
            ],
            outputs=[io.Float.Output(), io.Int.Output()],
        )

    @classmethod
    def execute(
        cls,
        value: float,
        min: float,
        max: float,
        rounding: int,
    ) -> io.NodeOutput:
        actual_value = min + value * (max - min)
        if rounding > 0:
            actual_value = round(actual_value, rounding)
        else:
            actual_value = int(actual_value)
        return io.NodeOutput(actual_value, int(actual_value))

