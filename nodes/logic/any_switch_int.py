import inspect
from comfy_api.latest import io


class AnySwitchInt(io.ComfyNode):
    SELECT_MAX = 10
    @classmethod
    def define_schema(cls) -> io.Schema:
        dyn_inpts = [
            io.Int.Input("select", default=1, min=1, max=cls.SELECT_MAX, step=1),
            io.Custom("*").Input("input_1", optional=True, lazy=True),
        ]
        for i in range(2, cls.SELECT_MAX + 1):
            dyn_inpts.append(
                io.Custom("*").Input(f"input_{i}", optional=True, lazy=True)
            )
        return io.Schema(
            node_id="1hew_AnySwitchInt",
            display_name="Any Switch Int",
            category="1hewNodes/logic",
            inputs=dyn_inpts,
            outputs=[io.Custom("*").Output(display_name="output")],
        )

    @classmethod
    def validate_inputs(cls, **kwargs):
        return True

    @classmethod
    def check_lazy_status(cls, select, **kwargs):
        try:
            idx = int(select)
        except Exception:
            idx = 1
        if idx < 1:
            idx = 1
        if idx > cls.SELECT_MAX:
            idx = cls.SELECT_MAX
        return [f"input_{idx}"]

    @classmethod
    async def execute(cls, select: int, **kwargs) -> io.NodeOutput:
        try:
            idx = int(select)
            if idx < 1:
                idx = 1
            if idx > cls.SELECT_MAX:
                idx = cls.SELECT_MAX
            key = f"input_{idx}"
            if key in kwargs:
                return io.NodeOutput(kwargs[key])
            return io.NodeOutput(None)
        except Exception as e:
            print(f"AnySwitchInt error: {e}")
            return io.NodeOutput(None)
