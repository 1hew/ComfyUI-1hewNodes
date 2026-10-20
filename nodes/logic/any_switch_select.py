from comfy_api.latest import io


class AnySwitchSelect(io.ComfyNode):
    SELECT_MAX = 10

    @classmethod
    def define_schema(cls) -> io.Schema:
        # Every dynamic port must be declared here, even though the front-end
        # adds input_2..input_10 on demand. ComfyUI's TopologicalSort only
        # skips an upstream branch when the port's schema entry carries
        # "lazy"; undeclared inputs have no schema info and are treated as
        # eager, which schedules every connected branch at once.
        inputs = [
            io.Int.Input("select", default=1, min=1, max=cls.SELECT_MAX, step=1),
            io.Custom("*").Input("input_1", optional=True, lazy=True),
        ]
        for i in range(2, cls.SELECT_MAX + 1):
            inputs.append(
                io.Custom("*").Input(f"input_{i}", optional=True, lazy=True)
            )
        return io.Schema(
            node_id="1hew_AnySwitchSelect",
            display_name="Any Switch Select",
            category="1hewNodes/logic",
            inputs=inputs,
            outputs=[
                io.Custom("*").Output(display_name="output"),
                io.Int.Output(display_name="select"),
            ],
        )

    @classmethod
    def validate_inputs(cls, **kwargs):
        return True

    @classmethod
    def check_lazy_status(cls, select, **kwargs):
        key = cls._selected_key(select)
        if key not in kwargs:
            return []
        if kwargs.get(key) is None:
            return [key]
        return []

    @classmethod
    async def execute(cls, select: int, **kwargs) -> io.NodeOutput:
        try:
            idx = cls._clamp_select(select)
            key = cls._selected_key(select)
            if key in kwargs:
                return io.NodeOutput(kwargs[key], idx)
            return io.NodeOutput(None, idx)
        except Exception as e:
            print(f"AnySwitchSelect error: {e}")
            return io.NodeOutput(None, 1)

    @classmethod
    def _clamp_select(cls, select):
        try:
            idx = int(select)
        except Exception:
            idx = 1
        return max(1, min(idx, cls.SELECT_MAX))

    @classmethod
    def _selected_key(cls, select):
        return f"input_{cls._clamp_select(select)}"
