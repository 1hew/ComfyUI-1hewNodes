from comfy_api.latest import io


class MultiSwitchSelect(io.ComfyNode):
    SELECT_MAX = 10

    @classmethod
    def define_schema(cls):
        inputs = [
            io.Int.Input("select", default=1, min=1, max=cls.SELECT_MAX, step=1),
            io.Custom("*").Input("input_1", optional=True, lazy=True),
        ]
        for i in range(2, cls.SELECT_MAX + 1):
            inputs.append(
                io.Custom("*").Input(f"input_{i}", optional=True, lazy=True)
            )
        return io.Schema(
            node_id="1hew_MultiSwitchSelect",
            display_name="Multi Switch Select",
            category="1hewNodes/logic",
            inputs=inputs,
            outputs=[
                io.Custom("*").Output(display_name=f"output_{i}")
                for i in range(1, cls.SELECT_MAX + 1)
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
    async def execute(cls, select, **kwargs) -> io.NodeOutput:
        idx = cls._clamp_select(select)
        key = cls._selected_key(select)
        value = kwargs.get(key)

        results = [None] * cls.SELECT_MAX
        results[idx - 1] = value
        return io.NodeOutput(*results)

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
