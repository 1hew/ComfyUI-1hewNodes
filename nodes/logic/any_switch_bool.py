from comfy_api.latest import io


class AnySwitchBool(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_AnySwitchBool",
            display_name="Any Switch Bool",
            category="1hewNodes/logic",
            inputs=[
                io.Boolean.Input("boolean", default=True),
                io.Custom("*").Input("on_true", optional=True, lazy=True),
                io.Custom("*").Input("on_false", optional=True, lazy=True),
            ],
            outputs=[io.Custom("*").Output(display_name="output")],
        )

    @classmethod
    def validate_inputs(cls, **kwargs):
        return True

    @classmethod
    async def execute(cls, boolean: bool, **kwargs) -> io.NodeOutput:
        try:
            key = cls._selected_key(boolean)
            return io.NodeOutput(kwargs.get(key))
        except Exception as e:
            print(f"AnySwitchBool error: {e}")
            return io.NodeOutput(None)

    @classmethod
    def check_lazy_status(cls, boolean, **kwargs):
        key = cls._selected_key(boolean)
        if key not in kwargs:
            return []
        if kwargs.get(key) is None:
            return [key]
        return []

    @staticmethod
    def _selected_key(boolean):
        return "on_true" if bool(boolean) else "on_false"
