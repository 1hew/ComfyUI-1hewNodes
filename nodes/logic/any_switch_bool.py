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
    async def execute(
        cls, boolean: bool, on_true=None, on_false=None
    ) -> io.NodeOutput:
        try:
            return io.NodeOutput(on_true if boolean else on_false)
        except Exception as e:
            print(f"AnySwitchBool error: {e}")
            return io.NodeOutput(None)

    @classmethod
    def check_lazy_status(cls, boolean, on_true=None, on_false=None):
        if boolean:
            return ["on_true"]
        return ["on_false"]
