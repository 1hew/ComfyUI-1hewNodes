from comfy_api.latest import io

from ...utils._logic_ops import (
    is_empty,
)


class AnyEmptyBool(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_AnyEmptyBool",
            display_name="Any Empty Bool",
            category="1hewNodes/logic",
            inputs=[
                io.Custom("*").Input("any", optional=True),
            ],
            outputs=[io.Boolean.Output(display_name="bool")],
        )

    @classmethod
    def validate_inputs(cls, **kwargs):
        return True

    @classmethod
    async def execute(cls, any=None) -> io.NodeOutput:
        try:
            return io.NodeOutput(bool(is_empty(any)))
        except Exception as e:
            print(f"AnyEmptyBool error: {e}")
            return io.NodeOutput(True)
