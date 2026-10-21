from comfy_api.latest import io

from ...utils._logic_ops import (
    is_empty,
)


class AnyEmptyInt(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_AnyEmptyInt",
            display_name="Any Empty Int",
            category="1hewNodes/logic",
            inputs=[
                io.Custom("*").Input("any", optional=True),
                io.Int.Input("empty", default=0, min=-999999, max=999999, step=1),
                io.Int.Input(
                    "not_empty", default=1, min=-999999, max=999999, step=1
                ),
            ],
            outputs=[io.Int.Output(display_name="int")],
        )

    @classmethod
    def validate_inputs(cls, **kwargs):
        return True

    @classmethod
    async def execute(cls, any=None, empty=0, not_empty=1) -> io.NodeOutput:
        try:
            empty_value = is_empty(any)
            return io.NodeOutput(int(empty if empty_value else not_empty))
        except Exception as e:
            print(f"AnyEmptyInt error: {e}")
            return io.NodeOutput(int(empty))
