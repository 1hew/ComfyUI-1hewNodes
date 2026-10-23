from typing import Any

from comfy_api.latest import io


class AnyToAny(io.ComfyNode):
    """Wildcard pass-through that erases the declared link type on both ends.

    ComfyUI refuses to create a link when the upstream output type and the
    downstream input type differ. Declaring both ports as the wildcard type
    (*) makes any typed output connectable to any typed input; the value
    itself is forwarded untouched, so this node only lifts the connection
    restriction and never casts data.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_AnyToAny",
            display_name="Any to Any",
            category="1hewNodes/conversion",
            inputs=[
                io.Custom("*").Input(
                    "any",
                    optional=True,
                    tooltip="Any upstream value; forwarded unchanged.",
                ),
            ],
            outputs=[
                io.Custom("*").Output(display_name="any"),
            ],
        )

    @classmethod
    def validate_inputs(cls, **kwargs):
        # The wildcard input already accepts every type; overriding this also
        # keeps legacy workflows usable if an old link recorded another type.
        return True

    @classmethod
    async def execute(cls, any: Any = None) -> io.NodeOutput:
        return io.NodeOutput(any)
