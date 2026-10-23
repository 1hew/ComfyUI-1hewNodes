from typing import Any

from comfy_api.latest import io

from ...utils import make_ui_text
from ...utils._count_ops import collect_prefixed_values, count_text_items


class IntTextCount(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_IntTextCount",
            display_name="Int Text Count",
            category="1hewNodes/int",
            # Wildcard ports so STRING / INT / FLOAT (and BOOL) can share the
            # same dynamic text_X sockets.  is_input_list keeps an upstream
            # string list in one call so each item can be counted.
            is_input_list=True,
            accept_all_inputs=True,
            inputs=[
                io.Custom("*").Input("text_1", optional=True),
            ],
            outputs=[io.Int.Output(display_name="int")],
        )

    @classmethod
    async def execute(cls, text_1: Any = None, **kwargs) -> io.NodeOutput:
        count = sum(
            count_text_items(value)
            for value in collect_prefixed_values("text_", text_1, kwargs)
        )
        return io.NodeOutput(int(count), ui=make_ui_text(str(int(count))))
