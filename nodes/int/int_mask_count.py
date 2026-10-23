from typing import Any

from comfy_api.latest import io

from ...utils import make_ui_text
from ...utils._count_ops import collect_prefixed_values, count_mask_items


class IntMaskCount(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_IntMaskCount",
            display_name="Int Mask Count",
            category="1hewNodes/int",
            # mask_2..mask_N are appended by the front end and arrive through
            # **kwargs.  is_input_list keeps an upstream mask list in one call
            # so each item can be counted instead of mapping the node per item.
            is_input_list=True,
            accept_all_inputs=True,
            inputs=[
                io.Mask.Input("mask_1", optional=True),
            ],
            outputs=[io.Int.Output(display_name="int")],
        )

    @classmethod
    async def execute(cls, mask_1: Any = None, **kwargs) -> io.NodeOutput:
        count = sum(
            count_mask_items(value)
            for value in collect_prefixed_values("mask_", mask_1, kwargs)
        )
        return io.NodeOutput(int(count), ui=make_ui_text(str(int(count))))
