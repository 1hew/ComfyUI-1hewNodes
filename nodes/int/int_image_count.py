from typing import Any

from comfy_api.latest import io

from ...utils import make_ui_text
from ...utils._count_ops import collect_prefixed_values, count_image_items


class IntImageCount(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_IntImageCount",
            display_name="Int Image Count",
            category="1hewNodes/int",
            inputs=[
                io.Image.Input("image_1", optional=True),
            ],
            outputs=[io.Int.Output(display_name="int")],
        )

    @classmethod
    async def execute(
        cls,
        image_1: Any = None,
        **kwargs,
    ) -> io.NodeOutput:
        count = sum(
            count_image_items(value)
            for value in collect_prefixed_values("image_", image_1, kwargs)
        )
        return io.NodeOutput(
            int(count),
            ui=make_ui_text(str(int(count))),
        )
