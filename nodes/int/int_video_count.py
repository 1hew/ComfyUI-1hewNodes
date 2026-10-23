from typing import Any

from comfy_api.latest import io

from ...utils import make_ui_text
from ...utils._count_ops import collect_prefixed_values, count_video_items


class IntVideoCount(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_IntVideoCount",
            display_name="Int Video Count",
            category="1hewNodes/int",
            inputs=[
                io.Video.Input("video_1", optional=True),
            ],
            outputs=[io.Int.Output(display_name="int")],
        )

    @classmethod
    async def execute(
        cls,
        video_1: Any = None,
        **kwargs,
    ) -> io.NodeOutput:
        count = sum(
            count_video_items(value)
            for value in collect_prefixed_values("video_", video_1, kwargs)
        )
        return io.NodeOutput(
            int(count),
            ui=make_ui_text(str(int(count))),
        )
