from typing import Any

from comfy_api.latest import io

from ...utils import make_ui_text
from ...utils._count_ops import collect_prefixed_values, count_audio_items


class IntAudioCount(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_IntAudioCount",
            display_name="Int Audio Count",
            category="1hewNodes/int",
            # audio_2..audio_N are appended by the front end and arrive through
            # **kwargs.  is_input_list keeps an upstream audio list in one call
            # so each item can be counted instead of mapping the node per item.
            is_input_list=True,
            accept_all_inputs=True,
            inputs=[
                io.Audio.Input("audio_1", optional=True),
            ],
            outputs=[io.Int.Output(display_name="int")],
        )

    @classmethod
    async def execute(cls, audio_1: Any = None, **kwargs) -> io.NodeOutput:
        count = sum(
            count_audio_items(value)
            for value in collect_prefixed_values("audio_", audio_1, kwargs)
        )
        return io.NodeOutput(int(count), ui=make_ui_text(str(int(count))))
