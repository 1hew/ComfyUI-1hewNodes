import random

from comfy_api.latest import io

from ...utils import make_ui_text

from ...utils._text_ops import (
    split_text,
)


class StringRandom(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_StringRandom",
            display_name="String Random",
            category="1hewNodes/logic",
            inputs=[
                io.String.Input("custom_text", default="", multiline=True),
                io.Int.Input("seed", default=42, min=0, max=1125899906842624, step=1, display_mode=io.NumberDisplay.number),
            ],
            outputs=[
                io.String.Output(display_name="string"),
            ],
        )

    @classmethod
    async def execute(cls, custom_text: str, seed: int) -> io.NodeOutput:
        text = custom_text or ""
        items = split_text(cls, text)
        if not items:
            return io.NodeOutput("")

        random.seed(int(seed))
        chosen = random.choice(items)
        return io.NodeOutput(str(chosen), ui=make_ui_text(str(chosen)))

    @staticmethod
    def _strip_item(item: str) -> str:
        s = item.strip()
        if len(s) >= 2 and ((s[0] == s[-1]) and s[0] in ('"', "'")):
            s = s[1:-1].strip()
        return s
