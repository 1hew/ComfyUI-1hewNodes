
from comfy_api.latest import io

from ...utils import make_ui_text

from ...utils._text_ops import (
    split_text,
)


class ListCustomString(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_ListCustomString",
            display_name="List Custom String",
            category="1hewNodes/text",
            inputs=[
                io.String.Input("custom_text", default="", multiline=True),
            ],
            outputs=[
                io.String.Output(display_name="string_list", is_output_list=True),
                io.Int.Output(display_name="count"),
            ],
        )

    @classmethod
    async def execute(cls, custom_text: str) -> io.NodeOutput:
        text = custom_text or ""
        items = split_text(cls, text)
        if not items:
            items = ["default"]
        count = len(items)
        return io.NodeOutput(items, count, ui=make_ui_text(str(count)))

    @staticmethod
    def _strip_item(item: str) -> str:
        s = item.strip()
        if len(s) >= 2 and ((s[0] == s[-1]) and s[0] in ('"', "'")):
            s = s[1:-1].strip()
        return s
