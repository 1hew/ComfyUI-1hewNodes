"""Data Path Extract: pull values out of JSON-like text by path.

Everything arrives through the text widget: JSON, JSON inside a Markdown
fence, a Python literal, relaxed JSON, or plain text. The path indexes like
Python and reads keys like JSONPath: "[0].content", "[-1].bbox[0]" and
"data.items[0].zh" all work, "keys[*]" / "values[*]" pick the key or value
side of an object, and the default "values[*]" splits the payload into its
values. "values[0].ids" therefore reads the first image whether its key is
called reference_image_1 or image_1, and "keys[0]" tells you which one it
was.

The path is named data_path because "path" alone reads as a file path in
ComfyUI. The node only collects values: the matches are handed over as a
list, and output_type decides whether that list is native or JSON text.
Rendering it item by item is a separate step (List Text, Text List to
String, Save Text).
"""

from comfy_api.latest import io

from ...utils import make_ui_text
from ...utils._json_ops import as_text, json_text

from ...utils._json_path import is_count_path, query_all


class DataPathExtract(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_DataPathExtract",
            display_name="Data Path Extract",
            category="1hewNodes/data",
            inputs=[
                io.String.Input("text", default="", multiline=True),
                io.String.Input(
                    "data_path",
                    default="values[*]",
                    tooltip=(
                        "JSON 取值路径，如 [0].content、values[*].bbox、keys[0]；"
                        "空字符串表示整段内容本身。"
                    ),
                ),
                io.Combo.Input(
                    "output_type",
                    options=["list", "json"],
                    default="list",
                    tooltip=(
                        "list 输出原生列表（保持值的类型）；"
                        "json 输出一个紧凑 JSON 数组文本。"
                    ),
                ),
            ],
            outputs=[
                # value is always a list (one entry per match, [] when nothing
                # matches) so the port shows as a list and a single list-valued
                # match is never split by ComfyUI's per-item execution.
                io.Custom("*").Output(display_name="value", is_output_list=True),
                io.Int.Output(display_name="count"),
            ],
        )

    @classmethod
    async def execute(
        cls,
        text: str = "",
        data_path: str = "values[*]",
        output_type="list",
    ) -> io.NodeOutput:
        if text is None or (isinstance(text, str) and not text.strip()):
            return cls._empty(output_type)

        try:
            matches = query_all(text, data_path)
        except Exception as exc:
            print(f"data_path_extract error: {exc}")
            # value is a list output, so a failure yields an empty list rather
            # than a type the port never declares.
            return cls._empty(output_type)

        count = len(matches)
        if (
            len(matches) == 1
            and is_count_path(data_path)
            and isinstance(matches[0], int)
            and not isinstance(matches[0], bool)
        ):
            count = matches[0]

        if str(as_text(output_type)).strip().lower() == "json":
            return io.NodeOutput(
                json_text(matches), count, ui=make_ui_text(str(count))
            )
        return io.NodeOutput(matches, count, ui=make_ui_text(str(count)))

    @classmethod
    def _empty(cls, output_type) -> io.NodeOutput:
        if str(as_text(output_type)).strip().lower() == "json":
            return io.NodeOutput("[]", 0, ui=make_ui_text("0"))
        return io.NodeOutput([], 0, ui=make_ui_text("0"))
