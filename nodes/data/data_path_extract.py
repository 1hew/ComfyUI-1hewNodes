"""Data Path Extract: pull values out of JSON-like text by path.

Everything arrives through the text widget: JSON, JSON inside a Markdown
fence, a Python literal, relaxed JSON, or plain text. The path indexes like
Python and reads keys like JSONPath: "[0].content", "[-1].bbox[0]" and
"data.items[0].zh" all work, "keys[*]" / "values[*]" pick the key or value
side of an object, and the default "[*]" splits the payload into its
values. "values[0].ids" therefore reads the first image whether its key is
called reference_image_1 or image_1, and "keys[0]" tells you which one it
was.

The path is named data_path because "path" alone reads as a file path in
ComfyUI. The node only collects values: the matches are handed over as a
list on a list-typed port, with no other shape. Rendering or serialising it
is a separate step (List Text, Multi List, Save Text).
"""

from comfy_api.latest import io

from ...utils import make_ui_text

from ...utils._json_path import is_count_path, query_all


class DataPathExtract(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_DataPathExtract",
            display_name="Data Path Extract",
            category="1hewNodes/data",
            inputs=[
                io.Custom("*").Input(
                    "data",
                    tooltip=(
                        "要解析的数据（连线）。JSON 文本、代码块 JSON、"
                        "Python 字面量或已解析的 dict / list 都可以直接接。"
                    ),
                ),
                io.String.Input(
                    "data_path",
                    default="[*]",
                    tooltip=(
                        "JSON 取值路径，如 [0].content、[*].bbox、keys[0]；"
                        "空字符串表示整段内容本身。"
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
    def validate_inputs(cls, **kwargs):
        # The wildcard input accepts every type; overriding this also keeps
        # legacy workflows usable if an old link recorded another type.
        return True

    @classmethod
    async def execute(
        cls,
        data=None,
        data_path: str = "[*]",
    ) -> io.NodeOutput:
        if data is None or (isinstance(data, str) and not data.strip()):
            return io.NodeOutput([], 0, ui=make_ui_text("0"))

        try:
            matches = query_all(data, data_path)
        except Exception as exc:
            print(f"data_path_extract error: {exc}")
            # value is a list output, so a failure yields an empty list rather
            # than a type the port never declares.
            return io.NodeOutput([], 0, ui=make_ui_text("0"))

        count = len(matches)
        if (
            len(matches) == 1
            and is_count_path(data_path)
            and isinstance(matches[0], int)
            and not isinstance(matches[0], bool)
        ):
            count = matches[0]

        return io.NodeOutput(matches, count, ui=make_ui_text(str(count)))
