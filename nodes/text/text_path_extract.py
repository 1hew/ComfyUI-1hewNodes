"""Text Path Extract: pull values out of JSON-like text by path.

Everything arrives through the text widget: JSON, JSON inside a Markdown
fence, a Python literal, relaxed JSON, or plain text. The path indexes like
Python and reads keys like JSONPath: "[0].content", "[-1].bbox[0]" and
"data.items[0].zh" all work, "keys[*]" / "values[*]" pick the key or value
side of an object, and the default "values[*]" splits the payload into its
values. "values[0].ids" therefore reads the first image whether its key is
called reference_image_1 or image_1, and "keys[0]" tells you which one it
was.

The node only collects values: value is always a list with one entry per
match. Rendering that list into text is a separate step, so List Text,
Text List to String and Save Text can each format it the way they need.
"""

from comfy_api.latest import io

from ...utils import make_ui_text

from ...utils._json_path import is_count_path, query_all


class TextPathExtract(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_TextPathExtract",
            display_name="Text Path Extract",
            category="1hewNodes/text",
            inputs=[
                io.String.Input("text", default="", multiline=True),
                io.String.Input("path", default="values[*]"),
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
        path: str = "values[*]",
    ) -> io.NodeOutput:
        if text is None or (isinstance(text, str) and not text.strip()):
            return io.NodeOutput([], 0, ui=make_ui_text("0"))

        try:
            matches = query_all(text, path)
        except Exception as exc:
            print(f"text_path_extract error: {exc}")
            # value is a list output, so a failure yields an empty list rather
            # than a type the port never declares.
            return io.NodeOutput([], 0, ui=make_ui_text("0"))

        if not matches:
            return io.NodeOutput([], 0, ui=make_ui_text("0"))

        count = len(matches)
        if (
            len(matches) == 1
            and is_count_path(path)
            and isinstance(matches[0], int)
            and not isinstance(matches[0], bool)
        ):
            count = matches[0]
        return io.NodeOutput(matches, count, ui=make_ui_text(str(count)))
