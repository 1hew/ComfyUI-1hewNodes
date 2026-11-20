"""Text Path Extract: pull a value out of JSON-like text by path.

Everything arrives through the text widget: JSON, JSON inside a Markdown
fence, a Python literal, relaxed JSON, or plain text. The path indexes like
Python and reads keys like JSONPath: "[0].content", "[-1].bbox[0]" and
"data.items[0].zh" all work, "keys[*]" / "values[*]" pick the key or value
side of an object, and the default "values[*]" splits the payload into its
values. "values[0].ids" therefore reads the first image whether its key is
called reference_image_1 or image_1, and "keys[0]" tells you which one it
was.
"""

from comfy_api.latest import io

from ...utils import make_ui_text

from ...utils._json_path import format_value, is_count_path, query_all


class TextPathExtract(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_TextPathExtract",
            display_name="Text Path Extract",
            category="1hewNodes/text",
            inputs=[
                # Required on purpose: ComfyUI lists required widgets before
                # optional ones, so an optional text box would render below
                # path / output_format / separator / fallback.
                io.String.Input("text", default="", multiline=True),
                io.String.Input("path", default="values[*]"),
                io.Combo.Input(
                    "output_format",
                    options=["auto", "json", "text", "csv"],
                    default="auto",
                ),
                io.String.Input("separator", default="\\n"),
                io.String.Input("fallback", default=""),
            ],
            outputs=[
                io.String.Output(display_name="text"),
                # value is always a list (one item per match, [] when nothing
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
        output_format: str = "auto",
        separator: str = "\\n",
        fallback: str = "",
    ) -> io.NodeOutput:
        fallback = "" if fallback is None else str(fallback)

        if text is None or (isinstance(text, str) and not text.strip()):
            return io.NodeOutput(fallback, [], 0, ui=make_ui_text("0"))

        try:
            matches = query_all(text, path)
        except Exception as exc:
            print(f"text_path_extract error: {exc}")
            # The text port reports the declared fallback; the value port is an
            # empty list because the value port is a list output.
            return io.NodeOutput(fallback, [], 0, ui=make_ui_text("0"))

        if not matches:
            return io.NodeOutput(fallback, [], 0, ui=make_ui_text("0"))

        if len(matches) == 1:
            value = matches[0]
            count = len(matches)
            if is_count_path(path) and isinstance(value, int) and not isinstance(
                value, bool
            ):
                count = value
            return io.NodeOutput(
                format_value(value, output_format),
                [value],
                count,
                ui=make_ui_text(str(count)),
            )

        count = len(matches)
        if str(output_format or "").strip().lower() == "json":
            # One JSON array for the whole match set, so [*].bbox stays a
            # single list instead of one JSON value per line.
            return io.NodeOutput(
                format_value(matches, "json"),
                matches,
                count,
                ui=make_ui_text(str(count)),
            )

        rendered = [format_value(value, output_format) for value in matches]
        return io.NodeOutput(
            cls._join(rendered, separator),
            matches,
            count,
            ui=make_ui_text(str(count)),
        )

    @classmethod
    def _join(cls, values, separator) -> str:
        sep = str(separator if separator is not None else "\\n")
        sep = (
            sep.replace("\\r", "\r")
            .replace("\\n", "\n")
            .replace("\\t", "\t")
        )
        return sep.join(values)
