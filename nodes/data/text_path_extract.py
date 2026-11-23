"""Legacy compatibility: pre-rename workflows still reference 1hew_TextPathExtract.

This keeps the exact interface the active release was built against
(text/path/output_format/separator/fallback, three outputs), so a sealed
runtime can serve both the old release (bbox_decompose) and workflows that
already moved to 1hew_DataPathExtract. New work should use DataPathExtract.
"""

from comfy_api.latest import io

from ...utils import make_ui_text

from ...utils._json_path import format_value, is_count_path, query_all


class TextPathExtract(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_TextPathExtract",
            display_name="Text Path Extract (legacy)",
            category="1hewNodes/text",
            inputs=[
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
                io.Custom("*").Output(display_name="value"),
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
            return io.NodeOutput(fallback, fallback, 0, ui=make_ui_text("0"))

        try:
            matches = query_all(text, path)
        except Exception as exc:
            print(f"text_path_extract error: {exc}")
            return io.NodeOutput(fallback, fallback, 0, ui=make_ui_text("0"))

        if not matches:
            return io.NodeOutput(fallback, fallback, 0, ui=make_ui_text("0"))

        if len(matches) == 1:
            value = matches[0]
            count = len(matches)
            if is_count_path(path) and isinstance(value, int) and not isinstance(
                value, bool
            ):
                count = value
            return io.NodeOutput(
                format_value(value, output_format),
                value,
                count,
                ui=make_ui_text(str(count)),
            )

        count = len(matches)
        if str(output_format or "").strip().lower() == "json":
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
