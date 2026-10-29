"""Text Path Extract: pull a value out of JSON-like text by path.

Everything arrives through the text widget: JSON, JSON inside a Markdown
fence, a Python literal, relaxed JSON, or plain text. The path uses a
JavaScript-flavoured syntax, so "[0].content", "[-1].bbox[0]",
"data.items[0].zh" or the wildcard form "[*].content" all work; the
default "[*]" splits the whole payload into its items.
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
                io.String.Input("path", default="[*]"),
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
        path: str = "[*]",
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
            # Keep both outputs on the declared fallback. Returning None on the
            # value port made downstream nodes receive an unexpected type while
            # the text port already reported the fallback.
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
