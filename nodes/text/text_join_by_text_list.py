from comfy_api.latest import io


class TextJoinByTextList(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_TextJoinByTextList",
            display_name="Text Join by Text List",
            category="1hewNodes/text",
            inputs=[
                io.Custom("Any").Input("text_list"),
                io.String.Input("prefix", default=""),
                io.String.Input("suffix", default=""),
                io.String.Input("separator", default="\\n"),
            ],
            outputs=[io.String.Output(display_name="text")],
        )

    @classmethod
    def validate_inputs(cls, text_list, prefix, suffix, separator):
        return True

    @classmethod
    async def execute(
        cls, text_list, prefix: str, suffix: str, separator: str
    ) -> io.NodeOutput:
        sep = (
            separator.replace("\\n", "\n")
            .replace("\\t", "\t")
            .replace("\\r", "\r")
            .replace("\\\\", "\\")
        )

        if not isinstance(text_list, (list, tuple)):
            text_list = [text_list]

        formatted = [f"{prefix}{str(item)}{suffix}" for item in text_list]
        joined = sep.join(formatted)
        return io.NodeOutput(str(joined))
