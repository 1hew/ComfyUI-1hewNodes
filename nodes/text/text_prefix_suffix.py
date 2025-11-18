from comfy_api.latest import io


class TextPrefixSuffix(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_TextPrefixSuffix",
            display_name="Text Prefix Suffix",
            category="1hewNodes/text",
            inputs=[
                io.Custom("Any").Input("any_text"),
                io.String.Input("prefix", default=""),
                io.String.Input("suffix", default=""),
                io.String.Input("separator", default="\\n"),
            ],
            outputs=[io.String.Output(display_name="text")],
        )

    @classmethod
    def validate_inputs(cls, any_text, prefix, suffix, separator):
        return True

    @classmethod
    async def execute(
        cls, any_text, prefix: str, suffix: str, separator: str
    ) -> io.NodeOutput:
        sep = "\n" if separator == "\\n" else str(separator)

        if not isinstance(any_text, (list, tuple)):
            if hasattr(any_text, "__iter__") and not isinstance(any_text, str):
                any_text = list(any_text)
            else:
                any_text = [any_text]

        formatted = [f"{prefix}{str(item)}{suffix}" for item in any_text]
        result = sep.join(formatted)
        return io.NodeOutput(str(result))
