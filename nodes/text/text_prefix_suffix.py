from comfy_api.latest import io


class TextPrefixSuffix(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_TextPrefixSuffix",
            display_name="Text Prefix Suffix",
            category="1hewNodes/text",
            inputs=[
                io.Custom("*").Input("text"),
                io.String.Input("prefix", default=""),
                io.String.Input("suffix", default=""),
            ],
            outputs=[io.String.Output(display_name="text")],
        )

    @classmethod
    def validate_inputs(cls, **kwargs):
        return True

    @classmethod
    async def execute(
        cls, text, prefix: str, suffix: str
    ) -> io.NodeOutput:
        sep = "\n"

        if not isinstance(text, (list, tuple)):
            if hasattr(text, "__iter__") and not isinstance(text, str):
                text = list(text)
            else:
                text = [text]

        formatted = [f"{prefix}{str(item)}{suffix}" for item in text]
        result = sep.join(formatted)
        return io.NodeOutput(str(result))
