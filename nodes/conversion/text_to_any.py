from comfy_api.latest import io


class TextToAny(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_TextToAny",
            display_name="Text to Any",
            category="1hewNodes/conversion",
            inputs=[
                io.String.Input("text", default="", multiline=True),
            ],
            outputs=[
                io.Custom("*").Output(display_name="any"),
            ],
        )

    @classmethod
    def validate_inputs(cls, **kwargs):
        return True

    @classmethod
    async def execute(cls, text: str) -> io.NodeOutput:
        if isinstance(text, (list, tuple)):
            text = text[0] if len(text) > 0 else ""
        text = "" if text is None else str(text)
        return io.NodeOutput(text)
