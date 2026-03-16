from comfy_api.latest import io


class TextMatchRownum(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_text_match_rownum",
            display_name="Text Match Rownum",
            category="1hewNodes/logic",
            inputs=[
                io.String.Input("text_multiline", multiline=True),
                io.String.Input("text_single", multiline=False),
            ],
            outputs=[io.Int.Output(display_name="int")],
        )

    @classmethod
    def validate_inputs(cls, **kwargs):
        return True

    @classmethod
    async def execute(cls, text_multiline: str, text_single: str) -> io.NodeOutput:
        try:
            target = (text_single or "").strip()
            lines = (text_multiline or "").splitlines()

            for idx, line in enumerate(lines, start=1):
                if line.strip() == target:
                    return io.NodeOutput(int(idx))

            return io.NodeOutput(0)
        except Exception as e:
            print(f"text_match_rownum error: {e}")
            return io.NodeOutput(0)
