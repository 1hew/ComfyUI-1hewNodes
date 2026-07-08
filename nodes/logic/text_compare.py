import re

from comfy_api.latest import io


class TextCompare(io.ComfyNode):
    OPERATORS = ["==", "!=", "⊂", "⊃", "⊄", "⊅", "startswith", "endswith", "regex"]

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_TextCompare",
            display_name="Text Compare",
            category="1hewNodes/logic",
            inputs=[
                io.String.Input("a", default=""),
                io.Combo.Input("operator", options=cls.OPERATORS, default="=="),
                io.String.Input("b", default=""),
            ],
            outputs=[io.Boolean.Output(display_name="bool")],
        )

    @classmethod
    def validate_inputs(cls, **kwargs):
        return True

    @classmethod
    async def execute(cls, a: str, operator: str, b: str) -> io.NodeOutput:
        try:
            return io.NodeOutput(cls._compare(str(a), operator, str(b)))
        except Exception as e:
            print(f"TextCompare error: {e}")
            return io.NodeOutput(False)

    @staticmethod
    def _compare(a: str, operator: str, b: str) -> bool:
        if operator == "==":
            return a == b
        if operator == "!=":
            return a != b
        if operator == "⊂":
            return a in b
        if operator == "⊃":
            return b in a
        if operator == "⊄":
            return a not in b
        if operator == "⊅":
            return b not in a
        if operator == "startswith":
            return b.startswith(a)
        if operator == "endswith":
            return b.endswith(a)
        if operator == "regex":
            try:
                return bool(re.search(b, a))
            except re.error:
                return False
        return False
