from comfy_api.latest import io


class IntNumberCompare(io.ComfyNode):
    OPERATORS = ["==", "!=", ">", ">=", "<", "<="]

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_IntNumberCompare",
            display_name="Int Number Compare",
            category="1hewNodes/logic",
            inputs=[
                io.Int.Input("a", default=0, min=-999999999, max=999999999, step=1),
                io.Combo.Input("operator", options=cls.OPERATORS, default="=="),
                io.Int.Input("b", default=0, min=-999999999, max=999999999, step=1),
            ],
            outputs=[io.Boolean.Output(display_name="bool")],
        )

    @classmethod
    def validate_inputs(cls, **kwargs):
        return True

    @classmethod
    async def execute(cls, a: int, operator: str, b: int) -> io.NodeOutput:
        try:
            return io.NodeOutput(cls._compare(int(a), operator, int(b)))
        except Exception as e:
            print(f"IntNumberCompare error: {e}")
            return io.NodeOutput(False)

    @staticmethod
    def _compare(a: int, operator: str, b: int) -> bool:
        if operator == "==":
            return a == b
        if operator == "!=":
            return a != b
        if operator == ">":
            return a > b
        if operator == ">=":
            return a >= b
        if operator == "<":
            return a < b
        if operator == "<=":
            return a <= b
        return False
