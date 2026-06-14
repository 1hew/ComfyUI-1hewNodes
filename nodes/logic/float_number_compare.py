from comfy_api.latest import io


class FloatNumberCompare(io.ComfyNode):
    OPERATORS = ["==", "!=", ">", ">=", "<", "<="]

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_FloatNumberCompare",
            display_name="Float Number Compare",
            category="1hewNodes/logic",
            inputs=[
                io.Float.Input("a", default=0.0, min=-999999999.0, max=999999999.0, step=0.01),
                io.Combo.Input("operator", options=cls.OPERATORS, default="=="),
                io.Float.Input("b", default=0.0, min=-999999999.0, max=999999999.0, step=0.01),
            ],
            outputs=[io.Boolean.Output(display_name="bool")],
        )

    @classmethod
    def validate_inputs(cls, **kwargs):
        return True

    @classmethod
    async def execute(cls, a: float, operator: str, b: float) -> io.NodeOutput:
        try:
            return io.NodeOutput(cls._compare(float(a), operator, float(b)))
        except Exception as e:
            print(f"FloatNumberCompare error: {e}")
            return io.NodeOutput(False)

    @staticmethod
    def _compare(a: float, operator: str, b: float) -> bool:
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
