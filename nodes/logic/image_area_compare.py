import torch
from comfy_api.latest import io

from ...utils import first_torch_tensor


class ImageAreaCompare(io.ComfyNode):
    """Compare the pixel area (width x height) of two IMAGE inputs.

    Mirrors ``Int Number Compare``: the selected operator decides the boolean
    result, but the compared values are each image's total pixel count
    (``height * width``). Every frame in an IMAGE batch shares the same
    dimensions, so the batch size does not affect the area.
    """

    OPERATORS = ["==", "!=", ">", ">=", "<", "<="]

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_ImageAreaCompare",
            display_name="Image Area Compare",
            category="1hewNodes/logic",
            inputs=[
                io.Image.Input("a"),
                io.Combo.Input("operator", options=cls.OPERATORS, default="=="),
                io.Image.Input("b"),
            ],
            outputs=[io.Boolean.Output(display_name="bool")],
        )

    @classmethod
    def validate_inputs(cls, **kwargs):
        return True

    @classmethod
    async def execute(
        cls, a: torch.Tensor, operator: str, b: torch.Tensor
    ) -> io.NodeOutput:
        try:
            area_a = cls._area(a)
            area_b = cls._area(b)
            if area_a is None or area_b is None:
                return io.NodeOutput(False)
            return io.NodeOutput(cls._compare(area_a, operator, area_b))
        except Exception as e:
            print(f"ImageAreaCompare error: {e}")
            return io.NodeOutput(False)

    @staticmethod
    def _area(image) -> int | None:
        """Return the total pixel count (height * width) of an IMAGE tensor."""
        tensor = first_torch_tensor(image)
        if tensor is None or getattr(tensor, "ndim", 0) < 3:
            return None

        height = int(tensor.shape[-3])
        width = int(tensor.shape[-2])
        if height < 1 or width < 1:
            return None
        return height * width

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
