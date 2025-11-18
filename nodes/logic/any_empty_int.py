from comfy_api.latest import io
import numpy as np
import torch


class AnyEmptyInt(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_AnyEmptyInt",
            display_name="Any Empty Int",
            category="1hewNodes/logic",
            inputs=[
                io.Custom("*").Input("any"),
                io.Int.Input("empty", default=0, min=-999999, max=999999, step=1),
                io.Int.Input(
                    "not_empty", default=1, min=-999999, max=999999, step=1
                ),
            ],
            outputs=[io.Int.Output(display_name="int")],
        )

    @classmethod
    def validate_inputs(cls, **kwargs):
        return True

    @classmethod
    async def execute(cls, any, empty=0, not_empty=1) -> io.NodeOutput:
        try:
            is_empty = cls._is_empty(any)
            return io.NodeOutput(int(empty if is_empty else not_empty))
        except Exception as e:
            print(f"AnyEmptyInt error: {e}")
            return io.NodeOutput(int(empty))

    @classmethod
    def _is_empty(cls, value):
        if value is None:
            return True

        if isinstance(value, str):
            return len(value.strip()) == 0

        if isinstance(value, bool):
            return not value

        if isinstance(value, (int, float)):
            return value == 0

        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return True
            return torch.all(value == 0).item()

        if isinstance(value, np.ndarray):
            if value.size == 0:
                return True
            return np.all(value == 0)

        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                return True
            return all(cls._is_empty(item) for item in value)

        if isinstance(value, dict):
            return len(value) == 0

        if hasattr(value, "__len__"):
            try:
                return len(value) == 0
            except Exception:
                pass

        return False
