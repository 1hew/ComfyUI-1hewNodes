from __future__ import annotations

import torch
from comfy_api.latest import io

from ...utils import first_torch_tensor, make_ui_text


class FloatMaskAreaRatio(io.ComfyNode):
    """Mask area ratio of the whole frame as a 0-1 float.

    The mask's own H x W is the reference frame, so the value is simply the
    mean mask weight. Soft values are weighted (a 0.5 pixel counts as half a
    covered pixel), which keeps feathered and anti-aliased masks stable.
    """

    DEFAULT_ROUND = 2
    MAX_ROUND = 10

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_FloatMaskAreaRatio",
            display_name="Float Mask Area Ratio",
            category="1hewNodes/float",
            inputs=[
                io.Mask.Input("mask"),
                io.Int.Input(
                    "round",
                    default=cls.DEFAULT_ROUND,
                    min=0,
                    max=cls.MAX_ROUND,
                    step=1,
                    display_mode=io.NumberDisplay.number,
                ),
            ],
            outputs=[io.Float.Output(display_name="float")],
        )

    @classmethod
    async def execute(
        cls,
        mask: torch.Tensor,
        round: int = DEFAULT_ROUND,
    ) -> io.NodeOutput:
        ratio = cls._round_ratio(cls._coverage(mask), cls._clamp_round(round))
        return io.NodeOutput(ratio, ui=make_ui_text(str(ratio)))

    @classmethod
    def _coverage(cls, mask) -> float:
        tensor = cls._normalize_mask(mask)
        if tensor is None:
            return 0.0

        values = tensor.detach()
        if values.dtype != torch.float32:
            values = values.to(torch.float32)
        values = torch.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
        values = values.clamp(0.0, 1.0)
        # Accumulate in float64 so high `round` values do not expose the
        # float32 reduction noise on large masks.
        total = float(values.sum(dtype=torch.float64).item())
        return total / float(values.numel())

    @staticmethod
    def _normalize_mask(mask) -> torch.Tensor | None:
        tensor = first_torch_tensor(mask)
        if tensor is None:
            return None

        current = tensor.detach()
        if current.ndim == 2:
            current = current.unsqueeze(0)
        elif current.ndim == 4 and int(current.shape[-1]) >= 1:
            current = current[:, :, :, 0]
        if current.ndim != 3:
            return None
        if any(int(size) <= 0 for size in current.shape):
            return None
        return current

    @staticmethod
    def _round_ratio(value: float, digits: int) -> float:
        # Kept in a scope where the `round` builtin is not shadowed by the
        # widget parameter of the same name.
        return round(value, digits)

    @classmethod
    def _clamp_round(cls, digits) -> int:
        try:
            value = int(digits)
        except (TypeError, ValueError):
            return cls.DEFAULT_ROUND
        return max(0, min(cls.MAX_ROUND, value))
