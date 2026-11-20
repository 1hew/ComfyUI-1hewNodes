"""List Math Expression: the Math Expression node, applied element-wise to lists.

The expression grammar, the function set and the result checks are the same as
ComfyUI's core Math Expression (simpleeval, no arbitrary code). The difference
is the data: every input is a list, the expression runs once per position, and
the outputs are lists.

Inputs are the letters a, b, c, ... added on demand by Autogrow, exactly as in
the core node: each letter is the name used in the expression, and values is
the list of all inputs at the current position.

Alignment, per position:
  * a scalar input is a list of length 1;
  * a length-1 list is broadcast to the common length;
  * all other inputs must share one length, otherwise the node raises.
"""

from __future__ import annotations

import math
import string
from typing import Any

from comfy_api.latest import io
from simpleeval import simple_eval

from ...utils import make_ui_text

MAX_EXPONENT = 4000


def _variadic_sum(*args):
    """Support both sum(values) and sum(a, b, c)."""
    if len(args) == 1 and hasattr(args[0], "__iter__"):
        return sum(args[0])
    return sum(args)


def _safe_pow(base, exp):
    """Cap exponents: pow() as a callable bypasses simpleeval's ** guard."""
    if abs(exp) > MAX_EXPONENT:
        raise ValueError(f"Exponent {exp} exceeds maximum allowed ({MAX_EXPONENT})")
    return pow(base, exp)


# Same function set as ComfyUI's core Math Expression.
MATH_FUNCTIONS = {
    "sum": _variadic_sum,
    "min": min,
    "max": max,
    "abs": abs,
    "round": round,
    "pow": _safe_pow,
    "sqrt": math.sqrt,
    "ceil": math.ceil,
    "floor": math.floor,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "int": int,
    "float": float,
}


def _as_list(value: Any) -> list:
    """A scalar becomes a one-item list; a list / tuple is taken as-is."""
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _align(inputs: list[list]) -> int:
    """Return the common length, or raise when the inputs cannot be aligned."""
    lengths = {len(values) for values in inputs if len(values) != 1}
    if len(lengths) > 1:
        detail = ", ".join(str(len(values)) for values in inputs)
        raise ValueError(
            f"List Math Expression: 输入长度不一致（{detail}）；"
            "长度为 1 的输入会广播，其余输入必须长度相同"
        )
    if not lengths:
        # Every input is length 1: one position.
        return 1
    return lengths.pop()


def _check_result(result: Any, expression: str) -> float:
    if not isinstance(result, (int, float)):
        raise ValueError(
            f"Math Expression '{expression}' must evaluate to a numeric result, "
            f"got {type(result).__name__}: {result!r}"
        )
    try:
        float_result = float(result)
    except OverflowError:
        raise ValueError(
            f"Math Expression '{expression}' produced a result too large to "
            f"represent as a float: {result}"
        ) from None
    if not math.isfinite(float_result):
        raise ValueError(
            f"Math Expression '{expression}' produced a non-finite result: {result}"
        )
    return float_result


class ListMathExpression(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        # Autogrow with TemplateNames, the same mechanism as the core Math
        # Expression: the front end grows a, b, c, ... as they are connected,
        # and each letter is directly the name used in the expression.
        autogrow = io.Autogrow.TemplateNames(
            input=io.Custom("*").Input("value"),
            names=list(string.ascii_lowercase),
            min=1,
        )
        return io.Schema(
            node_id="1hew_ListMathExpression",
            display_name="List Math Expression",
            category="1hewNodes/list",
            # Lists in, lists out: the node receives each whole input list once,
            # so ComfyUI does not run it per element.
            is_input_list=True,
            inputs=[
                # Defaults to a single letter so the node works with the first
                # input alone; add b / c as more sockets are connected.
                io.String.Input("expression", default="a", multiline=True),
                io.Autogrow.Input("values", template=autogrow),
            ],
            outputs=[
                io.Float.Output(display_name="FLOAT", is_output_list=True),
                io.Int.Output(display_name="INT", is_output_list=True),
                io.Boolean.Output(display_name="BOOL", is_output_list=True),
            ],
        )

    @classmethod
    def validate_inputs(cls, **kwargs):
        return True

    @classmethod
    async def execute(
        cls,
        expression,
        values=None,
        **kwargs: Any,
    ) -> io.NodeOutput:
        expr = _first(expression)
        if not str(expr or "").strip():
            raise ValueError("Expression cannot be empty.")

        # Autogrow hands the built inputs over as one mapping (letter -> item).
        # A plain port, if one ever arrives instead of the mapping, is ignored.
        columns: list[list] = []
        aliases: list[str] = []
        if isinstance(values, dict):
            for name in sorted(values):
                if values[name] is None:
                    continue
                aliases.append(name)
                columns.append(_as_list(values[name]))
        if not columns:
            return io.NodeOutput([], [], [], ui=make_ui_text("0"))

        length = _align(columns)

        floats, ints, bools = [], [], []
        for position in range(length):
            row = [col[0] if len(col) == 1 else col[position] for col in columns]
            context = dict(zip(aliases, row))
            context["values"] = list(row)
            result = simple_eval(expr, names=context, functions=MATH_FUNCTIONS)
            # bool is a subclass of int: check the numeric type first, as the
            # core node does, then derive the INT and BOOL views.
            float_result = _check_result(result, expr)
            floats.append(float_result)
            ints.append(int(result))
            bools.append(bool(result))

        return io.NodeOutput(
            floats,
            ints,
            bools,
            ui=make_ui_text(str(length)),
        )


def _first(value: Any) -> Any:
    """is_input_list wraps widget values in a list; unwrap the single item."""
    if isinstance(value, (list, tuple)):
        return value[0] if value else ""
    return value
