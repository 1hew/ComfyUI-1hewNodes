"""Data Path Replace: write values back into JSON-like data at a path.

It pairs with Data Path Extract: the same path that reads a value also says
where a new value goes. The payload arrives on the wildcard `data` link, so
JSON text and an already-parsed dict / list both work.

`value` is a single typed box: one entry per call, broadcast to every match.
The edited document leaves through the `data` port with the same shape it
arrived with: a dict stays a dict, a list stays a list.
"""

from comfy_api.latest import io

from ...utils import make_ui_text
from ...utils._json_ops import (
    VALUE_TYPES,
    as_text,
    coerce_value,
    validate_json_value,
)
from ...utils._json_path import parse_content, replace_all


class DataPathReplace(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_DataPathReplace",
            display_name="Data Path Replace",
            category="1hewNodes/data",
            # value arrives as a wildcard link and must be received whole: with
            # INPUT_IS_LIST off ComfyUI slices one item per call, so a list
            # upstream would never reach the node intact.
            is_input_list=True,
            inputs=[
                io.Custom("*").Input(
                    "data",
                    tooltip=(
                        "要编辑的数据（连线）。JSON 文本、代码块 JSON、"
                        "Python 字面量或已解析的 dict / list 都可以直接接。"
                    ),
                ),
                io.String.Input(
                    "data_path",
                    default="[*]",
                    tooltip=(
                        "写入位置的 JSON 路径，语法与 Data Path Extract 相同，默认 [*]；"
                        "空字符串或 count 类路径会被拒绝。"
                    ),
                ),
                io.Custom("*").Input(
                    "value",
                    tooltip=(
                        "要写入的值，连线输入，按 value_type 转换。"
                        "接单个值时广播到所有匹配位置；"
                        "接列表时按顺序一一写入各匹配位置。"
                        "上游可用 Multi List / List Custom 等构造列表。"
                    ),
                ),
                io.Combo.Input(
                    "value_type",
                    options=VALUE_TYPES,
                    default="auto",
                    tooltip=(
                        "作用于 value 的每一项：auto 原样保留（文本就是文本），"
                        "int / float 转数字，dict / list 解析 JSON 文本。"
                    ),
                ),
            ],
            outputs=[
                # The edited document keeps the shape of the input: a dict
                # stays a dict, a list stays a list. No wrapper, so the port
                # never claims to be a list when the data is an object.
                io.Custom("*").Output(display_name="data"),
                io.Int.Output(display_name="count"),
            ],
        )

    @classmethod
    def validate_inputs(cls, **kwargs):
        return True

    @classmethod
    async def execute(
        cls,
        data=None,
        data_path="[*]",
        value=None,
        value_type="auto",
    ) -> io.NodeOutput:
        # is_input_list wraps every input in a list; the widgets carry one
        # item and the wildcard `value` keeps whatever the upstream sent.
        data = cls._first(data)
        data_path = as_text(cls._first(data_path))
        value_type = as_text(cls._first(value_type)) or "auto"

        items = cls._value_items(value)

        new_values = []
        for item in items:
            converted = coerce_value(item, value_type)
            validate_json_value(converted)
            new_values.append(converted)

        if not new_values:
            # Nothing to write: return the document unchanged rather than
            # letting replace_all raise a count mismatch for zero values.
            return io.NodeOutput(parse_content(data), 0, ui=make_ui_text("0"))

        try:
            edited, count = replace_all(data, data_path, new_values)
        except ValueError as exc:
            # Surface bad paths and mismatched counts instead of returning
            # the original data: a silent no-op would hide the mistake.
            raise ValueError(f"Data Path Replace: {exc}") from None

        return io.NodeOutput(edited, count, ui=make_ui_text(str(count)))

    @staticmethod
    def _first(wrapped):
        """Unwrap an is_input_list value: the node needs the single item."""
        if isinstance(wrapped, (list, tuple)):
            return wrapped[0] if wrapped else None
        return wrapped

    @classmethod
    def _value_items(cls, value) -> list:
        """Return the items to write, one per match or one to broadcast.

        A list arrives whole (is_input_list) and its items are written in
        order; anything else is a single value that replace_all broadcasts.
        An empty box / empty text writes nothing at all.
        """
        items = list(value) if isinstance(value, (list, tuple)) else [value]
        items = [item for item in items if item is not None and item != ""]
        return items
