"""Text Path Replace: write values back into JSON-like text at a path.

It pairs with Text Path Extract: the same path that reads a value also says
where a new value goes. Computation stays upstream (List Math Expression,
Multi Dict, string nodes, ...); this node only places the values.

`value` is a list (is_input_list). One item per match is applied in order; a
single item is broadcast to every match. Any other count is an error, so a
mismatch never silently drops or misplaces values.

Like Text Path Extract the node does not render: the edited document leaves
through the `value` port as a one-item list, and formatting is the next
node's job (Save Text, List Text, or a JSON dump).
"""

from comfy_api.latest import io

from ...utils import make_ui_text
from ...utils._json_ops import (
    as_text,
    coerce_value,
    validate_json_value,
)
from ...utils._json_path import parse_content, replace_all

_VALUE_TYPES = [
    "auto",
    "string",
    "int",
    "float",
    "boolean",
    "dict",
    "list",
    "json",
    "null",
]


class TextPathReplace(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_TextPathReplace",
            display_name="Text Path Replace",
            category="1hewNodes/data",
            # Lists are no longer needed on the write side: the value is typed
            # into a single-line box and converted by value_type.
            inputs=[
                io.String.Input("text", default="", multiline=True),
                io.String.Input(
                    "json_path",
                    default="values[*]",
                    tooltip=(
                        "写入位置的 JSON 路径，语法与 Text Path Extract 相同；"
                        "空字符串或 count 类路径会被拒绝。"
                    ),
                ),
                io.String.Input(
                    "value",
                    default="",
                    tooltip=(
                        "要写入的值，直接输入，按 value_type 转换。"
                        "需要数字时选 int / float；需要字典 / 列表时选 dict / list / json。"
                        "匹配到多处时这个值广播到所有位置。"
                    ),
                ),
                io.Combo.Input(
                    "value_type",
                    options=_VALUE_TYPES,
                    default="auto",
                    tooltip=(
                        "作用于 value：auto 原样保留（文本就是文本），"
                        "int / float 转数字，dict / list / json 解析 JSON 文本。"
                    ),
                ),
            ],
            outputs=[
                # The edited document wrapped in a one-item list, so a
                # root-level JSON array is not split by ComfyUI downstream.
                io.Custom("*").Output(display_name="value", is_output_list=True),
                io.Int.Output(display_name="count"),
            ],
        )

    @classmethod
    def validate_inputs(cls, **kwargs):
        return True

    @classmethod
    async def execute(
        cls,
        text,
        json_path,
        value="",
        value_type="auto",
    ) -> io.NodeOutput:
        text = as_text(text)
        json_path = as_text(json_path)
        value_type = as_text(value_type) or "auto"

        typed = as_text(value)
        # The box holds one entry; value_type decides what it means, so a JSON
        # array typed here becomes several entries through `list`.
        items = [typed] if typed != "" else []

        new_values = []
        for item in items:
            converted = coerce_value(item, value_type)
            validate_json_value(converted)
            new_values.append(converted)

        if not new_values:
            # Nothing to write: return the document unchanged rather than
            # letting replace_all raise a count mismatch for zero values.
            data = parse_content(text)
            return io.NodeOutput([data], 0, ui=make_ui_text("0"))

        try:
            data, count = replace_all(text, json_path, new_values)
        except ValueError as exc:
            # Surface bad paths and mismatched counts instead of returning
            # the original text: a silent no-op would hide the mistake.
            raise ValueError(f"Text Path Replace: {exc}") from None

        return io.NodeOutput([data], count, ui=make_ui_text(str(count)))
