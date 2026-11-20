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
from ...utils._json_path import replace_all

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
            # Lists in, so value can hold one entry per match.
            is_input_list=True,
            inputs=[
                io.String.Input("text", default="", multiline=True),
                io.String.Input("path", default="values[*]"),
                io.Custom("*").Input(
                    "value",
                    tooltip=(
                        "要写入的值。接入列表时按顺序一一替换匹配位置；"
                        "只有 1 个值时广播到所有匹配位置。"
                        "可接 Text Path Extract 的 value、List Math Expression、Multi Dict 等。"
                    ),
                ),
                io.Combo.Input(
                    "value_type",
                    options=_VALUE_TYPES,
                    default="auto",
                    tooltip="只作用于 value：与 Multi Dict 的 value_type 含义一致。",
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
        path,
        value,
        value_type,
    ) -> io.NodeOutput:
        text = as_text(text)
        path = as_text(path)
        value_type = as_text(value_type) or "auto"

        if value is None:
            items = []
        elif isinstance(value, (list, tuple)):
            items = list(value)
        else:
            items = [value]

        new_values = []
        for item in items:
            converted = coerce_value(item, value_type)
            validate_json_value(converted)
            new_values.append(converted)

        try:
            data, count = replace_all(text, path, new_values)
        except ValueError as exc:
            # Surface bad paths and mismatched counts instead of returning
            # the original text: a silent no-op would hide the mistake.
            raise ValueError(f"Text Path Replace: {exc}") from None

        return io.NodeOutput([data], count, ui=make_ui_text(str(count)))
