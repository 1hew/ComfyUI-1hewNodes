"""Data Path Replace: write values back into JSON-like data at a path.

It pairs with Data Path Extract: the same path that reads a value also says
where a new value goes. The payload arrives on the wildcard `data` link, so
JSON text and an already-parsed dict / list both work.

`value` is a single typed box: one entry per call, broadcast to every match.
Formatting is the next node's job, so the edited document leaves through the
`data` port as a one-item list of native values.
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
            # Lists are no longer needed on the write side: the value is typed
            # into a single-line box and converted by value_type.
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
                    options=VALUE_TYPES,
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
                # Named data to match the input port it goes back into.
                io.Custom("*").Output(display_name="data", is_output_list=True),
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
        value="",
        value_type="auto",
    ) -> io.NodeOutput:
        # data is the wildcard input: a string is parsed later by replace_all,
        # a dict / list is used as-is, so it is never stringified here.
        data_path = as_text(data_path)
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
            return io.NodeOutput([parse_content(data)], 0, ui=make_ui_text("0"))

        try:
            edited, count = replace_all(data, data_path, new_values)
        except ValueError as exc:
            # Surface bad paths and mismatched counts instead of returning
            # the original data: a silent no-op would hide the mistake.
            raise ValueError(f"Data Path Replace: {exc}") from None

        return io.NodeOutput([edited], count, ui=make_ui_text(str(count)))
