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
                        "填单个值会广播到所有匹配位置；"
                        "填 JSON 数组（如 [1,2,3]）则按顺序一一写入各匹配位置。"
                        "需要数字时选 int / float；需要字典 / 列表时选 dict / list。"
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
        value="",
        value_type="auto",
    ) -> io.NodeOutput:
        # data is the wildcard input: a string is parsed later by replace_all,
        # a dict / list is used as-is, so it is never stringified here.
        data_path = as_text(data_path)
        value_type = as_text(value_type) or "auto"

        items = cls._value_items(as_text(value), value_type)

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
    def _value_items(text: str, value_type: str) -> list:
        """Split the value box into the items to write.

        A blank box writes nothing. A JSON array gives one item per element,
        so several matches can each get their own value; anything else is a
        single item and replace_all broadcasts it to every match.
        """
        if text == "":
            return []
        # Only look for a list when the caller did not ask for a list value:
        # value_type=list means "this whole text is one array to write".
        if value_type != "list" and text.strip()[:1] == "[":
            parsed = parse_content(text.strip())
            if isinstance(parsed, list):
                return list(parsed)
        return [text]
