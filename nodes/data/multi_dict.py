from __future__ import annotations

from typing import Any

from comfy_api.latest import io

from ...utils import make_ui_text
from ...utils._json_ops import (
    as_text,
    coerce_value,
    collect_indexed_inputs,
    json_text,
    merge_dict_inputs,
    normalize_key,
)

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

# A JSON object key must be a string, so the key side only needs to decide
# whether a typed number stays text ("1") or is normalised ("1" from "1.0").
_KEY_TYPES = ["string", "auto"]


class MultiDict(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_MultiDict",
            display_name="Multi Dict",
            category="1hewNodes/data",
            # Dynamic dict_2..N sockets are not declared in the schema.
            accept_all_inputs=True,
            inputs=[
                io.Custom("DICT").Input(
                    "dict_1",
                    optional=True,
                    tooltip=(
                        "第一个字典；后续通过动态端口 dict_2、dict_3… 接入。"
                        "按端口顺序合并，后出现的同名键覆盖前面的。"
                    ),
                ),
                io.String.Input(
                    "key",
                    default="",
                    tooltip=(
                        "最后添加/覆盖的键，直接输入。留空则只合并 dict_1..N。"
                        "JSON 对象的键必须是字符串，key_type 决定数字如何书写。"
                    ),
                ),
                io.Combo.Input(
                    "key_type",
                    options=_KEY_TYPES,
                    default="string",
                    tooltip=(
                        "string：键按文本原样使用（推荐，JSON 键只能是字符串）；"
                        "auto：数字 / 布尔按键值规范化，如 1.0 → \"1\"、true → \"true\"。"
                    ),
                ),
                io.String.Input(
                    "value",
                    default="",
                    tooltip=(
                        "最后添加/覆盖的值，直接输入，按 value_type 转换。"
                        "需要数字时选 int / float；需要字典 / 列表时选 dict / list / json。"
                        "key 为空时忽略。"
                    ),
                ),
                io.Combo.Input(
                    "value_type",
                    options=_VALUE_TYPES,
                    default="auto",
                    tooltip=(
                        "作用于 value：auto 原样保留（文本就是文本）；"
                        "string 转文本；int/float 转数字；boolean 转布尔；"
                        "dict/list 把 JSON 文本解析成对象/数组；"
                        "json 解析任意 JSON 值；null 强制输出 null。"
                    ),
                ),
            ],
            outputs=[
                io.Custom("DICT").Output(display_name="dict"),
                io.String.Output(display_name="json"),
            ],
        )

    @classmethod
    async def execute(
        cls,
        value_type: Any = "auto",
        key: Any = "",
        key_type: Any = "string",
        value: Any = "",
        dict_1: Any = None,
        **kwargs: Any,
    ) -> io.NodeOutput:
        inputs = collect_indexed_inputs("dict_", dict_1, kwargs)
        payload = merge_dict_inputs(inputs)

        resolved_key = cls._resolve_key(key, key_type)
        if resolved_key.strip():
            typed = as_text(value)
            # An empty box means "no value", so the key gets JSON null, the
            # same result an unconnected value socket used to produce.
            payload[resolved_key] = (
                None if typed == "" else coerce_value(typed, value_type)
            )
        elif as_text(value) != "":
            print("[1hewNodes][Multi Dict] key 为空，value 已忽略")

        text = json_text(payload)
        return io.NodeOutput(payload, text, ui=make_ui_text(text))

    @staticmethod
    def _resolve_key(key: Any, key_type: Any) -> str:
        """Return the JSON key for the typed key text."""
        raw = as_text(key)
        if raw == "":
            return ""
        if str(as_text(key_type)).strip().lower() != "auto":
            return raw
        # auto: a numeric key normalises the way a linked scalar would, so
        # "1.0" and "1" land on the same key instead of creating two.
        text = raw.strip()
        try:
            number = float(text)
        except ValueError:
            number = None
        if number is not None:
            return normalize_key(int(number) if number.is_integer() else number)
        if text.lower() in {"true", "false"}:
            return normalize_key(text.lower() == "true")
        return raw
