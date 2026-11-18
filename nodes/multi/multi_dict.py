from __future__ import annotations

from typing import Any

from comfy_api.latest import io

from ...utils import make_ui_text
from ...utils._json_ops import (
    coerce_value,
    collect_indexed_inputs,
    json_text,
    merge_dict_inputs,
    normalize_key,
)


class MultiDict(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_MultiDict",
            display_name="Multi Dict",
            category="1hewNodes/multi",
            # Dynamic dict_2..N sockets are not declared in the schema.
            accept_all_inputs=True,
            inputs=[
                io.Combo.Input(
                    "value_type",
                    options=[
                        "auto",
                        "string",
                        "int",
                        "float",
                        "boolean",
                        "dict",
                        "list",
                        "json",
                        "null",
                    ],
                    default="auto",
                    tooltip=(
                        "只作用于 value：auto 原样保留；string 转文本；int/float 转数字；"
                        "boolean 转布尔；dict/list 把 JSON 文本解析成对象/数组；"
                        "json 解析任意 JSON 值；null 强制输出 null。"
                        "未连接 value 时始终是 null。"
                    ),
                ),
                io.Custom("*").Input(
                    "dict_1",
                    optional=True,
                    tooltip=(
                        "第一个字典；后续通过动态端口 dict_2、dict_3… 接入。"
                        "按端口顺序合并，后出现的同名键覆盖前面的。"
                    ),
                ),
                io.Custom("*").Input(
                    "key",
                    optional=True,
                    tooltip=(
                        "最后添加/覆盖的键；接字符串或标量。"
                        "未连接或为空时只合并 dict_1..N。"
                    ),
                ),
                io.Custom("*").Input(
                    "value",
                    optional=True,
                    tooltip=(
                        "最后添加/覆盖的值；按 value_type 转换后写入。"
                        "接字符串在 auto 下保持字符串，选 json 可把 JSON 文本变成字典/列表。"
                        "key 为空时忽略；未连接时输出 null。"
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
        value_type: str = "auto",
        dict_1: Any = None,
        key: Any = None,
        value: Any = None,
        **kwargs: Any,
    ) -> io.NodeOutput:
        inputs = collect_indexed_inputs("dict_", dict_1, kwargs)
        payload = merge_dict_inputs(inputs)
        resolved_key = normalize_key(key)
        if resolved_key.strip():
            payload[resolved_key] = coerce_value(value, value_type)
        elif value is not None:
            print("[1hewNodes][Multi Dict] key 为空，value 已忽略")
        text = json_text(payload)
        return io.NodeOutput(payload, text, ui=make_ui_text(text))
