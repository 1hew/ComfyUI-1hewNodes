from __future__ import annotations

from typing import Any

from comfy_api.latest import io

from ...utils import make_ui_text
from ...utils._json_ops import (
    VALUE_TYPES,
    as_text,
    coerce_value,
    collect_indexed_inputs,
    json_text,
    merge_dict_inputs,
)

# A JSON object key must be a string, so the key side only needs to decide
# whether a typed number stays text ("1") or is normalised ("1" from "1.0").
# A Python dict accepts any hashable key, so the key side can produce a real
# int / float / bool / None key. The `json` output stringifies them, because
# JSON object keys are always strings.
_KEY_TYPES = ["auto", "string", "int", "float", "boolean", "null"]

# Marker for "no key typed", so an explicit None key stays distinguishable.
_NO_KEY = object()


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
                        "key_type 决定键的类型；json 输出的键一律写成字符串。"
                    ),
                ),
                io.Combo.Input(
                    "key_type",
                    options=_KEY_TYPES,
                    default="auto",
                    tooltip=(
                        "键的类型。auto：像 Python 字面量一样推断，1→int、1.5→float、"
                        "true/false→bool、null→None，其余保持文本；"
                        "string：键永远按文本使用。"
                        "其余选项强制转成该类型。"
                        "注意 json 输出的键一律是字符串。"
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
                    options=VALUE_TYPES,
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
        key_type: Any = "auto",
        value: Any = "",
        dict_1: Any = None,
        **kwargs: Any,
    ) -> io.NodeOutput:
        inputs = collect_indexed_inputs("dict_", dict_1, kwargs)
        payload = merge_dict_inputs(inputs)

        resolved_key = cls._resolve_key(key, key_type)
        # _NO_KEY means the box was blank; None is a real key (key_type=null).
        if resolved_key is not _NO_KEY:
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
    def _resolve_key(key: Any, key_type: Any) -> Any:
        """Return the key for the typed text, typed by key_type.

        A Python dict accepts any hashable key, so int / float / bool / None
        keys are produced as real values; the json output stringifies them.
        Returns _NO_KEY when the box is blank, which is different from an
        explicit None key produced by key_type=null.
        """
        raw = as_text(key)
        if raw.strip() == "":
            return _NO_KEY
        mode = str(as_text(key_type)).strip().lower() or "auto"
        if mode not in _KEY_TYPES:
            raise ValueError(
                f"key_type 仅支持 {', '.join(_KEY_TYPES)}，收到 {mode!r}"
            )
        if mode == "string":
            return raw

        text = raw.strip()
        # auto infers like a Python literal; the explicit modes force a type
        # and let coercion raise when the text does not fit.
        if mode in ("auto", "boolean"):
            lowered = text.lower()
            if lowered in {"true", "false"}:
                return lowered == "true"
        if mode in ("auto", "null"):
            if text.lower() == "null":
                return None
        if mode in ("auto", "int"):
            try:
                return int(text)
            except ValueError:
                if mode == "int":
                    raise ValueError(f"无法把键 {raw!r} 转成 int") from None
        if mode in ("auto", "float"):
            try:
                number = float(text)
            except ValueError:
                if mode == "float":
                    raise ValueError(f"无法把键 {raw!r} 转成 float") from None
                return raw
            return number
        return raw
