from __future__ import annotations

from typing import Any

from comfy_api.latest import io

from ...utils import make_ui_text
from ...utils._json_ops import as_text, collect_indexed_inputs, extend_values, json_text


class MultiList(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_MultiList",
            display_name="Multi List",
            category="1hewNodes/data",
            accept_all_inputs=True,
            inputs=[
                io.Combo.Input(
                    "mode",
                    options=["extend", "append"],
                    default="extend",
                    tooltip=(
                        "extend：列表/元组输入摊平一层，相当于 Python 的 list.extend；"
                        "append：每个输入作为一个元素，相当于 list.append，列表保持嵌套。"
                        "字符串和字典在两种模式下都是单个元素。"
                    ),
                ),
                io.Custom("*").Input(
                    "any_1",
                    optional=True,
                    tooltip="第一个输入，任意类型；后续通过动态端口 any_2、any_3… 接入。",
                ),
                io.Combo.Input(
                    "output_type",
                    options=["list", "json"],
                    default="list",
                    tooltip="list 输出原生列表，json 输出紧凑 JSON 文本。",
                ),
            ],
            outputs=[
                io.Custom("*").Output(display_name="data"),
            ],
        )

    @classmethod
    async def execute(
        cls,
        mode: str = "extend",
        any_1: Any = None,
        output_type: Any = "list",
        **kwargs: Any,
    ) -> io.NodeOutput:
        inputs = collect_indexed_inputs("any_", any_1, kwargs)
        values = extend_values(inputs, mode)
        text = json_text(values)
        if str(as_text(output_type)).strip().lower() == "json":
            return io.NodeOutput(text, ui=make_ui_text(text))
        return io.NodeOutput(values, ui=make_ui_text(text))
