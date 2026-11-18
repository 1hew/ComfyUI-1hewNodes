from __future__ import annotations

from typing import Any

from comfy_api.latest import io

from ...utils import make_ui_text
from ...utils._json_ops import collect_indexed_inputs, extend_values, json_text


class MultiList(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_MultiList",
            display_name="Multi List",
            category="1hewNodes/multi",
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
                    "item_1",
                    optional=True,
                    tooltip="第一个元素；后续通过动态端口 item_2、item_3… 接入。",
                ),
            ],
            outputs=[
                io.Custom("*").Output(display_name="list"),
                io.String.Output(display_name="json"),
            ],
        )

    @classmethod
    async def execute(
        cls,
        mode: str = "extend",
        item_1: Any = None,
        **kwargs: Any,
    ) -> io.NodeOutput:
        inputs = collect_indexed_inputs("item_", item_1, kwargs)
        values = extend_values(inputs, mode)
        text = json_text(values)
        return io.NodeOutput(values, text, ui=make_ui_text(text))
