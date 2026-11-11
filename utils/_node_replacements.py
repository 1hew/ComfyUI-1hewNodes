"""旧 node_id → 新 node_id 的改名迁移表（ComfyUI Node replacement）。

ComfyUI 只在扩展的 ``on_load`` 阶段接受替换注册。注册之后：

* 旧 API prompt 在进入执行队列前由 ``NodeReplaceManager.apply_replacements`` 改写；
* 打开旧 UI 工作流时，「缺失节点」面板会把新节点作为替换项，可一键替换。

注意两点实现细节：

* 服务端 ``apply_replacements`` 会先清空旧节点的 ``inputs``，再按
  ``input_mapping`` 回填，所以即使新旧输入同名同序，也必须显式列出每个输入；
* ``old_widget_ids`` 是旧节点 widget 的顺序（工作流按位置保存
  ``widgets_values``），前端用它把旧值搬到新节点的同名 widget 上。
"""

from __future__ import annotations

import logging

# old_node_id -> (new_node_id, widget_ids, input_ids)
LEGACY_NODE_REPLACEMENTS: dict[str, tuple[str, list[str], list[str]]] = {
    # v4.5.0：缩写 Txt 改为完整单词 Text，与类名 / display_name 一致。
    "1hew_SaveTxt": (
        "1hew_SaveText",
        # 旧工作流的 widgets_values 顺序；text 是可选插槽，不占 widget。
        ["filename", "format", "auto_increment", "save_output", "save_metadata"],
        [
            "text",
            "filename",
            "format",
            "auto_increment",
            "save_output",
            "save_metadata",
        ],
    ),
    "1hew_LoadTxt": (
        "1hew_LoadText",
        ["file", "index", "include_subdir"],
        ["file", "index", "include_subdir"],
    ),
}


async def register_legacy_node_replacements() -> None:
    """把 ``LEGACY_NODE_REPLACEMENTS`` 里的每条改名登记到 ComfyUI。"""
    try:
        from comfy_api.latest import ComfyAPI, io
    except Exception as exc:  # pragma: no cover - 极旧的 comfy_api
        logging.warning(
            "[1hewNodes] 跳过 node replacement 注册（comfy_api 版本过旧）：%s", exc
        )
        return

    node_replace = getattr(io, "NodeReplace", None)
    if node_replace is None:
        logging.warning(
            "[1hewNodes] 当前 ComfyUI 不支持 Node replacement；"
            "旧工作流里的 1hew_SaveTxt / 1hew_LoadTxt 需要手动换成新节点"
        )
        return

    api = ComfyAPI()
    for old_node_id, (new_node_id, widget_ids, input_ids) in (
        LEGACY_NODE_REPLACEMENTS.items()
    ):
        try:
            await api.node_replacement.register(
                node_replace(
                    new_node_id=new_node_id,
                    old_node_id=old_node_id,
                    old_widget_ids=list(widget_ids),
                    input_mapping=[
                        {"new_id": input_id, "old_id": input_id}
                        for input_id in input_ids
                    ],
                )
            )
        except Exception as exc:  # pragma: no cover - 运行期保护
            logging.warning(
                "[1hewNodes] 注册节点替换失败 %s -> %s：%s",
                old_node_id,
                new_node_id,
                exc,
            )
