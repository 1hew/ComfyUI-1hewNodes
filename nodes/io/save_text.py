# 保存路径说明：
# - 默认文件名 "text/ComfyUI"（相对路径）会将文本保存到 ComfyUI 配置的 output 目录下。
# - output 目录由 folder_paths.get_output_directory() 决定：
#   * 无 --output-directory 参数 → 默认 <comfyui_root>/output
#   * 带 --output-directory <路径> 参数 → 自定义路径
# - 输入绝对路径时会跳过 output 目录，直接保存到指定位置。
from __future__ import annotations

import asyncio
import json
import os
from typing import Optional

import folder_paths
from comfy_api.latest import io, ui

from ...utils._text_metadata import build_metadata_block

from ...utils._file_ops import (
    is_absolute_like,
    sanitize_filename_prefix,
    split_prefix_to_dir_and_stem,
)

_PATH_LOCK: Optional[asyncio.Lock] = None


def _next_increment_path(directory: str, stem: str, extension: str = "txt") -> str:
    counter = 1
    while True:
        candidate = os.path.join(directory, f"{stem}_{counter:05}_.{extension}")
        if not os.path.exists(candidate):
            return candidate
        counter += 1


def _format_payload(text: str, file_format: str) -> str:
    """json 格式在内容合法时做缩进格式化，其它格式原样保存。"""
    if file_format != "json":
        return text
    try:
        return json.dumps(json.loads(text), indent=2, ensure_ascii=False)
    except (TypeError, ValueError):
        return text


def _collect_workflow_metadata(cls: type) -> dict:
    """Collect the UI workflow and API prompt for drag-and-drop replay."""
    hidden = getattr(cls, "hidden", None)
    if hidden is None:
        return {}

    metadata: dict = {}
    extra = getattr(hidden, "extra_pnginfo", None)
    if isinstance(extra, dict):
        workflow = extra.get("workflow")
        if isinstance(workflow, str):
            try:
                workflow = json.loads(workflow)
            except ValueError:
                pass
        if workflow is not None:
            metadata["workflow"] = workflow

    prompt = getattr(hidden, "prompt", None)
    if prompt is not None:
        metadata["prompt"] = prompt
    return metadata


def _stringify_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="replace")
        except Exception:
            return str(value)
    if isinstance(value, (dict, list, tuple)):
        try:
            return json.dumps(value, ensure_ascii=False, indent=2, default=str)
        except Exception:
            return str(value)
    return str(value)


class SaveText(io.ComfyNode):
    FORMAT_EXTENSIONS = {
        "txt": "txt",
        "csv": "csv",
        "md": "md",
        "json": "json",
    }
    DEFAULT_FORMAT = "txt"

    @staticmethod
    def _preview_ui(
        text: str,
        output_path: str = "",
        base_dir: Optional[str] = None,
    ) -> dict:
        text_value = str(text)
        ui_data = {
            "text": [text_value],
            "output": [text_value],
        }
        if not output_path:
            return ui_data

        abs_output = os.path.abspath(output_path)
        ui_data["file_path"] = [abs_output]
        if not base_dir:
            return ui_data

        try:
            rel_path = os.path.relpath(abs_output, os.path.abspath(base_dir))
            if rel_path.startswith("..") or os.path.isabs(rel_path):
                return ui_data
            rel_path = rel_path.replace("\\", "/")
            subfolder, filename = os.path.split(rel_path)
            if filename:
                ui_data["texts"] = [
                    ui.SavedResult(
                        filename,
                        "" if subfolder == "." else subfolder,
                        io.FolderType.output,
                    )
                ]
        except Exception:
            return ui_data

        return ui_data

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="1hew_SaveTxt",
            display_name="Save Text",
            category="1hewNodes/io",
            inputs=[
                io.String.Input(
                    "text",
                    optional=True,
                    force_input=True,
                    tooltip="要保存的文本；不连接时为空（None），节点预览会同步清空。",
                ),
                io.String.Input("filename", default="text/ComfyUI"),
                io.Combo.Input(
                    "format",
                    default=cls.DEFAULT_FORMAT,
                    options=list(cls.FORMAT_EXTENSIONS),
                    tooltip=(
                        "输出格式：txt / csv / md / json，默认 txt。"
                        "json 会在内容合法时自动缩进格式化；csv/json 不追加元数据。"
                    ),
                ),
                io.Boolean.Input(
                    "auto_increment",
                    default=True,
                    tooltip="为 True 时自动编号；为 False 时固定文件名并覆盖。",
                ),
                io.Boolean.Input(
                    "save_output",
                    default=True,
                    tooltip="为 True 时写入文件；为 False 时不保存到磁盘。",
                ),
                io.Boolean.Input(
                    "save_metadata",
                    default=True,
                    tooltip=(
                        "为 True 时将 workflow/prompt 元数据写入文件末尾；"
                        "把生成的文件拖入 ComfyUI 即可复现工作流。"
                        "csv/json 格式会跳过元数据，避免破坏文件结构。"
                    ),
                ),
            ],
            outputs=[
                io.String.Output(display_name="file_path"),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def validate_inputs(cls, **kwargs):
        return True

    @classmethod
    async def execute(
        cls,
        text=None,
        filename: str = "text/ComfyUI",
        format: str = DEFAULT_FORMAT,
        auto_increment: bool = True,
        save_output: bool = True,
        save_metadata: bool = True,
    ) -> io.NodeOutput:
        if text is None:
            # 显式发送空 UI，前端 onExecuted 会清空上一次的预览内容。
            return io.NodeOutput("", ui={"text": [""], "output": [""]})

        final_format = str(format or cls.DEFAULT_FORMAT).strip().lower()
        if final_format not in cls.FORMAT_EXTENSIONS:
            raise ValueError(
                f"format 仅支持 {', '.join(cls.FORMAT_EXTENSIONS)}，收到 {format!r}"
            )
        extension = cls.FORMAT_EXTENSIONS[final_format]

        text_value = _stringify_value(text)
        if not save_output:
            return io.NodeOutput("", ui=cls._preview_ui(text_value))

        file_text = _format_payload(text_value, final_format)
        if save_metadata:
            if final_format in {"txt", "md"}:
                file_text = build_metadata_block(
                    file_text, _collect_workflow_metadata(cls)
                )
            else:
                print(
                    f"[1hewNodes][Save Text] format={final_format} 时跳过元数据，"
                    "避免破坏文件结构"
                )

        safe_filename_prefix = sanitize_filename_prefix(filename, "text")

        global _PATH_LOCK
        if _PATH_LOCK is None:
            _PATH_LOCK = asyncio.Lock()

        raw_prefix = str(filename or "").strip()
        abs_prefix = is_absolute_like(raw_prefix)
        base_dir = folder_paths.get_output_directory()

        if abs_prefix:
            target_dir_raw, stem = split_prefix_to_dir_and_stem(raw_prefix)
            target_dir = (
                os.path.abspath(target_dir_raw)
                if target_dir_raw
                else os.path.abspath(base_dir)
            )
        else:
            rel_dir_raw, stem = split_prefix_to_dir_and_stem(safe_filename_prefix)
            rel_dir = rel_dir_raw.replace("/", os.sep) if rel_dir_raw else ""
            target_dir = os.path.abspath(os.path.join(base_dir, rel_dir))

        os.makedirs(target_dir, exist_ok=True)

        async with _PATH_LOCK:
            if auto_increment:
                out_path = _next_increment_path(target_dir, stem, extension)
            else:
                out_path = os.path.join(target_dir, f"{stem}.{extension}")

            await asyncio.to_thread(cls._write_text_file, out_path, file_text)

        abs_out_path = os.path.abspath(out_path)
        return io.NodeOutput(
            abs_out_path,
            ui=cls._preview_ui(
                text_value,
                output_path=abs_out_path,
                base_dir=base_dir,
            ),
        )

    @staticmethod
    def _write_text_file(path: str, text: str) -> None:
        # 与 ComfyUI 原生 Save Text 一致，固定 UTF-8。
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(text)
