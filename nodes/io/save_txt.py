from __future__ import annotations

import asyncio
import json
import os
from typing import Optional

import folder_paths
from comfy_api.latest import io

_PATH_LOCK: Optional[asyncio.Lock] = None


def _sanitize_filename_prefix(filename: str) -> str:
    if not filename:
        return "txt/ComfyUI"

    normalized = filename.replace("\\", "/").strip()

    drive, tail = os.path.splitdrive(normalized)
    if os.path.isabs(normalized) or drive:
        if drive:
            drive_token = drive.rstrip(":").replace("/", "_").replace("\\", "_")
            tail = tail.lstrip("/\\")
            normalized = f"{drive_token}/{tail}" if tail else drive_token
        else:
            normalized = normalized.lstrip("/\\")

    parts: list[str] = []
    for part in normalized.split("/"):
        if not part or part in {".", ".."}:
            continue
        parts.append(part)

    return "/".join(parts) if parts else "txt/ComfyUI"


def _is_absolute_like(path_value: str) -> bool:
    raw = str(path_value or "").strip().replace("\\", "/")
    if not raw:
        return False
    drive, _ = os.path.splitdrive(raw)
    return bool(drive) or raw.startswith("/")


def _sanitize_filename_stem(name: str) -> str:
    stem = str(name or "").strip()
    if not stem:
        return "ComfyUI"
    for ch in '<>:"/\\|?*':
        stem = stem.replace(ch, "_")
    stem = stem.rstrip(". ")
    return stem if stem else "ComfyUI"


def _split_prefix_to_dir_and_stem(prefix: str) -> tuple[str, str]:
    raw = str(prefix or "").strip().replace("\\", "/")
    if not raw:
        return "", "ComfyUI"

    is_trailing_sep = raw.endswith("/")
    cleaned = raw.rstrip("/")
    if not cleaned:
        return "", "ComfyUI"

    if is_trailing_sep:
        return cleaned, "ComfyUI"
    if "/" not in cleaned:
        stem = os.path.splitext(cleaned)[0]
        return "", _sanitize_filename_stem(stem)

    dir_part, stem_part = cleaned.rsplit("/", 1)
    stem = os.path.splitext(stem_part)[0]
    return dir_part, _sanitize_filename_stem(stem)


def _next_increment_path(directory: str, stem: str) -> str:
    counter = 1
    while True:
        candidate = os.path.join(directory, f"{stem}_{counter:05}_.txt")
        if not os.path.exists(candidate):
            return candidate
        counter += 1


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


class SaveTxt(io.ComfyNode):
    @staticmethod
    def _preview_ui(text: str) -> dict:
        return {"text": (str(text),)}

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="1hew_SaveTxt",
            display_name="Save Txt",
            category="1hewNodes/io",
            inputs=[
                io.Custom("*").Input("any"),
                io.String.Input("filename", default="txt/ComfyUI"),
                io.Boolean.Input(
                    "auto_increment",
                    default=True,
                    tooltip="为 True 时自动编号；为 False 时固定文件名并覆盖。",
                ),
                io.Boolean.Input(
                    "save_output",
                    default=True,
                    tooltip="为 True 时写入 txt 文件；为 False 时不保存到磁盘。",
                ),
                io.Combo.Input(
                    "encode",
                    default="utf-8",
                    options=["utf-8", "utf-8-sig", "gbk"],
                ),
            ],
            outputs=[
                io.String.Output(display_name="file_path"),
            ],
            is_output_node=True,
        )

    @classmethod
    def validate_inputs(cls, **kwargs):
        return True

    @classmethod
    async def execute(
        cls,
        any,
        filename: str,
        auto_increment: bool,
        save_output: bool,
        encode: str,
    ) -> io.NodeOutput:
        text_value = _stringify_value(any)
        if not save_output:
            return io.NodeOutput("", ui=cls._preview_ui(text_value))

        safe_filename_prefix = _sanitize_filename_prefix(filename)

        global _PATH_LOCK
        if _PATH_LOCK is None:
            _PATH_LOCK = asyncio.Lock()

        raw_prefix = str(filename or "").strip()
        abs_prefix = _is_absolute_like(raw_prefix)
        base_dir = folder_paths.get_output_directory()

        if abs_prefix:
            target_dir_raw, stem = _split_prefix_to_dir_and_stem(raw_prefix)
            target_dir = (
                os.path.abspath(target_dir_raw)
                if target_dir_raw
                else os.path.abspath(base_dir)
            )
        else:
            rel_dir_raw, stem = _split_prefix_to_dir_and_stem(safe_filename_prefix)
            rel_dir = rel_dir_raw.replace("/", os.sep) if rel_dir_raw else ""
            target_dir = os.path.abspath(os.path.join(base_dir, rel_dir))

        os.makedirs(target_dir, exist_ok=True)

        async with _PATH_LOCK:
            if auto_increment:
                out_path = _next_increment_path(target_dir, stem)
            else:
                out_path = os.path.join(target_dir, f"{stem}.txt")

            await asyncio.to_thread(cls._write_text_file, out_path, text_value, encode)

        return io.NodeOutput(
            os.path.abspath(out_path),
            ui=cls._preview_ui(text_value),
        )

    @staticmethod
    def _write_text_file(path: str, text: str, encode: str) -> None:
        with open(path, "w", encoding=encode, newline="") as f:
            f.write(text)
