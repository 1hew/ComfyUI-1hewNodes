from __future__ import annotations

import hashlib
import os

import folder_paths
from comfy_api.latest import io

VALID_TEXT_EXTENSIONS = {
    ".txt",
}


def _candidate_paths(file: str) -> list[str]:
    raw = str(file or "").strip()
    if not raw:
        return []

    normalized = raw.replace("\\", os.sep).replace("/", os.sep)
    candidates: list[str] = []

    if os.path.isabs(normalized):
        candidates.append(os.path.abspath(normalized))
        return candidates

    for getter_name in (
        "get_input_directory",
        "get_output_directory",
        "get_temp_directory",
    ):
        getter = getattr(folder_paths, getter_name, None)
        if callable(getter):
            try:
                base_dir = getter()
            except Exception:
                continue
            if base_dir:
                candidates.append(os.path.abspath(os.path.join(base_dir, normalized)))

    candidates.append(os.path.abspath(normalized))
    return candidates


def _resolve_existing_path(file: str) -> str:
    for candidate in _candidate_paths(file):
        if os.path.isfile(candidate) or os.path.isdir(candidate):
            return candidate
    return ""


def _read_text(path: str, encoding: str) -> str:
    encoding_name = str(encoding or "auto").strip().lower()
    if encoding_name == "auto":
        for candidate_encoding in ("utf-8", "utf-8-sig", "gbk", "utf-16"):
            try:
                with open(path, "r", encoding=candidate_encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    with open(path, "r", encoding=encoding_name, errors="replace") as f:
        return f.read()


def _get_filename_stem(path: str) -> str:
    base_name = os.path.basename(str(path or "").strip())
    stem, _ext = os.path.splitext(base_name)
    return stem or "txt"


class LoadTxt(io.ComfyNode):
    @staticmethod
    def _preview_ui(text: str) -> dict:
        return {"text": (str(text),)}

    @staticmethod
    def get_text_paths(path: str, include_subdir: bool) -> list[str]:
        resolved = _resolve_existing_path(path)
        if os.path.isfile(resolved):
            ext = os.path.splitext(resolved)[1].lower()
            return [resolved] if ext in VALID_TEXT_EXTENSIONS else []

        if not os.path.isdir(resolved):
            return []

        text_paths: list[str] = []
        if include_subdir:
            for root, _dirs, files in os.walk(resolved):
                for file_name in files:
                    if os.path.splitext(file_name)[1].lower() in VALID_TEXT_EXTENSIONS:
                        text_paths.append(os.path.join(root, file_name))
        else:
            for file_name in os.listdir(resolved):
                file_path = os.path.join(resolved, file_name)
                if (
                    os.path.isfile(file_path)
                    and os.path.splitext(file_name)[1].lower() in VALID_TEXT_EXTENSIONS
                ):
                    text_paths.append(file_path)

        text_paths.sort(key=lambda x: x.lower())
        return text_paths

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="1hew_LoadTxt",
            display_name="Load Txt",
            category="1hewNodes/io",
            inputs=[
                io.String.Input("file", default=""),
                io.Combo.Input(
                    "encode",
                    default="auto",
                    options=["auto", "utf-8", "utf-8-sig", "gbk", "utf-16"],
                ),
                io.Int.Input("index", default=0, min=-8192, max=8192, step=1),
                io.Boolean.Input("include_subdir", default=False),
            ],
            outputs=[
                io.String.Output(display_name="text"),
                io.String.Output(display_name="filename"),
            ],
        )

    @classmethod
    def IS_CHANGED(
        cls,
        file: str,
        encode: str,
        index: int,
        include_subdir: bool,
        **kwargs,
    ):
        paths = cls.get_text_paths(file, include_subdir)
        if not paths:
            return float("nan")

        m = hashlib.sha256()
        m.update(
            f"index:{int(index)}|include_subdir:{bool(include_subdir)}|encode:{encode}".encode(
                "utf-8"
            )
        )
        for path in paths:
            try:
                stat = os.stat(path)
                m.update(f"{path}|{stat.st_mtime_ns}|{stat.st_size}".encode("utf-8"))
            except OSError:
                continue
        return m.hexdigest()

    @classmethod
    def execute(
        cls,
        file: str,
        encode: str,
        index: int,
        include_subdir: bool,
    ) -> io.NodeOutput:
        paths = cls.get_text_paths(file, include_subdir)
        if not paths:
            raise ValueError(f"Txt file not found: {file}")

        idx = int(index) % len(paths)
        path = paths[idx]
        text = _read_text(path, encode)
        return io.NodeOutput(
            text,
            _get_filename_stem(path),
            ui=cls._preview_ui(text),
        )
