"""Shared file-path helpers for 1hewNodes.

Filename prefix sanitising, absolute-path detection and save retries used
by the io save nodes. This is a library module, not a node: files under
nodes/<category>/ must define exactly one ComfyNode.
"""

from __future__ import annotations

import asyncio
import os


def sanitize_filename_prefix(
    filename_prefix: str, default_folder: str = "image"
) -> str:
    """Normalize user prefix to a safe relative path for ComfyUI."""
    if not filename_prefix:
        return default_folder + "/ComfyUI"

    normalized = filename_prefix.replace("\\", "/").strip()

    # ComfyUI forbids absolute paths; map them to a relative subfolder path.
    drive, tail = os.path.splitdrive(normalized)
    if os.path.isabs(normalized) or drive:
        if drive:
            drive_token = drive.rstrip(":").replace("/", "_").replace("\\", "_")
            tail = tail.lstrip("/\\")
            normalized = f"{drive_token}/{tail}" if tail else drive_token
        else:
            normalized = normalized.lstrip("/\\")

    parts = []
    for part in normalized.split("/"):
        if not part or part == ".":
            continue
        if part == "..":
            continue
        parts.append(part)

    return "/".join(parts) if parts else default_folder + "/ComfyUI"


def is_absolute_like(path_value: str) -> bool:
    raw = str(path_value or "").strip().replace("\\", "/")
    if not raw:
        return False
    drive, _ = os.path.splitdrive(raw)
    return bool(drive) or raw.startswith("/")


def sanitize_filename_stem(name: str) -> str:
    stem = str(name or "").strip()
    if not stem:
        return "ComfyUI"
    for ch in '<>:"/\\|?*':
        stem = stem.replace(ch, "_")
    stem = stem.rstrip(". ")
    return stem if stem else "ComfyUI"


def split_prefix_to_dir_and_stem(prefix: str) -> tuple[str, str]:
    raw = str(prefix or "").strip().replace("\\", "/")
    if not raw:
        return "", "ComfyUI"

    is_trailing_sep = raw.endswith("/")
    cleaned = raw.rstrip("/")
    if not cleaned:
        return "", "ComfyUI"

    if is_trailing_sep:
        if len(cleaned) == 2 and cleaned[1] == ":":
            cleaned += "/"
        return cleaned, "ComfyUI"
    if "/" not in cleaned:
        stem = os.path.splitext(cleaned)[0]
        return "", sanitize_filename_stem(stem)

    dir_part, stem_part = cleaned.rsplit("/", 1)
    if len(dir_part) == 2 and dir_part[1] == ":":
        dir_part += "/"
    stem = os.path.splitext(stem_part)[0]
    return dir_part, sanitize_filename_stem(stem)


async def remove_file_with_retries(
    path: str,
    *,
    attempts: int = 20,
    base_delay_s: float = 0.05,
) -> None:
    for attempt in range(attempts):
        try:
            os.remove(path)
            return
        except FileNotFoundError:
            return
        except PermissionError:
            if attempt >= attempts - 1:
                return
            await asyncio.sleep(base_delay_s * (attempt + 1))
