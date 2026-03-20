from __future__ import annotations

import asyncio
import json
import os
from typing import Optional

import folder_paths
import numpy as np
import torch
from comfy_api.latest import io, ui
from PIL import Image, PngImagePlugin

_PATH_LOCK: Optional[asyncio.Lock] = None


def _sanitize_filename_prefix(filename_prefix: str) -> str:
    """Normalize user prefix to a safe relative path for ComfyUI."""
    if not filename_prefix:
        return "image/ComfyUI"

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

    return "/".join(parts) if parts else "image/ComfyUI"


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


def _tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    img = image_tensor.detach().to(torch.float32).cpu().clamp(0.0, 1.0).numpy()
    if img.ndim != 3:
        raise ValueError("Save Image expects [H, W, C] tensor per image.")
    channels = int(img.shape[2])
    if channels >= 4:
        arr = np.clip(img[:, :, :4] * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGBA")
    if channels == 3:
        arr = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")
    if channels == 1:
        arr = np.clip(img[:, :, 0] * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="L")
    raise ValueError(f"Unsupported image channel count: {channels}")


def _to_batch(image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 4:
        return image
    if image.ndim == 3:
        return image.unsqueeze(0)
    raise ValueError("Save Image expects IMAGE tensor with shape [B,H,W,C] or [H,W,C].")


def _collect_png_text_metadata(cls: type) -> dict[str, str]:
    hidden = getattr(cls, "hidden", None)
    if hidden is None:
        return {}
    out: dict[str, str] = {}
    extra = getattr(hidden, "extra_pnginfo", None)
    prompt = getattr(hidden, "prompt", None)
    if isinstance(extra, dict):
        for k, v in extra.items():
            out[str(k)] = json.dumps(v, ensure_ascii=False, separators=(",", ":"))
    if prompt is not None:
        out["prompt"] = json.dumps(prompt, ensure_ascii=False, separators=(",", ":"))
    return out


def _build_pnginfo(metadata: dict[str, str]) -> Optional[PngImagePlugin.PngInfo]:
    if not metadata:
        return None
    pnginfo = PngImagePlugin.PngInfo()
    for key, value in metadata.items():
        pnginfo.add_text(str(key), str(value))
    return pnginfo


def _next_increment_path(directory: str, stem: str) -> str:
    counter = 1
    while True:
        candidate = os.path.join(directory, f"{stem}_{counter:05}_.png")
        if not os.path.exists(candidate):
            return candidate
        counter += 1


class SaveImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="1hew_SaveImage",
            display_name="Save Image",
            category="1hewNodes/io",
            inputs=[
                io.Image.Input("image", optional=True),
                io.String.Input("filename", default="image/ComfyUI"),
                io.Boolean.Input(
                    "auto_increment",
                    default=True,
                    tooltip="为 True 时使用 ComfyUI 自动编号规则；为 False 时固定文件名并覆盖。",
                ),
                io.Boolean.Input(
                    "save_output",
                    default=True,
                    tooltip="为 True 时写入文件；为 False 时不保存到磁盘并返回空路径。",
                ),
                io.Boolean.Input("save_metadata", default=True),
            ],
            outputs=[
                io.String.Output(display_name="file_path"),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    async def execute(
        cls,
        image: Optional[torch.Tensor],
        filename: str,
        auto_increment: bool,
        save_output: bool,
        save_metadata: bool,
    ) -> io.NodeOutput:
        if image is None:
            return io.NodeOutput()

        folder_type = io.FolderType.output
        safe_filename_prefix = _sanitize_filename_prefix(filename)

        global _PATH_LOCK
        if _PATH_LOCK is None:
            _PATH_LOCK = asyncio.Lock()

        class _Hidden:
            prompt = None
            extra_pnginfo = None

        class _NoMetadataNode:
            hidden = _Hidden()

        effective_cls = cls if save_metadata else _NoMetadataNode

        if not save_output:
            async with _PATH_LOCK:
                results = ui.ImageSaveHelper.save_images(
                    image,
                    filename_prefix=safe_filename_prefix,
                    folder_type=io.FolderType.temp,
                    cls=effective_cls,
                    compress_level=4,
                )
            return io.NodeOutput("", ui=ui.SavedImages(results))

        use_helper = (not _is_absolute_like(filename)) and bool(auto_increment)

        if use_helper:
            async with _PATH_LOCK:
                results = ui.ImageSaveHelper.save_images(
                    image,
                    filename_prefix=safe_filename_prefix,
                    folder_type=folder_type,
                    cls=effective_cls,
                    compress_level=4,
                )

            file_paths = []
            if results:
                base_dir = (
                    folder_paths.get_output_directory()
                    if folder_type == io.FolderType.output
                    else folder_paths.get_temp_directory()
                )
                for result in results:
                    file_paths.append(
                        os.path.abspath(
                            os.path.join(base_dir, result.subfolder, result.filename)
                        )
                    )
            return io.NodeOutput("\n".join(file_paths), ui=ui.SavedImages(results))

        image_batch = _to_batch(image)
        metadata = _collect_png_text_metadata(cls) if save_metadata else {}
        pnginfo = _build_pnginfo(metadata)

        raw_prefix = str(filename or "").strip()
        abs_prefix = _is_absolute_like(raw_prefix)
        base_dir = (
            folder_paths.get_output_directory()
            if folder_type == io.FolderType.output
            else folder_paths.get_temp_directory()
        )

        if abs_prefix:
            target_dir_raw, stem = _split_prefix_to_dir_and_stem(raw_prefix)
            target_dir = os.path.abspath(target_dir_raw) if target_dir_raw else os.path.abspath(base_dir)
        else:
            rel_dir_raw, stem = _split_prefix_to_dir_and_stem(safe_filename_prefix)
            rel_dir = rel_dir_raw.replace("/", os.sep) if rel_dir_raw else ""
            target_dir = os.path.abspath(os.path.join(base_dir, rel_dir))

        os.makedirs(target_dir, exist_ok=True)

        file_paths: list[str] = []
        async with _PATH_LOCK:
            batch_count = int(image_batch.shape[0])
            for i in range(batch_count):
                pil_img = _tensor_to_pil(image_batch[i])
                if auto_increment:
                    out_path = _next_increment_path(target_dir, stem)
                else:
                    if i == 0:
                        out_path = os.path.join(target_dir, f"{stem}.png")
                    else:
                        out_path = os.path.join(target_dir, f"{stem}_{i:03d}.png")

                pil_img.save(out_path, format="PNG", compress_level=4, pnginfo=pnginfo)
                file_paths.append(os.path.abspath(out_path))

        return io.NodeOutput("\n".join(file_paths))
