from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
import time
from typing import Optional, Tuple

import requests

from comfy_api.latest import io


_CACHE_DIR = os.path.join(tempfile.gettempdir(), "1hew_video_url_cache")
_CACHE_FILE = os.path.join(_CACHE_DIR, "url_mapping.json")

_EXT_TO_MIME = {
    ".mp4": "video/mp4",
    ".m4v": "video/x-m4v",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
    ".mkv": "video/x-matroska",
    ".avi": "video/x-msvideo",
    ".flv": "video/x-flv",
    ".wmv": "video/x-ms-wmv",
    ".gif": "image/gif",
}

_UPLOAD_URL = "https://ai.kefan.cn/api/upload/local"


def _ext_from_path(path: str) -> str:
    ext = os.path.splitext(str(path or ""))[1].lower()
    return ext if ext.startswith(".") else ""


def _mime_for_ext(ext: str) -> str:
    return _EXT_TO_MIME.get(ext.lower(), "video/mp4")


def _download_bytes(url: str, timeout: int) -> Optional[bytes]:
    try:
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()
        buf = bytearray()
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                buf.extend(chunk)
        return bytes(buf)
    except Exception:
        return None


def _read_file_bytes(path: str) -> Optional[bytes]:
    try:
        with open(path, "rb") as f:
            return f.read()
    except Exception:
        return None


def _read_filelike(file_obj) -> bytes:
    data = file_obj.read()
    if isinstance(data, str):
        data = data.encode("utf-8")
    return data


def _filelike_ext(file_obj) -> str:
    name = getattr(file_obj, "name", None)
    return _ext_from_path(str(name)) if name else ""


def _video_to_bytes(video) -> Tuple[bytes, str]:
    """从视频对象中提取原始字节与扩展名（含点）。"""
    ext = ".mp4"

    get_stream_source = getattr(video, "get_stream_source", None)
    if callable(get_stream_source):
        try:
            source = get_stream_source()
        except Exception:
            source = None

        if isinstance(source, str):
            if source.startswith(("http://", "https://")):
                data = _download_bytes(source, 30)
                if data:
                    return data, _ext_from_path(source) or ext
            elif os.path.isfile(source):
                data = _read_file_bytes(source)
                if data is not None:
                    return data, _ext_from_path(source) or ext
        elif hasattr(source, "read"):
            try:
                return _read_filelike(source), _filelike_ext(source) or ext
            except Exception:
                pass

    for attr in ("path", "source_path"):
        path = getattr(video, attr, None)
        if isinstance(path, str) and os.path.isfile(path):
            data = _read_file_bytes(path)
            if data is not None:
                return data, _ext_from_path(path) or ext

    file_obj = getattr(video, "_VideoFromFile__file", None)
    if hasattr(file_obj, "read"):
        try:
            return _read_filelike(file_obj), _filelike_ext(file_obj) or ext
        except Exception:
            pass
    if isinstance(file_obj, str) and os.path.isfile(file_obj):
        data = _read_file_bytes(file_obj)
        if data is not None:
            return data, _ext_from_path(file_obj) or ext

    save_to = getattr(video, "save_to", None)
    if callable(save_to):
        tmp_path = os.path.join(
            tempfile.gettempdir(),
            f"1hew_video_to_url_{int(time.time() * 1000)}.mp4",
        )
        try:
            save_to(tmp_path)
            data = _read_file_bytes(tmp_path)
            if data is not None:
                return data, ext
        except Exception:
            pass
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    raise ValueError("无法从视频对象中提取文件内容")


def _ensure_cache_dir() -> None:
    os.makedirs(_CACHE_DIR, exist_ok=True)


def _load_cache() -> dict[str, dict[str, object]]:
    _ensure_cache_dir()
    if not os.path.isfile(_CACHE_FILE):
        return {}

    try:
        with open(_CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_cache(data: dict[str, dict[str, object]]) -> None:
    _ensure_cache_dir()
    with open(_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _get_cached_url(video_hash: str) -> Optional[str]:
    cached = _load_cache().get(video_hash)
    if isinstance(cached, str):
        return cached
    if isinstance(cached, dict):
        url = cached.get("url")
        if isinstance(url, str) and url.strip():
            return url.strip()
    return None


def _cache_url(video_hash: str, url: str) -> None:
    data = _load_cache()
    data[video_hash] = {
        "url": url,
        "timestamp": time.time(),
    }
    _save_cache(data)


def _upload_bytes_to_kefan(data: bytes, ext: str, timeout: int) -> Optional[str]:
    video_hash = hashlib.md5(data).hexdigest()
    cached_url = _get_cached_url(video_hash)
    if cached_url:
        return cached_url

    temp_path = os.path.join(
        tempfile.gettempdir(),
        f"1hew_video_to_url_{int(time.time() * 1000)}{ext}",
    )

    try:
        with open(temp_path, "wb") as f:
            f.write(data)

        with open(temp_path, "rb") as f:
            files = {
                "file": (os.path.basename(temp_path), f, _mime_for_ext(ext)),
            }
            response = requests.post(_UPLOAD_URL, files=files, timeout=timeout)

        if response.status_code == 200:
            payload = response.json()
            image_url = payload.get("data")
            if payload.get("success") is True and isinstance(image_url, str) and image_url.strip():
                image_url = image_url.strip()
                _cache_url(video_hash, image_url)
                return image_url
    except Exception:
        pass
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

    return None


class VideoToURL(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_VideoToURL",
            display_name="Video to URL",
            category="1hewNodes/conversion",
            inputs=[
                io.Video.Input("video", optional=True),
                io.Int.Input("timeout", default=30, min=5, max=300, step=1),
            ],
            outputs=[
                io.String.Output(display_name="url"),
            ],
        )

    @classmethod
    async def execute(
        cls,
        video=None,
        timeout: int = 30,
    ) -> io.NodeOutput:
        if video is None:
            return io.NodeOutput("")

        data, ext = await asyncio.to_thread(_video_to_bytes, video)
        if not data:
            return io.NodeOutput("")

        url = await asyncio.to_thread(_upload_bytes_to_kefan, data, ext, timeout)
        if not url:
            raise RuntimeError("视频上传失败，未能获取 kefan URL")

        return io.NodeOutput(url)
