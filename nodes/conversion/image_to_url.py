from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import tempfile
import time
from io import BytesIO
from typing import Optional

import numpy as np
import requests
import torch
from comfy_api.latest import io
from PIL import Image


_CACHE_DIR = os.path.join(tempfile.gettempdir(), "1hew_image_url_cache")
_CACHE_FILE = os.path.join(_CACHE_DIR, "url_mapping.json")


def _to_batch(image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 4:
        return image
    if image.ndim == 3:
        return image.unsqueeze(0)
    raise ValueError("Image to URL expects IMAGE tensor with shape [B,H,W,C] or [H,W,C].")


def _tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    img = image_tensor.detach().to(torch.float32).cpu().clamp(0.0, 1.0).numpy()
    if img.ndim != 3:
        raise ValueError("Image to URL expects a single image tensor with shape [H,W,C].")

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


def _tensor_hash(image_tensor: torch.Tensor) -> str:
    data = image_tensor.detach().to(torch.float32).cpu().numpy().tobytes()
    return hashlib.md5(data).hexdigest()


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


def _get_cached_url(image_hash: str) -> Optional[str]:
    cached = _load_cache().get(image_hash)
    if isinstance(cached, str):
        return cached
    if isinstance(cached, dict):
        url = cached.get("url")
        if isinstance(url, str) and url.strip():
            return url.strip()
    return None


def _cache_url(image_hash: str, url: str) -> None:
    data = _load_cache()
    data[image_hash] = {
        "url": url,
        "timestamp": time.time(),
    }
    _save_cache(data)


def _tensor_to_data_url(image_tensor: torch.Tensor) -> str:
    pil_image = _tensor_to_pil(image_tensor)
    buffer = BytesIO()

    if pil_image.mode == "RGBA":
        pil_image.save(buffer, format="PNG", optimize=True)
        mime = "image/png"
    else:
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        pil_image.save(buffer, format="JPEG", quality=95, optimize=True)
        mime = "image/jpeg"

    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def _upload_tensor_to_kefan(
    image_tensor: torch.Tensor,
    timeout: int,
) -> Optional[str]:
    image_hash = _tensor_hash(image_tensor)
    cached_url = _get_cached_url(image_hash)
    if cached_url:
        return cached_url

    pil_image = _tensor_to_pil(image_tensor)
    is_rgba = pil_image.mode == "RGBA"
    image_format = "PNG" if is_rgba else "JPEG"
    mime = "image/png" if is_rgba else "image/jpeg"
    suffix = ".png" if is_rgba else ".jpg"

    temp_path = os.path.join(
        tempfile.gettempdir(),
        f"1hew_image_to_url_{int(time.time() * 1000)}{suffix}",
    )

    try:
        if not is_rgba and pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        if is_rgba:
            pil_image.save(temp_path, format=image_format, optimize=True)
        else:
            pil_image.save(temp_path, format=image_format, quality=95, optimize=True)

        upload_url = "https://ai.kefan.cn/api/upload/local"
        try:
            with open(temp_path, "rb") as img_file:
                files = {
                    "file": (os.path.basename(temp_path), img_file, mime),
                }
                response = requests.post(upload_url, files=files, timeout=timeout)

            if response.status_code == 200:
                payload = response.json()
                image_url = payload.get("data")
                if payload.get("success") is True and isinstance(image_url, str) and image_url.strip():
                    image_url = image_url.strip()
                    _cache_url(image_hash, image_url)
                    return image_url
        except Exception:
            pass

        return None
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


class ImageToURL(io.ComfyNode):
    MODE_OPTIONS = ["auto", "kefan", "data"]

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_ImageToURL",
            display_name="Image to URL",
            category="1hewNodes/conversion",
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input(
                    "mode",
                    options=cls.MODE_OPTIONS,
                    default="auto",
                ),
                io.Int.Input("timeout", default=30, min=5, max=300, step=1),
            ],
            outputs=[
                io.String.Output(display_name="url"),
            ],
        )

    @classmethod
    async def execute(
        cls,
        image: torch.Tensor,
        mode: str,
        timeout: int,
    ) -> io.NodeOutput:
        image_batch = _to_batch(image)
        if image_batch.shape[0] == 0:
            return io.NodeOutput("")

        urls: list[str] = []
        for idx in range(image_batch.shape[0]):
            single_image = image_batch[idx]

            if mode == "data":
                url = _tensor_to_data_url(single_image)
            elif mode == "kefan":
                url = await asyncio.to_thread(
                    _upload_tensor_to_kefan,
                    single_image,
                    timeout,
                )
                if not url:
                    raise RuntimeError("图像上传失败，未能获取 kefan URL")
            else:
                url = await asyncio.to_thread(
                    _upload_tensor_to_kefan,
                    single_image,
                    timeout,
                )
                if not url:
                    url = _tensor_to_data_url(single_image)

            urls.append(url)

        return io.NodeOutput("\n".join(urls))
