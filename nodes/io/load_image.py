from __future__ import annotations

import base64
import hashlib
from io import BytesIO
import math
import os
import time

from aiohttp import web
from comfy_api.latest import io, ui
import folder_paths
import numpy as np
from PIL import Image, ImageOps
from server import PromptServer
import torch
import torch.nn.functional as F


VALID_IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tiff",
    ".webp",
}


class LoadImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="1hew_LoadImage",
            display_name="Load Image",
            category="1hewNodes/io",
            inputs=[
                io.Image.Input("get_image_size", optional=True),
                io.String.Input("path", default=""),
                io.Int.Input("index", default=0, min=-8192, max=8192, step=1),
                io.Boolean.Input("include_subdir", default=True),
                io.Boolean.Input("all", default=False),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Mask.Output(display_name="mask"),
            ],
        )

    @staticmethod
    def load_image(path):
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)

        if img.mode == "I":
            img = img.point(lambda v: v * (1 / 255))

        rgb = img.convert("RGB")
        mask = LoadImage._extract_mask(img, rgb.size)
        return rgb, mask

    @staticmethod
    def _extract_mask(img: Image.Image, rgb_size: tuple[int, int]) -> torch.Tensor:
        w, h = rgb_size

        if "A" in img.getbands():
            alpha = np.array(img.getchannel("A")).astype(np.float32) / 255.0
            base = 1.0 - torch.from_numpy(alpha)
        elif img.mode == "P" and "transparency" in getattr(img, "info", {}):
            alpha = np.array(img.convert("RGBA").getchannel("A")).astype(np.float32) / 255.0
            base = 1.0 - torch.from_numpy(alpha)
        else:
            base = torch.zeros((h, w), dtype=torch.float32)

        editor = LoadImage._load_sidecar_mask(img, w, h)
        if editor is None:
            return base

        return torch.maximum(base, editor)

    @staticmethod
    def _load_sidecar_mask(
        img: Image.Image, w: int, h: int
    ) -> torch.Tensor | None:
        try:
            src_path = getattr(img, "filename", "") or ""
        except Exception:
            src_path = ""

        if not src_path:
            return None

        src_path = os.path.abspath(src_path)
        base_dir = os.path.dirname(src_path)
        base_name = os.path.basename(src_path)
        stem, _ext = os.path.splitext(base_name)

        candidates = [
            os.path.join(base_dir, f"{stem}_mask.png"),
            os.path.join(base_dir, f"{stem}.mask.png"),
            os.path.join(base_dir, f"{base_name}_mask.png"),
            os.path.join(base_dir, f"{base_name}.mask.png"),
            os.path.join(base_dir, f"{stem}_mask.webp"),
            os.path.join(base_dir, f"{stem}.mask.webp"),
            os.path.join(base_dir, f"{base_name}_mask.webp"),
            os.path.join(base_dir, f"{base_name}.mask.webp"),
        ]

        mask_path = next((p for p in candidates if os.path.isfile(p)), "")
        if not mask_path:
            return None

        try:
            m = Image.open(mask_path)
            m = ImageOps.exif_transpose(m)
        except Exception:
            return None

        if m.size != (w, h):
            try:
                m = m.resize((w, h), Image.Resampling.NEAREST)
            except Exception:
                return None

        if "A" in m.getbands():
            arr = np.array(m.getchannel("A")).astype(np.float32) / 255.0
        else:
            arr = np.array(m.convert("L")).astype(np.float32) / 255.0

        return torch.from_numpy(arr)

    @staticmethod
    def pil2tensor(image):
        arr = np.array(image).astype(np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)

    @staticmethod
    def tensor2pil(image):
        arr = np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(
            np.uint8
        )
        return Image.fromarray(arr)

    @staticmethod
    def crop_and_resize(tensor, target_h, target_w):
        """
        先按比例裁剪（Center Crop），再缩放到目标尺寸。
        输入 tensor shape: [1, H, W, C]
        """
        if tensor.shape[1] == target_h and tensor.shape[2] == target_w:
            return tensor

        # 转换为 PIL 以利用 Pillow 的处理逻辑，或者手动计算 tensor 操作
        # 这里为了简单复用逻辑，可以手动计算 tensor 的切片

        curr_h, curr_w = tensor.shape[1], tensor.shape[2]
        curr_ratio = curr_w / curr_h
        target_ratio = target_w / target_h

        # Permute to [B, C, H, W] for processing
        t = tensor.permute(0, 3, 1, 2)

        if curr_ratio > target_ratio:
            # 原图更宽，以高为基准，裁剪宽度
            # new_w = curr_h * target_ratio
            # 实际上我们需要保留的高度就是 curr_h
            # 需要保留的宽度
            crop_w = int(curr_h * target_ratio)
            crop_h = curr_h
        else:
            # 原图更高，以宽为基准，裁剪高度
            crop_w = curr_w
            crop_h = int(curr_w / target_ratio)

        # Center Crop
        start_x = (curr_w - crop_w) // 2
        start_y = (curr_h - crop_h) // 2

        # [B, C, H, W]
        t = t[:, :, start_y : start_y + crop_h, start_x : start_x + crop_w]

        # Resize
        t = F.interpolate(
            t,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )

        # Permute back to [B, H, W, C]
        return t.permute(0, 2, 3, 1)

    @staticmethod
    def get_image_paths(folder, include_subdir):
        if not folder:
            return []

        folder = folder.strip().strip('"').strip("'")
        if os.path.isfile(folder):
            ext = os.path.splitext(folder)[1].lower()
            if ext in VALID_IMAGE_EXTENSIONS:
                return [os.path.abspath(folder)]
            return []

        if not os.path.isdir(folder):
            return []

        image_paths = []

        if include_subdir:
            for root, _dirs, files in os.walk(folder):
                for file in files:
                    if os.path.splitext(file)[1].lower() in VALID_IMAGE_EXTENSIONS:
                        image_paths.append(os.path.join(root, file))
        else:
            for file in os.listdir(folder):
                path = os.path.join(folder, file)
                if (
                    os.path.isfile(path)
                    and os.path.splitext(file)[1].lower() in VALID_IMAGE_EXTENSIONS
                ):
                    image_paths.append(path)

        # Case-insensitive sort for cross-platform consistency
        image_paths.sort(key=lambda x: x.lower())
        return image_paths

    @classmethod
    def IS_CHANGED(cls, path, include_subdir, **kwargs):
        if not path:
            return float("nan")

        image_paths = cls.get_image_paths(path, include_subdir)
        if not image_paths:
            return float("nan")
        m = hashlib.sha256()
        for file_path in image_paths:
            try:
                mtime = os.path.getmtime(file_path)
                m.update(f"{file_path}:{mtime}".encode())
            except OSError:
                continue

        return m.hexdigest()

    @staticmethod
    def crop_and_resize_pil(pil_img, target_w, target_h):
        """
        PIL 版本的 crop_and_resize，保持与 tensor 版本逻辑一致。
        """
        curr_w, curr_h = pil_img.size
        if curr_w == target_w and curr_h == target_h:
            return pil_img

        curr_ratio = curr_w / curr_h
        target_ratio = target_w / target_h

        if curr_ratio > target_ratio:
            # 原图更宽，裁剪宽度
            crop_w = int(curr_h * target_ratio)
            crop_h = curr_h
        else:
            # 原图更高，裁剪高度
            crop_w = curr_w
            crop_h = int(curr_w / target_ratio)

        # Center Crop
        left = (curr_w - crop_w) // 2
        top = (curr_h - crop_h) // 2
        right = left + crop_w
        bottom = top + crop_h

        img_cropped = pil_img.crop((left, top, right, bottom))
        img_resized = img_cropped.resize(
            (target_w, target_h),
            Image.Resampling.BILINEAR,
        )
        return img_resized

    @classmethod
    async def execute(
        cls,
        path: str,
        index: int,
        all: bool,
        include_subdir: bool,
        get_image_size: torch.Tensor | None = None,
    ) -> io.NodeOutput:
        image_paths = cls.get_image_paths(path, include_subdir)
        count = len(image_paths)

        if count == 0:
            return io.NodeOutput(None, None)

        target_h = 0
        target_w = 0

        # 如果提供了参考图片，优先使用参考图片的尺寸
        if get_image_size is not None:
            target_h, target_w = get_image_size.shape[1:3]

        if all:
            images_tensors = []
            masks_tensors = []

            # 如果没有参考图片，加载第一张图片来确定基准尺寸
            start_idx = 0
            if target_h == 0 or target_w == 0:
                try:
                    first_img, first_mask = cls.load_image(image_paths[0])
                    first_tensor = cls.pil2tensor(first_img)
                    first_mask_tensor = first_mask.unsqueeze(0).unsqueeze(
                        -1
                    )

                    target_h, target_w = first_tensor.shape[1:3]
                    images_tensors.append(first_tensor)
                    masks_tensors.append(first_mask_tensor)
                    start_idx = 1
                except Exception as e:
                    print(f"Error loading first image {image_paths[0]}: {e}")
                    return io.NodeOutput(None, None)

            for file_path in image_paths[start_idx:]:
                try:
                    img, mask = cls.load_image(file_path)
                    tensor = cls.pil2tensor(img)
                    mask_tensor = mask.unsqueeze(0).unsqueeze(-1)

                    # 裁剪并缩放
                    tensor = cls.crop_and_resize(tensor, target_h, target_w)
                    mask_tensor = cls.crop_and_resize(mask_tensor, target_h, target_w)

                    images_tensors.append(tensor)
                    masks_tensors.append(mask_tensor)
                except Exception as e:
                    print(f"Error loading image {file_path}: {e}")

            if not images_tensors:
                return io.NodeOutput(None, None)

            output_image = torch.cat(images_tensors, dim=0)
            output_mask = torch.cat(masks_tensors, dim=0).squeeze(-1)

        else:
            idx = index % count
            path = image_paths[idx]
            try:
                img, mask = cls.load_image(path)
                tensor = cls.pil2tensor(img)
                mask_tensor = mask.unsqueeze(0).unsqueeze(-1)

                # 如果有参考尺寸，单张图片也进行裁剪缩放
                if target_h > 0 and target_w > 0:
                    tensor = cls.crop_and_resize(tensor, target_h, target_w)
                    mask_tensor = cls.crop_and_resize(mask_tensor, target_h, target_w)

                output_image = tensor
                output_mask = mask_tensor.squeeze(-1)
            except Exception as e:
                print(f"Error loading image {path}: {e}")
                return io.NodeOutput(None, None)

        return io.NodeOutput(
            output_image,
            output_mask,
        )


def _safe_filename(name: str) -> str:
    if not name:
        return "image"
    base = os.path.basename(name)
    base = base.replace("\\", "_").replace("/", "_").strip()
    return base or "image"


def _get_upload_base_dir() -> str:
    get_input_dir = getattr(folder_paths, "get_input_directory", None)
    if callable(get_input_dir):
        return get_input_dir()
    return folder_paths.get_temp_directory()


def _sanitize_relative_path(name: str) -> str:
    if not name:
        return "image"

    rel = name.replace("\\", "/").lstrip("/").strip()
    rel = os.path.normpath(rel)
    rel = rel.lstrip("\\/").strip()

    if rel in {".", ""}:
        return "image"

    parts = [p for p in rel.split(os.sep) if p not in {"", ".", ".."}]
    if not parts:
        return "image"

    return os.path.join(*parts)


def _sidecar_mask_path(image_path: str) -> str:
    base_dir = os.path.dirname(image_path)
    stem, _ext = os.path.splitext(os.path.basename(image_path))
    return os.path.join(base_dir, f"{stem}_mask.png")


def _decode_data_url(data_url: str) -> bytes:
    if not data_url:
        raise ValueError("empty data url")

    head, sep, tail = data_url.partition(",")
    if not sep:
        raise ValueError("invalid data url")
    if "base64" not in head:
        raise ValueError("missing base64")

    return base64.b64decode(tail.encode("ascii"), validate=False)


def _mask_to_rgba(mask_img: Image.Image) -> Image.Image:
    if "A" in mask_img.getbands():
        alpha = mask_img.getchannel("A")
    else:
        alpha = mask_img.convert("L")

    out = Image.new("RGBA", mask_img.size, (0, 0, 0, 0))
    out.putalpha(alpha)
    return out


@PromptServer.instance.routes.post("/1hew/upload_images")
async def upload_images(request):
    reader = await request.multipart()

    base_dir = _get_upload_base_dir()
    target_root = os.path.join(base_dir, "1hew_uploads_images")
    folder_id = str(time.time_ns())
    target_dir = os.path.abspath(os.path.join(target_root, folder_id))
    os.makedirs(target_dir, exist_ok=True)

    saved: list[str] = []

    while True:
        field = await reader.next()
        if field is None:
            break

        rel = _sanitize_relative_path(field.filename or "image")
        ext = os.path.splitext(rel)[1].lower()
        if ext not in VALID_IMAGE_EXTENSIONS:
            continue

        out_path = os.path.abspath(os.path.join(target_dir, rel))
        try:
            common = os.path.commonpath([target_dir, out_path])
        except ValueError:
            continue
        if common != target_dir:
            continue

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "wb") as f:
            while True:
                chunk = await field.read_chunk()
                if not chunk:
                    break
                f.write(chunk)

        saved.append(out_path)

    if not saved:
        return web.json_response({"error": "no valid files"}, status=400)

    return web.json_response({"folder": target_dir, "count": len(saved)})


@PromptServer.instance.routes.get("/1hew/view_image_from_folder")
async def view_image_from_folder(request):
    folder = request.query.get("path") or request.query.get("folder")
    index_str = request.query.get("index", "0")
    include_subdir = (
        (request.query.get("include_subdir") or "true").lower()
        == "true"
    )
    all_images = request.query.get("all", "false").lower() == "true"
    return_list = request.query.get("return_list", "false").lower() == "true"

    if not folder:
        return web.Response(status=404)

    folder = folder.strip().strip('"').strip("'")
    if os.path.isfile(folder):
        if os.path.splitext(folder)[1].lower() not in VALID_IMAGE_EXTENSIONS:
            return web.Response(status=404)

        if return_list:
            return web.json_response({"count": 1, "paths": [os.path.abspath(folder)]})

        return web.FileResponse(folder)

    if not os.path.isdir(folder):
        return web.Response(status=404)

    try:
        index = int(index_str)
    except ValueError:
        index = 0

    image_paths = LoadImage.get_image_paths(folder, include_subdir)
    if not image_paths:
        return web.Response(status=404)

    if return_list:
        return web.json_response({"count": len(image_paths), "paths": image_paths})

    if all_images:
        # 批量预览模式：生成网格图
        # 限制预览数量，避免过慢
        max_preview = 200
        paths_to_show = image_paths[:max_preview]

        if not paths_to_show:
            return web.Response(status=404)

        images = []

        # 确定基准尺寸：读取第一张图
        try:
            first_img, _ = LoadImage.load_image(paths_to_show[0])
            # 计算预览用的目标尺寸，限制最大边长，但保持长宽比
            base_w, base_h = first_img.size
            max_side = 256
            scale = min(max_side / base_w, max_side / base_h)
            # 至少为 1
            if scale > 1:
                scale = 1

            target_w = int(base_w * scale)
            target_h = int(base_h * scale)

            # 处理第一张图
            images.append(LoadImage.crop_and_resize_pil(first_img, target_w, target_h))

            # 处理后续图片
            for p in paths_to_show[1:]:
                try:
                    img, _ = LoadImage.load_image(p)
                    # 使用与 execute 相同的 crop_and_resize 逻辑
                    processed_img = LoadImage.crop_and_resize_pil(
                        img, target_w, target_h
                    )
                    images.append(processed_img)
                except Exception:
                    continue

        except Exception as e:
            print(f"Preview error: {e}")
            return web.Response(status=404)

        if not images:
            return web.Response(status=404)

        # 计算网格行列
        count = len(images)
        cols = math.ceil(math.sqrt(count))
        rows = math.ceil(count / cols)

        # 创建网格背景
        # 既然所有图片都 resize 到了 target_w, target_h，这里直接用
        cell_w = target_w
        cell_h = target_h
        # ComfyUI 原生 Preview Image 如果是 batch，通常是紧密排列或有很小间距
        # 这里为了对齐效果，我们不加额外大间距，或者只加一点点
        spacing = 0

        grid_w = cols * cell_w + (cols - 1) * spacing
        grid_h = rows * cell_h + (rows - 1) * spacing

        # 使用黑色背景，类似 ComfyUI
        grid_img = Image.new("RGB", (grid_w, grid_h), (0, 0, 0))

        for i, img in enumerate(images):
            r = i // cols
            c = i % cols
            x = c * (cell_w + spacing)
            y = r * (cell_h + spacing)
            grid_img.paste(img, (x, y))

        # 保存到内存流
        stream = BytesIO()
        grid_img.save(stream, format="JPEG", quality=85)
        return web.Response(body=stream.getvalue(), content_type="image/jpeg")

    # 单张预览模式
    idx = index % len(image_paths)
    path = image_paths[idx]

    return web.FileResponse(path)


@PromptServer.instance.routes.get("/1hew/resolve_image_from_folder")
async def resolve_image_from_folder(request):
    folder = request.query.get("path") or request.query.get("folder")
    index_str = request.query.get("index", "0")
    include_subdir = (
        (request.query.get("include_subdir") or "true").lower()
        == "true"
    )
    all_images = request.query.get("all", "false").lower() == "true"

    if not folder:
        return web.json_response({"error": "missing path"}, status=400)

    folder = folder.strip().strip('"').strip("'")
    if os.path.isfile(folder):
        if os.path.splitext(folder)[1].lower() not in VALID_IMAGE_EXTENSIONS:
            return web.json_response({"error": "invalid file"}, status=404)
        return web.json_response({"path": os.path.abspath(folder), "count": 1})

    if not os.path.isdir(folder):
        return web.json_response({"error": "invalid folder"}, status=404)

    try:
        index = int(index_str)
    except ValueError:
        index = 0

    image_paths = LoadImage.get_image_paths(folder, include_subdir)
    if not image_paths:
        return web.json_response({"error": "empty folder"}, status=404)

    if all_images:
        idx = 0
    else:
        idx = index % len(image_paths)

    return web.json_response(
        {"path": os.path.abspath(image_paths[idx]), "count": len(image_paths)}
    )


@PromptServer.instance.routes.post("/1hew/save_sidecar_mask")
async def save_sidecar_mask(request):
    payload = await request.json()
    image_path = str(payload.get("image_path") or "").strip()
    mask_data_url = str(payload.get("mask_data_url") or "").strip()

    if not image_path or not mask_data_url:
        return web.json_response({"error": "missing fields"}, status=400)

    image_path = image_path.strip().strip('"').strip("'")
    if not os.path.isfile(image_path):
        return web.json_response({"error": "image not found"}, status=404)

    try:
        raw = _decode_data_url(mask_data_url)
    except Exception:
        return web.json_response({"error": "invalid mask payload"}, status=400)

    try:
        mask_img = Image.open(BytesIO(raw))
        mask_img = ImageOps.exif_transpose(mask_img)
    except Exception:
        return web.json_response({"error": "invalid mask image"}, status=400)

    try:
        with Image.open(image_path) as src_img:
            src_img = ImageOps.exif_transpose(src_img)
            w, h = src_img.size
    except Exception:
        return web.json_response({"error": "invalid source image"}, status=400)

    if mask_img.size != (w, h):
        mask_img = mask_img.resize((w, h), Image.Resampling.NEAREST)

    out = _mask_to_rgba(mask_img)
    out_path = _sidecar_mask_path(image_path)
    try:
        out.save(out_path, format="PNG")
    except Exception:
        return web.json_response({"error": "save failed"}, status=500)

    return web.json_response({"saved": True, "mask_path": out_path})
