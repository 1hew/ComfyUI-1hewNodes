from __future__ import annotations

import hashlib
import math
import os
import time
from io import BytesIO
from typing import Any

from aiohttp import web
from comfy_api.latest import io
import numpy as np
from PIL import Image
from server import PromptServer
import torch


VALID_PS_EXTENSIONS = {".psd", ".psb"}
MAX_PREVIEW_LAYERS = 24
UPLOAD_READ_CHUNK_SIZE = 4 * 1024 * 1024


class LoadPS(io.ComfyNode):
    GROUP_MODES = ["layer", "merged"]
    OUTPUT_MODES = ["single_layer", "all_layers", "merged"]

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="1hew_LoadPS",
            display_name="Load PS",
            category="1hewNodes/io",
            inputs=[
                io.String.Input("file", default=""),
                io.Int.Input("index", default=0, min=-8192, max=8192, step=1),
                io.Boolean.Input("include_hidden", default=False),
                io.Combo.Input("group_mode", options=cls.GROUP_MODES, default="layer"),
                io.Combo.Input("output_mode", options=cls.OUTPUT_MODES, default="all_layers"),
                io.Boolean.Input("preview", default=False),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Mask.Output(display_name="mask"),
                io.String.Output(display_name="filename"),
                io.String.Output(display_name="layer_name"),
            ],
        )

    @staticmethod
    def _resolve_input_relative_path(path_value: str) -> str:
        raw = str(path_value or "").strip().strip('"').strip("'")
        if not raw:
            return ""
        drive, _tail = os.path.splitdrive(raw)
        if drive or os.path.isabs(raw):
            return os.path.abspath(raw)

        try:
            import folder_paths

            get_input_dir = getattr(folder_paths, "get_input_directory", None)
            base_dir = get_input_dir() if callable(get_input_dir) else folder_paths.get_temp_directory()
        except Exception:
            base_dir = os.getcwd()
        normalized = raw.replace("\\", os.sep).replace("/", os.sep)
        return os.path.abspath(os.path.join(base_dir, normalized))

    @classmethod
    def IS_CHANGED(
        cls,
        file,
        index=None,
        include_hidden=None,
        group_mode=None,
        output_mode=None,
        preview=None,
        **kwargs,
    ):
        del preview, kwargs
        path = cls._resolve_input_relative_path(file)
        if not path or not os.path.isfile(path):
            return float("nan")
        try:
            m = hashlib.sha256()
            cache_key = {
                "path": path,
                "mtime": os.path.getmtime(path),
                "index": int(index or 0),
                "include_hidden": bool(include_hidden),
                "group_mode": str(group_mode or "layer"),
                "output_mode": str(output_mode or "all_layers"),
            }
            m.update(repr(cache_key).encode("utf-8", errors="ignore"))
            return m.hexdigest()
        except OSError:
            return float("nan")

    @classmethod
    async def execute(
        cls,
        file: str,
        index: int = 0,
        include_hidden: bool = False,
        group_mode: str = "layer",
        output_mode: str = "all_layers",
        preview: bool = False,
    ) -> io.NodeOutput:
        del preview
        path = cls._resolve_input_relative_path(file)
        if not path or not os.path.isfile(path):
            return cls._empty_output()

        ext = os.path.splitext(path)[1].lower()
        if ext not in VALID_PS_EXTENSIONS:
            return cls._empty_output()

        psd = cls._open_psd(path)
        filename = cls._filename_stem(path)

        mode = output_mode if output_mode in cls.OUTPUT_MODES else "all_layers"
        group = group_mode if group_mode in cls.GROUP_MODES else "layer"

        if mode == "merged":
            image = cls._render_psd(psd)
            return cls._build_output([image], filename, ["merged"])

        items = cls._collect_items(psd, group, bool(include_hidden))
        if not items:
            return cls._empty_output(filename)

        if mode == "single_layer":
            item = items[int(index) % len(items)]
            image = cls._render_layer_item(item["node"], psd, bool(include_hidden))
            if cls._is_empty_rgba(image):
                return cls._empty_output(filename)
            return cls._build_output([image], filename, [item["name"]])

        images: list[Image.Image] = []
        names: list[str] = []
        for item in items:
            image = cls._render_layer_item(item["node"], psd, bool(include_hidden))
            if cls._is_empty_rgba(image):
                continue
            images.append(image)
            names.append(item["name"])

        if not images:
            return cls._empty_output(filename)
        return cls._build_output(images, filename, names)

    @staticmethod
    def _open_psd(path: str) -> Any:
        try:
            from psd_tools import PSDImage
        except Exception as exc:
            raise RuntimeError(f"Load PS 需要 psd-tools 依赖，请先安装: python -m pip install psd-tools ({exc})") from exc
        LoadPS._patch_psd_tools_metadata_padding()
        return PSDImage.open(path)

    @staticmethod
    def _patch_psd_tools_metadata_padding() -> None:
        try:
            from psd_tools.psd.bin_utils import read_fmt, read_length_block
            from psd_tools.psd.descriptor import DescriptorBlock
            from psd_tools.psd.tagged_blocks import MetadataSetting
        except Exception:
            return

        if getattr(MetadataSetting, "_comfy1hew_padding_patch", False):
            return

        def read(cls, fp, **kwargs):
            del kwargs
            signature = read_fmt("4s", fp)[0]
            if signature not in cls._KNOWN_SIGNATURES:
                # Some PSDs include zero padding between metadata settings. If
                # the real next signature follows that padding, realign instead
                # of failing the whole document parse.
                pos = fp.tell() - 4
                fp.seek(pos)
                probe = fp.read(7)
                for pad in range(1, 4):
                    if probe[:pad] == b"\0" * pad and probe[pad : pad + 4] in cls._KNOWN_SIGNATURES:
                        fp.seek(pos + pad + 4)
                        signature = probe[pad : pad + 4]
                        break
                else:
                    raise AssertionError("Invalid signature %r" % signature)

            key, copy_on_sheet = read_fmt("4s?3x", fp)
            data: Any = read_length_block(fp)

            pos = fp.tell()
            probe = fp.read(7)
            fp.seek(pos)
            for pad in range(1, 4):
                if probe[:pad] == b"\0" * pad and probe[pad : pad + 4] in cls._KNOWN_SIGNATURES:
                    fp.seek(pos + pad)
                    break

            if key in (b"mdyn", b"sgrp"):
                with BytesIO(data) as f:
                    data = read_fmt("I", f)[0]
            elif key in cls._KNOWN_KEYS:
                data = DescriptorBlock.frombytes(data, padding=4)

            return cls(signature, key, copy_on_sheet, data)

        MetadataSetting.read = classmethod(read)
        MetadataSetting._comfy1hew_padding_patch = True

    @classmethod
    def _collect_items(cls, psd: Any, group_mode: str, include_hidden: bool) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for child in cls._iter_children(psd):
            cls._collect_from_node(
                child,
                items,
                group_mode=group_mode,
                include_hidden=include_hidden,
                parent_visible=True,
                path_parts=[],
            )
        return items

    @classmethod
    def _collect_from_node(
        cls,
        node: Any,
        items: list[dict[str, Any]],
        *,
        group_mode: str,
        include_hidden: bool,
        parent_visible: bool,
        path_parts: list[str],
    ) -> None:
        node_visible = cls._is_visible(node)
        effective_visible = parent_visible and node_visible
        if not include_hidden and not effective_visible:
            return

        name = cls._node_name(node)
        next_path = [*path_parts, name] if name else path_parts

        if cls._is_group(node):
            children = cls._iter_children(node)
            if group_mode == "merged":
                if children:
                    items.append({"node": node, "name": cls._format_name(next_path)})
                return
            for child in children:
                cls._collect_from_node(
                    child,
                    items,
                    group_mode=group_mode,
                    include_hidden=include_hidden,
                    parent_visible=effective_visible,
                    path_parts=next_path,
                )
            return

        items.append({"node": node, "name": cls._format_name(next_path)})

    @staticmethod
    def _iter_children(node: Any) -> list[Any]:
        try:
            # psd-tools stores the top layer at the end of the group list.
            # Reverse it so index/order matches Photoshop's layer panel: top to bottom.
            return list(reversed(list(node)))
        except Exception:
            return []

    @staticmethod
    def _is_group(node: Any) -> bool:
        is_group = getattr(node, "is_group", None)
        if callable(is_group):
            try:
                return bool(is_group())
            except Exception:
                return False
        return bool(getattr(node, "is_group", False))

    @staticmethod
    def _is_visible(node: Any) -> bool:
        is_visible = getattr(node, "is_visible", None)
        if callable(is_visible):
            try:
                return bool(is_visible())
            except Exception:
                return True
        return bool(getattr(node, "visible", True))

    @staticmethod
    def _node_name(node: Any) -> str:
        name = str(getattr(node, "name", "") or "").strip()
        return name or "Layer"

    @staticmethod
    def _format_name(parts: list[str]) -> str:
        clean = [str(part).strip() for part in parts if str(part).strip()]
        return "/".join(clean) if clean else "Layer"

    @classmethod
    def _render_psd(cls, psd: Any) -> Image.Image:
        pil = cls._composite(psd, viewport=cls._canvas_viewport(psd))
        if pil is None:
            return cls._transparent_canvas(psd)
        return cls._ensure_canvas_rgba(pil, psd, None)

    @classmethod
    def _render_layer_item(cls, node: Any, psd: Any, include_hidden: bool) -> Image.Image:
        layer_filter = (lambda _layer: True) if include_hidden else None
        pil = cls._composite(node, viewport=cls._canvas_viewport(psd), layer_filter=layer_filter)
        return cls._ensure_canvas_rgba(pil, psd, node)

    @staticmethod
    def _composite(
        node: Any,
        viewport: tuple[int, int, int, int] | None = None,
        layer_filter: Any | None = None,
    ) -> Image.Image | None:
        composite = getattr(node, "composite", None)
        if not callable(composite):
            return None
        attempts = [
            {"viewport": viewport, "force": True, "apply_icc": True, "layer_filter": layer_filter},
            {"viewport": viewport, "force": True, "layer_filter": layer_filter},
            {"viewport": viewport, "layer_filter": layer_filter},
            {"viewport": viewport, "force": True, "apply_icc": True},
            {"viewport": viewport, "force": True},
            {"viewport": viewport},
            {},
        ]
        for kwargs in attempts:
            try:
                return composite(**{k: v for k, v in kwargs.items() if v is not None})
            except TypeError:
                continue
        return None

    @staticmethod
    def _canvas_size(psd: Any) -> tuple[int, int]:
        width = max(1, int(getattr(psd, "width", 1) or 1))
        height = max(1, int(getattr(psd, "height", 1) or 1))
        return width, height

    @classmethod
    def _canvas_viewport(cls, psd: Any) -> tuple[int, int, int, int]:
        width, height = cls._canvas_size(psd)
        return (0, 0, width, height)

    @classmethod
    def _transparent_canvas(cls, psd: Any) -> Image.Image:
        return Image.new("RGBA", cls._canvas_size(psd), (0, 0, 0, 0))

    @classmethod
    def _ensure_canvas_rgba(cls, pil: Image.Image | None, psd: Any, node: Any | None) -> Image.Image:
        canvas_w, canvas_h = cls._canvas_size(psd)
        if pil is None:
            return Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))

        rgba = pil.convert("RGBA")
        if rgba.size == (canvas_w, canvas_h):
            return rgba

        canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        left, top = cls._node_offset(node)
        canvas.alpha_composite(rgba, dest=(left, top))
        return canvas

    @staticmethod
    def _node_offset(node: Any | None) -> tuple[int, int]:
        if node is None:
            return (0, 0)
        bbox = getattr(node, "bbox", None)
        if bbox is None:
            return (0, 0)
        try:
            return (int(bbox[0]), int(bbox[1]))
        except Exception:
            pass
        left = getattr(bbox, "x1", getattr(bbox, "left", 0))
        top = getattr(bbox, "y1", getattr(bbox, "top", 0))
        return (int(left or 0), int(top or 0))

    @staticmethod
    def _is_empty_rgba(image: Image.Image) -> bool:
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        alpha = np.asarray(image.getchannel("A"), dtype=np.uint8)
        return bool(alpha.size == 0 or int(alpha.max(initial=0)) <= 0)

    @classmethod
    def _build_output(cls, images: list[Image.Image], filename: str, names: list[str]) -> io.NodeOutput:
        image_tensors: list[torch.Tensor] = []
        mask_tensors: list[torch.Tensor] = []
        for image in images:
            rgba = image.convert("RGBA")
            arr = np.asarray(rgba).astype(np.float32) / 255.0
            tensor = torch.from_numpy(arr).to(torch.float32)
            image_tensors.append(tensor)
            mask_tensors.append(tensor[:, :, 3].clone())

        if not image_tensors:
            return cls._empty_output(filename)

        return io.NodeOutput(
            torch.stack(image_tensors, dim=0),
            torch.stack(mask_tensors, dim=0),
            filename,
            "\n".join(names),
        )

    @staticmethod
    def _filename_stem(path: str) -> str:
        base_name = os.path.basename(str(path or "").strip())
        stem, _ext = os.path.splitext(base_name)
        return stem or "ps"

    @staticmethod
    def _empty_output(filename: str = "") -> io.NodeOutput:
        image = torch.zeros((0, 64, 64, 4), dtype=torch.float32)
        mask = torch.zeros((0, 64, 64), dtype=torch.float32)
        return io.NodeOutput(image, mask, filename, "")


def _get_upload_base_dir() -> str:
    try:
        import folder_paths

        get_input_dir = getattr(folder_paths, "get_input_directory", None)
        if callable(get_input_dir):
            return get_input_dir()
        return folder_paths.get_temp_directory()
    except Exception:
        return os.getcwd()


def _sanitize_relative_path(name: str) -> str:
    if not name:
        return "file.psd"
    rel = name.replace("\\", "/").lstrip("/").strip()
    rel = os.path.normpath(rel)
    rel = rel.lstrip("\\/").strip()
    if rel in {".", ""}:
        return "file.psd"
    parts = [p for p in rel.split(os.sep) if p not in {"", ".", ".."}]
    return os.path.join(*parts) if parts else "file.psd"


def _preview_resize(image: Image.Image, max_side: int = 256) -> Image.Image:
    rgba = image.convert("RGBA")
    width, height = rgba.size
    if width <= 0 or height <= 0:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    scale = min(float(max_side) / float(width), float(max_side) / float(height), 1.0)
    target = (max(1, int(width * scale)), max(1, int(height * scale)))
    if target == rgba.size:
        return rgba
    return rgba.resize(target, Image.Resampling.LANCZOS)


def _preview_to_response(image: Image.Image) -> web.Response:
    stream = BytesIO()
    image.convert("RGBA").save(stream, format="PNG")
    return web.Response(body=stream.getvalue(), content_type="image/png")


def _grid_preview(images: list[Image.Image]) -> Image.Image | None:
    if not images:
        return None
    thumbs = [_preview_resize(img) for img in images[:120]]
    if not thumbs:
        return None
    cell_w = max(img.size[0] for img in thumbs)
    cell_h = max(img.size[1] for img in thumbs)
    count = len(thumbs)
    cols = max(1, math.ceil(math.sqrt(count)))
    rows = math.ceil(count / cols)
    grid = Image.new("RGBA", (cols * cell_w, rows * cell_h), (0, 0, 0, 255))
    for i, img in enumerate(thumbs):
        row = i // cols
        col = i % cols
        x = col * cell_w + (cell_w - img.size[0]) // 2
        y = row * cell_h + (cell_h - img.size[1]) // 2
        grid.alpha_composite(img, dest=(x, y))
    return grid


@PromptServer.instance.routes.post("/1hew/upload_ps")
async def upload_ps(request):
    reader = await request.multipart()

    base_dir = _get_upload_base_dir()
    target_root = os.path.join(base_dir, "1hew_uploads_ps")
    folder_id = str(time.time_ns())
    target_dir = os.path.abspath(os.path.join(target_root, folder_id))
    os.makedirs(target_dir, exist_ok=True)

    saved: list[str] = []
    while True:
        field = await reader.next()
        if field is None:
            break

        rel = _sanitize_relative_path(field.filename or "file.psd")
        ext = os.path.splitext(rel)[1].lower()
        if ext not in VALID_PS_EXTENSIONS:
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
                chunk = await field.read_chunk(size=UPLOAD_READ_CHUNK_SIZE)
                if not chunk:
                    break
                f.write(chunk)
        saved.append(out_path)

    if not saved:
        return web.json_response({"error": "no valid psd files"}, status=400)
    return web.json_response({"path": saved[0], "count": len(saved), "files": saved})


@PromptServer.instance.routes.get("/1hew/view_ps")
async def view_ps(request):
    path = request.query.get("file") or request.query.get("path")
    if not path:
        return web.Response(status=404)

    path = LoadPS._resolve_input_relative_path(path)
    if not os.path.isfile(path) or os.path.splitext(path)[1].lower() not in VALID_PS_EXTENSIONS:
        return web.Response(status=404)

    try:
        index = int(request.query.get("index", "0"))
    except ValueError:
        index = 0
    include_hidden = (request.query.get("include_hidden") or "false").lower() == "true"
    group_mode = request.query.get("group_mode") or "layer"
    output_mode = request.query.get("output_mode") or "all_layers"
    group_mode = group_mode if group_mode in LoadPS.GROUP_MODES else "layer"
    output_mode = output_mode if output_mode in LoadPS.OUTPUT_MODES else "all_layers"

    try:
        psd = LoadPS._open_psd(path)
        if output_mode == "merged":
            return _preview_to_response(LoadPS._render_psd(psd))

        items = LoadPS._collect_items(psd, group_mode, include_hidden)
        if not items:
            return web.Response(status=404)

        if output_mode == "single_layer":
            item = items[index % len(items)]
            image = LoadPS._render_layer_item(item["node"], psd, include_hidden)
            if LoadPS._is_empty_rgba(image):
                return web.Response(status=404)
            return _preview_to_response(image)

        images: list[Image.Image] = []
        for item in items[:MAX_PREVIEW_LAYERS]:
            image = LoadPS._render_layer_item(item["node"], psd, include_hidden)
            if not LoadPS._is_empty_rgba(image):
                images.append(image)
        grid = _grid_preview(images)
        if grid is None:
            return web.Response(status=404)
        return _preview_to_response(grid)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)
