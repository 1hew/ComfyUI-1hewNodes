"""Seedream 坐标（prompt 里的 <bbox>，或图层拆分的 bounding_box）转遮罩。

只处理 bbox，不处理 <point>（点没有面积，属于另一类标记）。坐标只有两种单位，
必须由 `coord_mode` 明确指定，不做自动猜测：

- `normalized`（默认）：千分位 0-1000，对应官方交互编辑指南的
  `<bbox>x1 y1 x2 y2</bbox>`，正向换算是 x = round(x_px / width * 1000)，
  反解就是 x_px = round(x / 1000 * width)。图层拆分的 `bounding_box.normalized` 也是这个单位。
- `absolute`：像素，对应图层拆分的 `bounding_box.absolute`。

两种单位都是 `[left, top, right, bottom]`（xyxy），right / bottom 是边界而非宽高，
所以没有 box_format 选项。遮罩用于局部回贴、框选预览，或交给需要 mask 的下游节点
（Seedream 接口本身只吃 prompt 里的坐标标记，不吃 mask）。

解析不绑定单一格式：带标签（<bbox> / <box> / <rect>）、函数式 bbox(...)、JSON 数组或对象、
以及每行裸数字都可以；`图1` / `Image 1` 之类前缀会被忽略，不会污染数字列表。
"""

from __future__ import annotations

import ast
import json
import math
import re

import numpy as np
import torch
from comfy_api.latest import io


_BOX_TAGS = ("bbox", "box", "rect")
_TAG_NAMES = "|".join(_BOX_TAGS)
_BOX_KEYS = ("bbox", "box", "rect", "xyxy", "bbox_xyxy")

# <bbox>120 180 640 760</bbox>；闭合标签名不强制一致
_TAG_RE = re.compile(
    r"<\s*(%s)\s*>\s*(.*?)\s*<\s*/\s*(?:%s)\s*>" % (_TAG_NAMES, _TAG_NAMES),
    re.IGNORECASE | re.DOTALL,
)
# 兜底：只有开标签、没有闭标签
_OPEN_TAG_RE = re.compile(r"<\s*(%s)\s*>\s*([^<]*)" % _TAG_NAMES, re.IGNORECASE)
# 函数式写法：bbox(...) / bbox[...] / bbox: x1,y1,x2,y2
_CALL_RE = re.compile(r"\b(%s)\s*[\(\[:=]\s*([^)\]\n]*)" % _TAG_NAMES, re.IGNORECASE)
# 任意数字（含小数、负号）；逗号/分号/顿号/空格等自然成为分隔符
_NUM_RE = re.compile(r"[-+]?\d*\.?\d+")
# 整行注释：只有行首才算，避免把 JSON 里 https:// 的斜杠当成注释
_COMMENT_LINE_RE = re.compile(r"^\s*(?:#|//|--)")
# 图1 / 图片 2 / Image 1 / img#3 前缀：序号不能当成坐标
_IMAGE_PREFIX_RE = re.compile(
    r"(?:图片|图像|图|image|img|picture|photo)\s*[#第]?\s*(?:\d+|[一二三四五六七八九十]+)",
    re.IGNORECASE,
)

_FULLWIDTH = {
    "\uff10": "0", "\uff11": "1", "\uff12": "2", "\uff13": "3", "\uff14": "4",
    "\uff15": "5", "\uff16": "6", "\uff17": "7", "\uff18": "8", "\uff19": "9",
    "\uff0c": ",", "\uff0e": ".", "\u3002": ".", "\uff1b": ";", "\uff1a": ":",
    "\uff08": "(", "\uff09": ")", "\uff1c": "<", "\uff1e": ">", "\uff0f": "/",
    "\uff3b": "[", "\uff3d": "]", "\uff5b": "{", "\uff5d": "}", "\u3001": ",",
    "\u3000": " ", "\u00a0": " ",
}

# 图层拆分的 bounding_box 字段 -> 该字段自带的单位
_FIELD_MODES = {"absolute": "absolute", "normalized": "normalized"}


class SeedreamBBoxToMask(io.ComfyNode):
    """把一个或多个 bbox 栅格化成矩形遮罩。"""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_SeedreamBBoxToMask",
            display_name="Seedream BBox to Mask",
            category="1hewNodes/conversion",
            description=(
                "把 Seedream 的 <bbox>x1 y1 x2 y2</bbox> 或图层 bounding_box "
                "还原成矩形遮罩；坐标单位由 coord_mode 指定。"
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="参考图像，仅用于提供输出 mask 的宽高；批次内尺寸一致。",
                ),
                io.String.Input(
                    "coordinates_string",
                    default="",
                    multiline=True,
                    tooltip=(
                        "每行一条，格式固定为 xyxy（left top right bottom）。"
                        "支持 <bbox>x1 y1 x2 y2</bbox>、bbox(...)、JSON 数组/对象、"
                        "带标签的行（如 xyxy: 120 180 640 760）以及裸数字。"
                        "行首 # / // / -- 是整行注释，解析时忽略；"
                        "只解析 bbox，<point> 之类非矩形标记会被忽略。"
                    ),
                ),
                io.Combo.Input(
                    "coord_mode",
                    options=["normalized", "absolute"],
                    default="normalized",
                    tooltip=(
                        "坐标单位。normalized = 千分位 0-1000（prompt 里的 <bbox>、"
                        "bounding_box.normalized）；absolute = 像素（bounding_box.absolute）。"
                    ),
                ),
                io.Combo.Input(
                    "output_mode",
                    options=["separate", "merge"],
                    default="separate",
                    tooltip="separate：每个框一张 mask（批次）；merge：全部合并成一张。",
                ),
            ],
            outputs=[
                io.Mask.Output(display_name="mask"),
                io.Int.Output(display_name="count"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        coordinates_string: str = "",
        coord_mode: str = "normalized",
        output_mode: str = "separate",
    ) -> io.NodeOutput:
        if not isinstance(image, torch.Tensor) or image.ndim < 3:
            raise ValueError("image is required and must be [H,W,C] or [B,H,W,C]")

        frames = image if image.ndim == 4 else image.unsqueeze(0)
        height = int(frames.shape[1])
        width = int(frames.shape[2])
        device = image.device

        mode = cls.resolve_mode(coord_mode)
        items = cls.parse_items(coordinates_string or "", mode)
        boxes = cls.to_boxes(items, mode, width=width, height=height)

        separate = str(output_mode).strip().lower() != "merge"
        if not boxes:
            empty = torch.zeros((0 if separate else 1, height, width),
                                dtype=torch.float32, device=device)
            return io.NodeOutput(empty, 0)

        if separate:
            canvases = []
            for x1, y1, x2, y2 in boxes:
                canvas = np.zeros((height, width), dtype=np.float32)
                canvas[y1:y2, x1:x2] = 1.0
                canvases.append(canvas)
        else:
            union = np.zeros((height, width), dtype=np.float32)
            for x1, y1, x2, y2 in boxes:
                union[y1:y2, x1:x2] = 1.0
            canvases = [union]

        masks = torch.from_numpy(np.stack(canvases, axis=0)).to(dtype=torch.float32, device=device)
        return io.NodeOutput(masks, len(boxes))

    # ------------------------------------------------------------------ parse
    @staticmethod
    def resolve_mode(coord_mode: str) -> str:
        mode = str(coord_mode).strip().lower()
        return "absolute" if mode in ("absolute", "pixel", "px") else "normalized"

    @classmethod
    def parse_items(
        cls,
        text: str,
        mode: str = "normalized",
    ) -> list[tuple[list[float], str | None]]:
        """把文本解析成 [(四个坐标, 该条自带的单位或 None)]；<point> 之类会被忽略。"""
        normalized = cls._normalize_text(cls._strip_comments(text))
        if not normalized.strip():
            return []

        tagged = list(_TAG_RE.finditer(normalized))
        if tagged:
            return cls._collect(tagged)

        opened = list(_OPEN_TAG_RE.finditer(normalized))
        if opened:
            return cls._collect(opened)

        called = list(_CALL_RE.finditer(normalized))
        if called:
            return cls._collect(called)

        stripped = normalized.strip()
        if stripped[:1] in "[{":
            container = cls._load_container(stripped)
            if container is not None:
                items = cls._items_from_container(container, mode)
                if items:
                    return items

        return cls._parse_lines(normalized)

    @classmethod
    def _collect(cls, matches) -> list[tuple[list[float], str | None]]:
        items: list[tuple[list[float], str | None]] = []
        for match in matches:
            numbers = cls._numbers(match.group(2))
            if len(numbers) >= 4:
                items.append((numbers[:4], None))
        return items

    @classmethod
    def _parse_lines(cls, text: str) -> list[tuple[list[float], str | None]]:
        items: list[tuple[list[float], str | None]] = []
        for raw in text.splitlines():
            numbers = cls._numbers(_IMAGE_PREFIX_RE.sub(" ", raw))
            # 行首的行号（如 "1. 520 460 640 760"）会被当成多余数字，去掉再分组
            if len(numbers) >= 5 and len(numbers) % 4 == 1:
                numbers = numbers[1:]
            if len(numbers) >= 4 and len(numbers) % 4 == 0:
                items.extend(
                    (numbers[index:index + 4], None)
                    for index in range(0, len(numbers), 4)
                )
        return items

    @staticmethod
    def _numbers(raw: str) -> list[float]:
        values: list[float] = []
        for token in _NUM_RE.findall(raw or ""):
            try:
                values.append(float(token))
            except ValueError:
                continue
        return values

    @staticmethod
    def _strip_comments(text: str) -> str:
        """丢掉整行注释（#、//、--），方便在坐标块里写 xyxy / 单位之类的说明。"""
        lines = [
            line for line in str(text).splitlines()
            if not _COMMENT_LINE_RE.match(line)
        ]
        return "\n".join(lines)

    @staticmethod
    def _normalize_text(text: str) -> str:
        return str(text).translate(_FULLWIDTH).replace("\r\n", "\n").replace("\r", "\n")

    @staticmethod
    def _load_container(text: str):
        try:
            return json.loads(text)
        except Exception:
            pass
        try:
            return ast.literal_eval(text)
        except Exception:
            return None

    @classmethod
    def _items_from_container(
        cls,
        value,
        prefer: str = "normalized",
        space: str | None = None,
    ) -> list[tuple[list[float], str | None]]:
        items: list[tuple[list[float], str | None]] = []
        if isinstance(value, dict):
            lowered = {str(key).strip().lower(): item for key, item in value.items()}
            # 图层拆分：bounding_box 同时带 absolute 与 normalized，先进入内层再按 prefer 取
            nested = lowered.get("bounding_box")
            if isinstance(nested, dict):
                return cls._items_from_container(nested, prefer, space)
            for key in _BOX_KEYS:
                if key in lowered:
                    return cls._items_from_container(lowered[key], prefer, space)
            if all(key in lowered for key in ("x1", "y1", "x2", "y2")):
                return [([float(lowered[key]) for key in ("x1", "y1", "x2", "y2")], space)]
            if all(key in lowered for key in ("x", "y", "w", "h")):
                # 显式带 w/h 的对象在解析阶段就换算成 xyxy
                x = float(lowered["x"])
                y = float(lowered["y"])
                return [([x, y, x + float(lowered["w"]), y + float(lowered["h"])], space)]

            # 只有 absolute / normalized 这种成对字段时才需要挑一个：
            # 两个都在就按 coord_mode，只剩一个就用它自己的单位，永远不会猜错
            chosen = prefer if prefer in lowered else next(
                (key for key in _FIELD_MODES if key in lowered), None
            )
            if chosen is not None:
                return cls._items_from_container(lowered[chosen], prefer, _FIELD_MODES[chosen])

            for item in value.values():
                items.extend(cls._items_from_container(item, prefer, space))
            return items

        if isinstance(value, (list, tuple)):
            if value and cls._is_number_sequence(value):
                numbers = [float(item) for item in value]
                if len(numbers) == 4:
                    return [(numbers, space)]
                return []
            for item in value:
                items.extend(cls._items_from_container(item, prefer, space))
            return items

        return items

    @staticmethod
    def _is_number_sequence(value) -> bool:
        return all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value)

    # --------------------------------------------------------------- geometry
    @classmethod
    def to_boxes(
        cls,
        items,
        mode: str,
        *,
        width: int,
        height: int,
    ) -> list[tuple[int, int, int, int]]:
        boxes: list[tuple[int, int, int, int]] = []
        for values, space in items:
            current = space or mode
            cls._check_range(values, current)
            x1, y1, x2, y2 = (float(value) for value in values[:4])
            boxes.append((
                cls._clamp_box(
                    cls._scale(x1, width, current),
                    cls._scale(y1, height, current),
                    cls._scale(x2, width, current),
                    cls._scale(y2, height, current),
                    width=width,
                    height=height,
                )
            ))
        return boxes

    @staticmethod
    def _check_range(values, space: str) -> None:
        if space != "normalized":
            return
        for value in values:
            if value < -0.5 or value > 1000.5:
                raise ValueError(
                    "坐标 %s 超出 normalized 的 0-1000 范围；"
                    "这看起来是像素坐标，请把 coord_mode 改成 absolute" % value
                )

    @staticmethod
    def _scale(value: float, extent: int, space: str) -> float:
        if space == "absolute":
            return float(value)
        return float(value) / 1000.0 * float(extent)

    @staticmethod
    def _clamp_box(x1, y1, x2, y2, *, width: int, height: int) -> tuple[int, int, int, int]:
        left = int(math.floor(min(x1, x2) + 0.5))
        top = int(math.floor(min(y1, y2) + 0.5))
        right = int(math.floor(max(x1, x2) + 0.5))
        bottom = int(math.floor(max(y1, y2) + 0.5))
        left = max(0, min(left, width - 1))
        top = max(0, min(top, height - 1))
        right = max(left + 1, min(right, width))
        bottom = max(top + 1, min(bottom, height))
        return (left, top, right, bottom)
