"""Shared text helpers for 1hewNodes.

Splitting, normalising and preset matching used by the text/list nodes.
This is a library module, not a node: files under nodes/<category>/ must
define exactly one ComfyNode.
"""
from __future__ import annotations

import math
import re


def normalize_text(text: str) -> str:
        return (
            (text or "")
            .replace("【", "[")
            .replace("】", "]")
            .replace("（", "(")
            .replace("）", ")")
            .replace("，", ",")
            .replace("：", ":")
            .replace("；", ",")
            .replace(";", ",")
        )


def split_text(cls, text: str) -> list[str]:
        t = text or ""
        if not t.strip():
            return []

        dash_line_pattern = r"^\s*-+\s*$"
        if re.search(dash_line_pattern, t, flags=re.MULTILINE):
            parts = re.split(dash_line_pattern, t, flags=re.MULTILINE)
        elif ("\n" in t) or ("\r" in t):
            parts = re.split(r"\r?\n", t)
        elif ("。" in t) or ("." in t):
            parts = re.split(r"[。\.]+", t)
        elif ("；" in t) or (";" in t):
            parts = re.split(r"[；;]+", t)
        else:
            parts = re.split(r"[，,]+", t)

        out: list[str] = []
        for p in parts:
            s = cls._strip_item(p)
            if s:
                out.append(str(s))
        return out


def match_ratio(cls, width: int, height: int) -> str:
        image_ratio = width / max(height, 1)
        best_label = cls.RATIO_OPTIONS[0][0]
        best_diff = float("inf")

        for label, rw, rh in cls.RATIO_OPTIONS:
            ratio = rw / rh
            diff = abs(math.log(image_ratio) - math.log(ratio))
            if diff < best_diff:
                best_diff = diff
                best_label = label

        return best_label


def match_resolution(cls, width: int, height: int) -> str:
        area = max(width * height, 1)
        best_label = cls.RESOLUTION_OPTIONS[0][0]
        best_diff = float("inf")

        for label, target_area in cls.RESOLUTION_OPTIONS:
            diff = abs(math.log(area) - math.log(target_area))
            if diff < best_diff:
                best_diff = diff
                best_label = label

        return best_label
