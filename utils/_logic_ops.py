"""Shared value helpers for 1hewNodes.

Comparison and emptiness checks used by the Logic nodes. This is a shared
module, not a node: files under nodes/<category>/ must define exactly one
ComfyNode, so reusable code lives here instead of inside a node file.
"""

from __future__ import annotations


import numpy as np
import torch


def compare(a: int, operator: str, b: int) -> bool:
    if operator == "==":
        return a == b
    if operator == "!=":
        return a != b
    if operator == ">":
        return a > b
    if operator == ">=":
        return a >= b
    if operator == "<":
        return a < b
    if operator == "<=":
        return a <= b
    return False


def is_empty(value):
    if value is None:
        return True

    if isinstance(value, str):
        return len(value.strip()) == 0

    if isinstance(value, bool):
        return not value

    if isinstance(value, (int, float)):
        return value == 0

    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return True
        return torch.all(value == 0).item()

    if isinstance(value, np.ndarray):
        if value.size == 0:
            return True
        return np.all(value == 0)

    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return True
        return all(is_empty(item) for item in value)

    if isinstance(value, dict):
        return len(value) == 0

    if hasattr(value, "__len__"):
        try:
            return len(value) == 0
        except Exception:
            pass

    return False
