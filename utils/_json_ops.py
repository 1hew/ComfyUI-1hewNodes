"""Shared JSON helpers for the Multi Dict / Multi List nodes.

The nodes target the JSON data model only: object, array, string, number,
boolean and null. Python-only values (set, bytes, complex, Decimal, datetime,
tensors) are rejected with a path so a bad value never reaches a JSON file
silently.
"""

from __future__ import annotations

import json
import math
from typing import Any

__all__ = [
    "as_text",
    "coerce_value",
    "collect_indexed_inputs",
    "extend_values",
    "json_text",
    "merge_dict_inputs",
    "normalize_key",
    "pretty_json_text",
    "validate_json_value",
]

_LIST_MODES = ("extend", "append")
# One shared list so Multi Dict and Data Path Replace stay in step.
# `auto` keeps whatever arrives; the rest coerce to that JSON type.
VALUE_TYPES = (
    "auto",
    "dict",
    "list",
    "string",
    "int",
    "float",
    "boolean",
    "null",
)
_TRUE_TEXT = {"true", "1", "yes", "y", "on"}
_FALSE_TEXT = {"false", "0", "no", "n", "off", ""}


def as_text(value: Any) -> str:
    """Normalise a widget value that may arrive as a single-item list."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return as_text(value[0]) if value else ""
    return str(value)


def normalize_key(key: Any) -> str:
    """Return a JSON object key: strings pass through, scalars are stringified."""
    if key is None:
        return ""
    if isinstance(key, str):
        return key
    if isinstance(key, bool):
        return "true" if key else "false"
    if isinstance(key, int):
        return str(key)
    if isinstance(key, float):
        if not math.isfinite(key):
            raise ValueError(f"键 {key!r} 不是有限数字")
        return str(key)
    raise ValueError(f"键必须是字符串或标量，收到 {type(key).__name__}")


def _reject_constant(value: str) -> Any:
    raise ValueError(f"{value} 不是合法 JSON 值")


def _require_finite(value: float, label: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{label} 不能是 {value!r}")


def coerce_value(value: Any, value_type: Any = "auto") -> Any:
    """Convert a connected value to the requested JSON-friendly type."""
    mode = as_text(value_type).strip().lower() or "auto"
    if mode not in VALUE_TYPES:
        raise ValueError(
            f"value_type 仅支持 {', '.join(VALUE_TYPES)}，收到 {mode!r}"
        )
    if value is None:
        return None
    if mode == "auto":
        return value
    if mode == "null":
        return None
    if mode == "string":
        return _value_to_text(value)
    if mode == "int":
        return _value_to_int(value)
    if mode == "float":
        return _value_to_float(value)
    if mode == "boolean":
        return _value_to_bool(value)
    if mode == "dict":
        parsed = _value_from_json(value)
        if isinstance(parsed, dict):
            return parsed
        raise ValueError(
            f"value_type=dict 需要字典/JSON 对象，收到 {type(parsed).__name__}"
        )
    if mode == "list":
        parsed = _value_from_json(value)
        if isinstance(parsed, (list, tuple)):
            return list(parsed)
        raise ValueError(
            f"value_type=list 需要列表/JSON 数组，收到 {type(parsed).__name__}"
        )
    return _value_from_json(value)


def _value_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list, tuple)):
        return json_text(value)
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        _require_finite(value, "value")
        return str(value)
    raise ValueError(f"无法把 {type(value).__name__} 转成 string")


def _value_to_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        _require_finite(value, "value")
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        try:
            return int(text)
        except ValueError:
            try:
                return int(float(text))
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"无法把 {value!r} 转成 int") from exc
    raise ValueError(f"无法把 {type(value).__name__} 转成 int")


def _value_to_float(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int):
        return float(value)
    if isinstance(value, float):
        _require_finite(value, "value")
        return value
    if isinstance(value, str):
        try:
            parsed = float(value.strip())
        except (TypeError, ValueError) as exc:
            raise ValueError(f"无法把 {value!r} 转成 float") from exc
        _require_finite(parsed, "value")
        return parsed
    raise ValueError(f"无法把 {type(value).__name__} 转成 float")


def _value_to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, float):
        _require_finite(value, "value")
        return value != 0
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _TRUE_TEXT:
            return True
        if text in _FALSE_TEXT:
            return False
        try:
            parsed = float(text)
        except ValueError as exc:
            raise ValueError(f"无法把 {value!r} 转成 boolean") from exc
        _require_finite(parsed, "value")
        return parsed != 0
    raise ValueError(f"无法把 {type(value).__name__} 转成 boolean")


def _value_from_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            parsed = json.loads(value, parse_constant=_reject_constant)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"无法按 JSON 解析：{exc}") from exc
        if isinstance(parsed, float):
            _require_finite(parsed, "value")
        return parsed
    if isinstance(value, (dict, list, tuple)) or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        _require_finite(value, "value")
        return value
    raise ValueError(f"无法把 {type(value).__name__} 当作 JSON 值")


def collect_indexed_inputs(
    prefix: str,
    first: Any,
    kwargs: dict[str, Any],
) -> list[tuple[str, Any]]:
    """Return the prefix+1..N inputs ordered by numeric suffix."""
    indexed: list[tuple[int, str, Any]] = [(1, f"{prefix}1", first)]
    for key, value in kwargs.items():
        if not isinstance(key, str) or not key.startswith(prefix):
            continue
        suffix = key[len(prefix):]
        if not suffix.isdigit():
            continue
        index = int(suffix)
        if index <= 1:
            continue
        indexed.append((index, key, value))
    indexed.sort(key=lambda item: item[0])
    return [(name, value) for _, name, value in indexed]


def _json_key(key: Any, path: str) -> str:
    """A JSON object key is always a string, so stringify the scalar keys.

    A Python dict accepts int / float / bool / None keys, and json.dumps would
    silently write them as "1" / "true" / "null". Doing it here keeps the
    check explicit and rejects the keys JSON has no form for.
    """
    if isinstance(key, str):
        return key
    if isinstance(key, bool):
        return "true" if key else "false"
    if key is None:
        return "null"
    if isinstance(key, int):
        return str(key)
    if isinstance(key, float):
        _require_finite(key, f"{path} 的键")
        return str(key)
    raise ValueError(
        f"{path} 的键必须是字符串或标量，收到 {type(key).__name__}"
    )


def validate_json_value(value: Any, path: str = "$") -> None:
    """Raise a path-aware error when value cannot be represented as JSON."""
    if isinstance(value, dict):
        for key, item in value.items():
            _json_key(key, path)
            validate_json_value(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            validate_json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} 是 {value!r}，不是合法 JSON 数字")
        return
    raise ValueError(f"{path} 的类型 {type(value).__name__} 不能转成 JSON")


def json_text(value: Any) -> str:
    """Compact JSON text; raises instead of silently stringifying bad values."""
    validate_json_value(value)
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def pretty_json_text(value: Any) -> str:
    """Indented JSON text used by Dict previews and Save Text."""
    validate_json_value(value)
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)


def merge_dict_inputs(inputs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Shallow-merge dict_N inputs in port order; later keys overwrite earlier ones."""
    merged: dict[str, Any] = {}
    for name, value in inputs:
        if value is None:
            continue
        if not isinstance(value, dict):
            raise ValueError(
                f"{name} 需要是字典，收到 {type(value).__name__}；"
                "列表请拆成单独的 dict_N 输入"
            )
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(
                    f"{name} 的键必须是字符串，收到 {type(key).__name__}"
                )
            merged[key] = item
    return merged


def extend_values(inputs: list[tuple[str, Any]], mode: Any = "extend") -> list[Any]:
    """Combine item_N inputs into a list; extend splices list/tuple inputs."""
    normalized = as_text(mode).strip().lower() or "extend"
    if normalized not in _LIST_MODES:
        raise ValueError(
            f"mode 仅支持 {', '.join(_LIST_MODES)}，收到 {normalized!r}"
        )
    result: list[Any] = []
    for _, value in inputs:
        if value is None:
            continue
        if normalized == "extend" and isinstance(value, (list, tuple)):
            result.extend(value)
        else:
            result.append(value)
    return result
