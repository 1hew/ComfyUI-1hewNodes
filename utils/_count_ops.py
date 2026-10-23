"""Shared counting helpers for the Int * Count nodes.

Presence is exactly the ``is_empty`` rule used by ``Any Empty Bool``:
``None``, empty or all-zero tensors, blank strings, ``0`` / ``False`` scalars, and
empty containers are absent.  Counting is per item: lists/tuples are flattened
first (an OUTPUT_IS_LIST upstream contributes one item per element), and an
IMAGE/MASK batch contributes one count per frame.
"""

from __future__ import annotations

from typing import Any, Iterator

import torch

from ._logic_ops import is_empty

_IMAGE_BATCH_NDIM = 4
_MASK_BATCH_NDIM = 3


def iter_count_items(value: Any) -> Iterator[Any]:
    """Yield leaf values, flattening nested lists/tuples from list outputs."""
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from iter_count_items(item)
        return
    yield value


def _count_tensor_items(value: Any, batch_ndim: int) -> int:
    """Count non-empty entries of an IMAGE/MASK tensor, frame by frame."""
    if not isinstance(value, torch.Tensor):
        return 0
    if value.ndim < batch_ndim:
        return 0 if is_empty(value) else 1
    frames = int(value.shape[0])
    if frames <= 0:
        return 0
    return sum(1 for index in range(frames) if not is_empty(value[index : index + 1]))


def count_image_items(value: Any) -> int:
    return sum(
        _count_tensor_items(item, _IMAGE_BATCH_NDIM)
        for item in iter_count_items(value)
    )


def count_mask_items(value: Any) -> int:
    return sum(
        _count_tensor_items(item, _MASK_BATCH_NDIM)
        for item in iter_count_items(value)
    )


def count_video_items(value: Any) -> int:
    return sum(1 for item in iter_count_items(value) if not is_empty(item))


def count_audio_items(value: Any) -> int:
    return sum(1 for item in iter_count_items(value) if not is_empty(item))


def count_text_items(value: Any) -> int:
    """Count non-empty STRING / INT / FLOAT leaves; other types are ignored."""
    return sum(
        1
        for item in iter_count_items(value)
        if isinstance(item, (str, int, float)) and not is_empty(item)
    )


def collect_prefixed_values(
    prefix: str,
    first_value: Any,
    kwargs: dict[str, Any],
) -> list[Any]:
    """Return the ``<prefix>1..N`` values ordered by numeric suffix."""
    indexed: list[tuple[int, Any]] = [(1, first_value)]
    for key, value in kwargs.items():
        if not isinstance(key, str) or not key.startswith(prefix):
            continue
        suffix = key[len(prefix) :]
        if not suffix.isdigit():
            continue
        index = int(suffix)
        if index == 1:
            continue
        indexed.append((index, value))
    indexed.sort(key=lambda item: item[0])
    return [value for _, value in indexed]
