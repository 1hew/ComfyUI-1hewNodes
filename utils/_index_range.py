"""Shared index-expression parser for Image Batch Index and Multi Switch Select.

The grammar is intentionally identical to the original Image Batch Index
implementation, so both nodes accept exactly the same range syntax:

    0,2,6             single values / lists
    0-10, 10-0        inclusive dash ranges (optional ":step")
    [0,10] [0,10)     bracket ranges with inclusive/exclusive endpoints
    (0,10] (0,10)
    [0,20):2          optional step after the closing bracket
    full-width brackets / commas / colons are normalized
    -1, [-3,-1]       negative values, resolved by the caller's resolve hook

Unknown separators (".", "、", "/", whitespace, ...) are skipped, matching the
loose behavior of the original parser.
"""

import re


_BRACKET_RANGE_RE = re.compile(
    r"(?P<open>[\[\(])\s*(?P<start>-?\d+)\s*,\s*(?P<end>-?\d+)\s*"
    r"(?P<close>[\]\)])\s*(?::\s*(?P<step>\d+))?"
)
_DASH_RANGE_RE = re.compile(
    r"(?P<start>-?\d+)\s*-\s*(?P<end>-?\d+)\s*(?::\s*(?P<step>\d+))?"
)
_INT_RE = re.compile(r"(?P<value>-?\d+)")


def normalize_text(text):
    return (
        (text or "")
        .replace("【", "[")
        .replace("】", "]")
        .replace("（", "(")
        .replace("）", ")")
        .replace("，", ",")
        .replace("：", ":")
        .replace("；", ",")
    )


def parse_index_expressions(text, resolve=None):
    """Parse an index expression into a flat list of integer values.

    resolve is applied to every single value and to both range endpoints
    before a range is expanded, preserving the original node's handling of
    negative endpoints. Duplicates and input order are preserved; callers
    decide whether they matter.
    """
    if resolve is None:
        resolve = lambda value: value

    normalized = normalize_text(text)
    if not normalized.strip():
        return []

    indices = []
    pos = 0
    length = len(normalized)

    while pos < length:
        ch = normalized[pos]
        if ch.isspace() or ch in ",;":
            pos += 1
            continue

        match = _BRACKET_RANGE_RE.match(normalized, pos)
        if match:
            start = resolve(int(match.group("start")))
            end = resolve(int(match.group("end")))
            indices.extend(
                _expand_bracket_range(
                    match.group("open"),
                    start,
                    end,
                    match.group("close"),
                    int(match.group("step") or 1),
                )
            )
            pos = match.end()
            continue

        match = _DASH_RANGE_RE.match(normalized, pos)
        if match:
            start = resolve(int(match.group("start")))
            end = resolve(int(match.group("end")))
            indices.extend(
                _expand_dash_range(start, end, int(match.group("step") or 1))
            )
            pos = match.end()
            continue

        match = _INT_RE.match(normalized, pos)
        if match:
            indices.append(resolve(int(match.group("value"))))
            pos = match.end()
            continue

        pos += 1

    return indices


def _expand_bracket_range(open_bracket, start, end, close_bracket, step):
    direction = 1 if start <= end else -1
    actual_step = max(1, step)
    first = start if open_bracket == "[" else start + direction
    last = end if close_bracket == "]" else end - direction
    return _build_range(first, last, actual_step, direction)


def _expand_dash_range(start, end, step):
    direction = 1 if start <= end else -1
    return _build_range(start, end, max(1, step), direction)


def _build_range(first, last, step, direction):
    if direction > 0:
        if first > last:
            return []
        return list(range(first, last + 1, step))

    if first < last:
        return []
    return list(range(first, last - 1, -step))
