"""JSON-like text parsing and path queries shared by the text nodes.

The public surface is small on purpose:

    data = parse_content(value)          # str / dict / list / number / ...
    value = query(data, "[0].content")   # first match, or a default
    values = query_all(data, "[*].content")

parse_content accepts whatever a ComfyUI input hands over: a STRING widget
(plain text, JSON, JSON inside a Markdown fence, a Python literal, relaxed
JSON with single quotes / trailing commas / True|False|None, or JSON with
unquoted keys), an already-parsed dict / list, or any other value, which is
returned unchanged.

query walks that object with a JavaScript-flavoured path:

    [0].content          list index, then key
    [-1].bbox            negative index
    content              bare key == ["content"]
    $.data.items[0].zh   optional JSONPath-ish root
    [0]["content"]       quoted bracket key
    [0].'content'        single quotes work too
    data[*].content      wildcard: collect every element's content
    data.*.content       dot form of the wildcard
    [*].content.count    count / length / size / [#]: how many
    [0].bbox.length      length of the matched list (4)

Full-width brackets / quotes / commas from a Chinese IME are normalised, so
a path typed as [0].“content” behaves like [0]."content".

This module is a library, not a node: files under nodes/<category>/ must
define exactly one ComfyNode, so shared helpers live in utils/_*.py.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Mapping, Sequence

__all__ = [
    "clean_json_text",
    "parse_json_text",
    "parse_content",
    "parse_path",
    "query",
    "query_all",
    "is_count_path",
    "format_value",
]


# ---------------------------------------------------------------------------
# JSON-like text parsing
# ---------------------------------------------------------------------------

_FENCE = chr(96) * 3
_PY_LITERAL_VALUE = {"True": "true", "False": "false", "None": "null"}
_PY_LITERAL_RE = re.compile(r"\b(True|False|None)\b")
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")
_KV_PAIR_RE = re.compile(r'"([^"]+)"\s*:\s*"([^"]+)"')


def clean_json_text(text) -> str:
    """Strip Markdown fences and non-JSON noise around the first JSON block."""
    text = str(text or "").strip()
    if not text:
        return ""

    if _FENCE in text:
        for block in text.split(_FENCE)[1::2]:
            body = block.strip()
            if body[:4].lower() == "json":
                body = body[4:].lstrip()
            if body[:1] in "[{":
                text = body
                break

    start = -1
    for index, char in enumerate(text):
        if char in "[{":
            start = index
            break
    if start == -1:
        return text

    text = text[start:]
    end = -1
    for index in range(len(text) - 1, -1, -1):
        if text[index] in "]}":
            end = index + 1
            break
    if end != -1:
        text = text[:end]
    return text


def parse_json_text(value):
    """Parse JSON-like text, returning None when nothing can be parsed.

    The fallback chain is, in order: strict JSON, Python literal
    (single quotes, True/False/None), relaxed JSON (trailing commas and
    Python literals), a hand-written loose parser (unquoted keys, single
    quotes inside arrays, bare tokens) and finally a regex scrape of
    "key": "value" pairs so a truncated payload still yields a dict.
    """
    if value is None:
        return None
    text = value if isinstance(value, str) else str(value)

    # A JSON string that wraps a payload ("{\"a\": 1}") must be decoded
    # before the fence/brace trimming below, otherwise the escaped quotes
    # would be read as literal characters.
    stripped = text.strip()
    if stripped[:1] == '"' and stripped[-1:] == '"':
        wrapped = _try_json_loads(stripped)
        if isinstance(wrapped, str):
            return wrapped

    text = clean_json_text(text)
    if not text:
        return None

    for candidate in (text, _TRAILING_COMMA_RE.sub(r"\1", text)):
        parsed = _try_json_loads(candidate)
        if parsed is not None:
            return parsed
        parsed = _try_literal_eval(candidate)
        if parsed is not None:
            return parsed
        parsed = _try_relaxed_json(candidate)
        if parsed is not None:
            return parsed
        parsed = _try_loose_parse(candidate)
        if parsed is not None:
            return parsed

    pairs = _KV_PAIR_RE.findall(text)
    if pairs:
        return {key: value for key, value in pairs}
    return None


def parse_content(value):
    """Normalise a ComfyUI input value into a queryable Python object.

    Strings are parsed as JSON-like text when possible (including JSON that
    was itself serialised into a string); every other value -- dict, list,
    number, bool, None, tensor -- is returned untouched.
    """
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        value = bytes(value).decode("utf-8", "replace")
    if not isinstance(value, str):
        return value

    parsed = parse_json_text(value)
    if parsed is None:
        return value
    if isinstance(parsed, str):
        stripped = parsed.strip()
        if stripped[:1] in "[{" and stripped[-1:] in "]}":
            inner = parse_json_text(stripped)
            if inner is not None and not isinstance(inner, str):
                return inner
    return parsed


def _try_json_loads(text: str):
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def _try_literal_eval(text: str):
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return None


def _try_relaxed_json(text: str):
    relaxed = _TRAILING_COMMA_RE.sub(r"\1", text)
    relaxed = _PY_LITERAL_RE.sub(
        lambda match: _PY_LITERAL_VALUE.get(match.group(1), match.group(1)), relaxed
    )
    if relaxed == text:
        return None
    try:
        return json.loads(relaxed)
    except (json.JSONDecodeError, ValueError):
        return None


def _try_loose_parse(text: str):
    stripped = text.strip()
    if not stripped:
        return None

    start_idx = None
    start_char = None
    for idx, char in enumerate(stripped):
        if char in "[{":
            start_idx = idx
            start_char = char
            break
    if start_idx is None or start_char is None:
        return None

    end_char = "}" if start_char == "{" else "]"
    end_idx = stripped.rfind(end_char)
    if end_idx == -1 or end_idx <= start_idx:
        return None

    container = stripped[start_idx : end_idx + 1]
    try:
        if start_char == "{":
            parsed, _ = _loose_parse_dict(container, 0)
            return parsed
        parsed, _ = _loose_parse_list(container, 0)
        return parsed
    except Exception:
        return None


def _loose_parse_dict(text: str, start: int):
    if start >= len(text) or text[start] != "{":
        raise ValueError("Expected '{'")

    idx = start + 1
    result = {}
    while idx < len(text):
        idx = _skip_ws(text, idx)
        if idx >= len(text):
            break
        if text[idx] == "}":
            return result, idx + 1
        if text[idx] == ",":
            idx += 1
            continue

        key, idx = _read_key(text, idx)
        idx = _skip_ws(text, idx)
        if idx < len(text) and text[idx] == ":":
            idx += 1
        idx = _skip_ws(text, idx)

        value, idx = _loose_parse_value(text, idx)
        if key:
            result[key] = value

        idx = _skip_ws(text, idx)
        if idx < len(text) and text[idx] == ",":
            idx += 1
    return result, idx


def _loose_parse_list(text: str, start: int):
    if start >= len(text) or text[start] != "[":
        raise ValueError("Expected '['")

    idx = start + 1
    result = []
    while idx < len(text):
        idx = _skip_ws(text, idx)
        if idx >= len(text):
            break
        if text[idx] == "]":
            return result, idx + 1
        if text[idx] == ",":
            idx += 1
            continue

        value, idx = _loose_parse_value(text, idx)
        result.append(value)

        idx = _skip_ws(text, idx)
        if idx < len(text) and text[idx] == ",":
            idx += 1
    return result, idx


def _loose_parse_value(text: str, start: int):
    idx = _skip_ws(text, start)
    if idx >= len(text):
        return None, idx

    char = text[idx]
    if char in ('"', "'"):
        return _read_loose_quoted_string(text, idx)
    if char == "{":
        return _loose_parse_dict(text, idx)
    if char == "[":
        return _loose_parse_list(text, idx)

    token, idx = _read_token(text, idx)
    lowered = token.lower()
    if lowered in {"null", "none"}:
        return None, idx
    if lowered == "true":
        return True, idx
    if lowered == "false":
        return False, idx

    if re.fullmatch(r"-?\d+", token):
        try:
            return int(token), idx
        except ValueError:
            return token, idx
    if re.fullmatch(r"-?\d+\.\d+", token):
        try:
            return float(token), idx
        except ValueError:
            return token, idx
    return token, idx


def _skip_ws(text: str, idx: int) -> int:
    while idx < len(text) and text[idx].isspace():
        idx += 1
    return idx


def _read_key(text: str, start: int):
    idx = start
    if idx >= len(text):
        return "", idx
    char = text[idx]
    if char in ('"', "'"):
        quote = char
        idx += 1
        buf = []
        while idx < len(text):
            current = text[idx]
            if current == "\\" and idx + 1 < len(text):
                buf.append(text[idx + 1])
                idx += 2
                continue
            if current == quote:
                idx += 1
                break
            buf.append(current)
            idx += 1
        return "".join(buf).strip(), idx

    buf = []
    while idx < len(text):
        current = text[idx]
        if current == ":" or current.isspace() or current in ",}":
            break
        buf.append(current)
        idx += 1
    return "".join(buf).strip(), idx


def _read_loose_quoted_string(text: str, start: int):
    quote = text[start]
    idx = start + 1
    buf = []
    while idx < len(text):
        current = text[idx]
        if current == "\\" and idx + 1 < len(text):
            buf.append(current)
            buf.append(text[idx + 1])
            idx += 2
            continue
        if current == quote:
            look = _skip_ws(text, idx + 1)
            if look >= len(text) or text[look] in ",}]":
                return _decode_loose_string("".join(buf), quote), idx + 1
            buf.append(current)
            idx += 1
            continue
        buf.append(current)
        idx += 1
    return _decode_loose_string("".join(buf), quote), idx


def _decode_loose_string(raw: str, quote: str) -> str:
    candidate = f"{quote}{raw}{quote}"
    try:
        return json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        pass
    try:
        return ast.literal_eval(candidate)
    except (ValueError, SyntaxError):
        return raw


def _read_token(text: str, start: int):
    idx = start
    buf = []
    while idx < len(text):
        current = text[idx]
        if current in ",}]":
            break
        buf.append(current)
        idx += 1
    return "".join(buf).strip(), idx


# ---------------------------------------------------------------------------
# Path parsing and lookup
# ---------------------------------------------------------------------------

_WILDCARD = object()
_MISSING = object()
_INT_TOKEN_RE = re.compile(r"[+-]?\d+")
_COUNT_TOKENS = {"count", "length", "size", "len"}


class _Key(str):
    """A quoted key: always a literal, never a wildcard or a count operator."""

    __slots__ = ()


def _is_count_token(token) -> bool:
    return type(token) is str and token.strip().lower() in _COUNT_TOKENS
_FULL_WIDTH = str.maketrans(
    {
        "\u3010": "[",
        "\u3011": "]",
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
        "\uff0c": ",",
        "\uff0e": ".",
        "\u3002": ".",
    }
)


def parse_path(path) -> list:
    """Tokenise a path such as [0].content into keys / indices / wildcards."""
    text = str(path or "").translate(_FULL_WIDTH)
    tokens = []
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char in " \t\r\n":
            index += 1
            continue
        if char in ".$":
            index += 1
            continue
        if char == "[":
            token, index = _read_bracket(text, index)
            if token is not None:
                tokens.append(token)
            continue
        if char in "\"'":
            token, index = _read_quoted(text, index)
            if token:
                tokens.append(token)
            continue

        start = index
        while index < length and text[index] not in ".[]":
            index += 1
        raw = text[start:index].strip()
        if raw:
            tokens.append(_WILDCARD if raw == "*" else raw)
    return tokens


def query(content, path, default=None):
    """Return the first value matched by path, or default when nothing matches."""
    matches = query_all(content, path)
    return matches[0] if matches else default


def is_count_path(path) -> bool:
    """True when the path ends with a count / length / size / [#] operator."""
    tokens = parse_path(path)
    return bool(tokens) and _is_count_token(tokens[-1])


def query_all(content, path) -> list:
    """Return every value matched by path; an empty path returns the whole content.

    A trailing count / length / size / [#] token counts instead of indexing:
    the matches of everything before it are counted when there are several,
    otherwise the matched value itself is measured (list, dict or string).
    An unquoted key that really is named count still wins, so a payload with
    a count field keeps working; quote it ("count") to force key access.
    """
    data = parse_content(content)
    tokens = parse_path(path)
    if tokens and _is_count_token(tokens[-1]):
        return _count(data, tokens[:-1], tokens[-1])
    return _walk(data, tokens)


def _count(data, tokens: list, token) -> list:
    subject = _walk(data, tokens)
    if len(subject) == 1:
        node = subject[0]
        if isinstance(node, Mapping) and token in node:
            return [node[token]]
        if isinstance(node, Mapping):
            return [len(node)]
        if isinstance(node, Sequence) and not isinstance(
            node, (str, bytes, bytearray)
        ):
            return [len(node)]
        if isinstance(node, str):
            return [len(node)]
        return [1]
    return [len(subject)]


def _walk(node, tokens: list) -> list:
    if not tokens:
        return [node]

    head, rest = tokens[0], tokens[1:]
    if head is _WILDCARD:
        matches = []
        for child in _iter_children(node):
            matches.extend(_walk(child, rest))
        return matches

    child = _pick(node, head)
    if child is _MISSING:
        return []
    return _walk(child, rest)


def _iter_children(node):
    if isinstance(node, Mapping):
        return list(node.values())
    if isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
        return list(node)
    return []


def _pick(node, token):
    if isinstance(node, Mapping):
        if token in node:
            return node[token]
        text = str(token)
        if text in node:
            return node[text]
        return _MISSING

    if isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
        index = _as_index(token)
        if index is None:
            return _MISSING
        try:
            return node[index]
        except IndexError:
            return _MISSING

    return _MISSING


def _as_index(token):
    if isinstance(token, bool):
        return None
    if isinstance(token, int):
        return token
    if isinstance(token, str) and _INT_TOKEN_RE.fullmatch(token.strip()):
        try:
            return int(token.strip())
        except ValueError:
            return None
    return None


def _read_bracket(text: str, start: int):
    index = start + 1
    length = len(text)
    while index < length and text[index] in " \t\r\n":
        index += 1

    if index < length and text[index] in "\"'":
        token, index = _read_quoted(text, index)
        while index < length and text[index] in " \t\r\n":
            index += 1
        if index < length and text[index] == "]":
            index += 1
        return (token or None), index

    raw_start = index
    while index < length and text[index] != "]":
        index += 1
    raw = text[raw_start:index].strip()
    if index < length:
        index += 1

    if not raw or raw == "*":
        return _WILDCARD, index
    if raw == "#":
        return "count", index
    if _INT_TOKEN_RE.fullmatch(raw):
        return int(raw), index
    return raw, index


def _read_quoted(text: str, start: int):
    quote = text[start]
    index = start + 1
    length = len(text)
    buf = []
    while index < length:
        char = text[index]
        if char == "\\" and index + 1 < length:
            buf.append(char)
            buf.append(text[index + 1])
            index += 2
            continue
        if char == quote:
            index += 1
            break
        buf.append(char)
        index += 1
    return _Key(_decode_loose_string("".join(buf), quote)), index


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_value(value, style: str = "auto") -> str:
    """Render a queried value for a STRING port.

    style is one of:
      auto  -- strings stay raw, containers become compact JSON
      json  -- always JSON (objects/lists keep their structure)
      text  -- Python str() of the value
      csv   -- flat lists joined with ", "
    """
    style = str(style or "auto").strip().lower()
    if style == "json":
        return _to_json(value)
    if style == "text":
        return _to_text(value)
    if style == "csv":
        if isinstance(value, (list, tuple)) and value and all(
            not isinstance(item, (Mapping, list, tuple)) for item in value
        ):
            return ", ".join(_to_text(item) for item in value)
        if isinstance(value, (Mapping, list, tuple)):
            return _to_json(value)
        return _to_text(value)

    if isinstance(value, str):
        return value
    if isinstance(value, (Mapping, list, tuple)):
        return _to_json(value)
    return _to_text(value)


def _to_json(value) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return _to_text(value)


def _to_text(value) -> str:
    if value is None:
        return ""
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return str(value)
        if value.is_integer() and abs(value) < 1e16:
            return str(int(value))
        return repr(value)
    return str(value)
