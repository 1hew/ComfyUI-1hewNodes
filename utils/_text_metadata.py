"""Workflow metadata embedded at the end of a Save Text .txt/.md file.

A plain text file has no metadata chunks, so Save Text appends a small
HTML-comment block after the payload. The block is deliberately
ASCII-only (JSON escaped with ensure_ascii=True) so it can still be found
when reading older non-UTF-8 files, even if the payload itself is
mojibake. Load Text strips the block before returning the text.

Keep METADATA_BEGIN / METADATA_END in sync with
web/js/core/text_metadata.js.

This is a shared helper, not a node: it lives in utils/ next to
_index_range.py, because every file under nodes/<category>/ must define
exactly one ComfyNode.
"""

from __future__ import annotations

import json

METADATA_BEGIN = "<!-- 1hew_nodes_txt_metadata_begin"
METADATA_END = "1hew_nodes_txt_metadata_end -->"
METADATA_VERSION = 1


def build_metadata_block(text: str, metadata: dict | None) -> str:
    """Append workflow/prompt metadata to text.

    Returns text unchanged when there is nothing to embed, so a plain
    text save never gains an empty trailing comment block.
    """
    if not metadata:
        return text

    payload = {"version": METADATA_VERSION, "text_length": len(text)}
    payload.update(metadata)
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
            default=str,
        )
    except (TypeError, ValueError):
        return text

    return f"{text}\n\n{METADATA_BEGIN}\n{encoded}\n{METADATA_END}\n"


def _locate_block(text: str) -> tuple[int, dict] | None:
    end = text.rfind(METADATA_END)
    if end < 0:
        return None

    # Walk candidate begin markers backwards. Workflow values may contain
    # the marker text themselves, so the last begin marker is not
    # necessarily the real block start; the first candidate whose body is
    # valid JSON wins.
    search_end = len(text)
    while True:
        begin = text.rfind(METADATA_BEGIN, 0, search_end)
        if begin < 0 or begin > end:
            return None

        body_start = text.find("\n", begin)
        if body_start < 0 or body_start > end:
            search_end = begin
            continue

        raw = text[body_start + 1 : end].strip()
        if raw:
            try:
                data = json.loads(raw)
            except ValueError:
                data = None
            if isinstance(data, dict):
                return begin, data

        search_end = begin


def parse_metadata(text: str) -> dict:
    """Return the embedded metadata dict, or {} when there is none."""
    located = _locate_block(text)
    return located[1] if located else {}


def strip_metadata(text: str) -> str:
    """Remove the trailing metadata block, restoring the original text.

    The text_length field in the block is the authoritative boundary, so
    payload text that itself ends with blank lines is preserved exactly.
    When a user edited the file and the length no longer matches, fall
    back to cutting at the marker and trimming separator blank lines.
    """
    located = _locate_block(text)
    if located is None:
        return text

    begin, data = located
    prefix = text[:begin]
    length = data.get("text_length")
    if (
        isinstance(length, int)
        and not isinstance(length, bool)
        and 0 <= length <= len(prefix)
        and prefix[length:] in ("", "\n", "\n\n", "\r\n", "\r\n\r\n")
    ):
        return prefix[:length]

    return prefix.rstrip("\r\n")
