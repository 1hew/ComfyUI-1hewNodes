"""System prompt registry shared by the ``Text System Prompt`` node.

Prompt bodies live as plain `` .txt `` files under ``utils/system_prompt/``:

    utils/system_prompt/qwen_image_2.1/[en] edit.txt
    utils/system_prompt/qwen_image_2.1/[en] t2i.txt

A registry key is the file path relative to ``utils/system_prompt`` with the
``.txt`` suffix removed, so the two files above become
``qwen_image_2.1/[en] edit`` and ``qwen_image_2.1/[en] t2i``. The language tag is
part of the file name on purpose: one file holds exactly one language, so an
English and a Chinese variant of the same prompt are two files.

The key list is built once at import time (ComfyUI startup) and cached. Changing
only a body does not need a restart: :func:`load_system_prompt` re-reads a file
when its mtime changes. Adding or removing a file changes the dropdown, which
only a reload picks up.

Only self-contained prompts belong in the tree: a body must make sense on its
own, with no caption injected by another node and no extraction wrapper. The
bundled qwen_image_2.1 bodies are adapted from
benjiyaya/ComfyUI-Qwen-Image-2.1-Prompt-Enhancer (MIT): their JSON output
contracts were replaced by plain-text ones and the upstream size section was
dropped, so they answer in prose that can be wired straight into an encoder.
"""

from __future__ import annotations

import logging
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent / "system_prompt"
PROMPT_SUFFIX = ".txt"

_LOG_PREFIX = "[1hewNodes][TextSystemPrompt]"

# key -> file path, built once at import time
_REGISTRY: dict[str, Path] | None = None
# key -> (mtime_ns, body) so an edited body is picked up without a restart
_CACHE: dict[str, tuple[int, str]] = {}


def _iter_prompt_paths():
    """Yield every prompt file, skipping `_`/`.` prefixed names and folders."""
    if not PROMPT_DIR.is_dir():
        return
    for file_path in sorted(PROMPT_DIR.rglob(f"*{PROMPT_SUFFIX}")):
        relative = file_path.relative_to(PROMPT_DIR)
        if any(part.startswith(("_", ".")) for part in relative.parts):
            continue
        yield file_path


def _build_registry() -> dict[str, Path]:
    registry: dict[str, Path] = {}
    for file_path in _iter_prompt_paths():
        key = file_path.relative_to(PROMPT_DIR).with_suffix("").as_posix()
        registry[key] = file_path
    return registry


def system_prompt_registry() -> dict[str, Path]:
    """Return the ``key -> path`` mapping built at startup."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _build_registry()
    return _REGISTRY


def list_system_prompts() -> list[str]:
    """Every registry key, sorted, for the node's dropdown."""
    return list(system_prompt_registry())


def load_system_prompt(key: str) -> str:
    """Return the body of ``key`` verbatim; missing or unreadable keys give ``""``."""
    file_path = system_prompt_registry().get(key)
    if file_path is None:
        logging.warning("%s unknown prompt %r", _LOG_PREFIX, key)
        return ""

    try:
        mtime = file_path.stat().st_mtime_ns
    except OSError as exc:
        logging.warning("%s cannot stat %s: %s", _LOG_PREFIX, key, exc)
        return ""

    cached = _CACHE.get(key)
    if cached is not None and cached[0] == mtime:
        return cached[1]

    try:
        body = file_path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        logging.warning("%s cannot read %s: %s", _LOG_PREFIX, key, exc)
        return ""

    _CACHE[key] = (mtime, body)
    return body


def reload_system_prompts() -> list[str]:
    """Rebuild the registry and drop cached bodies (tests / dev only)."""
    global _REGISTRY
    _REGISTRY = _build_registry()
    _CACHE.clear()
    return list(_REGISTRY)
