# Save Text - Save text to disk

**Node Purpose:** `Save Text` writes the input to a file in the selected format (`txt` / `csv` / `md` / `json`) and returns the saved absolute path. The wildcard `text` input accepts strings as well as JSON-safe dicts, lists, numbers and booleans. It supports relative or absolute paths, auto-increment naming, overwrite mode, `format` selection, and preview-only mode without writing to disk. Files are always written as UTF-8, matching ComfyUI's native Save Text.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `text` | optional | * | - | - | Content to save: `str`, `bytes`, `dict`, `list`, `int`, `float`, `bool` or `None`. Containers are written as JSON text; `csv` stays text-only. When unconnected it is `None`, and the in-node preview is cleared. |
| `filename` | - | STRING | `text/ComfyUI` | - | Save prefix or absolute path. Relative paths are written under the ComfyUI output directory. |
| `format` | - | COMBO | `txt` | `txt` / `csv` / `md` / `json` | Output file format. `json` is pretty-printed when the payload is valid JSON; csv/json skip metadata. |
| `auto_increment` | - | BOOLEAN | `true` | - | When `true`, creates a new numbered file; when `false`, uses a fixed filename and overwrites it. |
| `save_output` | - | BOOLEAN | `true` | - | When `true`, writes to disk; when `false`, only previews the text in UI without saving. |
| `save_metadata` | - | BOOLEAN | `true` | - | When `true`, appends workflow/prompt metadata to text/Markdown files so dragging the file into ComfyUI restores the workflow; skipped for csv/json. |

## Input Type and Format Policy

| Input \ format | `txt` / `md` | `json` | `csv` |
| --- | --- | --- | --- |
| `str` | As-is | Parsed and pretty-printed when valid; otherwise as-is | As-is |
| `bytes` | UTF-8 decoded, then treated as `str` | Same | Same |
| `dict` / `list` / `tuple` | Pretty JSON text | Pretty JSON text | Error: csv is text-only |
| `int` / `float` | Plain text | JSON number (`NaN` / `Infinity` rejected) | Plain text |
| `bool` | `True` / `False` | `true` / `false` | `True` / `False` |
| `None` (unconnected) | Clear preview, empty output | Same | Same |
| tensor / audio / other | Error | Error | Error |

Object keys must be strings. A JSON document may have any JSON value as its root (object, array, string, number, boolean). Many downstream APIs expect an object or array root; wrap scalars with `Dict` when needed. Python-only values such as `set`, `bytes` inside a container, tensors, `NaN` and `Infinity` raise path-aware errors and are never silently stringified. Top-level `None` still means unconnected.

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `file_path` | STRING | Absolute saved file path; returns an empty string when `save_output=false`. |

## Features

- Wildcard input: accepts strings, dicts, lists, numbers, booleans and bytes; when unconnected it is treated as `None` and the in-node preview is cleared.
- Path flexibility: supports both relative and absolute paths; relative paths default to the output directory.
- Auto numbering: with `auto_increment=true`, generates names like `_00001_` to avoid overwriting existing files.
- Overwrite mode: with `auto_increment=false`, writes to a fixed `filename.<format>`.
- Format support: `format` accepts `txt` / `csv` / `md` / `json` (default `txt`); the extension follows the selected format.
- JSON formatting: `json` pretty-prints string payloads when valid and always serializes dict/list payloads as indented JSON; Python-only values raise an error.
- Preview mode: with `save_output=false`, skips disk output but still shows the text in the node UI.
- Workflow metadata: with `save_metadata=true`, an HTML-comment metadata block (UI workflow and API prompt) is appended after the text/Markdown payload. Drag the generated `.txt` onto the ComfyUI canvas to load that workflow, just like dragging a metadata-rich PNG or video. Metadata is skipped for csv/json to keep those files valid.
- Encoding: always UTF-8 (matching native Save Text); the `encode` input has been removed. The metadata block is ASCII-only (non-ASCII values are \uXXXX-escaped), so it remains readable to the front end.
- In-node preview: after execution the text is shown in a read-only panel at the bottom of the node with selection and scrolling; when there is no input the preview is cleared instead of keeping the previous content.

## Typical Usage

- Save prompts, logs, or JSON results as text files for external tools or later inspection.
- Save a prompt/result text file with metadata, then drag it back onto the canvas to quickly restore the workflow that produced it.
- Disable `save_output` during debugging to inspect the final text without writing files.

## Notes & Tips

- In absolute-path mode, make sure the target directory is writable.
- During concurrent saves, the node serializes path allocation to reduce filename collisions.
- With `save_metadata` enabled, text/Markdown files gain a trailing metadata comment block. Disable it when the file must stay pure text for external scripts or LLMs; `Load Text` strips the block automatically and still returns the original payload. Metadata is skipped for csv/json.
- Text file drag-and-drop restore relies on this pack's front-end script. Without the front-end extension loaded, the stock ComfyUI frontend does not read text metadata.
- The node ID is `1hew_SaveText` (renamed from `1hew_SaveTxt` in v4.5.0; the display name, ports, defaults and formats are unchanged). The extension registers a ComfyUI node replacement, so a workflow saved with the old ID can be swapped from the missing-node panel (links and widget values carry over), and API prompts that still send `1hew_SaveTxt` are rewritten before execution. Builds without node-replacement support need the node re-added manually.
- `format` sits directly below `filename`; `encode` has been removed (fixed UTF-8). Workflows saved before v4.0.0 may still carry the old `encode` widget value, so after replacing an old node recheck `auto_increment` / `save_output` / `save_metadata`.
- The input port was renamed from `any` to `text`; it is now a wildcard (`*`) input so dicts and lists can be wired directly. Old STRING links remain valid because a wildcard input accepts every type.
