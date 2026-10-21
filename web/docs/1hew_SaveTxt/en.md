# Save Text - Save text to disk

**Node Purpose:** `Save Text` writes the input text to a file in the selected format (`txt` / `csv` / `md` / `json`) and returns the saved absolute path. It supports relative or absolute paths, auto-increment naming, overwrite mode, `format` selection, and preview-only mode without writing to disk. Files are always written as UTF-8, matching ComfyUI's native Save Text.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `text` | optional | STRING | - | - | Text to save. When unconnected it is `None`, and the in-node preview is cleared. |
| `filename` | - | STRING | `text/ComfyUI` | - | Save prefix or absolute path. Relative paths are written under the ComfyUI output directory. |
| `format` | - | COMBO | `txt` | `txt` / `csv` / `md` / `json` | Output file format. `json` is pretty-printed when the payload is valid JSON; csv/json skip metadata. |
| `auto_increment` | - | BOOLEAN | `true` | - | When `true`, creates a new numbered file; when `false`, uses a fixed filename and overwrites it. |
| `save_output` | - | BOOLEAN | `true` | - | When `true`, writes to disk; when `false`, only previews the text in UI without saving. |
| `save_metadata` | - | BOOLEAN | `true` | - | When `true`, appends workflow/prompt metadata to text/Markdown files so dragging the file into ComfyUI restores the workflow; skipped for csv/json. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `file_path` | STRING | Absolute saved file path; returns an empty string when `save_output=false`. |

## Features

- Text input: accepts a STRING input; when unconnected it is treated as `None` and the in-node preview is cleared.
- Path flexibility: supports both relative and absolute paths; relative paths default to the output directory.
- Auto numbering: with `auto_increment=true`, generates names like `_00001_` to avoid overwriting existing files.
- Overwrite mode: with `auto_increment=false`, writes to a fixed `filename.<format>`.
- Format support: `format` accepts `txt` / `csv` / `md` / `json` (default `txt`); the extension follows the selected format.
- JSON formatting: `json` parses and pretty-prints the payload when valid; otherwise the text is written as-is.
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
- `format` now sits directly below `filename`. `encode` has been removed (fixed UTF-8); when an older workflow is loaded, the front-end automatically migrates the legacy `widgets_values` order and defaults missing `format` to `txt`, so no manual reconfiguration is needed.
- The input port was renamed from `any` to `text` (STRING); rewire existing workflows to `text`.
