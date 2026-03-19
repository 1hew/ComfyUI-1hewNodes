# Save Txt - Save text to disk

**Node Purpose:** `Save Txt` converts any input into text, writes it to a `.txt` file, and returns the saved absolute path. It supports relative or absolute paths, auto-increment naming, overwrite mode, `encode` selection, and preview-only mode without writing to disk.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `any` | - | ANY (`*`) | - | - | Input content to save; automatically converted to string. Dict/list values are formatted as JSON text. |
| `filename` | - | STRING | `txt/ComfyUI` | - | Save prefix or absolute path. Relative paths are written under the ComfyUI output directory. |
| `auto_increment` | - | BOOLEAN | `true` | - | When `true`, creates a new numbered file; when `false`, uses a fixed filename and overwrites it. |
| `save_output` | - | BOOLEAN | `true` | - | When `true`, writes to disk; when `false`, only previews the text in UI without saving. |
| `encode` | - | COMBO | `utf-8` | `utf-8` / `utf-8-sig` / `gbk` | Encoding used when writing the text file. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `file_path` | STRING | Absolute saved file path; returns an empty string when `save_output=false`. |

## Features

- Generic input: accepts any input type and converts it to text; structured data is formatted as readable JSON.
- Path flexibility: supports both relative and absolute paths; relative paths default to the output directory.
- Auto numbering: with `auto_increment=true`, generates names like `_00001_` to avoid overwriting existing files.
- Overwrite mode: with `auto_increment=false`, writes to a fixed `filename.txt`.
- Preview mode: with `save_output=false`, skips disk output but still shows the text in the node UI.

## Typical Usage

- Save prompts, logs, or JSON results as text files for external tools or later inspection.
- Disable `save_output` during debugging to inspect the final text without writing files.

## Notes & Tips

- In absolute-path mode, make sure the target directory is writable.
- During concurrent saves, the node serializes path allocation to reduce filename collisions.
