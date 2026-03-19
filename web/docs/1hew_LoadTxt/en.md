# Load Txt - Load text files

**Node Purpose:** `Load Txt` reads `.txt` content from a file path or a folder and outputs the text plus the filename stem. It supports folder scan, optional recursive subfolders, `encode` selection, and indexed file selection via `index`.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `file` | - | STRING | `""` | - | Text file path or folder path. Supports absolute paths and also resolves relative paths against ComfyUI input/output/temp directories. |
| `encode` | - | COMBO | `auto` | `auto` / `utf-8` / `utf-8-sig` / `gbk` / `utf-16` | Text read option; `auto` tries common text formats in sequence. |
| `index` | - | INT | `0` | -8192-8192 | Text file index when `file` is a folder; supports negative indices via modulo selection. |
| `include_subdir` | - | BOOLEAN | `false` | - | Include subfolders when `file` points to a directory. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `text` | STRING | Loaded text content. |
| `filename` | STRING | Filename stem of the selected text file. |

## Features

- File or folder input: reads a single `.txt` file or selects from a directory.
- Encoding compatibility: `auto` mode tries `utf-8`, `utf-8-sig`, `gbk`, and `utf-16`.
- Stable ordering: folder entries are sorted case-insensitively for more predictable indexing.
- Indexed selection: use `index` to pick one file from a folder; negative indices wrap by modulo.
- UI preview: shows the loaded text directly in the node panel.

## Typical Usage

- Read prompt templates from a folder by index for batched workflows.
- Load one config text file and feed the `text` output into prompt or text-processing nodes.

## Notes & Tips

- Only `.txt` files are matched currently.
- The node raises an error when the path does not exist or no valid text files are found.
