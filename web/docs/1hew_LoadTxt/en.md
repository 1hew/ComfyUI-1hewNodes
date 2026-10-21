# Load Text - Load text files

**Node Purpose:** `Load Text` reads `.txt` content from a file path or a folder and outputs the text plus the filename stem. It supports folder scan, optional recursive subfolders, indexed file selection via `index`, and a `choose file to upload` button on the node.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `file` | - | STRING | `""` | - | Text file path or folder path. Supports absolute paths and also resolves relative paths against ComfyUI input/output/temp directories. |
| `index` | - | INT | `0` | -8192-8192 | Text file index when `file` is a folder; supports negative indices via modulo selection. |
| `include_subdir` | - | BOOLEAN | `false` | - | Include subfolders when `file` points to a directory. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `text` | STRING | Loaded text content. |
| `filename` | STRING | Filename stem of the selected text file. |

## Features

- File or folder input: reads a single `.txt` file or selects from a directory.
- File upload: the `choose file to upload` button opens a file picker, uploads the `.txt` to ComfyUI's `input/text`, and fills in the `file` widget automatically.
- Encoding: automatically detects `utf-8-sig` / `utf-16` / `gbk`; no manual encoding selection is needed.
- Stable ordering: folder entries are sorted case-insensitively for more predictable indexing.
- Indexed selection: use `index` to pick one file from a folder; negative indices wrap by modulo.
- Metadata stripping: when reading a Save Text text/Markdown file that carries metadata, the trailing workflow/prompt comment block is removed and the original payload is returned.
- UI preview: selecting/uploading a file, changing `index`, or loading a workflow immediately reads and shows the text in the bottom panel without queueing the workflow; supports selection, scrolling, and copy.

## Typical Usage

- Read prompt templates from a folder by index for batched workflows.
- Load one config text file and feed the `text` output into prompt or text-processing nodes.

## Notes & Tips

- Only `.txt` files are matched currently; `choose file to upload` also accepts `.txt` only.
- Uploaded files are stored under `ComfyUI/input/text/`.
- The metadata block written by `Save Text` only affects the file on disk; the `text` output of `Load Text` never includes it.
- The node raises an error when the path does not exist or no valid text files are found.
- The old `encode` widget is removed from the UI; loading an older workflow migrates the widget order automatically.
