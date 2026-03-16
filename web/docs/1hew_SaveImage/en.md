# Save Image - Save image batches to disk

**Node Purpose:** `Save Image` writes an IMAGE batch to the ComfyUI output or temp directory and returns the saved absolute file paths. It supports configurable filename prefixes and optional prompt/workflow metadata embedding.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `image` | optional | IMAGE | - | - | Image batch to save; when absent, the node finishes with an empty output. |
| `filename` | - | STRING | `image/ComfyUI` | - | Save prefix or absolute path; typically supports date placeholders (e.g. `%date:yyyy-MM-dd%`). |
| `auto_increment` | - | BOOLEAN | `true` | - | When true, uses auto-incrementing numbers; when false, uses fixed filename and overwrites. |
| `save_output` | - | BOOLEAN | `true` | - | Save to output directory when enabled; skip saving to disk when disabled. |
| `save_metadata` | - | BOOLEAN | `true` | - | Embed prompt/workflow metadata when enabled. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `file_path` | STRING | Absolute paths joined by newline, one per saved image. |

## Features

- Output routing: choose output directory or temp directory via `save_output`.
- Metadata control: toggle embedding of `prompt` and `extra_pnginfo` via `save_metadata`.
- Stable naming: uses ComfyUI path allocation to generate unique filenames.
- UI preview: shows a Saved Images gallery for the saved results.

## Typical Usage

- Save final results to the output directory while keeping intermediate steps in temp.
- Set `save_metadata=false` when exporting images for external pipelines.

## Notes & Tips

- The node returns absolute paths; multiple images are separated by newline.
- The node serializes filename allocation to keep counters consistent under
  concurrent execution.
