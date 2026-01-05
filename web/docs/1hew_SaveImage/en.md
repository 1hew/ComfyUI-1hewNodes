# Save Image - Save images to output or temp

**Node Purpose:** `Save Image` writes an IMAGE batch to disk and emits the absolute file paths. It supports saving into the ComfyUI output folder or the temp folder and provides UI previews.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `image` | optional | IMAGE | - | - | Image batch to save. |
| `filename_prefix` | - | STRING | `image/ComfyUI` | - | Output filename prefix; follows ComfyUI save path rules. |
| `save_output` | - | BOOLEAN | `true` | - | Save into output directory (true) or temp directory (false). |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `file_path` | STRING | Absolute paths joined by newlines for batch outputs. |

## Features

- Batch saving: saves every image in the input batch.
- ComfyUI-compatible paths: uses the same naming and folder rules as other save nodes.
- Preview UI: emits saved image previews for quick inspection.

## Typical Usage

- Save intermediate or final images while keeping the exact saved paths for downstream file-based nodes.

## Notes & Tips

- `filename_prefix` supports ComfyUI’s built-in formatting placeholders.

