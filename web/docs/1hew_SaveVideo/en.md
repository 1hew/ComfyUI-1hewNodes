# Save Video - Save VideoInput to disk

**Node Purpose:** `Save Video` saves a `VideoInput` into the output or temp directory and returns a UI preview. It supports optional input and can pass through when the input is empty.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `video` | optional | VIDEO | - | - | Video to save; when absent, the node completes with no output. |
| `filename_prefix` | - | STRING | `video/ComfyUI` | - | Save prefix; supports format placeholders (e.g., `%date:yyyy-MM-dd%`). |
| `save_output` | - | BOOLEAN | True | - | When True, saves into the output directory; when False, saves into the temp directory. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| - | - | Output node; returns a UI preview of the saved video. |

## Features

- Optional input handling: `video` can be omitted for conditional workflows.
- Auto container selection: uses `format=auto` and `codec=auto` to pick a suitable output format.
- Metadata embedding: when enabled in ComfyUI, writes prompt and extra PNG info as metadata.

## Typical Usage

- Save an upstream decoded video: connect a VIDEO output into `video` and set `filename_prefix`.
- Conditional save: drive `video` with an optional branch so the node saves only when a valid video is provided.

## Notes & Tips

- `save_output=False` is useful for preview-oriented workflows that keep artifacts in the temp directory.
