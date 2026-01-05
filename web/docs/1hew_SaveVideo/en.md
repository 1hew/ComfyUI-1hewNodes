# Save Video - Save a VIDEO object to disk

**Node Purpose:** `Save Video` writes a VIDEO object to disk and returns the saved path. It preserves the source container extension when available and provides an alpha-friendly preview for UI display.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `video` | optional | VIDEO | - | - | Video to save; omitted input yields a passthrough behavior. |
| `filename_prefix` | - | STRING | `video/ComfyUI` | - | Output filename prefix; supports formatting placeholders (e.g. `%date:yyyy-MM-dd%`). |
| `save_output` | - | BOOLEAN | `true` | - | Save into output directory (true) or temp directory (false). |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `file_path` | STRING | Absolute path of the saved video file. |

## Features

- Container-aware saving: uses the source file extension when the VIDEO input exposes a path.
- Metadata embedding: attaches prompt/extra metadata when ComfyUI metadata is enabled.
- Alpha preview workflow: detects alpha via ffprobe and generates a VP9 WebM preview for UI playback.
- Safe naming: uses ComfyUI save path allocation with counters to avoid collisions.

## Typical Usage

- Save a selected or processed VIDEO object and use the returned path for external tools.

## Notes & Tips

- Ensure `ffprobe` and `ffmpeg` are available for alpha detection and preview generation.

