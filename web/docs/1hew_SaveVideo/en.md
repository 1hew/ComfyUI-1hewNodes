# Save Video - Save a video to disk with optional metadata

**Node Purpose:** `Save Video` saves a VIDEO input into the ComfyUI output or temp directory and returns the saved absolute file path. It optionally remuxes prompt/workflow metadata into the container `comment` field and can generate a preview WEBM when the source contains an alpha channel.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `video` | optional | VIDEO | - | - | Video to save; when absent, the node finishes with an empty output. |
| `filename_prefix` | - | STRING | `video/ComfyUI` | - | Save prefix passed to ComfyUI path generation; typically supports date placeholders (e.g. `%date:yyyy-MM-dd%`). |
| `save_output` | - | BOOLEAN | `true` | - | Save to output directory when enabled; save to the temp directory when disabled. |
| `save_metadata` | - | BOOLEAN | `true` | - | Write prompt/workflow metadata into the container `comment` field when enabled. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `file_path` | STRING | Absolute file path of the saved video. |

## Features

- Output routing: choose output directory or temp directory via `save_output`.
- Extension inference: uses the source file extension when available.
- Metadata remux: uses `ffmpeg` to remux metadata into `comment` while keeping
  streams copied.
- Alpha preview: detects alpha via `ffprobe` and generates a WEBM preview in the
  temp directory for UI playback.
- UI preview: shows a Preview Video entry for quick inspection.

## Typical Usage

- Save upstream decoded/processed videos to output for archiving.
- Enable metadata to keep prompts attached to exported results.
- Use alpha preview to validate transparency workflows in the UI.

## Notes & Tips

- The node calls `ffprobe` for alpha/audio probing and calls `ffmpeg` for remux
  and optional preview generation.
- Path allocation runs under a lock to keep counters stable under concurrency.
