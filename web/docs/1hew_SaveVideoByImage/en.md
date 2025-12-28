# Save Video by Image - Encode image batches into video

**Node Purpose:** `Save Video by Image` encodes an IMAGE batch into a video file at the given FPS. It supports optional AUDIO muxing, alpha-aware codec/container selection, and returns a preview in the UI.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `image` | - | IMAGE | - | - | Input image batch `[B,H,W,C]` in 0–1 range. |
| `audio` | optional | AUDIO | - | - | Optional audio dict containing `waveform` and `sample_rate`. |
| `fps` | - | FLOAT | 8.0 | 0.01–120.0 | Frames per second for the output video. |
| `filename_prefix` | - | STRING | `video/ComfyUI` | - | Save path prefix under output or temp directory. |
| `save_output` | - | BOOLEAN | True | - | When True, saves into the output directory; when False, saves into the temp directory. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| - | - | Output node; returns a UI preview of the saved video. |

## Features

- Dimension alignment: auto-resizes to even width/height for video encoding compatibility.
- Alpha handling:
  - RGBA inputs (`C=4`) use alpha-capable encoding.
  - When `save_output=True` with RGBA, the node saves a `.mov` for final output and a `.webm` for UI preview.
- Audio muxing: when `audio` is provided, the node writes a temporary WAV and muxes it into the final container.
- FFmpeg piping: streams raw frames to FFmpeg via stdin for efficient encoding.

## Typical Usage

- Export a frame sequence as MP4: connect an IMAGE batch, set `fps`, keep `save_output=True`.
- Export an RGBA sequence: use RGBA images, set `save_output=True` to generate a `.mov` plus a `.webm` preview.
- Add audio: connect an AUDIO input from upstream nodes, and the node muxes audio into the result.

## Notes & Tips

- Ensure `ffmpeg` is available on the system PATH.
- Large batches increase encoding time; use a suitable FPS and batch length for workflow needs.
