# Save Video by Image - Encode an image batch into a video

**Node Purpose:** `Save Video by Image` encodes an IMAGE batch into a video file with a specified FPS. It supports optional audio muxing and alpha-aware output strategies.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `image` | - | IMAGE | - | - | Image batch used as frames. |
| `audio` | optional | AUDIO | - | - | Optional audio to mux into the output. |
| `fps` | - | FLOAT | `8.0` | 0.01-120 | Frames per second for encoding. |
| `filename_prefix` | - | STRING | `video/ComfyUI` | - | Output filename prefix; follows ComfyUI save path rules. |
| `save_output` | - | BOOLEAN | `true` | - | Save into output directory (true) or temp directory (false). |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `file_path` | STRING | Absolute path of the saved video file. |

## Features

- FFmpeg raw-video encoding: streams frames to ffmpeg for encoding.
- Audio support: saves provided audio to a temp WAV and muxes it during encoding.
- Alpha-aware behavior:
  - RGBA + `save_output=true`: saves a `.mov` (ProRes) to output and emits a `.webm` preview in temp.
  - RGBA + `save_output=false`: saves `.webm` with alpha to temp.
  - RGB: saves `.mp4` (H.264) to output or temp.
- Even-dimension handling: automatically resizes to even width/height for common codecs.

## Typical Usage

- Convert a processed frame batch into a previewable video and keep the output path for later steps.

## Notes & Tips

- Ensure `ffmpeg` is available in your environment for encoding.

