# Save Video by Image - Encode an image batch into a video

**Node Purpose:** `Save Video by Image` encodes an IMAGE batch as video frames and optionally muxes an AUDIO input. It returns the saved absolute file path and provides a UI preview entry. For alpha sequences, it uses WEBM (VP9) for preview, and can also export a high-quality MOV (ProRes) when saving to the output directory.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `image` | - | IMAGE | - | - | Image batch interpreted as frames in time order. |
| `audio` | optional | AUDIO | - | - | Optional audio input; muxed into the output when provided. |
| `fps` | - | FLOAT | `8.0` | 0.01-120.0 | Frame rate for encoding. |
| `filename_prefix` | - | STRING | `video/ComfyUI` | - | Save prefix passed to ComfyUI path generation; typically supports date placeholders (e.g. `%date:yyyy-MM-dd%`). |
| `save_output` | - | BOOLEAN | `true` | - | Save to output directory when enabled; save to the temp directory when disabled. |
| `save_metadata` | - | BOOLEAN | `true` | - | Write prompt/workflow metadata into the container `comment` field when enabled. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `file_path` | STRING | Absolute file path of the saved video. |

## Features

- Frame encoding: streams raw frames to `ffmpeg` via stdin for encoding.
- Audio muxing: converts the AUDIO input to a temporary WAV and muxes it into
  the output.
- Even-size alignment: ensures width/height are even for stable encoding.
- Alpha-aware outputs:
  - When frames include alpha: exports WEBM (VP9 + yuva420p) by default.
  - When frames include alpha and `save_output=true`: exports a preview WEBM to
    temp and a MOV (ProRes 4444) to output.
- Metadata embedding: writes prompt/workflow JSON into the `comment` field.
- UI preview: shows a Preview Video entry, using the preview file when present.

## Typical Usage

- Encode a generated frame batch into an MP4 for sharing.
- Encode an RGBA sequence with transparency; use preview WEBM for playback and
  MOV (ProRes) for high-quality export.
- Attach an audio track to the video output for final delivery.

## Notes & Tips

- The node calls `ffmpeg` for encoding and muxing.
- When `audio` is provided, an intermediate WAV is created in temp and removed
  after encoding completes.
