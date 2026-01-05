# Load Video to Image - Decode video into frames and audio

**Node Purpose:** `Load Video to Image` decodes a selected video into an image batch, audio, and FPS metadata. It supports folder scanning, indexed selection, and the same frame settings pipeline used by `Load Video`.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `path` | - | STRING | `""` | - | Video file path or folder path. |
| `frame_limit` | - | INT | `0` | 0-100000 | Maximum output frame count; `0` keeps all frames after other settings. |
| `fps` | - | FLOAT | `0.0` | 0-120 | Target FPS used for resampling; `0` keeps source FPS. |
| `start_skip` | - | INT | `0` | 0-100000 | Number of frames to skip from the start. |
| `end_skip` | - | INT | `0` | 0-100000 | Number of frames to skip from the end. |
| `format` | - | COMBO | `4n+1` | `n` / `2n+1` / `4n+1` / `6n+1` / `8n+1` | Frame-count constraint pattern applied after resampling. |
| `video_index` | - | INT | `0` | -8192-8192 | Selection index when `path` is a folder; supports negative indices via modulo selection. |
| `include_subdir` | - | BOOLEAN | `false` | - | Include subfolders when scanning a folder. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Decoded frame batch. |
| `audio` | AUDIO | Decoded audio data when available. |
| `fps` | FLOAT | Output FPS after resampling rules. |
| `frame_count` | INT | Number of frames in `image`. |

## Features

- Direct decode: outputs image frames and audio for immediate processing.
- Consistent settings: applies skip, FPS resampling, format constraint, and frame limit.
- Alpha-aware decode: prefers RGBA when the source stream indicates alpha support.
- Folder workflows: stable sorting plus indexing enables dataset-style access to clips.

## Typical Usage

- Convert a video into a frame batch for image-based pipelines, then encode back with `Save Video by Image`.
- Extract audio for separate processing while using frames for visual workflows.

## Notes & Tips

- Output `fps` reflects the effective frame rate after resampling configuration.

