# Load Video - Load a video object from file or folder

**Node Purpose:** `Load Video` selects a video file and returns a VIDEO object that can be consumed by downstream nodes. It supports folder scanning, indexed selection, and frame settings applied during decoding.

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
| `video` | VIDEO | Video object that decodes frames/audio with the given settings. |

## Features

- File or folder input: accepts a single video file or a directory of videos.
- Indexed selection: stable path sorting plus modulo selection for robust indexing.
- Frame settings: start/end skip, FPS resampling, format constraint, then frame limit.
- Alpha-aware decoding: prefers RGBA when the source stream indicates alpha support.
- Audio passthrough: decodes an audio stream when present and returns it as part of the video object.

## Typical Usage

- Select a clip from a folder and route the VIDEO output into `Save Video`.
- Apply frame trimming and FPS normalization before converting to images.

## Notes & Tips

- When the video container fails to decode via PyAV, decoding falls back to ffmpeg for frames.

