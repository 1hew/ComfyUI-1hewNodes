# Save Video RGBA - Save RGB/RGBA batches with alpha-safe codecs

**Node Purpose:** `Save Video RGBA` saves an image batch as a video with support for both RGB and RGBA inputs. It ensures dimensions are divisible by 2, selects alpha-safe containers/codecs, and returns a preview entry. When alpha is present, it produces a `webm` preview and writes a production `mov` (ProRes 4444) to the output folder.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `images` | - | IMAGE | - | - | Image batch `B×H×W×C` (`C=3`/`4`). |
| `fps` | - | FLOAT | `24.0` | `1.0–120.0` (step `1.0`) | Output frame rate. |
| `filename_prefix` | - | STRING | `video/ComfyUI` | - | Save path prefix; supports formatting placeholders handled by the framework. |
| `only_preview` | - | BOOLEAN | `False` | - | Save to temporary preview only; skips regular output entry when enabled. |
| `audio` | optional | AUDIO | - | - | Audio track to mux; uses `libopus` for `webm` and `aac` for `mp4/mov`. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| - | - | Output node; returns a UI preview element with saved result info. |

## Features

- Divisible-by-2 resize: when width or height is not divisible by 2, resizes frames down using LANCZOS to satisfy codec requirements.
- Alpha-aware formats: with alpha, preview uses `webm` + `libvpx-vp9` and production output uses `mov` + `prores_ks` (ProRes 4444, `yuva444p10le`). Without alpha, defaults to `mp4` + `h264`.
- Pixel formats: selects `yuva420p` for VP9, `yuva444p10le` for ProRes with alpha, and `yuv420p` for RGB-only outputs.
- Path and naming: computes folder, base name, and counter via `get_save_image_path`; final filename pattern `{base}_{counter:05}_.{ext}`.
- Async save: performs encoding on a worker thread and returns `PreviewVideo` with saved entries.
- Audio mux: resamples and crops audio to match video duration; uses `mono`/`stereo` layout based on channels.

## Typical Usage

- Preserve transparency: feed RGBA batches; preview uses `webm` (alpha), and the output folder contains a `mov` suitable for professional workflows.
- Preview-only exports: enable `only_preview` to generate a temporary video without adding a regular output entry.
- Deterministic naming: rely on the auto-incrementing counter to organize repeated runs under `filename_prefix`.

## Notes & Tips

- Alpha workflow: preview favors `webm` for broad browser compatibility; production output uses `mov` with ProRes 4444 for high-fidelity alpha.
- Audio handling: when provided, audio is cropped to match the video frame count at the specified `fps`.
- Layout and range: expects channels-last tensors; frames are converted to 8-bit per channel during encoding with values clamped to `[0,1]` in preprocessing.