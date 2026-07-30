# Video to Image - Extract image and video metadata

**Node Purpose:** `Video to Image` extracts image frames, audio, FPS, and frame count from a ComfyUI `VIDEO` object.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `video` | - | VIDEO | - | - | Input ComfyUI video object. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Image frames from the video as an image batch. |
| `audio` | AUDIO | Audio data from the video, when available. |
| `fps` | FLOAT | Frame rate of the video. |
| `frame_count` | INT | Number of image frames in the output batch. |

## Features

- Extracts image frames without decoding the video again.
- Preserves the image batch, audio data, and frame rate provided by the video object.
- Reports the number of frames in the image output.
- Returns empty outputs when no video or valid frames are available.

## Typical Usage

- Connect `Load Video` to `Video to Image` to process video frames with image nodes.
- Connect `URL to Video` to `Video to Image` to extract frames from a downloaded video.
- Use `fps` and `frame_count` for timing and batch-size calculations.

## Notes & Tips

- The input must be a ComfyUI `VIDEO` object with a `get_components()` method.
- The node does not change the frame rate, frame selection, or image format.
- Audio can be unavailable for videos without an audio stream.
