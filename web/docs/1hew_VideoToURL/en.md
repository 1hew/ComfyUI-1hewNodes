# Video to URL - Convert video to URL

**Node Purpose:** `Video to URL` uploads a ComfyUI `VIDEO` object to `kefan.cn` and returns a public URL string that can be passed to downstream nodes or external APIs.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `video` | - | VIDEO | - | - | Input video; optional, output is an empty string when not connected |
| `timeout` | - | INT | 30 | 5-300 | Upload timeout in seconds |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `url` | STRING | Public URL of the uploaded video; empty string when no video is connected |

## Features

- Uploads the video to `kefan.cn` and returns the resulting public URL.
- Reuses upload cache: repeated uploads of the same video content reuse the cached URL mapping when available.
- Optional input: when `video` is not connected, the node returns an empty string instead of raising an error.
- Raises an error when the upload fails (for example, due to network issues or service unavailability).

## Typical Usage

- Convert a loaded or generated `VIDEO` object into a public URL for downstream services that only accept public video links.
- Chain with `Load Video` or `Save Video` workflows to hand off videos to external APIs.

## Notes & Tips

- Uploading depends on external network access and third-party service availability.
- Video files are usually much larger than images, so uploading may take longer; the `timeout` control lets you adjust the limit.
