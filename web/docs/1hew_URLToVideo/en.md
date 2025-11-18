# URL to Video - Download video from URL

**Node Purpose:** `URL to Video` downloads a video from an HTTP(S) URL and outputs a video object for downstream processing.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `video_url` | - | STRING | - | http(s) | Direct URL to a video resource (must start with `http://` or `https://`). |
| `timeout` | - | INT | 30 | 5-300 | Download timeout in seconds. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `video` | VIDEO | Video object constructed from downloaded bytes. |

## Features

- Streaming download: fetches content in chunks with a desktop browser user-agent.
- Error handling: logs failures and raises exceptions for invalid URLs or download issues.
- Compatibility: returns a `VideoFromFile` object suitable for downstream nodes.

## Typical Usage

- Provide a direct link to a video file hosted over HTTP(S).

## Notes & Tips

- Ensure the URL points to downloadable video content (not an HTML page or DRM stream).