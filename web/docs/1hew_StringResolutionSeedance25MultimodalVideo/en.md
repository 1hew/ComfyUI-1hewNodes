# String Resolution Seedance 2.5 Multimodal Video - Seedance 2.5 Video Resolution Selector

**Node Purpose:** `String Resolution Seedance 2.5 Multimodal Video` outputs the nearest Seedance 2.5 resolution tier label (`480p` / `720p` / `1080p` / `2k` / `4k`) based on the input video's pixel area. If no video is connected, it passes through the manually selected label. It is designed to pair with `RunningHub Seedance 2.5 Multimodal Video` in ComfyUI-1hewOffice (`nodes/api/runninghub/runninghub_seedance_25_multimodal_video.py`).

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `selection` | - | COMBO | `480p` | `480p` / `720p` / `1080p` / `2k` / `4k` | Manual resolution label used when `video` is not connected. |
| `video` | optional | VIDEO | - | - | Input video used to infer the nearest resolution tier. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `string` | STRING | Resolution label string, ready to connect to an API node's `resolution` input. |

## Tiers and target areas

| Tier | 16:9 standard size | Target area (pixels) |
| ---- | ------------------ | -------------------- |
| `480p` | `854x480` | 409,920 |
| `720p` | `1280x720` | 921,600 |
| `1080p` | `1920x1080` | 2,073,600 |
| `2k` | `2560x1440` | 3,686,400 |
| `4k` | `3840x2160` | 8,294,400 |

## Features

- Tiers: `480p` / `720p` / `1080p` / `2k` / `4k`, matching the usual `resolution` tiers of the Seedance 2.5 multimodal video node (`native1080p` is not part of the inference).
- Area-based matching: compares the video's total pixel area against the tier target areas and picks the nearest in log space, consistent with the image `String Resolution` nodes in this family.
- Portrait-safe: matching is by area, not by a single side, so a `1080x1920` video is still classified as `1080p`.
- Manual fallback: returns the selected label directly when no video is connected or its size cannot be read.

## Typical Usage

- Feed the input video to both this node and the Seedance 2.5 multimodal video node's reference-video input, then connect this node's output to `resolution` so the generated resolution tracks the source.
- Leave `video` unconnected to use it as a fixed resolution string selector.

## Notes & Tips

- Matching is based on total pixel area, not exact width/height ratio.
- Target areas use each tier's 16:9 standard size; `2k` is based on `2560x1440`.
- This node only outputs the string label: it does not resize the video and performs no network requests.
