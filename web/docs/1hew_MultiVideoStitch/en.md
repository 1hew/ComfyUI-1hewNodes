# Multi Video Stitch - Stitch Videos End to End

**Node Purpose:** Multi Video Stitch concatenates multiple videos end to end into a single video. Every segment is first aligned to one target size with crop, pad or stretch, so videos with different resolutions can be stitched together. The target size comes from the optional get_video_size reference video, or from the first connected segment when it is not connected.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| fps | - | STRING | video_1 | video_N / vN / a number | Output frame rate. video_1 (default) reuses the first segment frame rate; video_2 / v2 / v_2 target that segment; a number like 8 or 29.5 sets an explicit rate. |
| fit | - | COMBO | pad | crop / pad / stretch | Size alignment mode. |
| pad_color | - | STRING | 1.0 | color name/HEX/RGB/edge/average/extend/mirror | Padding color or strategy used by pad. |
| video_1 | optional | VIDEO | - | - | First segment; extra segments arrive through the dynamic video_2...video_N ports. |
| get_video_size | optional | VIDEO | - | - | Reference video; its width/height become the target size. Pinned below every video port. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| video | VIDEO | Single video with all segments concatenated in video_X order. |

## Features

- Dynamic segments: collects video_* inputs by numeric suffix so video_1, video_2, ... keep their order.
- Target size: get_video_size wins when connected, otherwise the first valid segment defines the size.
- Fit modes: stretch resizes directly; crop scales to cover then center-crops; pad scales to fit then pads with pad_color.
- Padding strategies: explicit colors plus extend, mirror, edge and average.
- Frame rate: video_1 (default) follows the first segment; video_N / vN / v_N reuse another segment's rate; a numeric fps is used directly and every segment is resampled to it while keeping its duration.
- Alpha: if any segment has alpha, all segments are promoted to RGBA and the result is written as WebM/VP9 so the bundled encoder can carry alpha where it is supported; otherwise MP4/H.264.
- Audio: segments with audio are resampled and concatenated, missing tracks become silence so the timeline stays aligned; a video without any audio keeps no audio track.

## Typical Usage

- Join clips with different resolutions or aspect ratios into one continuous video.
- Normalize a mixed set of clips before saving or feeding downstream video nodes.

## Notes & Tips

- get_video_size only provides the target size; it is never stitched into the output.
- When fps references a segment (video_1 ...) and segments disagree, each segment keeps its own duration.
- The result is encoded to a temporary MP4/WebM file and returned as a video object.
