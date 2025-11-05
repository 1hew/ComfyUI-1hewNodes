# Image Batch Extract

**Node Function:** The `Image Batch Extract` node is used to extract specific images from a batch of images, supporting multiple extraction modes: custom indices, step intervals, and automatic frame count calculation, commonly used for video frame extraction and image sequence processing.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
|--|--|--|--|--|--|
| `image` | Required | IMAGE | - | - | Input batch of images |
| `mode` | - | COMBO[STRING] | step | index, step, uniform | Extraction mode: index (custom indices), step (step interval), uniform (uniform distribution) |
| `index` | - | STRING | "0" | Custom indices | Custom indices string, supports comma-separated index list like "0,2,5,10" |
| `step` | - | INT | 4 | 1-8192 | Step interval, extract one image every N frames |
| `uniform` | - | INT | 4 | 0-8192 | Uniform distribution count, extract specified number of images evenly from total frames |
| `max_keep` | - | INT | 10 | 0-8192 | Maximum keep count, limits the final output image count |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `image` | IMAGE | Extracted image batch |
